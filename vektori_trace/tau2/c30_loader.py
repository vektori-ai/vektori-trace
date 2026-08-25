"""Load the frozen C30 replay-prefix pool for Tau2 ReOPD.

One replay state needs three things that live in three different files, and the
whole point of this module is to join them without letting them drift apart:

    c30_prefix_manifest.json   which prefixes, in what order, and their hashes
    rows.tokenized.jsonl       Qwen `input_ids` -> the exact student prompt
    rows.semantic.jsonl        the chat messages -> what DeepSeek re-renders

Why both renderings are needed
------------------------------
The student and the teacher do not share a tokenizer, and neither one can be
derived from the other. The student is sampled from `prompt_token_ids` taken
verbatim from the frozen corpus, because re-rendering at training time would
silently retokenize under whatever tokenizer happens to be installed. The
teacher scores under *its own* template, so it needs the semantic messages;
handing it Qwen ids would be meaningless.

`tau2_swr_canary.py` reads only the tokenized side, which is correct for a
sampling-only canary. Scoring needs the semantic side too, and that join is
what this module adds.

Why C30Prefix is its own type
-----------------------------
C30's frozen identity is `task_id#position`, assigned by the manifest and
hashed into it. The shared replay dataclass derives its identity from other
fields instead, so adopting it would mean synthesising values to reproduce the
frozen string -- feeding invented data to the very checks that exist to catch
substitution. This type carries `prefix_id` explicitly and satisfies the
structural contract the loss path reads (`prefix_id`, `step_index`, `task`).

What it refuses
---------------
Every check here has a matching silent failure:

- a manifest whose hash is not the expected one -- a different prefix pool is
  not comparable to the branch that trained against the frozen one;
- corpus files whose bytes moved under the split -- the rows would be a
  different experiment wearing the same manifest;
- a per-row `semantic_hash` mismatch between the two corpus files -- that is
  precisely a mis-join, the failure mode this module exists to prevent;
- a row whose labels are entirely masked, or whose prompt boundary is absent --
  there is no action to score and no prefix to sample from.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

IGNORE = -100

#: The partition this loader will read. Named, not parameterised: S16 and F38
#: are evaluation partitions whose contents must never reach an optimizer, and
#: W30 already trained `A_warm`.
PARTITION = "C30"


class C30LoadError(RuntimeError):
    """A frozen artifact is missing, moved, or does not join."""


@dataclass(frozen=True)
class C30Prefix:
    """One frozen replay state, with both renderings joined.

    `step_index` mirrors `position` so this satisfies the structural contract
    `replay_opd` reads while keeping C30's own frozen identity.
    """

    prefix_id: str                       # "42#5" -- frozen, not derived
    task_id: str
    #: The selected trace's hash from `eligibility_report.json`. Real
    #: provenance, never synthesised: the archive records it as the
    #: reproducibility key, and an invented value would make it claim a lineage
    #: that cannot be reproduced.
    trace_id: str
    position: int
    action_type: str
    tool_names: list[str]
    semantic_hash: str

    #: Qwen ids for the prompt only: `input_ids` up to the first supervised
    #: label. This is what the student is sampled from, verbatim.
    prompt_token_ids: list[int]

    #: The chat history DeepSeek re-renders under its own template, **including
    #: the system policy**. `rows.semantic.jsonl` stores only `decision.prompt`,
    #: but the tokenized row the student samples from was built as
    #: `[system] + prompt + [target]` with the retail tool schemas passed to the
    #: template (`export.build_row`). Handing the teacher the bare prompt would
    #: score the student's action under a strictly smaller context than the
    #: student saw -- a finite loss computed against the wrong conditioning,
    #: with nothing in any metric to show for it.
    canonical_messages: list[dict[str, Any]]

    #: The tool schemas the student's prompt was rendered with. The teacher
    #: needs them to render an equivalent state; passing them separately rather
    #: than folding them into the messages is how both providers' templates
    #: expect to receive tools.
    tools: list[dict[str, Any]]

    #: The recorded DeepSeek action at this state. ReOPD does not train on it
    #: -- that is `A_sft_new`'s signal -- but `replay_opd` accepts stored
    #: teacher actions for its identical-action diagnostics, and reporting the
    #: overlap between sampled and recorded actions is worth having.
    stored_teacher_action: dict[str, Any]

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def step_index(self) -> int:
        """Position within the trace. `replay_opd` reads this for reporting."""
        return self.position

    @property
    def task(self) -> str:
        """`replay_archive` reads `.task`; C30's task identity is `task_id`."""
        return self.task_id

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_ids_from_row(row: dict[str, Any]) -> list[int]:
    """`input_ids` up to the first supervised token.

    A corpus row is the whole training sequence -- prompt followed by the
    recorded action -- with `labels` masked to -100 across the prompt. The
    first non-masked label is therefore the action's first token, and
    everything before it is exactly what the student should be prompted with.

    Deriving the boundary from the labels rather than re-rendering the messages
    is what makes this byte-exact: it cannot disagree with the corpus, because
    it is read out of the corpus.
    """
    labels = row["labels"]
    start = next((i for i, l in enumerate(labels) if l != IGNORE), None)
    if start is None:
        raise C30LoadError(
            f"{row['task_id']}#{row['position']}: every label is masked, so the "
            "row carries no supervised action and no prompt boundary"
        )
    if start == 0:
        raise C30LoadError(
            f"{row['task_id']}#{row['position']}: the first token is supervised, "
            "so there is no prompt to sample from"
        )
    return [int(t) for t in row["input_ids"][:start]]


def _verify_corpus_bytes(artifacts: str) -> dict[str, str]:
    """Fail unless every corpus file is byte-identical to the frozen hash.

    Per-row `semantic_hash` values would all still match a corpus rebuilt under
    a different tokenizer or template -- they cover the semantic row, not the
    ids. Only the file hash catches that.
    """
    hash_path = os.path.join(artifacts, "artifact_hashes.json")
    if not os.path.exists(hash_path):
        raise C30LoadError(f"missing {hash_path}")
    frozen = json.load(open(hash_path))
    for fn, want in frozen.items():
        path = os.path.join(artifacts, fn)
        if not os.path.exists(path):
            raise C30LoadError(f"frozen artifact {fn} is missing from {artifacts}")
        got = sha256_file(path)
        if got != want:
            raise C30LoadError(
                f"{fn} hash {got[:16]} != frozen {want[:16]}; the corpus moved "
                "under the split, so these rows are a different experiment"
            )
    return frozen


def _eligibility_report(artifacts: str) -> dict[str, Any]:
    """The corpus's own record of which trace it selected per task.

    Written by `tau2_build_corpus` as `eligibility_report.json`, keyed
    `per_task[task_id]` -- not `eligibility.json`/`selected_traces`, which do
    not exist. Reading the wrong name is not a gradient bug, but it makes every
    archived `trace_id` fall back to the task id, so the archive claims a
    provenance it never read.
    """
    path = os.path.join(artifacts, "eligibility_report.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def selected_trace_hashes(artifacts: str) -> dict[str, str]:
    """`task_id -> trace_hash` for the trace each task's rows were built from."""
    per_task = (_eligibility_report(artifacts).get("per_task") or {})
    return {str(t): rec["trace_hash"]
            for t, rec in per_task.items()
            if isinstance(rec, dict) and rec.get("trace_hash")}


def selected_source_files(artifacts: str) -> dict[str, str]:
    """`task_id -> source simulation file` for the selected trace.

    The system policy is not stored in the artifacts directory; it came from
    `simulation["info"]["environment_info"]["policy"]` at corpus build. These
    are the files to recover it from, named by the corpus itself rather than
    guessed.
    """
    per_task = (_eligibility_report(artifacts).get("per_task") or {})
    return {str(t): rec["source_file"]
            for t, rec in per_task.items()
            if isinstance(rec, dict) and rec.get("source_file")}


def _policy_from_simulation_file(path: str) -> str | None:
    """The environment policy, read the way `tau2_build_corpus` reads it.

    The builder takes it from the **file's** top-level info block:

        data["info"]["environment_info"]["policy"]

    Not from inside each simulation record -- an earlier draft here invented
    that nesting level and failed on every real file. The per-simulation lookup
    is kept only as a fallback for files that carry it there instead, and never
    as the primary path.
    """
    data = json.load(open(path))

    info = (data.get("info") or {}) if isinstance(data, dict) else {}
    policy = ((info.get("environment_info") or {}).get("policy"))
    if policy:
        return policy

    sims = data.get("simulations") if isinstance(data, dict) else data
    for sim in (sims or []):
        if not isinstance(sim, dict):
            continue
        env = (sim.get("info") or {}).get("environment_info") or {}
        if env.get("policy"):
            return env["policy"]
    return None


def recover_system_policy(
    artifacts: str,
    *,
    simulations_dir: str | None = None,
    task_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Recover the system policy from the simulations the corpus selected.

    Every selected trace must carry an identical policy string. If two differ,
    the corpus was built across a policy revision and there is no single
    context to score under -- that must stop the run rather than be resolved by
    picking one.

    Returns `(policy, report)` where the report carries the policy's sha256 for
    the run manifest. Recovery is not proof: only `assert_render_parity`
    establishes that this string is the one the frozen ids were rendered from.
    """
    sources = selected_source_files(artifacts)
    if not sources:
        raise C30LoadError(
            f"no per_task source files in {artifacts}/eligibility_report.json; "
            "cannot recover the policy the corpus was built with"
        )
    if task_ids:
        keep = {str(t) for t in task_ids}
        sources = {t: f for t, f in sources.items() if t in keep}

    seen: dict[str, list[str]] = {}
    for task_id, fn in sorted(sources.items()):
        path = fn if os.path.isabs(fn) else os.path.join(
            simulations_dir or os.path.dirname(fn) or ".", os.path.basename(fn)
        )
        if not os.path.exists(path):
            raise C30LoadError(
                f"simulation file for task {task_id} not found: {path}. Pass "
                "--simulations-dir pointing at the directory the corpus was "
                "built from."
            )
        policy = _policy_from_simulation_file(path)
        if not policy:
            raise C30LoadError(
                f"no environment policy in {path}. `tau2_build_corpus` reads it "
                'at `data["info"]["environment_info"]["policy"]`; this file has '
                "neither that nor a per-simulation copy."
            )
        seen.setdefault(policy, []).append(task_id)

    if len(seen) > 1:
        sizes = {hashlib.sha256(p.encode()).hexdigest()[:12]: len(t)
                 for p, t in seen.items()}
        raise C30LoadError(
            f"selected traces carry {len(seen)} different policies {sizes}. The "
            "corpus spans a policy revision, so there is no single context to "
            "score under."
        )

    policy = next(iter(seen))
    return policy, {
        "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "policy_chars": len(policy),
        "n_tasks_agreeing": len(sources),
        "n_source_files": len({os.path.basename(f) for f in sources.values()}),
    }


def load_policy_and_tools(
    artifacts: str,
    *,
    system_policy: str,
    tools: list[dict[str, Any]] | None = None,
    domain: str = "retail",
) -> tuple[str, list[dict[str, Any]]]:
    """The system policy and tool schemas the corpus was rendered with.

    Neither is stored in the artifacts directory: the policy came from the
    simulation files and the schemas from the live Tau2 registry
    (`tau2_build_corpus`). Both must therefore be supplied or re-derived, and
    then *verified* against what the corpus recorded -- an unverified
    reconstruction is how the teacher ends up scoring under a different policy
    revision than the student was trained on, which no downstream check would
    notice.

    `system_policy` has no recorded hash to check against in the current
    corpus metadata, so it is the caller's responsibility to pass the same
    string `tau2_build_corpus` used. The tool schemas *are* hashed, and a
    mismatch here is fatal.
    """
    from .tools import load_domain_tools, tools_hash

    if not system_policy or not system_policy.strip():
        raise C30LoadError(
            "system_policy is required: the student's prompt ids were rendered "
            "with it, so scoring without it grades the action under a strictly "
            "smaller context than the student saw"
        )

    schemas = list(tools) if tools is not None else load_domain_tools(domain)

    # Both files record it; `eligibility_report.json` is the name the corpus
    # builder actually writes. Reading a name that does not exist made this
    # check silently pass for every real corpus.
    recorded = None
    for cand in ("eligibility_report.json", "data_census.json"):
        path = os.path.join(artifacts, cand)
        if os.path.exists(path):
            recorded = json.load(open(path)).get("tools_hash") or recorded
    if recorded:
        got = tools_hash(schemas)
        if got != recorded:
            raise C30LoadError(
                f"tool schema hash {got} != corpus-recorded {recorded}. The "
                "student's prompts were rendered against a different tool set; "
                "scoring against this one compares two different states."
            )
    return system_policy, schemas


def load_c30_prefixes(
    artifacts: str,
    *,
    system_policy: str,
    tools: list[dict[str, Any]] | None = None,
    domain: str = "retail",
    trace_ids: dict[str, str] | None = None,
    manifest_path: str | None = None,
    expect_manifest_hash: str | None = None,
) -> tuple[list[C30Prefix], dict[str, Any]]:
    """Join the frozen manifest with both corpus renderings.

    Returns the prefixes **in the manifest's frozen sampling order**, not in
    corpus order. The order is part of what was frozen (V2 §8, task-first /
    position-second); a branch that re-derives it independently breaks the
    match with its sibling as surely as a different row set would.
    """
    man_path = manifest_path or os.path.join(artifacts, "c30_prefix_manifest.json")
    rows_path = os.path.join(artifacts, "rows.tokenized.jsonl")
    sem_path = os.path.join(artifacts, "rows.semantic.jsonl")
    for p in (man_path, rows_path, sem_path):
        if not os.path.exists(p):
            raise C30LoadError(f"missing {p}")

    manifest = json.load(open(man_path))
    got_hash = manifest.get("prefix_manifest_hash")
    if expect_manifest_hash and got_hash != expect_manifest_hash:
        raise C30LoadError(
            f"prefix manifest hash {got_hash} != expected {expect_manifest_hash}; "
            "this is a different prefix pool and its results are not comparable "
            "to anything trained against the frozen one"
        )
    if manifest.get("partition") != PARTITION:
        raise C30LoadError(
            f"manifest partition is {manifest.get('partition')!r}, not {PARTITION!r}"
        )

    _verify_corpus_bytes(artifacts)
    policy, schemas = load_policy_and_tools(
        artifacts, system_policy=system_policy, tools=tools, domain=domain
    )

    # Real trace provenance, from the frozen eligibility record when it is
    # present. Absent, `trace_id` falls back to the task id rather than to a
    # fabricated identifier -- an archive that records a synthetic trace hash
    # claims a lineage that cannot be reproduced.
    traces = dict(trace_ids or {})
    if not traces:
        traces = selected_trace_hashes(artifacts)

    want = {p["prefix_id"]: p for p in manifest["prefixes"]}

    tokenized: dict[str, dict] = {}
    for line in open(rows_path):
        r = json.loads(line)
        key = f"{r['task_id']}#{r['position']}"
        if key in want:
            tokenized[key] = r

    semantic: dict[str, dict] = {}
    for line in open(sem_path):
        r = json.loads(line)
        key = f"{r['task_id']}#{r['position']}"
        if key in want:
            semantic[key] = r

    missing_tok = sorted(set(want) - set(tokenized))
    missing_sem = sorted(set(want) - set(semantic))
    if missing_tok:
        raise C30LoadError(
            f"{len(missing_tok)} frozen prefixes absent from rows.tokenized.jsonl: "
            f"{missing_tok[:4]}"
        )
    if missing_sem:
        raise C30LoadError(
            f"{len(missing_sem)} frozen prefixes absent from rows.semantic.jsonl: "
            f"{missing_sem[:4]}"
        )

    prefixes: list[C30Prefix] = []
    for pid in manifest["sampling_order"]:
        w, tok, sem = want[pid], tokenized[pid], semantic[pid]

        # Three-way hash agreement. The manifest froze one hash; both corpus
        # files carry their own. A mis-join -- the tokenized row of one prefix
        # paired with the messages of another -- is invisible in every
        # downstream metric and produces a finite loss, so it is checked here
        # rather than trusted.
        for name, row in (("tokenized", tok), ("semantic", sem)):
            if row["semantic_hash"] != w["semantic_hash"]:
                raise C30LoadError(
                    f"{pid}: {name} semantic_hash {row['semantic_hash'][:16]} != "
                    f"frozen {w['semantic_hash'][:16]}"
                )

        if len(tok["input_ids"]) != len(tok["labels"]):
            raise C30LoadError(f"{pid}: input_ids/labels length mismatch")
        if len(tok["input_ids"]) != w["n_tokens"]:
            raise C30LoadError(
                f"{pid}: {len(tok['input_ids'])} tokens != frozen {w['n_tokens']}"
            )

        messages = sem.get("prompt")
        if not messages:
            raise C30LoadError(f"{pid}: semantic row has an empty prompt")
        target = sem.get("target")
        if not target:
            raise C30LoadError(f"{pid}: semantic row has no target action")

        # The system policy is prepended here, not stored in the semantic row.
        # `export.build_row` rendered the student's ids from
        # `[system] + prompt + [target]`; reconstructing the same head is what
        # makes the teacher's context equivalent rather than merely similar.
        if messages[0].get("role") == "system":
            raise C30LoadError(
                f"{pid}: semantic prompt already begins with a system message; "
                "prepending the policy would duplicate it"
            )
        # Tools ride on the system message, not as a separate argument.
        # `render_teacher_prefix` takes only (messages, thinking_mode), and
        # `encoding_dsv4` reads `msg["tools"]` off the system/developer turn
        # (`_render_system`, line ~352) converting it with
        # `tools_from_openai_format` -- the same OpenAI shape
        # `load_domain_tools` returns. Carrying `tools` on the prefix object
        # alone would leave DeepSeek unconditioned on them while the student's
        # ids contain the full schema block.
        canonical = [{"role": "system", "content": policy, "tools": schemas}]
        canonical += list(messages)

        prefixes.append(C30Prefix(
            prefix_id=pid,
            task_id=w["task_id"],
            trace_id=traces.get(str(w["task_id"]), str(w["task_id"])),
            position=int(w["position"]),
            action_type=w["action_type"],
            tool_names=list(w.get("tool_names") or []),
            semantic_hash=w["semantic_hash"],
            prompt_token_ids=prompt_ids_from_row(tok),
            canonical_messages=canonical,
            tools=schemas,
            stored_teacher_action=target,
            meta={
                "n_supervised_tokens": w["n_supervised_tokens"],
                "message_index": sem.get("message_index"),
            },
        ))

    if len(prefixes) != manifest["n_prefixes"]:
        raise C30LoadError(
            f"joined {len(prefixes)} prefixes but the manifest froze "
            f"{manifest['n_prefixes']}"
        )

    report = {
        "prefix_manifest_hash": got_hash,
        "split_manifest_hash": manifest.get("split_manifest_hash"),
        "partition": PARTITION,
        "n_prefixes": len(prefixes),
        "n_tasks": len({p.task_id for p in prefixes}),
        "seed": manifest.get("seed"),
        "prompt_tokens": {
            "min": min(p.n_prompt_tokens for p in prefixes),
            "max": max(p.n_prompt_tokens for p in prefixes),
            "total": sum(p.n_prompt_tokens for p in prefixes),
        },
    }
    return prefixes, report


def assert_render_parity(
    prefixes: list[C30Prefix],
    tokenizer: Any,
    *,
    max_length: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Prove the reconstructed context re-renders to the frozen prompt ids.

    This is the check that makes the policy string *proven* rather than merely
    plausible. `load_policy_and_tools` verifies the tool hash, but nothing
    verifies the policy text -- and a policy that is close but not identical
    produces a valid-looking prefix whose every downstream assertion passes.

    Re-rendering `[system+tools] + prompt` under the pinned Qwen tokenizer must
    reproduce `prompt_token_ids` exactly. If it does for all 289, the
    reconstruction is the corpus's own, byte for byte.

    Run this before the first paid call, not during training: it costs one
    tokenizer render per prefix and answers a question that cannot be answered
    later from any log.
    """
    from ..dataset import tokenize_messages

    checked = mismatched = 0
    failures: list[dict[str, Any]] = []
    for p in prefixes[:limit] if limit else prefixes:
        messages = list(p.canonical_messages) + [p.stored_teacher_action]
        supervise = [False] * (len(messages) - 1) + [True]
        ex = tokenize_messages(
            messages, tokenizer, supervise,
            max_length=max_length, truncate=False,
            template_kwargs={"tools": p.tools, "enable_thinking": True},
            mask_think_wrapper=True,
        )
        checked += 1
        if ex is None:
            failures.append({"prefix_id": p.prefix_id, "why": "over length"})
            mismatched += 1
            continue
        n_target = sum(1 for l in ex.labels if l != -100)
        got = list(ex.input_ids[: len(ex.input_ids) - n_target])
        if got != p.prompt_token_ids:
            mismatched += 1
            if len(failures) < 5:
                first = next(
                    (i for i, (a, b) in enumerate(zip(got, p.prompt_token_ids))
                     if a != b),
                    min(len(got), p.n_prompt_tokens),
                )
                failures.append({
                    "prefix_id": p.prefix_id,
                    "n_rendered": len(got),
                    "n_frozen": p.n_prompt_tokens,
                    "first_divergence": first,
                })

    if mismatched:
        raise C30LoadError(
            f"render parity failed for {mismatched}/{checked} prefixes: "
            f"{failures}. The reconstructed system policy or tool schema is not "
            "what the corpus was built with, so the teacher would score under "
            "different conditioning than the student sampled under."
        )
    return {"n_checked": checked, "all_exact": True}


def cycle_updates(
    prefixes: list[C30Prefix], *, n_per_update: int, n_updates: int
) -> list[list[C30Prefix]]:
    """Chunk the frozen order into fixed-size updates, wrapping at the end.

    The manifest's `sampling_order` is already a task-first round-robin, so
    consecutive slices are task-balanced by construction and no reshuffling is
    needed or wanted -- reshuffling here would discard the property the freeze
    exists to guarantee.

    Wrapping continues from where the previous pass ended rather than
    restarting, so a pool that does not divide evenly still gives every prefix
    equal exposure across the run instead of over-weighting its head.
    """
    if n_per_update <= 0 or n_updates <= 0:
        raise C30LoadError("n_per_update and n_updates must both be > 0")
    if not prefixes:
        raise C30LoadError("no prefixes to cycle")

    batches: list[list[C30Prefix]] = []
    i = 0
    for _ in range(n_updates):
        batch = [prefixes[(i + k) % len(prefixes)] for k in range(n_per_update)]
        batches.append(batch)
        i = (i + n_per_update) % len(prefixes)
    return batches


__all__ = [
    "C30LoadError",
    "C30Prefix",
    "PARTITION",
    "assert_render_parity",
    "cycle_updates",
    "load_c30_prefixes",
    "load_policy_and_tools",
    "prompt_ids_from_row",
]
