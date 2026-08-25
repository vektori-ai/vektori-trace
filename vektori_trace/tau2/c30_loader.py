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

Why not `ReplayPrefix`
----------------------
`replay_select.ReplayPrefix` computes its identity as `f"{trace_id}@{step_index}"`
from a Harbor trace id. C30's frozen identity is `task_id#position`. Faking a
`trace_id` to reproduce that string would put a fabricated value into the
manifest checks that exist to catch exactly this kind of substitution. So this
is a separate type that carries `prefix_id` explicitly and satisfies the same
structural contract the downstream reads (`prefix_id`, `step_index`).

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
    `replay_opd` reads without inheriting Harbor's trace-derived identity.
    """

    prefix_id: str                       # "42#5" -- frozen, not derived
    task_id: str
    position: int
    action_type: str
    tool_names: list[str]
    semantic_hash: str

    #: Qwen ids for the prompt only: `input_ids` up to the first supervised
    #: label. This is what the student is sampled from, verbatim.
    prompt_token_ids: list[int]

    #: The chat history DeepSeek re-renders under its own template.
    canonical_messages: list[dict[str, Any]]

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


def load_c30_prefixes(
    artifacts: str,
    *,
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

        prefixes.append(C30Prefix(
            prefix_id=pid,
            task_id=w["task_id"],
            position=int(w["position"]),
            action_type=w["action_type"],
            tool_names=list(w.get("tool_names") or []),
            semantic_hash=w["semantic_hash"],
            prompt_token_ids=prompt_ids_from_row(tok),
            canonical_messages=messages,
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
    "cycle_updates",
    "load_c30_prefixes",
    "prompt_ids_from_row",
]
