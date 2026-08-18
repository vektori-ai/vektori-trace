"""Build the Stage A format corpus by slicing the repaired set.

`docs/SFT-SCRATCH-PLAN.md` step 3. Stage A teaches one thing: emit a Terminus
action when the user message ends. So every row is a cold start — a prompt with
no prior *action* to copy a protocol from — and the target is the last message,
which is the only shape Qwen3's template can mask correctly (step 1).

183 rows, all sliced from `/data/sft-repaired`. Nothing is rebuilt: the message
bytes are the frozen hybrid the repair produced, and re-running the capture join
would mean a new sha and a new argument. What changes is which turns a row
contains and which one carries the loss.

    turn_1                 117  segment_index 0, the production first turn
    post_compaction_first   48  first action after a compaction boundary
                           ---
                           165

The 48 are already inside the 165 firsts (segments 1..N of a rollout), so they
are not additional; counting them twice was an early error in the plan.

**Parse-error recoveries are deferred to Stage B**, against the plan's original
183. They were included because `parse_error_recovery` fails 2 of 3 prefixes at
every checkpoint and reads as a cold start — no *valid* prior action to copy a
protocol from. Measured, they are not cold starts by position:

    turn_1                 952 ..  2,234 tokens
    post_compaction_first  2,567 .. 6,494
    parse_error_recovery   5,302 .. 33,179   (median 15,059; 14 of 18 over 8k)

They sit at turns 13-75 with the whole trajectory behind them. Admitting them
means either a max_length near 33k — where 10% of the rows would carry ~47% of
the input tokens and Stage A stops being the short, fast format run the plan
costed — or keeping only the 4 that happen to fit, which selects recoveries by
trajectory length and so by task and repo. Stage B's one-action-per-row rows
already run to 40k; that is where these belong. `--recoveries stage-a` builds
the 183-row version for comparison.

Repo mass is rebalanced because the corpus is not: 110 of 165 firsts are pallets
(click + jinja), so an unweighted Stage A would be a Click opener model. Weights
are `1/n_task x repo_weight`, which flattens the 3..10 rows-per-task spread as
well. Rows are never dropped to hit a target — a downweighted row is still gold
and still reachable, it just stops eating optimizer steps.

    python scripts/sft_stage_a_dataset.py --src /data/sft-repaired \\
        --out /data/sft-stage-a

Writes `stage_a.jsonl`, `mix_report.json`, `preflight_report.json` and — only on
a clean pass — `tokenization_fingerprint.json`. A fingerprint from a failed
build would let the trainer's drift guard pass against a set nobody validated.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.dataset import IGNORE_INDEX, THINK_WRAPPER_TEXT, tokenize_messages

TEMPLATE_KWARGS = {"enable_thinking": True}
MAX_LENGTH = 8192

# The repaired corpus this is sliced from. Pinned so a rebuilt source cannot
# quietly become the input: the whole point of slicing is that these bytes are
# the ones the Phase 2 join verified.
EXPECTED_SRC_SHA = "7ecfee31"

TURN_1 = "turn_1"
POST_COMPACTION_FIRST = "post_compaction_first"
PARSE_ERROR_RECOVERY = "parse_error_recovery"

EXPECTED_COUNTS = {TURN_1: 117, POST_COMPACTION_FIRST: 48}
EXPECTED_COUNTS_WITH_RECOVERIES = {**EXPECTED_COUNTS, PARSE_ERROR_RECOVERY: 18}

# Sampler mass per repo. pallets is capped well under its natural 67%; the other
# three are lifted, but only to ~1.65x, comfortably inside the 3x cap — a repo
# with 15 rows should not be stretched to look like one with 110.
REPO_TARGET_MASS = {"click": None, "jinja": None, "anyio": 0.25, "hatch": 0.15,
                    "prefect": 0.15}
PALLETS = ("click", "jinja")
PALLETS_MASS = 0.45
REPO_FLOOR = 0.10
MAX_UPSAMPLE = 3.0


def repo_of(task: str) -> str:
    """`pallets__click-3484` -> `click`."""
    return task.split("__", 1)[1].rsplit("-", 1)[0]


def _is_native_action(text: str) -> tuple[bool, str]:
    """(ok, why). harbor's parser is the authority, plus the v1 envelope check."""
    for marker in ("<tool_call>", "</tool_call>"):
        if marker in text:
            return False, f"target contains {marker}"
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    parsed = TerminusJSONPlainParser().parse_response(text)
    if parsed.error:
        return False, f"harbor rejects the target: {parsed.error}"
    return True, ""


def slice_rows(
    rows: list[dict], manifest: list[dict], *, recoveries: bool
) -> list[dict]:
    """One row per cold start, each ending on its target."""
    out: list[dict] = []
    for row, man in zip(rows, manifest, strict=True):
        meta = man["turn_meta"]
        if len(meta) != len(row["messages"]):
            raise SystemExit(
                f"{man['task']} seg{man['segment_index']}: turn_meta has "
                f"{len(meta)} entries for {len(row['messages'])} messages"
            )
        task, seg = man["task"], man["segment_index"]

        targets: list[tuple[int, str]] = []
        first = next(i for i, sup in enumerate(row["supervise"]) if sup)
        targets.append((first, TURN_1 if seg == 0 else POST_COMPACTION_FIRST))

        # The action *after* a rejected completion. Harbor will hit this state
        # whenever the model's JSON is wrong, and the previous sweep failed it
        # on 2 of 3 prefixes at every checkpoint — a cold start in everything but
        # name, since the visible assistant turn above is malformed.
        for i, m in enumerate(meta):
            if not recoveries or m.get("kind") != "parse_error":
                continue
            nxt = next(
                (j for j in range(i + 1, len(meta)) if meta[j].get("kind") == "action"),
                None,
            )
            # Two rejected completions in a row resolve to the same recovery
            # action; that is one training row, not two.
            already = {i for i, _ in targets}
            if nxt is not None and row["supervise"][nxt] and nxt not in already:
                targets.append((nxt, PARSE_ERROR_RECOVERY))

        for idx, kind in targets:
            messages = row["messages"][: idx + 1]
            out.append({
                "source_id": f"{task}|seg{seg}|msg{idx}",
                "task": task,
                "repo": repo_of(task),
                "kind": kind,
                "segment_index": seg,
                "rollout_index": man.get("rollout_index"),
                # Prior assistant turns are visible context. For turn_1 this is
                # 0 by construction; the other two kinds carry prose (a handoff
                # answer, a malformed completion) and never a prior action.
                "n_prior_assistant_turns": sum(
                    1 for m in messages[:-1] if m["role"] == "assistant"
                ),
                "messages": messages,
                "supervise": [False] * idx + [True],
            })
    return out


def compute_weights(rows: list[dict]) -> dict[str, float]:
    """`1/n_task x repo_weight`, normalised so each repo holds its target mass.

    Within a repo, mass is split evenly across tasks and then evenly across that
    task's rows, so a task with 10 openers does not outvote one with 3.
    """
    by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    targets = dict(REPO_TARGET_MASS)
    pallets_rows = sum(len(by_repo.get(p, [])) for p in PALLETS)
    for p in PALLETS:
        n = len(by_repo.get(p, []))
        targets[p] = PALLETS_MASS * (n / pallets_rows) if pallets_rows else 0.0

    weights: dict[str, float] = {}
    for repo, repo_rows in by_repo.items():
        tasks: dict[str, list[dict]] = collections.defaultdict(list)
        for r in repo_rows:
            tasks[r["task"]].append(r)
        per_task = targets[repo] / len(tasks)
        for task_rows in tasks.values():
            per_row = per_task / len(task_rows)
            for r in task_rows:
                weights[r["source_id"]] = per_row
    return weights


def mix_report(rows: list[dict], weights: dict[str, float]) -> dict[str, Any]:
    n = len(rows)
    uniform = 1.0 / n
    by_repo: dict[str, dict[str, float]] = {}
    for r in rows:
        e = by_repo.setdefault(r["repo"], {"rows": 0, "mass": 0.0})
        e["rows"] += 1
        e["mass"] += weights[r["source_id"]]
    for e in by_repo.values():
        e["row_share"] = e["rows"] / n
        # >1 means the sampler draws this repo more often than its row count
        # would; the cap exists so a 15-row repo is not stretched to look large.
        e["upsample"] = e["mass"] / e["row_share"]
    return {
        "rows": n,
        "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
        "by_repo": by_repo,
        "tasks": len({r["task"] for r in rows}),
        "max_row_upsample": max(weights.values()) / uniform,
        "min_row_upsample": min(weights.values()) / uniform,
        "pallets_mass": sum(by_repo.get(p, {}).get("mass", 0.0) for p in PALLETS),
        "first_command_top": dict(
            collections.Counter(_first_command(r) for r in rows).most_common(5)
        ),
    }


def _first_command(row: dict) -> str:
    try:
        obj = json.loads(row["messages"][-1]["content"])
        return (obj["commands"][0]["keystrokes"].splitlines() or [""])[0][:40]
    except Exception:
        return "<unparsed>"


def check_mix(rep: dict[str, Any], expected: dict[str, int]) -> list[str]:
    """Every condition `docs/SFT-SCRATCH-PLAN.md` says aborts before a GPU."""
    bad = []
    if rep["pallets_mass"] > PALLETS_MASS + 1e-6:
        bad.append(f"pallets hold {rep['pallets_mass']:.3f} of sampler mass, "
                   f"cap is {PALLETS_MASS}")
    for repo in ("anyio", "hatch", "prefect"):
        mass = rep["by_repo"].get(repo, {}).get("mass", 0.0)
        if mass < REPO_FLOOR - 1e-6:
            bad.append(f"{repo} holds {mass:.3f} of sampler mass, floor is {REPO_FLOOR}")
    for repo, e in rep["by_repo"].items():
        if e["upsample"] > MAX_UPSAMPLE + 1e-6:
            bad.append(f"{repo} is upsampled {e['upsample']:.2f}x, cap is {MAX_UPSAMPLE}")
    for kind, want in expected.items():
        got = rep["by_kind"].get(kind, 0)
        if got != want:
            bad.append(f"{kind}: {got} rows, expected {want}")
    for kind in set(rep["by_kind"]) - set(expected):
        bad.append(f"{kind}: {rep['by_kind'][kind]} unexpected rows")
    return bad


def row_digest(example: Any) -> str:
    h = hashlib.sha256()
    for key in ("input_ids", "labels", "attention_mask"):
        h.update(key.encode())
        h.update(b",".join(str(x).encode() for x in getattr(example, key)))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--expect-src-sha", default=EXPECTED_SRC_SHA,
                    help="prefix of the repaired corpus sha256 this may slice")
    ap.add_argument("--recoveries", choices=("defer", "stage-a"), default="defer",
                    help="'defer' (default) leaves the 18 parse-error recoveries "
                         "for Stage B; they run to 33k tokens and would carry ~47%% "
                         "of Stage A's input for 10%% of its rows. 'stage-a' builds "
                         "the plan's original 183 — raise --max-length with it.")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args()

    data = args.src / "sft_repaired.jsonl"
    src_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    if not src_sha.startswith(args.expect_src_sha):
        print(f"source is {src_sha[:16]}, expected {args.expect_src_sha}* — refusing "
              "to slice a corpus this build was not specified against", file=sys.stderr)
        return 2
    print(f"source {data} sha256 {src_sha[:16]} (pinned)")

    rows = [json.loads(ln) for ln in data.read_text().splitlines() if ln.strip()]
    manifest = [
        json.loads(ln)
        for ln in (args.src / "repair_manifest.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    want_recoveries = args.recoveries == "stage-a"
    sliced = slice_rows(rows, manifest, recoveries=want_recoveries)
    print(f"sliced {len(sliced)} rows from {len(rows)} segments: "
          f"{dict(collections.Counter(r['kind'] for r in sliced))}")

    # ---- targets must be actions harbor would run --------------------------
    failures: list[str] = []
    kept = []
    for r in sliced:
        ok, why = _is_native_action(r["messages"][-1]["content"])
        (kept if ok else failures).append(r if ok else f"{r['source_id']}: {why}")
    if failures:
        print(f"dropped {len(failures)} unusable target(s)", file=sys.stderr)
        for f in failures[:10]:
            print(f"  {f}", file=sys.stderr)
    sliced = kept

    weights = compute_weights(sliced)
    rep = mix_report(sliced, weights)
    mix_problems = check_mix(
        rep, EXPECTED_COUNTS_WITH_RECOVERIES if want_recoveries else EXPECTED_COUNTS
    )

    # ---- tokenize, and verify the mask on every row ------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    wrapper = tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)
    fingerprint: list[str] = []
    tok_problems: list[str] = []
    lengths, supervised = [], []
    for r in sliced:
        try:
            ex = tokenize_messages(
                r["messages"], tok, r["supervise"], max_length=args.max_length,
                template_kwargs=TEMPLATE_KWARGS, truncate=False,
            )
        except Exception as e:  # a refusal here is a build failure, not a drop
            tok_problems.append(f"{r['source_id']}: {type(e).__name__}: {e}")
            continue
        if ex is None:
            tok_problems.append(f"{r['source_id']}: over {args.max_length} tokens or nothing supervised")
            continue
        n_sup = sum(1 for lab in ex.labels if lab != IGNORE_INDEX)
        first = next(i for i, lab in enumerate(ex.labels) if lab != IGNORE_INDEX)
        # The span must start at the action, not at the wrapper: supervising the
        # wrapper teaches an empty reasoning block on every row.
        if ex.input_ids[first - len(wrapper):first] != wrapper:
            tok_problems.append(f"{r['source_id']}: reasoning wrapper is not immediately "
                                "before the supervised span")
        if not tok.decode(ex.input_ids[first:]).lstrip().startswith("{"):
            tok_problems.append(f"{r['source_id']}: supervised span does not start at the object")
        if any(lab != IGNORE_INDEX for lab in ex.labels[:first]):
            tok_problems.append(f"{r['source_id']}: label before the target span")
        lengths.append(len(ex.input_ids))
        supervised.append(n_sup)
        fingerprint.append(row_digest(ex))

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "stage_a.jsonl"
    jsonl.write_text("".join(
        json.dumps({**r, "weight": weights[r["source_id"]]}) + "\n" for r in sliced
    ))
    out_sha = hashlib.sha256(jsonl.read_bytes()).hexdigest()

    (args.out / "mix_report.json").write_text(json.dumps(rep, indent=2))
    preflight = {
        "source_sha256": src_sha,
        "dataset_sha256": out_sha,
        "rows": len(sliced),
        "dropped_targets": failures,
        "tokens": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "total": sum(lengths),
        },
        "supervised_tokens": {
            "min": min(supervised) if supervised else 0,
            "total": sum(supervised),
            "fraction": (sum(supervised) / sum(lengths)) if lengths else 0.0,
        },
        "mix_problems": mix_problems,
        "tokenization_problems": tok_problems,
    }
    (args.out / "preflight_report.json").write_text(json.dumps(preflight, indent=2))

    print(f"\nrows {len(sliced)}  tokens {preflight['tokens']['min']}"
          f"..{preflight['tokens']['max']}  supervised "
          f"{preflight['supervised_tokens']['fraction']:.1%}")
    print("repo mass: " + "  ".join(
        f"{k} {v['mass']:.3f} ({v['rows']}r, {v['upsample']:.2f}x)"
        for k, v in sorted(rep["by_repo"].items())
    ))
    print(f"row upsample range {rep['min_row_upsample']:.2f}x..{rep['max_row_upsample']:.2f}x")
    print(f"first commands: {rep['first_command_top']}")

    problems = mix_problems + tok_problems + failures
    if problems:
        print(f"\nBUILD FAILED: {len(problems)} problem(s) — no fingerprint written",
              file=sys.stderr)
        for p in problems[:20]:
            print(f"  {p}", file=sys.stderr)
        return 1

    (args.out / "tokenization_fingerprint.json").write_text(json.dumps({
        "model": args.model,
        "template_kwargs": TEMPLATE_KWARGS,
        "max_length": args.max_length,
        "dataset_sha256": out_sha,
        "per_row": fingerprint,
    }, indent=2))
    print(f"\nBUILD PASSED — wrote {args.out}/stage_a.jsonl ({out_sha[:16]}) "
          "and its three reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
