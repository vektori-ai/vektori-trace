"""Freeze the Phase 7 prefix suites.

A *prefix* is the conversation immediately before a teacher action: everything
the model would have in context at the moment it must emit one. The captured
teacher response is the reference; nothing here is invented.

"Fixed" means every checkpoint sees byte-identical inputs, so the manifest
records the corpus, line number and message index of each prefix and is written
once. Re-running with the same seed and inputs reproduces it; the sha256 in the
report is what pins it.

Three cells, not two. Training consumed only *passing* rollouts, and every one
of the 26 held-out tasks has zero passing rollouts — so a held-out prefix can
only come from a rollout that failed. That confounds two variables at once:

                    | passing rollout      | failing rollout
    trained task    | acquisition          | control
    held-out task   | (does not exist)     | generalization

`acquisition -> control` moves only rollout outcome. `control -> generalization`
moves only task familiarity. Without the control cell, a low generalization
number cannot be read: "did not generalize" and "harder conversations" produce
the same digit.

Usage (on the box, where the corpora live):

    python scripts/phase7_manifest.py \
        --acquisition   /data/sft-repaired \
        --control       /data/phase7-corpora/trained-failing \
        --generalization /data/phase7-corpora/heldout-failing \
        --selection-per-category 5 --tripwire-per-category 1 \
        --out /data/phase7/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.evaluate.phase7 import (
    SELECTION_CATEGORIES,
    SELECTION_SUITES,
    TRIPWIRE_CATEGORIES,
    has_edit,
    has_read,
    has_test,
)

# Categories are *conversation states*, not tasks. The envelope failure is not
# task-dependent — emitting flat JSON has nothing to do with click versus jinja
# — but it is very plausibly state-dependent: correct at turn 1 with 3k of
# context, drifting at turn 30 with 35k after a compaction. Sampling 50 tasks
# all at turn 1 would buy almost nothing over sampling 5.
CATEGORIES = (
    "orientation",           # first action of the run
    "repo_present",          # terminal output has already shown the repo
    "first_inspection",      # first real source read
    "first_edit",            # the rarest and most valuable behavior: 146 ops
    "test_exec",             # 635 ops
    "parse_error_recovery",  # the turn after the parser rejected a response
    "post_compaction",       # first action of a segment after a handoff
    "long_context",          # the longest prefixes, where drift is likeliest
)

# The split into "selects checkpoints" and "tripwire" belongs to
# `evaluate.phase7` — the manifest builds what selection reads, so a category
# in one and not the other is either an ungraded prefix or a gate with nothing
# to grade.
assert set(CATEGORIES) == set(SELECTION_CATEGORIES) | set(TRIPWIRE_CATEGORIES)

GIT_PRESENT_RE = re.compile(r"\.git\b|On branch |nothing to commit|git status")


def _repo(task: str) -> str:
    """`pallets__click-3126` -> `click`.

    Splitting on the org instead would put click and jinja in one bucket — two
    of the five repos share `pallets` — and a suite drawn from "two orgs" can
    still be a suite drawn from one repo.
    """
    tail = task.split("__", 1)[-1]
    return tail.rsplit("-", 1)[0] or tail


def _ops(content: str) -> tuple[bool, bool, bool]:
    """(read, edit, test) present in this action's commands.

    Per command, never on the newline-joined blob: `EDIT_RE` and `TEST_RE` are
    `^`-anchored and compiled without MULTILINE, so a joined search only ever
    tested command 0 and an action that read a file then patched it was
    classified as a read. The classifiers live in `evaluate.phase7` — one home,
    so the manifest's `first_edit` means what the gate's `edit_emission` means.
    """
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return (False, False, False)
    ks = [
        c.get("keystrokes", "")
        for c in (obj.get("commands") or [])
        if isinstance(c, dict)
    ]
    return (has_read(ks), has_edit(ks), has_test(ks))


def load_corpus(root: Path) -> list[dict[str, Any]]:
    """A corpus written by `scripts/sft_repair_dataset.py`.

    `sft_repaired.jsonl` and `repair_manifest.jsonl` are written in the same
    loop over the same list, so line N of one is line N of the other. The join
    is positional by construction, and asserted rather than assumed.
    """
    rows = [
        json.loads(ln)
        for ln in (root / "sft_repaired.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    meta = [
        json.loads(ln)
        for ln in (root / "repair_manifest.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    if len(rows) != len(meta):
        raise ValueError(
            f"{root}: {len(rows)} segments but {len(meta)} manifest lines — the "
            "positional join is invalid"
        )
    for r, m in zip(rows, meta, strict=True):
        if r["task"] != m["task"]:
            raise ValueError(f"{root}: positional join disagrees at task {r['task']}")
        r["_meta"] = m
    return rows


def candidates(corpus: list[dict[str, Any]], corpus_name: str) -> list[dict[str, Any]]:
    """Every supervised action turn, classified into at most one category.

    A turn is a candidate prefix; the prefix itself is `messages[:i]` and the
    reference is `messages[i]`.
    """
    out: list[dict[str, Any]] = []
    for line_no, seg in enumerate(corpus):
        messages = seg["messages"]
        supervise = seg["supervise"]
        meta = seg["_meta"]
        turn_meta = meta.get("turn_meta") or []
        seg_index = meta.get("segment_index", 0)

        seen_action = 0
        seen_edit = False
        seen_test = False
        for i, (msg, sup) in enumerate(zip(messages, supervise, strict=True)):
            if not sup:
                continue
            read_op, edit_op, test_op = _ops(msg["content"])
            prev_kind = (
                turn_meta[i - 2].get("kind") if i >= 2 and i - 2 < len(turn_meta) else None
            )
            prior = "\n".join(m["content"] for m in messages[max(0, i - 4) : i])

            category: str | None = None
            if prev_kind == "parse_error":
                category = "parse_error_recovery"
            elif seen_action == 0 and seg_index > 0:
                category = "post_compaction"
            elif seen_action == 0:
                category = "orientation"
            elif edit_op and not seen_edit:
                category = "first_edit"
            elif test_op and not seen_test:
                category = "test_exec"
            elif seen_action == 1 and read_op:
                category = "first_inspection"
            elif GIT_PRESENT_RE.search(prior):
                category = "repo_present"

            seen_action += 1
            seen_edit = seen_edit or edit_op
            seen_test = seen_test or test_op
            if category is None:
                category = "long_context"  # ranked by size, trimmed below

            out.append(
                {
                    "corpus": corpus_name,
                    "line_no": line_no,
                    "message_index": i,
                    "task": seg["task"],
                    "repo": _repo(seg["task"]),
                    "segment_index": seg_index,
                    "turn_ordinal": seen_action,
                    "category": category,
                    "prefix_chars": sum(len(m["content"]) for m in messages[:i]),
                    "reference": msg["content"],
                    "git_present": bool(GIT_PRESENT_RE.search(prior)),
                    "jobs_dir": meta.get("jobs_dir"),
                    "rollout_index": meta.get("rollout_index"),
                }
            )
    return out


def pick(
    cands: list[dict[str, Any]],
    *,
    counts: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stratify by category, then spread across repos and tasks.

    Repo spread is not decoration. The five repos differ in file layout, test
    invocation and terminal-output shape; drawing every held-out prefix from
    click would test one repo's idiom and call it generalization. With
    `counts[cat] == 5` and five repos the round-robin gives exactly one prefix
    per repo.

    One prefix per task, with no fallback. The old code fell back to reusing a
    task once a repo's fresh pool ran out (`pickable = fresh or bucket`), which
    turns "5 prefixes" into "5 prefixes, two of them the same conversation at
    two turns" — correlated draws counted as independent evidence. A category
    that cannot be filled from distinct tasks comes back short and `main`
    refuses to freeze, rather than being quietly padded.

    Returns `(chosen, shortfall)` where shortfall maps category -> how many
    prefixes were asked for and not found.
    """
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_cat[c["category"]].append(c)

    chosen: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    used_tasks: set[str] = set()
    repo_used: dict[str, int] = defaultdict(int)
    for cat in CATEGORIES:
        want = counts.get(cat, 0)
        if want <= 0:
            continue
        pool = by_cat.get(cat, [])
        if pool and cat == "long_context":
            pool = sorted(pool, key=lambda c: -c["prefix_chars"])[: max(8, want * 4)]
        by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in pool:
            by_repo[c["repo"]].append(c)
        for repo in by_repo:
            rng.shuffle(by_repo[repo])
        # Start each category at the least-used repo so far. Ordering by bucket
        # size alone always starts at the largest, which puts all eight
        # categories in one repo whenever the count is 1 — the exact collapse
        # the stratification exists to prevent.
        order = sorted(by_repo, key=lambda r: (repo_used[r], -len(by_repo[r])))
        taken: list[dict[str, Any]] = []
        progress = True
        while len(taken) < want and progress:
            progress = False
            for repo in order:
                if len(taken) >= want:
                    break
                bucket = by_repo[repo]
                fresh = [c for c in bucket if c["task"] not in used_tasks]
                if not fresh:
                    continue
                c = fresh[0]
                bucket.remove(c)
                used_tasks.add(c["task"])
                repo_used[repo] += 1
                taken.append(c)
                progress = True
        if len(taken) < want:
            shortfall[cat] = want - len(taken)
        chosen.extend(taken)
    return chosen, shortfall


def prefix_id(entry: dict[str, Any]) -> str:
    key = f"{entry['corpus']}:{entry['line_no']}:{entry['message_index']}"
    return f"{entry['category']}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acquisition", type=Path, required=True,
                    help="corpus root of trained tasks / passing rollouts")
    ap.add_argument("--control", type=Path, default=None,
                    help="trained tasks / FAILING rollouts — isolates the "
                         "rollout-outcome confound from task familiarity")
    ap.add_argument("--generalization", type=Path, required=True,
                    help="held-out tasks (necessarily failing rollouts)")
    ap.add_argument("--selection-per-category", type=int, default=5,
                    help="prefixes per selection category per suite "
                         f"({', '.join(SELECTION_CATEGORIES)}). 5 x 3 "
                         "categories x 3 suites = the 45 the plan selects on.")
    ap.add_argument("--tripwire-per-category", type=int, default=1,
                    help="prefixes per non-selecting category per suite. These "
                         "are tripwires: reported, never gating.")
    ap.add_argument("--allow-short", action="store_true",
                    help="freeze even if a selection category could not be "
                         "filled from distinct tasks. Off by default — a short "
                         "suite silently shrinks the selection set.")
    ap.add_argument("--tokenizer", default=None,
                    help="HF id or path. Records each prefix's exact token "
                         "count under the serving chat template and refuses to "
                         "write a manifest containing a prefix that will not "
                         "fit. A prompt vLLM rejects becomes an infra failure, "
                         "and an ungraded prefix blocks its checkpoint from "
                         "being selected — so an over-long prefix does not "
                         "produce a bad number, it produces no number.")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    suites = {"acquisition": args.acquisition, "generalization": args.generalization}
    if args.control:
        suites["control"] = args.control

    counts = {
        c: (args.selection_per_category if c in SELECTION_CATEGORIES
            else args.tripwire_per_category)
        for c in CATEGORIES
    }
    rng = random.Random(args.seed)
    shortfalls: dict[str, dict[str, int]] = {}
    entries: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for suite, root in suites.items():
        corpus = load_corpus(root)
        cands = candidates(corpus, suite)
        got, short = pick(cands, counts=counts, rng=rng)
        if short:
            shortfalls[suite] = short
        for e in got:
            e["suite"] = suite
            e["prefix_id"] = prefix_id(e)
            e["corpus_root"] = str(root)
            e["selection"] = (
                e["category"] in SELECTION_CATEGORIES and suite in SELECTION_SUITES
            )
        entries.extend(got)
        stats[suite] = {
            "segments": len(corpus),
            "tasks": len({s["task"] for s in corpus}),
            "candidates": len(cands),
            "selected": len(got),
            "by_category": {
                c: sum(1 for e in got if e["category"] == c) for c in CATEGORIES
            },
            "repos": sorted({e["repo"] for e in got}),
            "selection_prefixes": sum(1 for e in got if e["selection"]),
            "distinct_tasks_selected": len({e["task"] for e in got}),
        }
        print(
            f"{suite:15} {len(corpus):3} segments, {stats[suite]['tasks']:2} tasks, "
            f"{len(cands):4} candidates -> {len(got)} prefixes "
            f"({stats[suite]['selection_prefixes']} selecting) "
            f"across {len(stats[suite]['repos'])} repos"
        )

    sel = [e for e in entries if e["selection"]]
    want_sel = len(SELECTION_CATEGORIES) * args.selection_per_category * len(
        [s for s in suites if s in SELECTION_SUITES]
    )
    print(f"\nselection set: {len(sel)} prefixes "
          f"({len({e['task'] for e in sel})} distinct tasks, "
          f"{len({e['repo'] for e in sel})} repos); expected {want_sel}")
    if shortfalls:
        print(f"\nSHORT: {shortfalls}", file=sys.stderr)
    if len(sel) != want_sel and not args.allow_short:
        print(
            f"\nselection set is {len(sel)}, not {want_sel}: refusing to "
            "freeze. A short set is not a smaller number, it is a different "
            "question — selection would be read off whatever survived. Pass "
            "--allow-short only if you mean it.",
            file=sys.stderr,
        )
        return 1
    dup = len(sel) - len({(e["suite"], e["task"]) for e in sel})
    if dup:
        print(f"\n{dup} selection prefix(es) reuse a task within their suite; "
              "refusing to freeze", file=sys.stderr)
        return 1

    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        budget = args.max_model_len - args.max_new_tokens
        corpora: dict[str, list] = {}
        over = []
        for e in entries:
            root = e["corpus_root"]
            if root not in corpora:
                corpora[root] = [
                    json.loads(ln)
                    for ln in (Path(root) / "sft_repaired.jsonl").read_text().splitlines()
                    if ln.strip()
                ]
            msgs = corpora[root][e["line_no"]]["messages"][: e["message_index"]]
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            e["prefix_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
            if e["prefix_tokens"] > budget:
                over.append(e)
        lengths = sorted(e["prefix_tokens"] for e in entries)
        print(f"\nprefix tokens: min {lengths[0]:,}  median "
              f"{lengths[len(lengths) // 2]:,}  max {lengths[-1]:,}  "
              f"(budget {budget:,} = {args.max_model_len:,} - {args.max_new_tokens})")
        if over:
            for e in over:
                print(f"  OVER BUDGET {e['prefix_tokens']:,}  {e['suite']}/"
                      f"{e['category']}  {e['task']}", file=sys.stderr)
            print(f"\n{len(over)} prefix(es) exceed the budget; refusing to "
                  "write the manifest", file=sys.stderr)
            return 1

    # Matched suites or the comparison is not a comparison. A category present
    # in one suite and absent from another silently changes what the two
    # numbers mean.
    per_suite_cats = {
        s: {e["category"] for e in entries if e["suite"] == s} for s in suites
    }
    common = set.intersection(*per_suite_cats.values()) if per_suite_cats else set()
    unmatched = {s: sorted(c - common) for s, c in per_suite_cats.items() if c - common}
    if unmatched:
        print(f"\nWARNING: categories not present in every suite: {unmatched}",
              file=sys.stderr)
        print("  cross-suite rates are only comparable on the matched "
              f"categories: {sorted(common)}", file=sys.stderr)

    payload = {
        "version": 1,
        "seed": args.seed,
        "selection_per_category": args.selection_per_category,
        "tripwire_per_category": args.tripwire_per_category,
        "selection_categories": list(SELECTION_CATEGORIES),
        "selection_suites": sorted(s for s in suites if s in SELECTION_SUITES),
        "selection_prefix_ids": sorted(e["prefix_id"] for e in sel),
        "suites": {k: str(v) for k, v in suites.items()},
        "matched_categories": sorted(common),
        "tokenizer": args.tokenizer,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "stats": stats,
        "prefixes": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True)
    args.out.write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    (args.out.parent / "manifest_sha256.txt").write_text(digest + "\n")
    print(f"\nwrote {args.out}  ({len(entries)} prefixes)\nmanifest sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
