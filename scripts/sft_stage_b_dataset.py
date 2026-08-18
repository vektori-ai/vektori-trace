"""Build the Stage B trajectory corpus by slicing the repaired set.

`docs/SFT-SCRATCH-PLAN.md` step 8. Stage A taught the model to emit a Terminus
action at a cold start; Stage B teaches it what to *do* — inspect, edit, run
tests — by supervising actions drawn from later in the same trajectories.

Same source, same pin, same slicing discipline as `sft_stage_a_dataset.py`:
nothing is rebuilt, only re-sliced. What changes is which action carries the
loss and how much context sits behind it.

    later_first_edit   the segment's first action containing an edit op
    later_first_test   its first action containing a test invocation
    later_last         its final action
    later_mid          one evenly-placed middle action
    later_read_only    segments with neither an edit nor a test: 4 evenly
                       spaced ordinals including the last
    parse_error_recovery  the action after a rejected completion (the 18)
    cold_replay        Stage A's cold starts, replayed — see below

**Stage A's first action is never reused.** Exclusion is by `source_id`
(`task|segN|msgI`), read out of Stage A's own jsonl, so it is exact on segment
and message index rather than on rendered text.

**The 18 parse-error recoveries live here**, deferred from Stage A as amendment
2. They sit at turns 13-75, median ~15k tokens, and only fit because Stage B
runs at `max_length` 40960. Fewer than 18 is a build failure, not a rounding
difference.

**Cold replay and the cold-token floor.** The plan's fail-closed line reads
"Stage B cold token share < 25%", and step 8 asks for cold draws >= 253 with a
target of 325 — a floor expressed in *supervised tokens*, on rows that carry no
prior action to copy a protocol from. Stage B's own rows are all later actions,
so they contribute no cold tokens at all: without a replay component the share
is 0 by construction and the floor could never be cleared. So Stage A's 165
cold starts are replayed as a weighted component, and the component's sampler
mass is *solved* for the token-share target rather than guessed. This is the
one part of step 8 the plan does not spell out; see `--cold-target` and the
report's `cold` block, and read the module docstring's note in the commit.

    python scripts/sft_stage_b_dataset.py --src /data/sft-repaired \\
        --stage-a /data/sft-stage-a --out /data/sft-stage-b

Writes `stage_b.jsonl`, `mix_report.json`, `preflight_report.json` and — only on
a clean pass — `tokenization_fingerprint.json`. A fingerprint from a failed
build would let a trainer's drift guard pass against a set nobody validated.
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
from vektori_trace.evaluate.phase7 import has_edit, has_read, has_test

TEMPLATE_KWARGS = {"enable_thinking": True}
MAX_LENGTH = 40960

# The repaired corpus this is sliced from, pinned exactly as Stage A pins it.
EXPECTED_SRC_SHA = "7ecfee31"

LATER_FIRST_EDIT = "later_first_edit"
LATER_FIRST_TEST = "later_first_test"
LATER_LAST = "later_last"
LATER_MID = "later_mid"
LATER_READ_ONLY = "later_read_only"
PARSE_ERROR_RECOVERY = "parse_error_recovery"
COLD_REPLAY = "cold_replay"

LATER_KINDS = (
    LATER_FIRST_EDIT, LATER_FIRST_TEST, LATER_LAST, LATER_MID, LATER_READ_ONLY,
)

# Exact, per amendment 2. A short count means recoveries were dropped by a
# parser or a length guard, which is a build failure.
EXPECTED_RECOVERIES = 18

# At most four later actions per segment (plan step 8). The cap is what keeps
# ~660 rows from becoming ~3,400 — one long trajectory would otherwise outvote
# thirty short ones.
MAX_LATER_PER_SEGMENT = 4

# Supervised-token share carried by cold-start rows. The plan's floor is 25%
# and its target 30%; the component mass is solved for the target and the build
# refuses below the floor.
COLD_TOKEN_FLOOR = 0.25
COLD_TOKEN_TARGET = 0.30
# Step 8's draw counts at those two shares. Reported, not gated: the binding
# constraint the fail-closed line names is the token share.
COLD_DRAWS_FLOOR = 253
COLD_DRAWS_TARGET = 325

# Sampler mass per repo — identical to Stage A's, so the two stages do not
# disagree about what a balanced mix is.
REPO_TARGET_MASS = {"click": None, "jinja": None, "anyio": 0.25, "hatch": 0.15,
                    "prefect": 0.15}
PALLETS = ("click", "jinja")
PALLETS_MASS = 0.45
REPO_FLOOR = 0.10
MAX_UPSAMPLE = 3.0


def repo_of(task: str) -> str:
    """`pallets__click-3484` -> `click`."""
    return task.split("__", 1)[1].rsplit("-", 1)[0]


def row_key(task: str, rollout: Any, seg: int, idx: int) -> str:
    """A row identity that is actually unique.

    Stage A's `source_id` is `task|segN|msgI`, which omits the rollout: the same
    task and segment recur across rollouts, so its 165 rows carry only **60**
    distinct ids, 46 of them duplicated. Anything keyed by it — sampler weights,
    per-row supervised-token counts — silently collapses onto one entry per
    collision. Stage A's dataset sha is pinned and its `source_id` is left
    exactly as it is; Stage B keys on this instead and reconstructs Stage A's
    exclusion keys in the same form.
    """
    return f"{task}|r{rollout}|seg{seg}|msg{idx}"


def stage_a_key(row: dict) -> str:
    """`row_key` for a row written by the Stage A builder.

    The message index is recovered from its `source_id` tail rather than from
    `len(messages) - 1`, so a row is identified by what it claims to be.
    """
    idx = int(str(row["source_id"]).rsplit("|msg", 1)[1])
    return row_key(
        row["task"], row.get("rollout_index"), row["segment_index"], idx
    )


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


def _ops(content: str) -> tuple[bool, bool, bool]:
    """(read, edit, test) present in this action's commands.

    Per command, never on the newline-joined blob — `EDIT_RE` and `TEST_RE` are
    `^`-anchored without MULTILINE, so a joined search only ever tests command
    0. The classifiers live in `evaluate.phase7`; this is the only sanctioned
    way to ask, and there is deliberately no fourth copy of the regexes here.
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


def evenly_spaced(items: list[int], n: int) -> list[int]:
    """`n` evenly spaced entries of `items`, always including the last.

    "even" in step 8 means evenly *spaced*, not even-numbered: "4 even ordinals
    incl. last" is unsatisfiable under the even-numbered reading whenever the
    last ordinal is odd, which is half of all segments. Spacing is anchored on
    the end so the final action is always drawn.
    """
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / (n - 1) if n > 1 else 0
    picked = sorted({items[round(i * step)] for i in range(n)})
    # Rounding can collide on short lists; backfill from the end so the count
    # is right and `last` is never the one dropped.
    for cand in reversed(items):
        if len(picked) >= n:
            break
        if cand not in picked:
            picked.append(cand)
    return sorted(picked)


def pick_later(
    later: list[int], contents: dict[int, str]
) -> list[tuple[int, str]]:
    """Up to 4 later actions for one segment, per plan step 8.

    first edit -> first test -> last -> one even mid. A segment with neither an
    edit nor a test is read-only, and gets 4 evenly spaced ordinals including
    the last instead — a read-only trajectory still teaches inspection order,
    which is most of what the corpus contains.
    """
    if not later:
        return []

    first_edit = next((i for i in later if _ops(contents[i])[1]), None)
    first_test = next((i for i in later if _ops(contents[i])[2]), None)

    if first_edit is None and first_test is None:
        return [(i, LATER_READ_ONLY)
                for i in evenly_spaced(later, MAX_LATER_PER_SEGMENT)]

    picks: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(idx: int | None, kind: str) -> None:
        if idx is None or idx in seen or len(picks) >= MAX_LATER_PER_SEGMENT:
            return
        seen.add(idx)
        picks.append((idx, kind))

    add(first_edit, LATER_FIRST_EDIT)
    add(first_test, LATER_FIRST_TEST)
    add(later[-1], LATER_LAST)
    # The midpoint of what is left, so the row is not adjacent to one already
    # taken whenever the segment is long enough to have a middle.
    remaining = [i for i in later if i not in seen]
    if remaining:
        add(remaining[len(remaining) // 2], LATER_MID)
    return sorted(picks)


def recovery_targets(meta: list[dict], supervise: list[bool]) -> list[int]:
    """Action indices that follow a rejected completion.

    Two rejected completions in a row resolve to the same recovery action; that
    is one training row, not two.
    """
    out: list[int] = []
    for i, m in enumerate(meta):
        if m.get("kind") != "parse_error":
            continue
        nxt = next(
            (j for j in range(i + 1, len(meta)) if meta[j].get("kind") == "action"),
            None,
        )
        if nxt is not None and supervise[nxt] and nxt not in out:
            out.append(nxt)
    return out


def slice_rows(
    rows: list[dict], manifest: list[dict], *, exclude: set[str]
) -> list[dict]:
    """One row per supervised later action, each ending on its target."""
    out: list[dict] = []
    for row, man in zip(rows, manifest, strict=True):
        meta = man["turn_meta"]
        if len(meta) != len(row["messages"]):
            raise SystemExit(
                f"{man['task']} seg{man['segment_index']}: turn_meta has "
                f"{len(meta)} entries for {len(row['messages'])} messages"
            )
        task, seg = man["task"], man["segment_index"]
        supervised = [i for i, sup in enumerate(row["supervise"]) if sup]
        if not supervised:
            continue

        contents = {i: row["messages"][i]["content"] for i in supervised}
        # Everything after the segment's first action. The first is Stage A's,
        # and `exclude` asserts that rather than assuming it.
        roll = man.get("rollout_index")
        later = [
            i for i in supervised[1:]
            if row_key(task, roll, seg, i) not in exclude
        ]

        targets: dict[int, str] = dict(pick_later(later, contents))
        # Recoveries are added whether or not the cap is already spent, and
        # their label wins: the count is exact and must not be absorbed into a
        # `later_first_edit` that happens to sit at the same index.
        for idx in recovery_targets(meta, row["supervise"]):
            if row_key(task, roll, seg, idx) in exclude:
                continue
            targets[idx] = PARSE_ERROR_RECOVERY

        for idx in sorted(targets):
            messages = row["messages"][: idx + 1]
            read_op, edit_op, test_op = _ops(messages[-1]["content"])
            out.append({
                "source_id": row_key(task, roll, seg, idx),
                "task": task,
                "repo": repo_of(task),
                "kind": targets[idx],
                "segment_index": seg,
                "rollout_index": roll,
                "turn_ordinal": supervised.index(idx),
                "n_prior_assistant_turns": sum(
                    1 for m in messages[:-1] if m["role"] == "assistant"
                ),
                "ops": {"read": read_op, "edit": edit_op, "test": test_op},
                "messages": messages,
                "supervise": [False] * idx + [True],
            })
    return out


def load_cold(stage_a: Path) -> list[dict]:
    """Stage A's rows, relabelled as the replay component."""
    path = stage_a / "stage_a.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    out = []
    for r in rows:
        r = dict(r)
        r.pop("weight", None)
        r["stage_a_source_id"] = r["source_id"]
        r["source_id"] = stage_a_key(r)
        r["kind"] = COLD_REPLAY
        r["ops"] = dict(zip(
            ("read", "edit", "test"),
            _ops(r["messages"][-1]["content"]),
            strict=True,
        ))
        out.append(r)
    return out


def compute_weights(rows: list[dict]) -> dict[str, float]:
    """`1/n_task x repo_weight`, normalised so each repo holds its target mass.

    Identical rule to Stage A's. Applied per component, so the combined mix
    satisfies the repo targets whatever the components' relative masses are.
    """
    by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    targets = dict(REPO_TARGET_MASS)
    pallets_rows = sum(len(by_repo.get(p, [])) for p in PALLETS)
    for p in PALLETS:
        n = len(by_repo.get(p, []))
        targets[p] = PALLETS_MASS * (n / pallets_rows) if pallets_rows else 0.0
    # A repo absent from one component would otherwise leave its mass unspent
    # and the component would not sum to 1.
    live = {r: t for r, t in targets.items() if by_repo.get(r) and t}
    scale = sum(live.values()) or 1.0

    weights: dict[str, float] = {}
    for repo, repo_rows in by_repo.items():
        tasks: dict[str, list[dict]] = collections.defaultdict(list)
        for r in repo_rows:
            tasks[r["task"]].append(r)
        per_task = (live.get(repo, 0.0) / scale) / len(tasks)
        for task_rows in tasks.values():
            per_row = per_task / len(task_rows)
            for r in task_rows:
                weights[r["source_id"]] = per_row
    return weights


def solve_cold_mass(cold_per_draw: float, new_per_draw: float, target: float) -> float:
    """Sampler mass for the cold component that lands `target` token share.

    share = m*A / (m*A + (1-m)*B)  ->  m = tB / (A(1-t) + tB)

    Solved rather than guessed because A and B are measured per-row supervised
    token means that differ by ~1.5x (first actions are shorter than later
    ones), so a mass picked to look like 30% of *rows* is not 30% of tokens.
    """
    if cold_per_draw <= 0 or not 0.0 < target < 1.0:
        return 0.0
    num = target * new_per_draw
    den = cold_per_draw * (1.0 - target) + target * new_per_draw
    return num / den if den else 0.0


def mix_report(
    rows: list[dict],
    weights: dict[str, float],
    supervised_tokens: dict[str, int],
    *,
    cold_mass: float,
) -> dict[str, Any]:
    n = len(rows)
    by_repo: dict[str, dict[str, float]] = {}
    for r in rows:
        e = by_repo.setdefault(r["repo"], {"rows": 0, "mass": 0.0})
        e["rows"] += 1
        e["mass"] += weights[r["source_id"]]
    for e in by_repo.values():
        e["row_share"] = e["rows"] / n
        e["upsample"] = e["mass"] / e["row_share"]

    cold = [r for r in rows if r["kind"] == COLD_REPLAY]
    tok_mass = sum(
        weights[r["source_id"]] * supervised_tokens.get(r["source_id"], 0) for r in rows
    )
    cold_tok_mass = sum(
        weights[r["source_id"]] * supervised_tokens.get(r["source_id"], 0) for r in cold
    )
    # What the floor is actually worth depends on who samples. No trainer in
    # this repo reads `weight`: they build `Dataset.from_list` and let TRL
    # shuffle uniformly. Under that sampler the cold share is the raw token
    # ratio, which is far below the weighted figure — so both are reported and
    # neither can be mistaken for the other.
    uniform_total = sum(supervised_tokens.get(r["source_id"], 0) for r in rows)
    uniform_cold = sum(supervised_tokens.get(r["source_id"], 0) for r in cold)
    # Draws per epoch-equivalent: what the sampler hands the trainer over `n`
    # draws. The plan's 253/325 are counts in these units.
    draws = sum(weights[r["source_id"]] for r in cold) * n
    return {
        "rows": n,
        "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
        "by_repo": by_repo,
        "tasks": len({r["task"] for r in rows}),
        # With the rollout, or this counts 60 where there are 165 — the same
        # collision `row_key` exists to prevent, one level up.
        "segments": len({
            (r["task"], r.get("rollout_index"), r["segment_index"]) for r in rows
        }),
        "max_row_upsample": max(weights.values()) * n,
        "min_row_upsample": min(weights.values()) * n,
        "pallets_mass": sum(by_repo.get(p, {}).get("mass", 0.0) for p in PALLETS),
        "ops": {
            k: sum(1 for r in rows if r["ops"][k]) for k in ("read", "edit", "test")
        },
        "cold": {
            "rows": len(cold),
            "component_mass": cold_mass,
            "token_share": (cold_tok_mass / tok_mass) if tok_mass else 0.0,
            "token_share_uniform": (uniform_cold / uniform_total) if uniform_total else 0.0,
            "requires_weighted_sampler": True,
            "draws_per_epoch": draws,
            "floor_token_share": COLD_TOKEN_FLOOR,
            "target_token_share": COLD_TOKEN_TARGET,
            "floor_draws": COLD_DRAWS_FLOOR,
            "target_draws": COLD_DRAWS_TARGET,
        },
    }


def check_mix(rep: dict[str, Any], *, allow_unweighted: bool = False) -> list[str]:
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

    got = rep["by_kind"].get(PARSE_ERROR_RECOVERY, 0)
    if got != EXPECTED_RECOVERIES:
        bad.append(
            f"{PARSE_ERROR_RECOVERY}: {got} rows, expected exactly "
            f"{EXPECTED_RECOVERIES} — amendment 2 moved these here and step 8 "
            "requires them"
        )
    share = rep["cold"]["token_share"]
    uniform = rep["cold"].get("token_share_uniform", 0.0)
    if not allow_unweighted and uniform < COLD_TOKEN_FLOOR - 1e-6 <= share:
        bad.append(
            f"the {share:.1%} cold share holds only under a weighted sampler; "
            f"uniformly it is {uniform:.1%}, under the {COLD_TOKEN_FLOOR:.0%} "
            "floor. No trainer in this repo reads `weight` — a Stage B trainer "
            "must sample by it, or the mix must be materialised. Pass "
            "--allow-unweighted once that is decided."
        )
    if share < COLD_TOKEN_FLOOR - 1e-6:
        bad.append(
            f"cold supervised-token share is {share:.1%}, floor is "
            f"{COLD_TOKEN_FLOOR:.0%} — Stage B would drift off the format Stage A "
            "just taught"
        )
    for kind in set(rep["by_kind"]) - set(LATER_KINDS) - {
        PARSE_ERROR_RECOVERY, COLD_REPLAY
    }:
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
    ap.add_argument("--stage-a", type=Path, required=True,
                    help="Stage A build dir; its rows are excluded from the "
                         "later-action pool and replayed as the cold component")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--expect-src-sha", default=EXPECTED_SRC_SHA)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    # argparse expands %% in help, and an f-string expands the format spec
    # first — the two cannot be combined, so the number is rendered up front.
    ap.add_argument("--cold-target", type=float, default=COLD_TOKEN_TARGET,
                    help="supervised-token share to solve the cold component's "
                         "mass for (floor "
                         + f"{COLD_TOKEN_FLOOR * 100:.0f} percent)")
    ap.add_argument("--allow-unweighted", action="store_true",
                    help="accept a mix whose cold floor holds only under a "
                         "weighted sampler. Only once a Stage B trainer samples "
                         "by `weight`, or the decision is recorded in the plan.")
    ap.add_argument("--no-cold-replay", action="store_true",
                    help="build without the cold component. The token-share "
                         "floor then cannot be met and the build will fail; for "
                         "inspecting the later-action pool alone.")
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

    cold_rows = load_cold(args.stage_a)
    exclude = {r["source_id"] for r in cold_rows}
    if len(exclude) != len(cold_rows):
        raise SystemExit(
            f"{len(cold_rows)} Stage A rows collapse to {len(exclude)} keys — "
            "row_key is not unique and every weight would collide"
        )
    print(f"excluding {len(exclude)} Stage A rows from the later pool")

    sliced = slice_rows(rows, manifest, exclude=exclude)
    print(f"sliced {len(sliced)} later rows from {len(rows)} segments: "
          f"{dict(collections.Counter(r['kind'] for r in sliced))}")

    reused = [r["source_id"] for r in sliced if r["source_id"] in exclude]
    if reused:
        raise SystemExit(f"{len(reused)} row(s) reuse a Stage A action: {reused[:5]}")

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

    all_rows = sliced + ([] if args.no_cold_replay else cold_rows)

    # ---- tokenize, and verify the mask on every row ------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    wrapper = tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)
    fingerprint: list[str] = []
    tok_problems: list[str] = []
    lengths: list[int] = []
    sup_by_id: dict[str, int] = {}
    usable: list[dict] = []
    for r in all_rows:
        try:
            ex = tokenize_messages(
                r["messages"], tok, r["supervise"], max_length=args.max_length,
                template_kwargs=TEMPLATE_KWARGS, truncate=False,
            )
        except Exception as e:  # a refusal here is a build failure, not a drop
            tok_problems.append(f"{r['source_id']}: {type(e).__name__}: {e}")
            continue
        if ex is None:
            tok_problems.append(
                f"{r['source_id']}: over {args.max_length} tokens or nothing supervised"
            )
            continue
        n_sup = sum(1 for lab in ex.labels if lab != IGNORE_INDEX)
        first = next(i for i, lab in enumerate(ex.labels) if lab != IGNORE_INDEX)
        if ex.input_ids[first - len(wrapper):first] != wrapper:
            tok_problems.append(f"{r['source_id']}: reasoning wrapper is not immediately "
                                "before the supervised span")
        if not tok.decode(ex.input_ids[first:]).lstrip().startswith("{"):
            tok_problems.append(f"{r['source_id']}: supervised span does not start at the object")
        if any(lab != IGNORE_INDEX for lab in ex.labels[:first]):
            tok_problems.append(f"{r['source_id']}: label before the target span")
        lengths.append(len(ex.input_ids))
        sup_by_id[r["source_id"]] = n_sup
        fingerprint.append(row_digest(ex))
        usable.append(r)

    # ---- weights: per component, then solved against the token floor -------
    new_rows = [r for r in usable if r["kind"] != COLD_REPLAY]
    cold_kept = [r for r in usable if r["kind"] == COLD_REPLAY]
    w_new = compute_weights(new_rows) if new_rows else {}
    w_cold = compute_weights(cold_kept) if cold_kept else {}

    def per_draw(rs: list[dict], w: dict[str, float]) -> float:
        return sum(w[r["source_id"]] * sup_by_id.get(r["source_id"], 0) for r in rs)

    cold_mass = 0.0
    if cold_kept and new_rows:
        cold_mass = solve_cold_mass(
            per_draw(cold_kept, w_cold), per_draw(new_rows, w_new), args.cold_target
        )
    elif cold_kept:
        cold_mass = 1.0

    weights = {
        **{k: v * (1.0 - cold_mass) for k, v in w_new.items()},
        **{k: v * cold_mass for k, v in w_cold.items()},
    }

    # One key per row, or every number below is computed on a dict that quietly
    # lost rows to collisions. This is exactly how the first build reported a
    # mix over 60 keys for 165 rows.
    if len(weights) != len(usable) or len(sup_by_id) != len(usable):
        raise SystemExit(
            f"{len(usable)} rows produced {len(weights)} weights and "
            f"{len(sup_by_id)} token counts — source_id collision"
        )

    rep = mix_report(usable, weights, sup_by_id, cold_mass=cold_mass)
    mix_problems = check_mix(rep, allow_unweighted=args.allow_unweighted)

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / "stage_b.jsonl"
    jsonl.write_text("".join(
        json.dumps({**r, "weight": weights[r["source_id"]]}) + "\n" for r in usable
    ))
    out_sha = hashlib.sha256(jsonl.read_bytes()).hexdigest()

    (args.out / "mix_report.json").write_text(json.dumps(rep, indent=2))
    supervised = [sup_by_id[r["source_id"]] for r in usable]
    preflight = {
        "source_sha256": src_sha,
        "stage_a_dir": str(args.stage_a),
        "dataset_sha256": out_sha,
        "rows": len(usable),
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

    print(f"\nrows {len(usable)}  tokens {preflight['tokens']['min']}"
          f"..{preflight['tokens']['max']}  supervised "
          f"{preflight['supervised_tokens']['fraction']:.1%}")
    print("kinds: " + "  ".join(f"{k} {v}" for k, v in sorted(rep["by_kind"].items())))
    print("repo mass: " + "  ".join(
        f"{k} {v['mass']:.3f} ({v['rows']}r, {v['upsample']:.2f}x)"
        for k, v in sorted(rep["by_repo"].items())
    ))
    c = rep["cold"]
    print(f"cold: {c['rows']} rows, mass {c['component_mass']:.3f}, "
          f"token share {c['token_share']:.1%} weighted / "
          f"{c['token_share_uniform']:.1%} uniform (floor {COLD_TOKEN_FLOOR:.0%}), "
          f"draws/epoch {c['draws_per_epoch']:.0f} "
          f"(floor {COLD_DRAWS_FLOOR}, target {COLD_DRAWS_TARGET})")
    print(f"ops: {rep['ops']}")

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
    print(f"\nBUILD PASSED — wrote {args.out}/stage_b.jsonl ({out_sha[:16]}) "
          "and its three reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
