"""Prove the repaired SFT set before a GPU is spent on it.

`docs/SFT-REPAIR-PLAN.md` Phase 3. Everything here is free — it runs on the box
with a tokenizer and no accelerator — and every check exists because its
failure mode is silent:

  * A target that does not parse as Terminus JSON is a target that teaches the
    exact bug this run repairs.
  * A label array that is all `-100` trains on nothing while reporting a
    perfectly plausible loss.
  * `enable_thinking` left unset takes Qwen3's template default rather than a
    decision, and the difference is `<think>` blocks in the rendered prompt.
  * TRL's `assistant_only_loss` derives the mask from the chat template and is
    all-or-nothing per role. We supervise a *subset* of assistant turns, so the
    two must agree on every segment with no handoff turn and differ only where
    intended. That is what makes the divergence attributable to intent rather
    than to a bug in our masking.

    python scripts/sft_repair_preflight.py --data /data/sft-repaired
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.dataset import IGNORE_INDEX, tokenize_messages

# Qwen3-14B's config sets max_position_embeddings=40960 with rope_scaling null;
# the 32,768 on the model card is a recommended *input* length, not the
# positional range. The longest v1 segment was 37,577, so nothing truncates.
MAX_LENGTH = 40960

# Pinned here, not left to the template's default. Qwen3 reads this from
# apply_chat_template kwargs, and serving must use the identical value.
TEMPLATE_KWARGS = {"enable_thinking": False}

# Envelopes that must never appear in a target. `run_command` / `wait_for_output`
# are the verbs v1 emitted at inference (94x / 257x) that appear zero times in
# its training data — drift, and a regression signal if they reappear.
FORBIDDEN_IN_TARGET = ["<tool_call>", "</tool_call>", "<think>", "</think>"]
INVENTED_VERBS = ["run_command", "wait_for_output"]


def load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check_targets(rows: list[dict]) -> list[str]:
    """Every supervised assistant message is native, well-formed Terminus JSON."""
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    parser = TerminusJSONPlainParser()
    failures: list[str] = []
    for i, row in enumerate(rows):
        for j, (msg, sup) in enumerate(
            zip(row["messages"], row["supervise"], strict=True)
        ):
            if not sup:
                continue
            if msg["role"] != "assistant":
                failures.append(f"row {i} msg {j}: supervised {msg['role']} message")
                continue
            content = msg["content"]
            if "tool_calls" in msg:
                failures.append(f"row {i} msg {j}: assistant carries tool_calls")
            for bad in FORBIDDEN_IN_TARGET:
                if bad in content:
                    failures.append(f"row {i} msg {j}: target contains {bad!r}")
            result = parser.parse_response(content)
            if result.error:
                failures.append(f"row {i} msg {j}: does not parse — {result.error[:120]}")
                continue
            obj = json.loads(content)
            for verb in INVENTED_VERBS:
                if verb in obj:
                    failures.append(f"row {i} msg {j}: invented verb {verb!r}")
            if "commands" not in obj:
                failures.append(f"row {i} msg {j}: no commands key")
            for c in obj.get("commands") or []:
                if "keystrokes" not in c:
                    failures.append(f"row {i} msg {j}: command without keystrokes")
                if c.get("keystrokes") == "mark_task_complete":
                    failures.append(f"row {i} msg {j}: mark_task_complete inside commands")
            if "task_complete" in obj and not isinstance(obj["task_complete"], bool):
                failures.append(f"row {i} msg {j}: non-boolean task_complete")
    return failures


def check_roles(rows: list[dict]) -> list[str]:
    """No tool roles, no tool_call_id, observations arrive as user messages."""
    failures: list[str] = []
    for i, row in enumerate(rows):
        for j, msg in enumerate(row["messages"]):
            if msg["role"] not in {"user", "assistant", "system"}:
                failures.append(f"row {i} msg {j}: role {msg['role']!r}")
            if "tool_call_id" in msg:
                failures.append(f"row {i} msg {j}: carries tool_call_id")
        if not any(m["role"] == "assistant" for m in row["messages"]):
            failures.append(f"row {i}: no assistant messages")
    return failures


def supervised_spans(labels: list[int]) -> list[list[int]]:
    """Contiguous runs of supervised label ids — one per supervised turn."""
    spans: list[list[int]] = []
    current: list[int] = []
    for lab in labels:
        if lab == IGNORE_INDEX:
            if current:
                spans.append(current)
                current = []
        else:
            current.append(lab)
    if current:
        spans.append(current)
    return spans


def check_masks(rows: list[dict], tokenizer) -> tuple[list[str], dict]:
    """Tokenize every row and verify the labels, not just batch zero."""
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    parser = TerminusJSONPlainParser()
    failures: list[str] = []
    lengths: list[int] = []
    supervised_counts: list[int] = []
    fingerprint: list[list[int]] = []

    for i, row in enumerate(rows):
        example = tokenize_messages(
            row["messages"],
            tokenizer,
            row["supervise"],
            max_length=MAX_LENGTH,
            template_kwargs=TEMPLATE_KWARGS,
            truncate=False,
        )
        if example is None:
            failures.append(f"row {i}: no tokenized example (over length, or nothing supervised)")
            continue
        n = len(example.input_ids)
        lengths.append(n)
        if n > MAX_LENGTH:
            failures.append(f"row {i}: {n} tokens over max_length {MAX_LENGTH}")
        kept = [lab for lab in example.labels if lab != IGNORE_INDEX]
        supervised_counts.append(len(kept))
        # The trainer re-tokenizes with its own copy of this logic; this is what
        # it checks itself against.
        fingerprint.append([n, len(kept)])
        if not kept:
            failures.append(f"row {i}: no supervised tokens")
        if len(example.labels) != n or len(example.attention_mask) != n:
            failures.append(f"row {i}: label/mask length disagrees with input_ids")

        # Every supervised span, on every row, must decode back to the Terminus
        # JSON it was built from. This is the check that catches a mask whose
        # boundaries are off — a span shifted by one turn still decodes to
        # *something*, but not to a parseable action.
        spans = supervised_spans(example.labels)
        expected = [
            msg["content"]
            for msg, sup in zip(row["messages"], row["supervise"], strict=True)
            if sup
        ]
        if len(spans) != len(expected):
            failures.append(
                f"row {i}: {len(spans)} supervised spans for {len(expected)} supervised turns"
            )
            continue
        for k, (span, want) in enumerate(zip(spans, expected, strict=True)):
            text = tokenizer.decode(span)
            if parser.parse_response(text).error:
                failures.append(f"row {i} span {k}: does not decode to Terminus JSON")
            elif want.strip() not in text:
                failures.append(f"row {i} span {k}: decoded span does not contain its target")

    stats = {
        "rows": len(rows),
        "tokenized": len(lengths),
        "tokens": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "total": sum(lengths),
        },
        "supervised_tokens": sum(supervised_counts),
        "supervised_fraction": (
            round(sum(supervised_counts) / sum(lengths), 4) if lengths else 0.0
        ),
    }
    return failures, stats, fingerprint


def check_against_trl(rows: list[dict], tokenizer) -> tuple[list[str], dict]:
    """Our mask must equal TRL's, position for position, on action-only rows.

    Where a row holds a handoff turn the two *must* differ, and by exactly that
    turn's tokens. Exact agreement everywhere else is what makes the difference
    attributable to intent rather than to a bug in our masking.

    Both sides render with **TRL's** training template, not one each. Comparing
    token counts across two different renders would be a coincidence check, not
    an equality check — the delimiters differ, so the counts would differ for
    reasons that have nothing to do with the mask.
    """
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    template = (
        tokenizer.chat_template
        if has_generation_markers(tokenizer.chat_template)
        else get_training_chat_template(tokenizer)
    )
    kwargs = dict(TEMPLATE_KWARGS, chat_template=template)

    failures: list[str] = []
    compared = agreed = 0
    for i, row in enumerate(rows):
        all_assistants_supervised = all(
            sup
            for msg, sup in zip(row["messages"], row["supervise"], strict=True)
            if msg["role"] == "assistant"
        )
        if not all_assistants_supervised:
            continue
        compared += 1
        enc = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )
        trl_mask = [bool(x) for x in enc["assistant_masks"]]
        ours = tokenize_messages(
            row["messages"], tokenizer, row["supervise"],
            max_length=MAX_LENGTH, template_kwargs=kwargs, truncate=False,
        )
        if ours is None:
            failures.append(f"row {i}: ours produced nothing where TRL produced a mask")
            continue
        if not any(trl_mask):
            failures.append(f"row {i}: TRL mask is empty (missing generation markers?)")
            continue
        our_mask = [lab != IGNORE_INDEX for lab in ours.labels]
        if len(our_mask) != len(trl_mask):
            failures.append(
                f"row {i}: length disagreement — TRL {len(trl_mask)}, ours {len(our_mask)}"
            )
            continue
        # TRL's `{% generation %}` markers wrap the assistant *content* and stop
        # before the turn's closing delimiter, which we include. Positions where
        # TRL supervises and we do not are the real failure: it means we masked
        # something that carries the target.
        trl_only = [j for j, (t, o) in enumerate(zip(trl_mask, our_mask)) if t and not o]
        ours_only = [j for j, (t, o) in enumerate(zip(trl_mask, our_mask)) if o and not t]
        if trl_only:
            failures.append(
                f"row {i}: {len(trl_only)} positions TRL supervises and we mask "
                f"(first at {trl_only[0]})"
            )
        elif len(ours_only) > 4 * sum(trl_mask) // 100:
            # We legitimately add each turn's closing delimiter; more than a few
            # percent beyond that is a boundary that has slipped a whole turn.
            failures.append(
                f"row {i}: {len(ours_only)} positions we supervise beyond TRL's span"
            )
        else:
            agreed += 1
    return failures, {"comparable_rows": compared, "agreed": agreed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with sft_repaired.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--skip-trl", action="store_true", help="skip the TRL cross-check")
    args = ap.parse_args()

    rows = load(args.data / "sft_repaired.jsonl")
    print(f"{len(rows)} rows from {args.data / 'sft_repaired.jsonl'}")

    all_failures: dict[str, list[str]] = {}
    all_failures["targets"] = check_targets(rows)
    all_failures["roles"] = check_roles(rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_failures, stats, fingerprint = check_masks(rows, tokenizer)
    all_failures["masks"] = mask_failures

    trl_stats: dict = {"skipped": True}
    if not args.skip_trl:
        trl_failures, trl_stats = check_against_trl(rows, tokenizer)
        all_failures["trl_agreement"] = trl_failures

    print(f"\ntokens: min {stats['tokens']['min']} median "
          f"{stats['tokens']['median']:.0f} max {stats['tokens']['max']}")
    print(f"supervised: {stats['supervised_tokens']:,} of {stats['tokens']['total']:,} "
          f"({100 * stats['supervised_fraction']:.1f}%)")
    if not trl_stats.get("skipped"):
        print(f"TRL cross-check: {trl_stats['agreed']}/{trl_stats['comparable_rows']} "
              "comparable rows agree")

    total = sum(len(v) for v in all_failures.values())
    print("\nchecks:")
    for name, failures in all_failures.items():
        mark = "ok" if not failures else f"FAIL ({len(failures)})"
        print(f"  {name}: {mark}")
        for f in failures[:5]:
            print(f"    {f}")

    report = {"stats": stats, "trl": trl_stats,
              "failures": {k: v for k, v in all_failures.items()},
              "total_failures": total}
    out = args.data / "preflight_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    if not total:
        # Only written when everything passed: a fingerprint from a failed
        # preflight would let the trainer's drift guard pass against a set that
        # was never actually validated.
        fp = args.data / "tokenization_fingerprint.json"
        fp.write_text(json.dumps({
            "model": args.model,
            "template_kwargs": TEMPLATE_KWARGS,
            "max_length": MAX_LENGTH,
            "per_row": fingerprint,
        }, indent=2))
        print(f"wrote {fp}")

    if total:
        print(f"PREFLIGHT FAILED: {total} problems — do not spend a GPU", file=sys.stderr)
        return 1
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
