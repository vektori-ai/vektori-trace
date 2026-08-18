"""Run the Stage A trainer's fingerprint check on CPU, before any GPU is booked.

`sft_stage_a_train_modal.py` verifies its inlined `tokenize_row` against the
builder's `tokenization_fingerprint.json` — but it does so *inside the Modal
container*, after the image build and the 14B download are already paid for. The
check is pure CPU and pure tokenizer, so there is no reason to discover a drift
there. This runs the identical comparison locally.

    .venv/bin/python scripts/sft_stage_a_verify_tokenization.py --data /data/sft-stage-a

Exit 0 means the rows the trainer will build are byte-identical to the rows the
builder validated, and a probe is worth approving. Non-zero means it is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Run as a plain script (`python scripts/...`) and the repo root is not on the
# path, so `scripts.` does not resolve. pytest gets this from rootdir; a bare
# invocation on the box has to do it itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sft_stage_a_train_modal import (
    IGNORE_INDEX,
    MAX_LENGTH,
    TEMPLATE_KWARGS,
    row_digest,
    tokenize_row,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Stage A dataset directory (stage_a.jsonl + fingerprint)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    data_path = args.data / "stage_a.jsonl"
    fp_path = args.data / "tokenization_fingerprint.json"
    for p in (data_path, fp_path):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    expected = json.loads(fp_path.read_text())
    rows = [json.loads(ln) for ln in data_path.read_text().splitlines() if ln.strip()]
    print(f"{len(rows)} rows from {data_path}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    problems: list[str] = []
    if expected.get("model") != args.model:
        problems.append(f"fingerprint model {expected.get('model')!r} != {args.model!r}")
    if expected.get("template_kwargs") != TEMPLATE_KWARGS:
        problems.append(
            f"fingerprint template_kwargs {expected.get('template_kwargs')} "
            f"!= trainer {TEMPLATE_KWARGS}"
        )
    if expected.get("max_length") != MAX_LENGTH:
        problems.append(
            f"fingerprint max_length {expected.get('max_length')} != "
            f"trainer MAX_LENGTH {MAX_LENGTH}"
        )
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if expected.get("dataset_sha256") not in (None, data_sha):
        problems.append(
            f"fingerprint dataset {expected['dataset_sha256'][:16]} != {data_sha[:16]}"
        )

    examples = [tokenize_row(r, tok) for r in rows]
    if any(e is None for e in examples):
        idx = [i for i, e in enumerate(examples) if e is None]
        problems.append(f"{len(idx)} row(s) produced no supervised example: {idx[:5]}")
        examples = [e for e in examples if e is not None]

    actual = [row_digest(e) for e in examples]
    per_row = expected.get("per_row", [])
    if per_row != actual:
        bad = [
            i for i, (a, b) in enumerate(zip(per_row, actual, strict=False)) if a != b
        ]
        problems.append(
            f"per-row digests disagree on {len(bad)} row(s) "
            f"(first row {bad[0] if bad else '?'}; {len(per_row)} vs {len(actual)})"
        )

    supervised = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    lengths = sorted(len(e["input_ids"]) for e in examples)
    # The wrapper mask, checked on the data rather than on a synthetic row: a
    # supervised span that opens with <think> is the step-1 failure the plan
    # says to catch on CPU.
    heads = [
        tok.decode([x for x in e["labels"] if x != IGNORE_INDEX][:12]).lstrip()
        for e in examples
    ]
    wrapped = [i for i, h in enumerate(heads) if h.startswith("<think>")]
    if wrapped:
        problems.append(
            f"{len(wrapped)} supervised span(s) start with <think> "
            f"(first row {wrapped[0]}) — the wrapper mask failed"
        )
    if lengths and lengths[-1] > MAX_LENGTH:
        problems.append(f"longest row {lengths[-1]} exceeds MAX_LENGTH {MAX_LENGTH}")

    print(f"dataset sha256   {data_sha}")
    print(f"supervised       {supervised}/{total} ({100 * supervised / total:.1f}%)")
    print(f"tokens           min {lengths[0]}  median {lengths[len(lengths) // 2]}  "
          f"max {lengths[-1]}  (MAX_LENGTH {MAX_LENGTH})")
    print(f"first span       {heads[0][:60]!r}")

    if problems:
        print("\nFAIL:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"\nOK — trainer tokenization matches the builder on all {len(actual)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
