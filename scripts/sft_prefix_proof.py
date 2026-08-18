"""Prove the step-1 assert on the real corpus, not on fixtures.

`docs/SFT-SCRATCH-PLAN.md` step 1 ends with a claim that has to be measured
rather than argued: **every packed multi-action row in the existing repaired
corpus is now refused.** If any row still tokenizes, the assert is not doing
what the plan says it does, and the failure it exists to catch would ride
through into a GPU run.

The unit tests cover the row *shapes*. This covers the corpus, which is the only
place a shape we did not anticipate can show up.

    python scripts/sft_prefix_proof.py --data /data/sft-repaired/sft_repaired.jsonl

Exit 0 only when no packed row tokenizes. Reports the first-action slice
alongside as context — that is a step-1 sanity signal, **not** the Stage A set,
which also carries the parse-error recovery rows (step 3).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.dataset import (
    NonLastSupervisionError,
    PrefixInstabilityError,
    tokenize_messages,
)

TEMPLATE_KWARGS = {"enable_thinking": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="the packed jsonl to prove refusal on")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--slice-max-length", type=int, default=8192,
                    help="max_length for the first-action slice report (Stage A's cap)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(ln) for ln in args.data.read_text().splitlines() if ln.strip()]

    packed: collections.Counter[str] = collections.Counter()
    accepted = 0
    for row in rows:
        try:
            ex = tokenize_messages(
                row["messages"], tok, row["supervise"], max_length=40960,
                template_kwargs=TEMPLATE_KWARGS, truncate=False,
            )
        except NonLastSupervisionError:
            packed["NonLastSupervisionError"] += 1
        except PrefixInstabilityError:
            packed["PrefixInstabilityError"] += 1
        except Exception as e:
            packed[type(e).__name__] += 1
        else:
            packed["ACCEPTED" if ex is not None else "None"] += 1
            accepted += ex is not None

    print(f"{args.data} — {len(rows)} rows as packed: {dict(packed)}")

    # Context, not the proof: does the shape step 3 will build actually tokenize?
    ok = over = refused = 0
    for row in rows:
        i = next(k for k, s in enumerate(row["supervise"]) if s)
        sup = [False] * i + [True]
        try:
            ex = tokenize_messages(
                row["messages"][: i + 1], tok, sup,
                max_length=args.slice_max_length,
                template_kwargs=TEMPLATE_KWARGS, truncate=False,
            )
        except Exception as e:
            refused += 1
            print(f"  slice refused: {type(e).__name__}: {e}"[:140])
        else:
            ok += ex is not None
            over += ex is None
    print(f"first-action slice @ max_length={args.slice_max_length}: "
          f"{ok} tokenize, {over} over-length, {refused} refused "
          f"(context only — Stage A also carries the parse-error recoveries)")

    if accepted:
        print(f"\nFAIL: {accepted} packed row(s) still tokenize. The step-1 assert "
              "is not catching the shape it was written for.", file=sys.stderr)
        return 1
    print("\nPASS: no packed row tokenizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
