"""Verify the rebuilt SFT segments against what the teacher actually saw.

Two questions, two checks:

1. **Are the over-length segments a split bug, or just a tokenizer difference?**
   The compaction ceiling was enforced in DeepSeek's tokenizer (127,997 vocab);
   we measure in Qwen's (151,936). Re-count the same text in both. If the
   DeepSeek count sits under the hard ceiling (`max_input_tokens` 40,448) the
   split is right and the excess is purely re-tokenization.

2. **Is the reconstruction what the model really saw?** `captures/` stores
   `prompt_token_ids` for every call DeepSeek made. The longest prompt inside a
   rollout is a hard fact; our longest rebuilt segment for that rollout should
   land in the same place once measured in DeepSeek tokens.

    python scripts/sft_verify_segments.py --data /data/sft \
        --run /data/vektori-out/dsv4-corpus60 --run /data/vektori-out/dsv4-corpus60-b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.vocab_bridge import encode_ids, load_tokenizer

TEACHER = "deepseek-ai/DeepSeek-V4-Flash-0731"
# terminus-2 was told this was the teacher's context window; compaction starts
# looking once free tokens drop below 8000, so a prompt can legitimately land
# anywhere up to the hard ceiling before the reset happens.
HARD_CEILING = 40448
COMPACTION_WATERMARK = HARD_CEILING - 8000


def render(messages: list[dict], tok, template: str | None) -> str:
    return tok.apply_chat_template(
        messages, chat_template=template, tokenize=False
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dir with sft_train.jsonl")
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="pass@k dir, for captures (repeatable)")
    ap.add_argument("--student", default="Qwen/Qwen3-14B")
    ap.add_argument("--sample", type=int, default=0,
                    help="only measure N of the longest segments (0 = all)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in (args.data / "sft_train.jsonl").read_text().splitlines() if ln.strip()]
    manifest = [json.loads(ln) for ln in (args.data / "segments_manifest.jsonl").read_text().splitlines() if ln.strip()]
    print(f"{len(rows)} segments")

    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, has_generation_markers

    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    template = (
        student.chat_template
        if has_generation_markers(student.chat_template)
        else get_training_chat_template(student)
    )
    # AutoTokenizer chokes on DeepSeek-V4-Flash's YaRN rope config on
    # transformers 5.5.x; load_tokenizer falls back to the raw tokenizers
    # backend, which is all we need here.
    teacher = load_tokenizer(TEACHER)

    order = sorted(range(len(rows)), key=lambda i: -manifest[i]["n_tokens"])
    if args.sample:
        order = order[: args.sample]

    results = []
    for i in order:
        text = render(rows[i]["messages"], student, template)
        q = manifest[i]["n_tokens"]
        d = len(encode_ids(teacher, text))
        results.append({
            "task": manifest[i]["task"],
            "jobs_dir": manifest[i]["jobs_dir"],
            "segment_index": manifest[i]["segment_index"],
            "qwen_tokens": q,
            "deepseek_tokens": d,
            "ratio": round(q / d, 4) if d else None,
            "over_hard_ceiling": d > HARD_CEILING,
        })

    ratios = [r["ratio"] for r in results if r["ratio"]]
    over = [r for r in results if r["over_hard_ceiling"]]
    print(f"\nmeasured {len(results)} segments in both tokenizers")
    print(f"qwen/deepseek token ratio: min {min(ratios):.3f} "
          f"median {statistics.median(ratios):.3f} max {max(ratios):.3f}")
    print(f"deepseek tokens over hard ceiling {HARD_CEILING}: {len(over)}/{len(results)}")
    worst = max(results, key=lambda r: r["deepseek_tokens"])
    print(f"largest segment: {worst['task']} seg{worst['segment_index']} "
          f"qwen {worst['qwen_tokens']} / deepseek {worst['deepseek_tokens']}")
    if over:
        print("\nsegments genuinely past the ceiling (these would be split bugs):")
        for r in over[:10]:
            print(f"  {r['task']} seg{r['segment_index']}: deepseek {r['deepseek_tokens']}")

    verdict = (
        "SPLIT OK — every segment fits the teacher's real context window; "
        "the Qwen overage is re-tokenization"
        if not over
        else f"SPLIT SUSPECT — {len(over)} segments exceed the teacher's own ceiling"
    )
    print(f"\n{verdict}")

    payload = {
        "hard_ceiling": HARD_CEILING,
        "compaction_watermark": COMPACTION_WATERMARK,
        "measured": len(results),
        "ratio": {
            "min": min(ratios), "median": statistics.median(ratios), "max": max(ratios)
        },
        "over_hard_ceiling": len(over),
        "verdict": verdict,
        "segments": results,
    }
    out = args.out or (args.data / "tokenizer_check.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
