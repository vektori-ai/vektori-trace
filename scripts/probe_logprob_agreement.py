#!/usr/bin/env python3
"""Do the trainer and the rollout engine agree on logprobs for identical input?

OPD's loss is defined against the student's own sampling distribution. If the
engine that *generates* a rollout and the engine that *scores* it compute
different logprobs for the same tokens, the on-policy assumption is quietly
false: the loss stays finite, the run completes, and the numbers look plausible.
Nothing in the logs shows it. This is the same failure signature as the 256-token
cap -- see docs/action-length-measurement.md.

Two known causes, and they are independent:

1. **Quantization path.** NF4 base under the trainer vs whatever SGLang
   dequantizes to. QLoRA's own result is that NF4+double-quant matches BF16,
   so this *should* be small -- but "should" is why it needs measuring.
2. **Kernel/dtype.** arXiv:2510.26788 ("Defeating the Training-Inference
   Mismatch via FP16") shows train and inference engines disagree even at
   matched precision, because BF16's 7 mantissa bits are coarse enough that
   different kernels round differently. FP16's 10 mantissa bits agree. This
   affects **BF16 too** -- it is not an NF4-only problem, so run the probe
   whichever dtype you pick.

Method: teacher-force one fixed text through both paths and compare per-token
logprobs. No sampling -- sampling noise would swamp the effect being measured.

    # 1. serve the base (matching the trainer's quantization)
    python3 -m sglang.launch_server --model-path Qwen/Qwen3-14B \
        --quantization bitsandbytes --port 30000

    # 2. probe
    python3 scripts/probe_logprob_agreement.py \
        --model Qwen/Qwen3-14B --load-4bit --sglang-url http://127.0.0.1:30000

Exit code is 1 if agreement fails the thresholds, so this is CI-able.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request

# Defaults chosen to catch drift, not to be lenient. A well-matched pair lands
# well under these; a genuinely broken pair blows past them.
MAX_ABS_DEFAULT = 0.05     # max |delta| on any single token's logprob
MAX_MEAN_DEFAULT = 0.01    # mean |delta| across all scored tokens
MAX_KL_DEFAULT = 1e-3      # DEPRECATED: kept only so --max-kl still parses

PROMPT = (
    "You are a customer service agent. The user asks to exchange a delivered "
    "item for a different variant. Call the exchange tool with the order id, "
    "the item id being returned, the replacement item id, and the payment method."
)


def post(url: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"sglang HTTP {e.code}: {e.read().decode()[:400]}")
    except urllib.error.URLError as e:
        sys.exit(f"sglang unreachable at {url}: {e.reason}\n"
                 "Is launch_server up, and does --quantization match the trainer?")


def trainer_logprobs(model_path: str, token_ids: list[int], load_4bit: bool,
                     dtype: str):
    """Score fixed token ids through the HF/trainer path. Teacher-forced."""
    import torch
    from transformers import AutoModelForCausalLM

    kw: dict = {"device_map": "cuda:0"}
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    if load_4bit:
        from transformers import BitsAndBytesConfig
        # double_quant matters: QLoRA's "matches BF16" result is specifically
        # with it on. Without it you are in the degraded regime.
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        kw["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.eval()

    ids = torch.tensor([token_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits.float()

    # token i is predicted by logits at i-1; drop the first (unconditioned)
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    picked = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in picked.cpu()], model.config


def sglang_logprobs(url: str, token_ids: list[int]):
    """Score the same ids through SGLang. max_new_tokens=0 => pure scoring."""
    out = post(
        f"{url.rstrip('/')}/generate",
        {
            "input_ids": token_ids,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )
    if isinstance(out, list):
        out = out[0]
    entries = (out.get("meta_info") or {}).get("input_token_logprobs")
    if not entries:
        sys.exit(f"no input_token_logprobs in response; got keys "
                 f"{list((out.get('meta_info') or {}).keys())}")
    # entries are [logprob, token_id, text]; first token is unconditioned (null)
    return [e[0] for e in entries if e[0] is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path, same one SGLang serves")
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    ap.add_argument("--load-4bit", action="store_true",
                    help="trainer loads NF4; match SGLang --quantization bitsandbytes")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"],
                    help="float16 tests the arXiv:2510.26788 recommendation")
    ap.add_argument("--text", default=PROMPT)
    ap.add_argument("--max-abs", type=float, default=MAX_ABS_DEFAULT)
    ap.add_argument("--max-mean", type=float, default=MAX_MEAN_DEFAULT)
    # Accepted and ignored, so an existing invocation does not break. The
    # metric it gated was not a KL divergence; see the verdict below.
    ap.add_argument("--max-kl", type=float, default=MAX_KL_DEFAULT,
                    help="DEPRECATED and ignored: the metric it gated was not a "
                         "valid KL estimate and has been removed.")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    token_ids = tok(a.text)["input_ids"]
    print(f"probe: {len(token_ids)} tokens | model={a.model} "
          f"| trainer={'nf4' if a.load_4bit else a.dtype}")

    tr, cfg = trainer_logprobs(a.model, token_ids, a.load_4bit, a.dtype)
    sg = sglang_logprobs(a.sglang_url, token_ids)

    # A length mismatch is FATAL, never a warning.
    #
    # This used to print a WARN and compare `min(len(tr), len(sg))` positions.
    # Truncating two differently-tokenized sequences aligns position i of one
    # with position i of the other, which are then unrelated tokens -- so the
    # probe can report close "agreement" between values that were never
    # comparable, and a tokenizer misalignment passes the very check meant to
    # catch it. If the two sides disagree on the token count they disagree on
    # the thing being measured.
    if len(tr) != len(sg):
        print(f"\n  FATAL length mismatch: trainer={len(tr)} sglang={len(sg)}.\n"
              f"  These are different tokenizations, so position-wise comparison "
              f"is meaningless.\n  Fix the tokenizer/offset alignment; do not "
              f"compare a truncated prefix.", file=sys.stderr)
        return 2
    n = len(tr)

    deltas = [abs(x - y) for x, y in zip(tr, sg)]
    mean_d = statistics.fmean(deltas)
    max_d = max(deltas)

    print(f"\n  mean |delta|  {mean_d:.6f}   (limit {a.max_mean})")
    print(f"  max  |delta|  {max_d:.6f}   (limit {a.max_abs})")

    worst = sorted(range(n), key=lambda i: -deltas[i])[:5]
    print("\n  worst tokens:")
    for i in worst:
        t = tok.decode([token_ids[i + 1]])
        print(f"    [{i:4d}] {t!r:16s} trainer={tr[i]:9.5f} sglang={sg[i]:9.5f} "
              f"d={deltas[i]:.5f}")

    # The verdict is mean and max absolute logprob delta, which is exactly the
    # agreement question this probe exists to answer.
    #
    # A third term used to gate it, printed as "mean KL": the sample mean of
    # exp(lp_p) * (lp_p - lp_q) over realized tokens. That is not KL divergence
    # and not a lower bound on it -- it is a signed statistic over one token per
    # position, so individual terms can be negative and a genuinely divergent
    # pair can sit under an upper threshold. Removed rather than repaired: the
    # absolute deltas already answer the question, and a metric that can pass a
    # bad pair is worse than no metric.
    ok = mean_d <= a.max_mean and max_d <= a.max_abs
    print(f"\n  {'PASS' if ok else 'FAIL'} -- train/serve logprobs "
          f"{'agree' if ok else 'DIVERGE'}")
    if not ok:
        print("  Do not trust an OPD run on this pair. Try: matching quantization "
              "on both sides, or --dtype float16 (arXiv:2510.26788).")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({
                "model": a.model, "trainer": "nf4" if a.load_4bit else a.dtype,
                "n_tokens": n, "mean_abs": mean_d, "max_abs": max_d,
                "pass": ok, "trainer_logprobs": tr, "sglang_logprobs": sg,
            }, f, indent=1)
        print(f"  wrote {a.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
