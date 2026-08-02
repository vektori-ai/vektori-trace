#!/usr/bin/env python3
"""Live capacity monitor for the served student — what the GPU can and cannot do.

vLLM exposes Prometheus metrics at `/metrics`. The one that matters is
`vllm:gpu_cache_usage_perc`: the fraction of the KV cache currently in use.
Everything else follows from it.

    KV usage near 100%  +  requests waiting  →  you are KV-bound.
        Nothing you do to the model helps. Either shorten context, cut
        concurrency, or move to a bigger card.

    KV usage low  +  requests waiting        →  you are NOT KV-bound.
        Something upstream is the bottleneck: the verifier containers,
        the GitHub API, or your own request rate.

That distinction is the whole point of watching this. "It's slow" is not
actionable; "KV is at 98% with 6 queued" is.

Usage:
    uv run python scripts/vllm_monitor.py --api-base $STUDENT_API_BASE
    uv run python scripts/vllm_monitor.py --api-base $STUDENT_API_BASE --once
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# Qwen3-8B: 36 layers x 8 GQA KV heads x 128 head_dim x 2 (K,V) x 2 bytes
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2

WANTED = {
    "vllm:gpu_cache_usage_perc": "kv_used_frac",
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_requests_swapped": "swapped",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "gen_tokens",
}


def scrape(api_base: str, timeout: float = 5.0) -> dict[str, float]:
    """Fetch /metrics and pull the handful of series we care about."""
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    with urllib.request.urlopen(root + "/metrics", timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")

    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if not m:
            continue
        key = WANTED.get(m.group(1))
        if key:
            try:
                # Multiple label-sets for one series: sum them.
                out[key] = out.get(key, 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def bar(frac: float, width: int = 32) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def verdict(kv: float, waiting: float, running: float) -> str:
    if waiting > 0 and kv >= 0.90:
        return "KV-BOUND — shorten context, lower concurrency, or use a bigger GPU"
    if waiting > 0 and kv < 0.50:
        return "NOT KV-bound — bottleneck is upstream (verifier / API / request rate)"
    if kv >= 0.95:
        return "KV nearly full — next request may be queued or preempted"
    if running == 0 and waiting == 0:
        return "idle"
    return "healthy — headroom available"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", required=True, help="e.g. https://...modal.run/v1")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--once", action="store_true", help="print one sample and exit")
    ap.add_argument("--kv-total-tokens", type=int, default=None,
                    help="total KV token budget, to show tokens not just %%")
    ap.add_argument("--log", default=None,
                    help="append each sample as JSON to this file. The bar is "
                         "for watching; this is for answering questions after "
                         "the run, joined against passk_log.jsonl by timestamp")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(sample: dict) -> None:
        if log_path is None:
            return
        with log_path.open("a") as fh:
            fh.write(json.dumps(sample) + "\n")
            fh.flush()

    prev_gen, prev_t = None, None
    print(f"watching {args.api_base}   (Ctrl-C to stop)\n")
    try:
        while True:
            try:
                m = scrape(args.api_base)
            except urllib.error.URLError as e:
                print(f"\r{time.strftime('%H:%M:%S')}  unreachable: {e.reason}",
                      end="", flush=True)
                # An unreachable endpoint is a data point, not a gap. A hole in
                # the log is ambiguous between "monitor was down" and "endpoint
                # was down"; this records which.
                _record({"logged_at": time.time(), "error": str(e.reason)})
                if args.once:
                    return 1
                time.sleep(args.interval)
                continue

            kv = m.get("kv_used_frac", 0.0)
            running = m.get("running", 0.0)
            waiting = m.get("waiting", 0.0)
            gen = m.get("gen_tokens", 0.0)

            now = time.time()
            tps = ""
            tok_per_s = None
            if prev_gen is not None and now > prev_t:
                rate = (gen - prev_gen) / (now - prev_t)
                if rate >= 0:
                    tok_per_s = rate
                    tps = f"  {rate:6.1f} tok/s"
            prev_gen, prev_t = gen, now

            _record({
                "logged_at": now,
                "kv_used_frac": kv,
                "kv_used_tokens": (
                    int(kv * args.kv_total_tokens) if args.kv_total_tokens else None
                ),
                "running": running,
                "waiting": waiting,
                "gen_tokens_total": gen,
                "tokens_per_sec": tok_per_s,
                "verdict": verdict(kv, waiting, running),
            })

            kv_tokens = ""
            if args.kv_total_tokens:
                kv_tokens = f" ({int(kv * args.kv_total_tokens):,}/{args.kv_total_tokens:,} tok)"

            print(f"{time.strftime('%H:%M:%S')}  KV [{bar(kv)}] {kv * 100:5.1f}%{kv_tokens}"
                  f"  run {int(running):>2}  wait {int(waiting):>2}{tps}")
            print(f"           {verdict(kv, waiting, running)}")

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
