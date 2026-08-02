"""`capture-proxy` — token-id capture in front of a vLLM endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_capture_proxy(args: argparse.Namespace) -> int:
    """Run a Phase 0.5 reverse proxy that injects return_token_ids and logs ids."""
    import signal

    from ...token_capture import CaptureProxy, load_captures

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy = CaptureProxy(
        upstream_api_base=args.upstream,
        capture_dir=out,
        host=args.host,
        port=args.port,
        inject_logprobs=args.logprobs,
    )
    api_base = proxy.start()
    print(f"capture proxy listening at {api_base}")
    print(f"upstream: {proxy.upstream_api_base}")
    print(f"captures → {out / 'token_captures.jsonl'}")
    print("point harbor --api-base (or --ak api_base=...) at this URL")
    print("Ctrl-C to stop")

    def _stop(signum, frame):
        proxy.stop()
        n = len(load_captures(out))
        print(f"\nstopped — {n} completion(s) captured under {out}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while True:
            signal.pause()
    except AttributeError:
        # Windows has no signal.pause; sleep-loop instead.
        import time

        while True:
            time.sleep(3600)
    return 0
