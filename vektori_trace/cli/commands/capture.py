"""`capture-proxy` — token-id capture in front of a vLLM endpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def cmd_capture_proxy(args: argparse.Namespace) -> int:
    """Run a Phase 0.5 reverse proxy that injects return_token_ids and logs ids."""
    import signal

    from ...runtime.token_capture import CaptureProxy, load_captures

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    top_logprobs = getattr(args, "top_logprobs", None)
    if top_logprobs is not None and top_logprobs < 1:
        raise SystemExit(f"--top-logprobs must be >= 1, got {top_logprobs}")
    # Asking for alternatives without asking for logprobs returns neither.
    inject_logprobs = bool(args.logprobs or top_logprobs is not None)
    key_env = getattr(args, "upstream_api_key_env", None)
    upstream_api_key = None
    if key_env:
        upstream_api_key = os.environ.get(key_env)
        if not upstream_api_key:
            raise SystemExit(
                f"--upstream-api-key-env {key_env} is set but ${key_env} is empty"
            )
    proxy = CaptureProxy(
        upstream_api_base=args.upstream,
        capture_dir=out,
        host=args.host,
        port=args.port,
        inject_logprobs=inject_logprobs,
        top_logprobs=top_logprobs,
        upstream_api_key=upstream_api_key,
    )
    api_base = proxy.start()
    print(f"capture proxy listening at {api_base}")
    print(f"upstream: {proxy.upstream_api_base}")
    print(f"captures → {out / 'token_captures.jsonl'}")
    if inject_logprobs:
        print(f"logprobs   on, top_logprobs={top_logprobs}")
    if upstream_api_key:
        print(f"auth       overriding Authorization from ${key_env}")
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

def register_capture_proxy(sub: argparse._SubParsersAction) -> None:
    """Register the `capture-proxy` subcommand on `sub`."""
    p_cap = sub.add_parser(
        "capture-proxy",
        help=(
            "Phase 0.5: reverse-proxy a vLLM server, inject return_token_ids, and "
            "persist sampled prompt/completion ids as JSONL"
        ),
        description=(
            "Harbor agents we do not control still need sampled token ids for OPD. "
            "This proxy sits in front of your vLLM OpenAI-compatible server, forces "
            "`return_token_ids: true` on every chat/completions request, forwards "
            "the response unchanged, and appends each capture to "
            "<out>/token_captures.jsonl. Point harbor's api_base at the printed URL."
        ),
    )
    p_cap.add_argument(
        "--upstream",
        required=True,
        help="real vLLM api base, e.g. http://127.0.0.1:8000/v1",
    )
    p_cap.add_argument(
        "--out",
        default="./vektori-out/token-captures",
        help="directory for token_captures.jsonl",
    )
    p_cap.add_argument("--host", default="127.0.0.1")
    p_cap.add_argument(
        "--port",
        type=int,
        default=0,
        help="local listen port (0 = ephemeral; the printed api_base always wins)",
    )
    p_cap.add_argument(
        "--logprobs",
        action="store_true",
        help="also request per-token logprobs alongside token ids",
    )
    p_cap.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        metavar="K",
        help=(
            "request the top-K alternatives at each generated position (implies "
            "--logprobs). Fireworks rejects K above the deployment's "
            "--max-logprobs, which defaults to 5 on serverless; not clamped, "
            "because K is part of the objective"
        ),
    )
    p_cap.add_argument(
        "--upstream-api-key-env",
        default=None,
        metavar="VAR",
        help=(
            "read the upstream bearer token from this env var and override the "
            "client's Authorization header (e.g. FIREWORKS_API_KEY)"
        ),
    )
    p_cap.set_defaults(func=cmd_capture_proxy)
