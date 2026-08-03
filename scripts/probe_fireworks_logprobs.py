#!/usr/bin/env python3
"""What can `accounts/fireworks/models/deepseek-v4-flash-0731` actually return?

Answers three questions before a real run burns an hour:

1. Does the model exist and serve chat/completions at all?
2. What is the **largest `top_logprobs` it accepts**? The docs say the ceiling is
   the serving deployment's `--max-logprobs`, default 5 on serverless -- but that
   is a per-deployment property, so the only trustworthy answer is empirical.
   Probed by walking K upward and recording the first rejection.
3. With `return_token_ids: true`, **where do the ids appear** in the response, and
   do the logprob entries carry `token_id`? `token_capture.extract_captured_completion`
   looks in a fixed set of places; if Fireworks nests them elsewhere every capture
   silently fails and the run produces an empty JSONL.

Read-only. Each probe is a 1-token generation off a 3-token prompt.

    FIREWORKS_API_KEY=... python3 scripts/probe_fireworks_logprobs.py \
        --model accounts/fireworks/models/deepseek-v4-flash-0731 --max-k 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://api.fireworks.ai/inference/v1"


def post(url: str, payload: dict, key: str, timeout: float = 120.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:800]}
    except urllib.error.URLError as e:
        return 0, {"unreachable": str(e)}


def walk(node, path="", hits=None):
    """Every path whose key mentions token_id / token_ids / logprob."""
    hits = hits if hits is not None else []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if any(s in k.lower() for s in ("token_id", "logprob")):
                if isinstance(v, list):
                    hits.append((p, f"list[{len(v)}]", str(v[:4])[:110]))
                else:
                    hits.append((p, type(v).__name__, str(v)[:110]))
            walk(v, p, hits)
    elif isinstance(node, list) and node:
        walk(node[0], f"{path}[0]", hits)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash-0731")
    ap.add_argument("--api-base", default=BASE)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2

    base = args.api_base.rstrip("/")
    report: dict = {"model": args.model, "api_base": base}
    msgs = [{"role": "user", "content": "Say hi."}]

    # -- 0. does it serve at all -------------------------------------------
    code, body = post(
        f"{base}/chat/completions",
        {"model": args.model, "messages": msgs, "max_tokens": 1, "temperature": 0},
        key,
    )
    report["reachable"] = {"status": code, "ok": 200 <= code < 300}
    print(f"[0] plain call            -> HTTP {code}")
    if not (200 <= code < 300):
        report["reachable"]["error"] = body
        print(json.dumps(body, indent=2)[:900])
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(report, fh, indent=2)
        return 1

    # -- 1. how high does top_logprobs go ----------------------------------
    print(f"[1] top_logprobs ladder 1..{args.max_k}")
    ladder = {}
    max_ok = 0
    for k in range(1, args.max_k + 1):
        code, body = post(
            f"{base}/chat/completions",
            {
                "model": args.model,
                "messages": msgs,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": k,
            },
            key,
        )
        ok = 200 <= code < 300
        width = None
        if ok:
            try:
                ent = body["choices"][0]["logprobs"]["content"][0]
                width = len(ent.get("top_logprobs") or [])
            except (KeyError, IndexError, TypeError):
                width = None
            max_ok = k
        ladder[k] = {"status": code, "ok": ok, "returned_width": width}
        if not ok:
            ladder[k]["error"] = (body.get("error") or body)
        mark = "ok" if ok else "REJECTED"
        print(f"    K={k:<3} HTTP {code:<4} {mark:<9} returned_width={width}")
        if not ok:
            break
    report["top_logprobs_ladder"] = ladder
    report["max_top_logprobs"] = max_ok
    print(f"    -> max accepted K = {max_ok}")

    # -- 2. response shape under return_token_ids --------------------------
    probe_k = max_ok or 1
    code, body = post(
        f"{base}/chat/completions",
        {
            "model": args.model,
            "messages": msgs,
            "max_tokens": 4,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": probe_k,
            "return_token_ids": True,
        },
        key,
    )
    print(f"[2] return_token_ids (K={probe_k}) -> HTTP {code}")
    report["return_token_ids"] = {"status": code, "ok": 200 <= code < 300}
    if 200 <= code < 300:
        hits = walk(body)
        report["return_token_ids"]["id_paths"] = [
            {"path": p, "type": t, "sample": s} for p, t, s in hits
        ]
        for p, t, s in hits:
            print(f"    {p:<52} {t:<12} {s}")
        top_ok = "prompt_token_ids" in body
        choice = (body.get("choices") or [{}])[0]
        ch_ok = "token_ids" in choice or "token_ids" in (choice.get("message") or {})
        report["return_token_ids"]["extractor_compatible"] = bool(top_ok and ch_ok)
        print(
            f"    extract_captured_completion compatible: "
            f"{top_ok and ch_ok}  (prompt@top={top_ok}, tokens@choice={ch_ok})"
        )
        try:
            e0 = body["choices"][0]["logprobs"]["content"][0]
            has_id = isinstance(e0.get("token_id"), int)
            alts = e0.get("top_logprobs") or []
            alt_id = bool(alts) and isinstance(alts[0].get("token_id"), int)
            report["return_token_ids"]["logprob_token_id"] = has_id
            report["return_token_ids"]["alt_token_id"] = alt_id
            print(f"    logprobs entry carries token_id: {has_id}; alternatives: {alt_id}")
        except (KeyError, IndexError, TypeError):
            print("    could not read logprobs.content[0]")
    else:
        report["return_token_ids"]["error"] = body
        print(json.dumps(body, indent=2)[:600])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
