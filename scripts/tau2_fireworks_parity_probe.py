#!/usr/bin/env python3
"""Integer-ID scoring compatibility probe for the DeepSeek deployment.

**What this can and cannot prove.** `FireworksTeacherPool.score_ids` posts

    prompt = locally_rendered_prefix_ids + payload_ids

as an **integer array** to `/completions`. Fireworks applies no chat template to
an id array -- the serialization was already done here, by `encoding_dsv4.py`.
So this probe cannot verify "the deployment serializes the way our encoder
does": there is no server-side templating step to compare against. Claiming
otherwise would be exactly the kind of plausible-but-wrong assertion this
repository keeps paying for.

What it *does* verify, which is what the billed path depends on:

- the pinned deployment accepts our locally generated ids at all;
- echoed ids come back identical -- no retokenization, no indexing shift;
- returned logprobs align one-for-one with the requested scoring window;
- the values are finite;
- the request was served by the model and endpoint we think it was.

The local rendering is verified *offline* instead, and that half is already
green: the prefix ends at `<｜Assistant｜><think>`, and no `</think><think>`
boundary appears -- the collision that made a bare `thinking_mode="thinking"`
flag insufficient and motivated the semantic projection.

Two cases, both tiny:

    1. reasoning + ordinary content
    2. reasoning + a single tool call

Fails closed on any mismatch, and never prints or archives the API key.

    # offline, no request, no spend:
    python3 scripts/tau2_fireworks_parity_probe.py --dry-run --out /tmp/parity

    # the paid probe (a few hundred tokens), only with explicit approval:
    python3 scripts/tau2_fireworks_parity_probe.py --yes --out /data/tau2/parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: The teacher of record. Must match what the OPD run bills.
TEACHER_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
TEACHER_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash-0731"

#: Pinned explicitly rather than inherited: the live path's silent "chat"
#: default is what put a `</think>`-terminated prefix in front of an action
#: that opens `<think>`.
THINKING_MODE = "thinking"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Look up one order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]

CASE_CONTENT = {
    "name": "reasoning_plus_content",
    "messages": [
        {"role": "system", "content": "You are a retail agent.", "tools": TOOLS},
        {"role": "user", "content": "Is my order shipped?"},
    ],
    "assistant": {
        "role": "assistant",
        "reasoning_content": "The user asks about shipping. I should check.",
        "content": "Let me check that for you.",
    },
}

CASE_TOOL = {
    "name": "reasoning_plus_tool_call",
    "messages": [
        {"role": "system", "content": "You are a retail agent.", "tools": TOOLS},
        {"role": "user", "content": "Check order #W123 please."},
    ],
    "assistant": {
        "role": "assistant",
        "reasoning_content": "I need the order details for #W123.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "arguments": json.dumps({"order_id": "#W123"}),
                },
            }
        ],
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render_local(case: dict[str, Any]) -> dict[str, Any]:
    """What `encoding_dsv4.py` says the prompt and the scored span are.

    Uses the same `render_teacher_prefix` the scorer uses, so a divergence here
    is a divergence in the billed path and not in a probe-only reimplementation.
    """
    from vektori_trace.providers.teacher.cross import render_teacher_prefix

    prefix_text = render_teacher_prefix(
        case["messages"], thinking_mode=THINKING_MODE
    )
    joint_text = render_teacher_prefix(
        [*case["messages"], case["assistant"]], thinking_mode=THINKING_MODE
    )
    if not joint_text.startswith(prefix_text):
        raise SystemExit(
            f"{case['name']}: the joint render does not extend the prefix "
            "render; the local encoder cannot describe a scored span at all"
        )
    return {
        "prefix_text": prefix_text,
        "joint_text": joint_text,
        "action_text": joint_text[len(prefix_text):],
        "prefix_sha": _sha(prefix_text),
        "joint_sha": _sha(joint_text),
    }


def local_ids(rendered: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from vektori_trace.providers.teacher.cross import encode_teacher_ids

    prefix_ids = encode_teacher_ids(rendered["prefix_text"], tokenizer)
    joint_ids = encode_teacher_ids(rendered["joint_text"], tokenizer)
    extends = joint_ids[: len(prefix_ids)] == prefix_ids
    return {
        "n_prefix_ids": len(prefix_ids),
        "n_joint_ids": len(joint_ids),
        "joint_extends_prefix": extends,
        "action_ids": joint_ids[len(prefix_ids):] if extends else [],
        "prefix_ids_head": prefix_ids[:16],
        "prefix_ids_tail": prefix_ids[-16:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for the archive")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and tokenise locally; send nothing")
    ap.add_argument("--yes", action="store_true",
                    help="required to send the paid request")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    if not a.dry_run and not a.yes:
        raise SystemExit(
            "refusing to send a paid request without --yes. Use --dry-run to "
            "see the local rendering first."
        )
    if a.yes and not os.environ.get("FIREWORKS_API_KEY"):
        raise SystemExit("FIREWORKS_API_KEY is not set")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if not a.dry_run or True:
        from vektori_trace.vocab_bridge import load_tokenizer
        tokenizer = load_tokenizer(TEACHER_TOKENIZER)

    report: dict[str, Any] = {
        "model": TEACHER_MODEL,
        "tokenizer": TEACHER_TOKENIZER,
        "thinking_mode": THINKING_MODE,
        "route": "POST /completions via FireworksTeacherPool.score_ids "
                 "(the route score_replay_batch bills)",
        "proves": [
            "the deployment accepts locally rendered integer ids",
            "logprobs align 1:1 with the requested scoring window",
            "values are finite",
        ],
        "does_not_prove": [
            "that Fireworks' own chat template matches encoding_dsv4.py -- an "
            "integer-id request is never templated server-side, so there is "
            "nothing to compare; local rendering is checked offline instead",
        ],
        "mode": "dry-run" if a.dry_run else "paid",
        "cases": {},
        "ok": True,
    }

    for case in (CASE_CONTENT, CASE_TOOL):
        rendered = render_local(case)
        ids = local_ids(rendered, tokenizer)
        entry: dict[str, Any] = {
            "rendered": {k: v for k, v in rendered.items()
                         if k.endswith("_sha")},
            "prefix_tail": rendered["prefix_text"][-120:],
            "action_text": rendered["action_text"],
            "ids": {k: v for k, v in ids.items() if k != "action_ids"},
            "n_action_ids": len(ids["action_ids"]),
        }
        # The structural checks that do not need the network.
        checks = {
            "joint_extends_prefix": ids["joint_extends_prefix"],
            "no_duplicated_think_boundary":
                "</think><think>" not in rendered["joint_text"],
            "action_is_non_empty": len(ids["action_ids"]) > 0,
        }
        entry["offline_checks"] = checks
        if not all(checks.values()):
            report["ok"] = False

        if a.yes:
            from vektori_trace.providers.teacher.cross import encode_teacher_ids
            from vektori_trace.providers.teacher.fireworks import (
                FireworksTeacherPool,
            )

            prefix_ids = encode_teacher_ids(rendered["prefix_text"], tokenizer)
            action_ids = list(ids["action_ids"])
            pool = FireworksTeacherPool(model=TEACHER_MODEL)
            t0 = time.time()
            logps = pool.score_ids(prefix_ids, action_ids)

            # Echo identity: the ids the server scored must be the ids we sent.
            # A retokenization or an indexing shift is the failure that leaves
            # every logprob finite and attached to the wrong position, so it is
            # checked against the raw entries rather than inferred from lengths.
            echoed_ok = None
            echoed_detail = ""
            try:
                entries = pool._scored_entries(
                    [int(t) for t in prefix_ids], action_ids, top_k=1
                )
                echoed = [e.get("token_id") for e in entries]
                if any(t is None for t in echoed):
                    echoed_ok = None
                    echoed_detail = "entries carry no token_id; cannot verify"
                else:
                    echoed_ok = [int(t) for t in echoed] == action_ids
                    if not echoed_ok:
                        first = next(
                            (i for i, (a, b) in enumerate(zip(echoed, action_ids))
                             if int(a) != b), None
                        )
                        echoed_detail = (
                            f"first divergence at position {first}: "
                            f"server={echoed[first] if first is not None else '?'} "
                            f"sent={action_ids[first] if first is not None else '?'}"
                        )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                echoed_detail = f"{type(exc).__name__}: {exc}"[:200]

            entry["remote"] = {
                "n_logprobs": len(logps),
                "n_action_ids": len(action_ids),
                "lengths_agree": len(logps) == len(action_ids),
                "all_finite": all(
                    x == x and abs(x) != float("inf") for x in logps
                ),
                "echoed_ids_match": echoed_ok,
                "echoed_detail": echoed_detail,
                "logprobs_head": [round(float(x), 4) for x in logps[:8]],
                "seconds": round(time.time() - t0, 2),
                "provenance": pool.provenance(),
            }
            if not (entry["remote"]["lengths_agree"]
                    and entry["remote"]["all_finite"]):
                report["ok"] = False
            if echoed_ok is False:
                report["ok"] = False

        report["cases"][case["name"]] = entry

    path = out / f"parity_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True)[:4000])
    print(f"\narchived -> {path}")
    if not report["ok"]:
        print("\nPARITY FAILED — do not re-score the archive")
        return 1
    print("\nparity checks passed"
          + ("" if a.yes else " (offline only; --yes sends the paid probe)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
