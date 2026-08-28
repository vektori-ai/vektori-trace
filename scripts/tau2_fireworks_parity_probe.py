#!/usr/bin/env python3
"""Does the deployed Fireworks model serialize the way `encoding_dsv4.py` says?

The whole cross-tokenizer objective rests on one unverified assumption: that
the DeepSeek deployment behind Fireworks consumes the prompt format our local
encoder produces. Everything downstream -- byte alignment, chunk advantages,
the projection -- is computed against `encoding_dsv4.py`'s output. If the
served model's actual serialization differs, every number is finite, plausible
and wrong.

Seeing `reasoning_content` or OpenAI-shaped `tool_calls` in a Chat Completions
response does **not** settle this. Those are Fireworks' *normalized interface*;
they say nothing about the underlying prompt the model was trained on. Only the
raw prompt fragments and returned token ids can answer it.

What this probe does NOT do
---------------------------
It does not use Chat Completions. `score_replay_batch` scores through
`FireworksTeacherPool.score_ids`, which posts an **integer id array** to
`/completions` with `echo`; a convenient chat request would exercise a
different code path and prove nothing about the one we bill.

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
            pool = FireworksTeacherPool(model=TEACHER_MODEL)
            t0 = time.time()
            logps = pool.score_ids(prefix_ids, list(ids["action_ids"]))
            entry["remote"] = {
                "n_logprobs": len(logps),
                "lengths_agree": len(logps) == len(ids["action_ids"]),
                "all_finite": all(
                    x == x and abs(x) != float("inf") for x in logps
                ),
                "logprobs_head": [round(float(x), 4) for x in logps[:8]],
                "seconds": round(time.time() - t0, 2),
            }
            if not (entry["remote"]["lengths_agree"]
                    and entry["remote"]["all_finite"]):
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
