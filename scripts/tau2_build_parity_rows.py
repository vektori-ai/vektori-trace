"""Build the span file both parser-parity checks consume. CPU only, no GPU.

`tau2_parser_parity_modal.py` and `tau2_live_parser_parity_modal.py` both read
`/tmp/parity_rows.jsonl`. That file was previously produced ad hoc, which meant
the 25/25 result could not be reproduced from the repository. This script is
that missing step.

One row per assistant target in the corpus:

    {"task_id", "position", "action_type",
     "decoded_target",   # what the model would emit, rendered as Qwen does
     "intended_calls"}   # [{"name", "arguments"}], possibly empty

`decoded_target` is rendered from the stored target with the *serving* markup --
`<tool_call>` blocks in Hermes form -- because that is the text both parsers
have to read. Conversational targets are included deliberately: a plain message
that either parser reads as a tool call is a failure worth catching.

Run on the box, where the corpus lives:

    python3 scripts/tau2_build_parity_rows.py \
        --artifacts /data/tau2/artifacts_16384 \
        --out /tmp/parity_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def render_target_text(target: dict) -> str:
    """The assistant turn as the model emits it, before any parser runs.

    Hermes form: content first (if any), then one `<tool_call>` block per call
    carrying `{"name": ..., "arguments": {...}}`. Tool-call *ids* are omitted
    because the model does not generate them -- vLLM assigns them at parse
    time, and including them here would test a string the student never emits.

    The `<think>` block is likewise absent: stored targets are DeepSeek's
    actions, which carry no Qwen reasoning. Reasoning splitting is covered by
    the unit tests; this file exists to compare *tool-call* extraction on real
    corpus content.
    """
    parts: list[str] = []
    content = target.get("content")
    if content:
        parts.append(content)
    for tc in target.get("tool_calls") or []:
        fn = tc.get("function") or tc
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        parts.append(
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": args or {}})
            + "\n</tool_call>"
        )
    return "".join(parts)


def intended_calls(target: dict) -> list[dict]:
    out = []
    for tc in target.get("tool_calls") or []:
        fn = tc.get("function") or tc
        args = fn.get("arguments")
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        out.append({"name": fn.get("name"), "arguments": args or {}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--out", default="/tmp/parity_rows.jsonl")
    a = ap.parse_args()

    src = os.path.join(a.artifacts, "rows.semantic.jsonl")
    n_msg = n_call = 0
    with open(a.out, "w") as fh:
        for line in open(src):
            r = json.loads(line)
            target = r.get("target")
            if not target:
                continue
            calls = intended_calls(target)
            action_type = "tool_call" if calls else "message"
            if calls:
                n_call += 1
            else:
                n_msg += 1
            fh.write(json.dumps({
                "task_id": r.get("task_id"),
                "position": r.get("position"),
                "action_type": action_type,
                "decoded_target": render_target_text(target),
                "intended_calls": calls,
            }) + "\n")

    total = n_msg + n_call
    print(f"wrote {total} spans to {a.out}  ({n_call} tool_call, {n_msg} message)")
    if total == 0:
        raise SystemExit(f"no targets found in {src}; wrong artifacts dir?")


if __name__ == "__main__":
    main()
