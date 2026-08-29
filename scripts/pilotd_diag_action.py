"""Read-only: diagnose one captured action's payload projection.

Usage: python scripts/pilotd_diag_action.py <actions.jsonl> <key>

Prints the parse (reasoning / content / tool calls), the reasoning byte span,
and where the student payload bytes and the DeepSeek-rendered payload bytes
diverge -- which is what a `payload_bytes_disagree` exclusion or a
`student/teacher payload bytes differ` skip is reporting.
"""
import base64
import json
import sys

sys.path.insert(0, "/data/vektori-trace")

from vektori_trace.tau2.live_agent import PARSER_VERSION, split_generation


def main():
    path, key = sys.argv[1], sys.argv[2]
    row = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("key") == key:
            row = r
            break
    if row is None:
        print("no action with key %r" % key)
        return

    raw = base64.b64decode(row["action_bytes_b64"]).decode("utf-8", "replace")
    print("parser %s | key %s" % (PARSER_VERSION, key))
    print("finish_reason  : %s" % row.get("finish_reason"))
    print("action tokens  : %d" % len(row.get("action_token_ids") or []))
    print("raw bytes      : %d" % len(raw.encode("utf-8")))

    reasoning, content, tools = split_generation(raw)
    print("\n== parse ==")
    print("reasoning bytes: %s" % (len(reasoning.encode()) if reasoning else None))
    print("content bytes  : %s" % (len(content.encode()) if content else None))
    print("tool calls     : %d" % len(tools))
    for t in tools:
        print("   name=%s args=%s" % (t.get("name"), json.dumps(t.get("arguments"))[:120]))

    print("\n== raw (first 700 chars) ==")
    print(raw[:700])
    print("\n== raw (last 400 chars) ==")
    print(raw[-400:])

    if reasoning:
        print("\n== reasoning head/tail ==")
        print("HEAD: %r" % reasoning[:250])
        print("TAIL: %r" % reasoning[-250:])
        # the shape that makes student and teacher payload bytes disagree
        for marker in ("<tool_call>", "</tool_call>", "<think>", "</think>",
                       "<|im_start|>", "<|im_end|>"):
            if marker in reasoning:
                print("  !! reasoning span CONTAINS %s at %d"
                      % (marker, reasoning.find(marker)))
    if content:
        print("\n== content ==")
        print("%r" % content[:400])


main()
