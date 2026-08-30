"""Verify the implicit_eof gate on real captured failures. No spend.

Usage: python scripts/pilotd_check_eof_gate.py <turns.jsonl> <turn_index>

Confirms the amended parser accepts an EOF-terminated reasoning span, that it
invents no bytes, and that a `length` finish still refuses.
"""
import json
import sys

sys.path.insert(0, "/data/vektori-trace")
from vektori_trace.tau2.live_agent import (  # noqa: E402
    PARSER_VERSION,
    _resolve_reasoning,
    split_generation,
)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    want = int(sys.argv[2])
    row = next((r for r in rows if r.get("turn_index") == want
                and r.get("kind") == "failed_turn"), None)
    if row is None:
        print("no failed_turn at index", want); return

    raw = row.get("raw_text") or ""
    if not raw and row.get("raw_bytes_hex"):
        raw = bytes.fromhex(row["raw_bytes_hex"]).decode("utf-8", "replace")
    fr = row.get("finish_reason")
    print("parser %s | turn %s | finish_reason=%s | %d bytes"
          % (PARSER_VERSION, want, fr, len(raw.encode())))

    span = _resolve_reasoning(raw, fr)
    if span is None:
        print("  REFUSED (no span)"); return
    print("  mode           : %s" % span.mode)
    print("  reasoning bytes: %d" % len(span.text.encode()))
    print("  rest bytes     : %d" % len(span.rest.encode()))

    reasoning, content, tools = split_generation(raw, fr)
    print("  content        : %r" % (content if content else None))
    print("  tool calls     : %d" % len(tools))

    # no invented bytes: the span must be a literal substring of the raw text
    assert span.text in raw, "span text is not a substring of raw -- bytes invented!"
    print("  no invented bytes: span is a literal substring of raw  OK")

    # a length cap must still refuse
    if _resolve_reasoning(raw, "length") is not None:
        print("  !! FAIL: a length cap was accepted")
    else:
        print("  length cap still refused                            OK")


main()
