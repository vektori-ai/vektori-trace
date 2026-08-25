"""Canonical form for a sampled Tau2 action, for diversity counting.

Two diversity questions, and they are not the same question:

- **byte-distinct**: did the sampler produce literally different bytes? This is
  the honest measure of raw sampling variation, and it is what a temperature
  probe is really asking.
- **canonical-distinct**: did the sampler produce *meaningfully* different
  actions? Two tool calls that differ only in whitespace, JSON key order, or a
  generated call id are the same decision, and counting them as two would
  overstate diversity exactly where it matters -- a collapsed policy emitting
  one action with cosmetic jitter would look healthy.

Both are reported. Byte-distinct alone overstates; canonical alone hides the
sampler's actual behaviour. A large gap between them is itself informative: it
means the model is varying its formatting rather than its decisions.

Canonicalization rules, each chosen because the alternative loses signal:

- object keys are **sorted**, because `{"a":1,"b":2}` and `{"b":2,"a":1}` are
  the same arguments;
- arrays keep their **order**, because `item_ids: [x, y]` is not the same
  request as `[y, x]` -- one returns a different set from the other in general,
  and Tau2's graders compare ordered lists;
- **multiple tool calls keep their order**, because "authenticate then mutate"
  and "mutate then authenticate" are different trajectories and the policy
  distinguishes them;
- generated **call ids are dropped** (`chatcmpl-tool-...`), because they are
  assigned per request and never equal across samples -- keeping them would
  make every sample trivially "distinct";
- a **parse failure is its own canonical value**, not a fallback to raw text.
  Malformed output collapsing into the text bucket would let a broken sampler
  read as diverse. Their rate is reported separately, because a high one means
  the diversity numbers describe garbage.
"""

from __future__ import annotations

import json
import re
from typing import Any

# vLLM/OpenAI-style generated ids. Never equal between two samples, so they
# would make every action distinct if left in.
_CALL_ID = re.compile(r"^(chatcmpl-tool-|call_|toolu_)[A-Za-z0-9_-]+$")

# Qwen3 emits tool calls wrapped in these; the hermes parser is what normally
# extracts them, but the canary sees raw completion text.
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# An opening marker on its own: the call was cut off or never closed.
_TOOL_CALL_OPEN = re.compile(r"<tool_call>")

PARSE_FAILURE = "<<unparseable>>"


def _canon_json(value: Any) -> Any:
    """Recursively sort object keys; leave array order alone."""
    if isinstance(value, dict):
        return {k: _canon_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canon_json(v) for v in value]
    if isinstance(value, str):
        # Arguments frequently arrive as a JSON *string*. Parse one level so
        # {"a":1} and { "a" : 1 } compare equal.
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                return _canon_json(json.loads(stripped))
            except (json.JSONDecodeError, ValueError):
                return value
        return value
    return value


def _strip_call_ids(obj: Any) -> Any:
    """Drop generated call ids wherever they appear."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("id", "tool_call_id") and isinstance(v, str) and _CALL_ID.match(v):
                continue
            out[k] = _strip_call_ids(v)
        return out
    if isinstance(obj, list):
        return [_strip_call_ids(v) for v in obj]
    return obj


def _normalize_text(text: str) -> str:
    """Collapse incidental whitespace in an ordinary assistant message.

    Not a semantic normalization -- two genuinely different sentences stay
    different. It only removes the formatting jitter that would otherwise make
    the same message read as two.
    """
    return " ".join(text.split())


def canonical_action(text: str) -> tuple[str, bool]:
    """Canonical form of one sampled action, and whether it parsed.

    Returns `(canonical, parsed_ok)`. `parsed_ok` is False only when the action
    *looked* like a tool call and could not be read as one; a plain message is a
    successful parse of a message.
    """
    if not isinstance(text, str):
        return PARSE_FAILURE, False

    blocks = _TOOL_CALL_BLOCK.findall(text)
    if not blocks:
        # An OPENING marker with no complete block is a truncated or malformed
        # tool call, not prose. Falling through to the message branch here would
        # classify broken output as a valid ordinary message -- and since each
        # truncation differs, a sampler emitting nothing but broken calls would
        # score as maximally diverse.
        if _TOOL_CALL_OPEN.search(text):
            return PARSE_FAILURE, False
        # No tool-call markup at all: an ordinary assistant message.
        return json.dumps(
            {"kind": "message", "text": _normalize_text(text)},
            sort_keys=True, ensure_ascii=False), True

    calls = []
    for block in blocks:
        try:
            parsed = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            # A malformed call is not "some other action" -- the sample is
            # unusable, and saying so is more honest than bucketing it with
            # whatever text happens to surround it.
            return PARSE_FAILURE, False
        if not isinstance(parsed, dict):
            return PARSE_FAILURE, False
        parsed = _strip_call_ids(parsed)
        # `arguments` is the payload the grader compares; normalize it hardest.
        if "arguments" in parsed:
            parsed["arguments"] = _canon_json(parsed["arguments"])
        calls.append(_canon_json(parsed))

    # Any prose accompanying the calls is part of the action -- a model that
    # says "let me check" and one that says nothing made different choices.
    prose = _normalize_text(_TOOL_CALL_BLOCK.sub("", text))
    return json.dumps(
        # Call ORDER is preserved: authenticate-then-mutate is not the same
        # trajectory as mutate-then-authenticate.
        {"kind": "tool_calls", "calls": calls, "text": prose},
        sort_keys=True, ensure_ascii=False), True


def diversity(actions: list[str]) -> dict:
    """Byte-distinct and canonical-distinct rates over one prefix's samples."""
    n = len(actions)
    if n == 0:
        raise ValueError("no actions to measure")

    canon, parsed_flags = [], []
    for a in actions:
        c, ok = canonical_action(a)
        canon.append(c)
        parsed_flags.append(ok)

    byte_distinct = len(set(actions))
    canon_distinct = len(set(canon))
    n_unparseable = sum(1 for ok in parsed_flags if not ok)

    counts: dict[str, int] = {}
    for c in canon:
        counts[c] = counts.get(c, 0) + 1

    return {
        "n_samples": n,
        "n_byte_distinct": byte_distinct,
        "byte_distinct_rate": byte_distinct / n,
        "n_canonical_distinct": canon_distinct,
        "canonical_distinct_rate": canon_distinct / n,
        # A policy emitting one decision with cosmetic jitter shows up here as
        # a large gap between the two rates.
        "formatting_only_variation": byte_distinct - canon_distinct,
        "largest_canonical_fraction": max(counts.values()) / n,
        "n_unparseable": n_unparseable,
        "unparseable_rate": n_unparseable / n,
    }
