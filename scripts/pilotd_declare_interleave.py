"""Declare the interleaved-tool-call exclusion as a CLASS, for all remaining updates.

Usage: python scripts/pilotd_declare_interleave.py <manifest.json> <out.json>

The frozen manifest declared one specific update-0 instance by episode id.
That does not scale and, worse, ratifies each occurrence only after seeing it.
This declares the *shape* prospectively, with the stop thresholds stated in
advance, so a later occurrence is covered by preregistration rather than by a
post-hoc edit.

Measured shape (from raw bytes, update 0 and update 1):

    <think>
    ...reasoning...
    <tool_call>{...}</tool_call>      <-- interleaved, inside the open block
    ...
    <|im_end|>                        <-- NO </think> anywhere

`_resolve_reasoning` applies vLLM 92762ed's implicit rule, so the reasoning
span ends at the first complete tool call -- but the span still carries Hermes
markup, which has no DeepSeek DSML counterpart. The payloads then disagree
byte-for-byte and the reasoning payload is skipped whole.

This is a declared scope limit of projection v4, not a defect: the alternative
(commit 980e06f, reverted) fed Hermes markup into DeepSeek `reasoning_content`
at 94.8% retention and was semantically contaminated.
"""
import json
import sys

DECLARATION = {
    "id": "interleaved_tool_calls_unclosed_think",
    "declared": "2026-08-29",
    "scope": "all updates of this run, prospectively",
    "supersedes": (
        "the per-episode declaration naming only u000-task76-seed0@9#0, which "
        "ratified one instance after the fact instead of declaring the shape"
    ),
    "shape": {
        "description": (
            "Hermes <tool_call> markup appears INSIDE the reasoning span. "
            "DeepSeek's rendering has no counterpart for those bytes -- its "
            "tool calls are DSML -- so the student and teacher reasoning "
            "payloads disagree byte-for-byte and the payload is skipped whole."
        ),
        "two_observed_surface_forms": {
            "A_unclosed_think_with_real_calls": (
                "<think>, reasoning, then one or more REAL <tool_call> blocks, "
                "no </think>, ends at <|im_end|>. Parser v3 applies vLLM "
                "92762ed's implicit boundary; the span still carries markup. "
                "Observed: u001-task4-seed1@4#0 (5 tool calls extracted)."
            ),
            "B_dead_markup_mid_reasoning": (
                "<tool_call> markup embedded mid-reasoning that the parser does "
                "NOT extract as a call (0 tool calls), with reasoning continuing "
                "past it and a normal visible-content payload following. "
                "Observed: u000-task76-seed0@9#0 (markup at byte 1119, 0 calls)."
            ),
        },
        "unifying_condition": (
            "presence of Hermes markup within the reasoning byte span -- NOT "
            "the presence or absence of </think>, and NOT whether the block "
            "parses as an executable call. Both were initially mis-described "
            "as one shape; they are not, and the class is defined by the "
            "condition that actually triggers the skip."
        ),
        "finish_reason": "stop in both (not length -- not a cap truncation)",
    },
    "handling": {
        "reasoning_payload": "skipped whole; student/teacher bytes cannot align",
        "visible_content_payload": (
            "supervised independently when it is authored text -- form B's "
            "604 bytes are exactly that case. Form A's content was markup "
            "only ('<|im_end|>'), already zero-weighted as outside_any_payload."
        ),
        "tool_calls": "conditioned on, never credited (projection v4, gate 5)",
        "episode": "remains sampled and trainable; other turns are unaffected",
    },
    "observed": [
        {"update": 0, "key": "u000-task76-seed0@9#0", "form": "B",
         "rate": "1/75 actions = 1.3%",
         "detail": ("2578 tokens; reasoning 8614 bytes with dead <tool_call> "
                    "markup at offset 1119; 0 tool calls extracted; visible "
                    "content 604 bytes of authored text")},
        {"update": 1, "key": "u001-task4-seed1@4#0", "form": "A",
         "rate": "1/77 actions = 1.3%",
         "detail": ("1137 tokens, 0 supervised; reasoning 3454 bytes with "
                    "<tool_call> at offset 2907; 5 tool calls; content was "
                    "'<|im_end|>' markup only, so nothing was recoverable")},
    ],
    "stop_rules": {
        "retention_below": 0.90,
        "interleaved_episodes_per_update_above": "2/8",
        "reasoning_bytes_excluded_above": 0.10,
        "note": (
            "unchanged from the frozen manifest; stated here so the class "
            "declaration cannot be read as relaxing them"
        ),
    },
    "long_term_fix": (
        "A semantic event IR (ReasoningSegment / ToolCall / ToolResult / "
        "FinalContent) rendered independently through Qwen's and DeepSeek's "
        "native templates, with monotonic span location and proof that each "
        "scored segment's teacher prefix carries the same prior semantic "
        "events. That is a separate projection version with its own canary "
        "and manifest -- deliberately NOT inserted into this frozen run."
    ),
}


def main():
    m = json.load(open(sys.argv[1]))
    plan_before = m.get("plan_hash")

    excl = m.setdefault("declared_exclusions", {})
    excl["interleaved_tool_calls_unclosed_think"] = DECLARATION

    m.setdefault("amendments", []).append({
        "date": "2026-08-29",
        "kind": "prospective_exclusion_class_declaration",
        "what_changed": [
            "declared_exclusions gains interleaved_tool_calls_unclosed_think, "
            "covering the shape for all remaining updates",
        ],
        "why": (
            "The frozen manifest declared a single update-0 episode id. Update 1 "
            "hit the same shape at the same 1.3% rate, and a per-instance "
            "declaration would require ratifying each occurrence after seeing it. "
            "Declaring the class with its thresholds stated in advance is the "
            "stronger preregistration."
        ),
        "unchanged": [
            "plan_hash and the frozen 80 (task, seed) pairs",
            "every stop rule and threshold",
            "policy weights, captured actions, score fingerprints",
            "the projection itself -- this declares behaviour, it does not change it",
        ],
    })

    assert m.get("plan_hash") == plan_before, "plan_hash must not change"

    with open(sys.argv[2], "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("declared class: interleaved_tool_calls_unclosed_think")
    print("plan_hash unchanged: %s" % m["plan_hash"])
    print("declared_exclusions keys: %s" % sorted(excl.keys()))


main()
