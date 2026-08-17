"""Protocol reconstruction tests — scripts/sft_repair_dataset.py.

The repair turns v1's `tool_calls` envelope back into the literal Terminus JSON
terminus-2 actually asks for. These tests pin the three places that can be
silently wrong: the inverse of terminus's `message` rendering, the classifier
that decides which assistant turns carry loss, and the duration clamp the join
has to account for.

Offline: no tokenizer, no Hub. Harbor's real parser is used where the code uses
it, because a hand-rolled `json.loads` would not be the same check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sft_repair_dataset import (
    parse_raw,
    ACTION,
    HANDOFF_QUESTION,
    SUMMARIZATION,
    UNKNOWN,
    Action,
    action_from_step,
    build_segment,
    classify,
    observation_text,
    split_on_compaction,
    split_rendered_message,
)

pytest.importorskip("harbor")


def _step(step_id: int, source: str, message: str, tool_calls=None, observation=None):
    return {
        "step_id": step_id,
        "source": source,
        "message": message,
        "tool_calls": tool_calls,
        "observation": observation,
    }


def _bash(call_id: str, keystrokes: str, duration: float = 0.1):
    return {
        "tool_call_id": call_id,
        "function_name": "bash_command",
        "arguments": {"keystrokes": keystrokes, "duration": duration},
    }


def _obs(content: str):
    return {"results": [{"source_call_id": None, "content": content}]}


# --------------------------------------------------------------------------
# The inverse of terminus_2.py:1338-1348
# --------------------------------------------------------------------------


def test_rendered_message_round_trips_through_the_split() -> None:
    action = Action("did a thing", "do another", [], False)
    a, p, problem = split_rendered_message(action.rendered_message())
    assert (a, p, problem) == ("did a thing", "do another", None)


def test_split_flags_an_analysis_that_contains_the_plan_marker() -> None:
    """Splitting is ambiguous here; the code must say so rather than guess."""
    message = "Analysis: first\nPlan: smuggled\nPlan: the real plan"
    a, p, problem = split_rendered_message(message)
    assert problem is not None and "2 occurrences" in problem
    # It still splits at the first marker — the flag is what makes it auditable.
    assert a == "first"


def test_split_handles_a_missing_half() -> None:
    assert split_rendered_message("Plan: only a plan")[:2] == ("", "only a plan")
    assert split_rendered_message("Analysis: only analysis")[:2] == ("only analysis", "")
    assert split_rendered_message("")[:2] == ("", "")


def test_message_with_neither_prefix_is_flagged() -> None:
    _, _, problem = split_rendered_message("just some prose")
    assert problem is not None and "neither" in problem


# --------------------------------------------------------------------------
# Action reconstruction
# --------------------------------------------------------------------------


def test_action_rebuilds_native_terminus_json() -> None:
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    step = _step(2, "agent", "Analysis: looking\nPlan: list files",
                 tool_calls=[_bash("call_1_1", "ls -la\n")])
    action, problem = action_from_step(step)
    assert problem is None
    result = TerminusJSONPlainParser().parse_response(action.to_json())
    assert result.error == ""
    assert [c.keystrokes for c in result.commands] == ["ls -la\n"]
    assert result.analysis == "looking"
    assert result.plan == "list files"


def test_mark_task_complete_becomes_a_flag_not_a_command() -> None:
    step = _step(9, "agent", "Analysis: done\nPlan: finish", tool_calls=[
        _bash("call_8_1", "pytest\n"),
        {"tool_call_id": "call_8_task_complete",
         "function_name": "mark_task_complete", "arguments": {}},
    ])
    action, _ = action_from_step(step)
    assert action.task_complete is True
    assert [c["keystrokes"] for c in action.commands] == ["pytest\n"]
    assert '"task_complete": true' in action.to_json()
    assert "mark_task_complete" not in action.to_json()


def test_task_complete_is_omitted_when_false() -> None:
    """The parser defaults it to false; the teacher writes it only when true."""
    action = Action("a", "p", [{"keystrokes": "ls\n", "duration": 0.1}], False)
    assert "task_complete" not in action.to_json()


def test_an_unknown_tool_verb_is_refused() -> None:
    step = _step(3, "agent", "Analysis: a\nPlan: p", tool_calls=[
        {"tool_call_id": "c", "function_name": "run_command", "arguments": {}},
    ])
    action, problem = action_from_step(step)
    assert action is None
    assert "run_command" in problem


# --------------------------------------------------------------------------
# The join signature
# --------------------------------------------------------------------------


def test_signature_clamps_duration_the_way_terminus_records_it() -> None:
    """terminus stores min(duration, 60), so a raw 120 must match a recorded 60."""
    raw = Action("a", "p", [{"keystrokes": "sleep 200\n", "duration": 120.0}], False)
    recorded = Action("a", "p", [{"keystrokes": "sleep 200\n", "duration": 60.0}], False)
    assert raw.signature() == recorded.signature()
    assert raw.join_key() == recorded.join_key()


def test_signature_separates_actions_that_share_analysis_and_plan() -> None:
    """A full analysis/plan string is not a sufficient key on its own."""
    a = Action("same", "same", [{"keystrokes": "ls\n", "duration": 0.1}], False)
    b = Action("same", "same", [{"keystrokes": "pytest\n", "duration": 0.1}], False)
    assert a.rendered_message() == b.rendered_message()
    assert a.join_key() != b.join_key()


def test_task_complete_participates_in_the_key() -> None:
    cmds = [{"keystrokes": "ls\n", "duration": 0.1}]
    assert Action("a", "p", cmds, False).join_key() != Action("a", "p", cmds, True).join_key()


# --------------------------------------------------------------------------
# Classification — only `action` is supervised
# --------------------------------------------------------------------------


def test_handoff_question_is_classified_by_its_prompt_not_its_content() -> None:
    prompt = "You are picking up work from a previous AI agent on this task:\n\n**Original Task:** x"
    step = _step(2, "agent", "Before I continue, I need clarifications: ...")
    assert classify(step, prompt) == HANDOFF_QUESTION


def test_summarization_prompts_are_classified() -> None:
    step = _step(2, "agent", '{"analysis": "Handoff summary..."}')
    assert classify(step, "You are about to hand off your work to another AI agent.") == SUMMARIZATION
    assert classify(step, "The next agent has a few questions for you, please answer") == SUMMARIZATION


def test_an_agent_turn_with_tool_calls_under_a_terminal_prompt_is_an_action() -> None:
    step = _step(2, "agent", "Analysis: a\nPlan: p", tool_calls=[_bash("call_1_1", "ls\n")])
    assert classify(step, "New Terminal Output:\n$ ") == ACTION


def test_parseable_json_with_no_tool_calls_is_unknown() -> None:
    """`unknown` is the bucket for a shape nobody predicted: the response parsed
    as an action, yet terminus recorded no tool calls. That combination has no
    explanation, so the audit fails rather than guessing."""
    step = _step(2, "agent", '{"analysis": "a", "plan": "p", "commands": []}')
    assert classify(step, "New Terminal Output:\n$ ") == UNKNOWN


# --------------------------------------------------------------------------
# Segment assembly
# --------------------------------------------------------------------------


def _rollout_steps():
    return [
        _step(1, "user", "You are an AI assistant tasked with solving command-line tasks"),
        _step(2, "agent", "Analysis: a1\nPlan: p1",
              tool_calls=[_bash("call_1_1", "ls -la\n")],
              observation=_obs("New Terminal Output:\ntotal 0")),
        _step(3, "agent", "Analysis: a2\nPlan: p2",
              tool_calls=[_bash("call_2_1", "pytest\n")],
              observation=_obs("New Terminal Output:\n1 failed")),
    ]


def test_observations_become_user_messages_and_actions_are_supervised(tmp_path) -> None:
    seg = build_segment(_rollout_steps(), {}, task="t", jobs_dir=tmp_path,
                        rollout_index=0, segment_index=0)
    assert [m["role"] for m in seg.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    # The trailing observation is dropped: it answers nothing.
    assert seg.supervise == [False, True, False, True]
    assert all("tool_calls" not in m for m in seg.messages)
    assert all("tool_call_id" not in m for m in seg.messages)
    assert seg.problems == []


def test_handoff_turns_stay_as_context_and_out_of_the_loss(tmp_path) -> None:
    steps = [
        _step(1, "user", "You are picking up work from a previous AI agent on this task:"),
        _step(2, "agent", "Before I continue implementing, I need clarifications:"),
        _step(3, "user", "Here are the answers the other agent provided."),
        _step(4, "agent", "Analysis: a\nPlan: p",
              tool_calls=[_bash("call_1_1", "ls\n")],
              observation=_obs("out")),
    ]
    seg = build_segment(steps, {}, task="t", jobs_dir=tmp_path,
                        rollout_index=0, segment_index=1)
    kinds = [m["kind"] for m in seg.turn_meta]
    assert kinds == ["prompt", HANDOFF_QUESTION, "prompt", ACTION]
    # The prose turn is present as context...
    assert seg.messages[1]["role"] == "assistant"
    # ...and carries no loss.
    assert seg.supervise == [False, False, False, True]


def test_unclassified_assistant_turns_fail_the_segment(tmp_path) -> None:
    steps = [
        _step(1, "user", "You are an AI assistant tasked with solving"),
        _step(2, "agent", '{"analysis": "a", "plan": "p", "commands": []}'),
    ]
    seg = build_segment(steps, {}, task="t", jobs_dir=tmp_path,
                        rollout_index=0, segment_index=0)
    assert any("unclassified" in p for p in seg.problems)
    assert seg.turn_meta[-1]["kind"] == UNKNOWN
    assert not any(seg.supervise)


def test_a_raw_capture_is_used_when_it_verifies(tmp_path) -> None:
    steps = _rollout_steps()
    action, _ = action_from_step(steps[1])
    raw = '{"analysis": "a1", "plan": "p1", "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}]}'
    seg = build_segment(steps, {action.join_key(): raw}, task="t", jobs_dir=tmp_path,
                        rollout_index=0, segment_index=0)
    assert seg.messages[1]["content"] == raw
    assert seg.turn_meta[1]["target_source"] == "raw_capture"
    assert seg.turn_meta[1]["equivalent"] is True
    # The unjoined action still lands, rebuilt.
    assert seg.turn_meta[3]["target_source"] == "atif_rebuilt"


def test_compaction_splits_into_the_conversations_the_model_saw(tmp_path) -> None:
    steps = [
        _step(1, "user", "start"),
        _step(2, "agent", "Analysis: a\nPlan: p", tool_calls=[_bash("c", "ls\n")],
              observation=_obs("o")),
        _step(3, "system", "Performed context summarization and handoff to continue task."),
        _step(4, "user", "Here are the answers the other agent provided."),
        _step(5, "agent", "Analysis: b\nPlan: q", tool_calls=[_bash("d", "pwd\n")],
              observation=_obs("o2")),
    ]
    segments = split_on_compaction(steps, tmp_path)
    assert len(segments) == 2
    assert [s["step_id"] for s in segments[0]] == [1, 2]
    assert [s["step_id"] for s in segments[1]] == [4, 5]


def test_a_step_with_split_observations_ends_the_segment(tmp_path) -> None:
    """More than one ObservationResult means this step is not shaped like the
    reply terminus's own loop sends, so the conversation cannot be continued."""
    step = _step(2, "agent", "Analysis: a\nPlan: p", tool_calls=[_bash("c", "ls\n")],
                 observation={"results": [{"content": "one"}, {"content": "two"}]})
    assert observation_text(step) is None


def test_a_bare_newline_keystroke_gets_a_label_not_a_crash() -> None:
    """Keystrokes go to a terminal verbatim, so pressing Enter is a real action
    whose text strips to nothing and has no first line."""
    from scripts.sft_repair_dataset import _command_label

    assert _command_label("\n") == "<enter>"
    assert _command_label("   ") == "<enter>"
    assert _command_label("") == "<empty>"
    assert _command_label("ls -la\n") == "ls -la"
    assert _command_label("C-c") == "C-c"


# --------------------------------------------------------------------------
# The teacher's own protocol failures
# --------------------------------------------------------------------------


def test_prose_that_does_not_parse_is_a_parse_error_not_unknown() -> None:
    """terminus records the raw response with no tool calls when the parser
    rejects it. Those are the teacher's failures, not ours to guess about."""
    from scripts.sft_repair_dataset import PARSE_ERROR

    step = _step(2, "agent", "I'll now revert the change.\n\n```bash\ngrep -n foo\n```")
    assert classify(step, "New Terminal Output:\n$ ") == PARSE_ERROR


def test_parse_error_prompt_is_recomputed_not_invented() -> None:
    """The error path writes no user step, so the reply must be re-derived from
    the same input with terminus's own formatting."""
    from scripts.sft_repair_dataset import parse_error_prompt

    reply = parse_error_prompt("not json at all")
    assert reply.startswith("Previous response had parsing errors:\nERROR: ")
    assert reply.endswith("Please fix these issues and provide a proper JSON response.")
    # A response that parses has no error reply.
    assert parse_error_prompt(
        '{"analysis": "a", "plan": "p", "commands": []}'
    ) is None


def test_parse_errors_stay_as_context_and_carry_no_loss(tmp_path) -> None:
    from scripts.sft_repair_dataset import PARSE_ERROR

    steps = [
        _step(1, "user", "You are an AI assistant tasked with solving"),
        _step(2, "agent", "prose with a ```bash fence, no JSON"),
        _step(3, "agent", "Analysis: recovering\nPlan: retry",
              tool_calls=[_bash("call_2_1", "ls\n")], observation=_obs("out")),
    ]
    seg = build_segment(steps, {}, task="t", jobs_dir=tmp_path,
                        rollout_index=0, segment_index=0)
    kinds = [m["kind"] for m in seg.turn_meta]
    assert kinds == ["prompt", PARSE_ERROR, "parse_error_reply", ACTION]
    assert seg.supervise == [False, False, False, True]
    # The reconstructed reply keeps the conversation alternating, so the
    # recovery turn is not left following another assistant turn.
    assert [m["role"] for m in seg.messages] == ["user", "assistant", "user", "assistant"]


def test_parse_error_prompt_matches_harbors_own_source() -> None:
    """The reconstruction is only reconstruction if it is terminus's template.

    Asserting the wrapper's prefix and suffix proves nothing about whether the
    wording is terminus's — it proves our own string is our own string. This
    reads the literals out of the installed harbor and fails if either drifts,
    which is what an upgrade would do silently. `result.error` itself cannot
    drift, because it comes from the same parser this code calls.
    """
    import inspect

    from harbor.agents.terminus_2 import terminus_2

    src = inspect.getsource(terminus_2)
    # terminus_2.py ~1355: the error prompt.
    assert '"Previous response had parsing errors:\\n{feedback}\\n\\n"' in src
    assert '"Please fix these issues and provide a proper "' in src
    # terminus_2.py ~1173: how `feedback` is assembled.
    assert 'feedback += f"ERROR: {result.error}"' in src
    assert 'feedback += f"\\nWARNINGS: {result.warning}"' in src
    # `_get_error_response_type()` returns this for the json parser.
    assert 'return "JSON response"' in src


def test_a_prose_wrapped_raw_capture_is_rebuilt_not_used(tmp_path) -> None:
    """Harbor tolerates text around the object; a target must not.

    The parser's job at runtime was to salvage whatever the teacher sent. Ours
    is to decide what the student learns to send, and a prose-wrapped action is
    most of the way back to the failure this run repairs.
    """
    from scripts.sft_repair_dataset import is_bare_json

    steps = _rollout_steps()
    action, _ = action_from_step(steps[1])
    wrapped = (
        "Here is my next step:\n\n"
        '{"analysis": "a1", "plan": "p1", '
        '"commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}]}'
    )
    # Harbor accepts it; we do not.
    assert parse_raw(wrapped) is not None
    assert not is_bare_json(wrapped)

    seg = build_segment(steps, {action.join_key(): wrapped}, task="t",
                        jobs_dir=tmp_path, rollout_index=0, segment_index=0)
    assert seg.turn_meta[1]["target_source"] == "atif_rebuilt"
    assert is_bare_json(seg.messages[1]["content"])
    assert any("not bare JSON" in p for p in seg.problems)
