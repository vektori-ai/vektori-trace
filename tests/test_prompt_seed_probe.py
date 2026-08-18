"""Tests for the calibration-seed probe agent.

The load-bearing property is *adjacency*: the demonstration must be the assistant
turn immediately preceding generation, with a short real terminal observation as
the final message. The first run seeded the pair before harbor's whole rendered
prompt, leaving a 5,277-char instruction block as the last message -- the same
number of prior assistant turns as Phase 7's passing prefix, and the opposite
result. Position, not count, is what these tests pin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prompt_seed_probe import (
    BEGIN_INSTRUCTION,
    CALIBRATION_ASSISTANT,
    SPLIT_MARKER,
    WAIT_INSTRUCTION,
    FirstTurnProtocolFailure,
    Terminus2PromptSeed,
    has_legacy_envelope,
    is_native_json,
)

RENDERED = (
    "You are an AI assistant.\n\nFormat your response as JSON ...\n\n"
    "Task Description:\nfix anyio\n\nCurrent terminal state:\nroot@abc:/workspace#"
)


class FakeChat:
    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.reset_calls = 0

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        self.reset_calls += 1

    def chat_append(self, prompt, response):
        self._messages.extend(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        )


class _Agent(Terminus2PromptSeed):
    def __init__(self):
        import logging

        self.logger = logging.getLogger("test")
        self._probe_log = None
        self._probe_turn = 0
        self._probe_episode = 1
        self._reseed_turn = -1
        self._seed_events = []
        self._first_completion = None


@pytest.fixture
def agent():
    return _Agent()


# ---- the split ---------------------------------------------------------------


def test_split_puts_the_terminal_state_in_the_tail():
    head, tail = Terminus2PromptSeed.split_initial_prompt(RENDERED)
    assert "Format your response as JSON" in head
    assert "fix anyio" in head
    assert SPLIT_MARKER not in head
    assert tail.startswith(SPLIT_MARKER)
    assert "root@abc:/workspace#" in tail


def test_split_round_trips():
    head, tail = Terminus2PromptSeed.split_initial_prompt(RENDERED)
    norm = lambda s: s.replace("\n", "").replace(" ", "")  # noqa: E731
    assert norm(head + tail) == norm(RENDERED)


@pytest.mark.parametrize("bad", ["no marker at all", f"{SPLIT_MARKER} a {SPLIT_MARKER} b"])
def test_split_fails_closed(bad):
    """A mis-split prompt would produce a result nobody could interpret."""
    with pytest.raises(FirstTurnProtocolFailure):
        Terminus2PromptSeed.split_initial_prompt(bad)


# ---- geometry ----------------------------------------------------------------


def test_generation_point_follows_the_real_terminal_state(agent):
    head, tail = agent.split_initial_prompt(RENDERED)
    chat = FakeChat(agent.seed_messages(head))
    chat.chat_append(tail + BEGIN_INSTRUCTION, "{}")

    roles = [m["role"] for m in chat.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # the demonstration is immediately above the generation point
    assert chat.messages[1]["content"] == CALIBRATION_ASSISTANT
    # and the last user message is the short real observation, not the 5k block
    assert chat.messages[2]["content"].startswith(SPLIT_MARKER)
    assert "root@abc:/workspace#" in chat.messages[2]["content"]
    assert len(chat.messages[2]["content"]) < len(chat.messages[0]["content"])


def test_no_terminal_output_is_fabricated(agent):
    """Everything in the final message comes from harbor, bar one added sentence."""
    _head, tail = agent.split_initial_prompt(RENDERED)
    final = tail + BEGIN_INSTRUCTION
    assert final.replace(BEGIN_INSTRUCTION, "") in RENDERED


def test_only_three_things_are_synthetic(agent):
    head, _ = agent.split_initial_prompt(RENDERED)
    seeded = agent.seed_messages(head)
    assert seeded[0]["content"] == head + WAIT_INSTRUCTION
    assert seeded[0]["content"].replace(WAIT_INSTRUCTION, "") in RENDERED
    assert seeded[1]["content"] == CALIBRATION_ASSISTANT


# ---- the demonstration -------------------------------------------------------


def test_demonstration_is_native_json_and_carries_no_legacy_envelope():
    assert is_native_json(CALIBRATION_ASSISTANT)
    assert not has_legacy_envelope(CALIBRATION_ASSISTANT)


def test_demonstration_carries_no_commands():
    demo = json.loads(CALIBRATION_ASSISTANT)
    assert demo["commands"] == []
    assert demo["task_complete"] is False


def test_a_verbatim_copy_fails_the_first_action_gate():
    demo = json.loads(CALIBRATION_ASSISTANT)
    assert not any(c.get("keystrokes", "").strip() for c in demo["commands"])


# ---- seed lifecycle ----------------------------------------------------------


def test_strip_removes_the_seed_and_restores_the_anchor(agent):
    head, tail = agent.split_initial_prompt(RENDERED)
    chat = FakeChat(agent.seed_messages(head))
    chat.chat_append(tail + BEGIN_INSTRUCTION, "{}")

    removed = agent._strip_seed(chat)

    assert removed == 2
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert chat.messages[0]["content"].startswith(SPLIT_MARKER)


def test_seed_present_tracks_installation_and_removal(agent):
    head, _ = agent.split_initial_prompt(RENDERED)
    chat = FakeChat()
    assert not agent._seed_present(chat)
    chat._messages = agent.seed_messages(head)
    assert agent._seed_present(chat)
    agent._strip_seed(chat)
    assert not agent._seed_present(chat)


# ---- format tiers ------------------------------------------------------------


def test_think_block_is_not_counted_as_a_legacy_envelope():
    """Run 3 dropped the <tool_call> tags but kept a <think> block. Conflating the
    two reported it as 'still emitting the envelope', which was false."""
    from prompt_seed_probe import has_think_block, has_v1_tool_schema

    run3 = (
        "<think>\n\n</think>\n\nAnalysis: We are in /workspace.\nPlan: List contents.\n"
        '{"name": "bash_command", "arguments": {"keystrokes": "ls -la"}}'
    )
    assert not has_legacy_envelope(run3)   # tags are gone
    assert has_think_block(run3)           # allowed, recorded separately
    assert has_v1_tool_schema(run3)        # the actual remaining defect
    assert not is_native_json(run3)


def test_legacy_envelope_detection_matches_v1_failure_shape():
    v1 = (
        "Analysis: We are in the workspace directory.\n"
        "Plan: List the contents.\n"
        '<tool_call>\n{"name": "bash_command", "arguments": {"keystrokes": "ls -la"}}\n</tool_call>'
    )
    assert has_legacy_envelope(v1)
    assert not is_native_json(v1)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"analysis": "a", "plan": "p", "commands": []}', True),
        ('```json\n{"analysis": "a"}\n```', False),
        ('Here is my answer:\n{"analysis": "a"}', False),
        ('<think>hm</think>{"analysis": "a"}', False),
        ("[1, 2, 3]", False),
    ],
)
def test_native_json_is_the_strict_tier(text, expected):
    """Strict tier is recorded but no longer gates: harbor tolerates a think block
    (warning only), so a rollout with `<think>` genuinely proceeds."""
    assert is_native_json(text) is expected


# ---- launch path -------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_name,expected",
    [
        ("claude_code", "claude-code"),
        ("terminus_2", "terminus-2"),
        (
            "scripts.prompt_seed_probe:Terminus2PromptSeed",
            "scripts.prompt_seed_probe:Terminus2PromptSeed",
        ),
        ("acp:opencode@1.3.9", "acp:opencode@1.3.9"),
    ],
)
def test_normalization_rule_matches_bare_names_only(agent_name, expected):
    normalized = agent_name if ":" in agent_name else agent_name.replace("_", "-")
    assert normalized == expected


def test_import_path_survives_agent_name_normalization():
    import inspect

    from vektori_trace.evaluate import validity

    assert 'if ":" not in agent:' in inspect.getsource(validity.run_trial)
