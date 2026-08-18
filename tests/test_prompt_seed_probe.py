"""Tests for the calibration-seed probe agent.

The load-bearing case is compaction: harbor rebuilds history as
`[chat.messages[0], question_prompt, questions]` and treats `messages[0]` as the
anchor carrying the Terminus contract and the task. A seeded history puts the
calibration message at index 0, so delegating without stripping would preserve the
calibration message as the anchor and silently drop the real instructions. That
would read as "the seed failed after compaction" when the cause is elsewhere,
which is exactly the misreading the probe's decision rule cannot afford.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prompt_seed_probe import (
    CALIBRATION_ASSISTANT,
    CALIBRATION_USER,
    Terminus2PromptSeed,
    has_legacy_envelope,
    is_native_json,
)

REAL_PROMPT = "Format your response as JSON ... Task Description: fix anyio"


class FakeChat:
    """Minimal stand-in exposing the surface the seed helpers touch."""

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self.reset_calls = 0

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        self.reset_calls += 1

    def chat_append(self, prompt, response):
        """Mirror of Chat.chat's history extension."""
        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )


class _Agent(Terminus2PromptSeed):
    """Bypass Terminus2.__init__, which wants an LLM backend."""

    def __init__(self):
        import logging

        self.logger = logging.getLogger("test")
        self._probe_log = None
        self._probe_turn = 0
        self._seed_events = []
        self._first_completion = None


@pytest.fixture
def agent():
    return _Agent()


def test_demonstration_is_native_json_and_carries_no_legacy_envelope():
    assert is_native_json(CALIBRATION_ASSISTANT)
    assert not has_legacy_envelope(CALIBRATION_ASSISTANT)


def test_demonstration_carries_no_commands():
    demo = json.loads(CALIBRATION_ASSISTANT)
    assert demo["commands"] == []
    assert demo["task_complete"] is False


def test_a_verbatim_copy_of_the_demonstration_fails_the_first_action_gate():
    """The reason the demo carries no commands rather than one empty keystroke.

    A copied `keystrokes: ""` would report n_commands == 1 and read as a working
    protocol; a copied `commands: []` cannot satisfy the gate at all.
    """
    demo = json.loads(CALIBRATION_ASSISTANT)
    keystrokes = [c.get("keystrokes", "") for c in demo["commands"]]
    assert not any(k.strip() for k in keystrokes)


def test_seed_lands_immediately_above_the_real_prompt(agent):
    chat = FakeChat()
    agent._append_seed(chat, reason="initial")
    chat.chat_append(REAL_PROMPT, "{}")

    roles = [m["role"] for m in chat.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert chat.messages[0]["content"] == CALIBRATION_USER
    assert chat.messages[1]["content"] == CALIBRATION_ASSISTANT
    assert chat.messages[2]["content"] == REAL_PROMPT


def test_strip_removes_both_calibration_messages_and_nothing_else(agent):
    chat = FakeChat()
    agent._append_seed(chat, reason="initial")
    chat.chat_append(REAL_PROMPT, "{}")

    removed = agent._strip_seed(chat)

    assert removed == 2
    assert [m["content"] for m in chat.messages] == [REAL_PROMPT, "{}"]


def test_strip_restores_the_real_prompt_as_the_compaction_anchor(agent):
    """The whole reason `_summarize` strips before delegating."""
    chat = FakeChat()
    agent._append_seed(chat, reason="initial")
    chat.chat_append(REAL_PROMPT, "{}")

    assert chat.messages[0]["content"] == CALIBRATION_USER  # would hijack the anchor
    agent._strip_seed(chat)
    assert chat.messages[0]["content"] == REAL_PROMPT  # harbor keeps the right one


def test_post_compaction_reseed_reproduces_turn_one_geometry(agent):
    """Simulate harbor's rebuild at terminus_2.py:939, then the reseed."""
    chat = FakeChat()
    agent._append_seed(chat, reason="initial")
    chat.chat_append(REAL_PROMPT, "{}")

    agent._strip_seed(chat)
    chat._messages = [  # harbor's replacement, verbatim in shape
        chat.messages[0],
        {"role": "user", "content": "question_prompt"},
        {"role": "assistant", "content": "model_questions"},
    ]
    agent._append_seed(chat, reason="post_compaction")
    chat.chat_append("handoff_prompt", "{}")

    contents = [m["content"] for m in chat.messages]
    assert contents[0] == REAL_PROMPT  # instructions survived compaction
    # calibration pair sits directly above the next real prompt, as at turn 1
    assert contents[-4] == CALIBRATION_USER
    assert contents[-3] == CALIBRATION_ASSISTANT
    assert contents[-2] == "handoff_prompt"


def test_reseed_is_idempotent_no_duplicate_pairs(agent):
    chat = FakeChat()
    agent._append_seed(chat, reason="initial")
    agent._strip_seed(chat)
    agent._append_seed(chat, reason="post_compaction")

    assert sum(1 for m in chat.messages if m["content"] == CALIBRATION_ASSISTANT) == 1


def test_seed_present_tracks_installation_and_removal(agent):
    chat = FakeChat()
    assert not agent._seed_present(chat)
    agent._append_seed(chat, reason="initial")
    assert agent._seed_present(chat)
    agent._strip_seed(chat)
    assert not agent._seed_present(chat)


def test_legacy_envelope_detection_matches_v1_failure_shape():
    v1_output = (
        "Analysis: We are in the workspace directory.\n"
        "Plan: List the contents.\n"
        '<tool_call>\n{"name": "bash_command", "arguments": {"keystrokes": "ls -la\\n"}}\n</tool_call>'
    )
    assert has_legacy_envelope(v1_output)
    assert not is_native_json(v1_output)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"analysis": "a", "plan": "p", "commands": []}', True),
        ('```json\n{"analysis": "a"}\n```', False),  # fenced: harbor salvages, strict tier fails
        ('Here is my answer:\n{"analysis": "a"}', False),  # prose preamble
        ('<think>hm</think>{"analysis": "a"}', False),  # thinking block
        ("[1, 2, 3]", False),  # valid JSON, wrong type
    ],
)
def test_native_json_is_the_strict_tier(text, expected):
    assert is_native_json(text) is expected


# ---- launch path -------------------------------------------------------------


def test_import_path_survives_agent_name_normalization():
    """`vektori passk` hyphenates agent names; an import path must be exempt.

    Without this, `scripts.prompt_seed_probe:Terminus2PromptSeed` reaches harbor as
    `scripts.prompt-seed-probe:Terminus2PromptSeed` and fails only after startup.
    """
    import inspect

    from vektori_trace.evaluate import validity
    from vektori_trace.evaluate.validity import run_trial  # noqa: F401

    src = inspect.getsource(validity.run_trial)
    assert 'if ":" not in agent:' in src, "normalization must be guarded on bare names"


@pytest.mark.parametrize(
    "agent,expected",
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
def test_normalization_rule_matches_bare_names_only(agent, expected):
    normalized = agent if ":" in agent else agent.replace("_", "-")
    assert normalized == expected
