"""What the proposer is shown.

The prompt's content is load-bearing and was wrong in a way no unit test would
have caught: given only the opening request and a turn count, the model has
nothing behavioural to reason about and proposes task domains ("Cloud Object
Storage Upload") instead of capabilities. Those can't be lacking, so every gap
downstream is measured against a list of subject headings. The planted-deficit
self-test caught it; these tests keep it caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vektori_trace.evaluate import diagnose
from vektori_trace.schema import ToolCall, Trace, Turn


def _trace(run_id: str, outcome: str, *, tool: str = "s3_upload", n_pad: int = 0) -> Trace:
    turns = [
        Turn(index=0, role="user", content=f"Do the {tool} thing"),
        Turn(
            index=1,
            role="assistant",
            thinking="I will call the tool",
            content="Working on it.",
            tool_calls=[ToolCall(id="c1", name=tool, args={"k": "v"})],
        ),
        Turn(index=2, role="tool", tool_call_id="c1", content='{"error": "AccessDenied"}'),
    ]
    turns += [Turn(index=3 + i, role="assistant", content=f"pad {i}") for i in range(n_pad)]
    return Trace(
        run_id=run_id, status="", turns=turns, outcome=outcome, source_path=Path(run_id)
    )


@pytest.fixture
def captured(monkeypatch):
    """Capture the prompt instead of calling OpenAI."""
    seen: dict[str, str] = {}

    def fake_call_json(system, user, schema_name, json_schema, model=None):
        seen["system"] = system
        seen["user"] = user
        return {"capabilities": [{"id": "x", "name": "X", "description": "d"}]}

    monkeypatch.setattr(diagnose, "call_json", fake_call_json)
    return seen


def test_prompt_contains_actual_trajectory_content(captured) -> None:
    """Tool names, arguments and results have to reach the model — they are the
    only evidence of behaviour in a trace."""
    diagnose.propose_capabilities([_trace("run-1", "win"), _trace("run-2", "loss")])
    prompt = captured["user"]

    assert "s3_upload" in prompt
    assert "AccessDenied" in prompt
    assert "I will call the tool" in prompt
    assert "run-1" in prompt and "run-2" in prompt


def test_prompt_never_reveals_outcome(captured) -> None:
    """Blind, like the labeller. Told which runs failed, the model proposes
    capabilities describing the failures it was shown and the separation comes
    out of the prompt instead of the data."""
    diagnose.propose_capabilities(
        [_trace("run-1", "win"), _trace("run-2", "loss"), _trace("run-3", "loss")]
    )
    prompt = captured["user"].lower()

    for banned in ("win", "loss", "succeeded", "failed the task", "outcome"):
        assert banned not in prompt, f"{banned!r} leaked into the proposer prompt"


def test_prompt_rules_out_task_domains(captured) -> None:
    """The exact failure observed: seven proposed 'capabilities', every one a
    task domain. The instruction has to name and exclude that."""
    diagnose.propose_capabilities([_trace("run-1", "win"), _trace("run-2", "loss")])
    prompt = captured["user"]

    assert "domain" in prompt.lower()
    assert "LACKS" in prompt  # the stated test for whether something qualifies


def test_large_corpora_are_sampled_not_truncated(captured) -> None:
    """Manifests arrive grouped by outcome, so showing the first k traces shows
    the model only wins. An even stride covers both ends."""
    traces = [_trace(f"win-{i}", "win") for i in range(30)]
    traces += [_trace(f"loss-{i}", "loss") for i in range(30)]

    diagnose.propose_capabilities(traces, max_traces=10)
    prompt = captured["user"]

    assert "sampled from 60" in prompt
    # Count excerpt headers, not bare ids: "win-1" is a substring of "win-10".
    assert prompt.count("--- ") == 10
    assert any(f"--- win-{i} (" in prompt for i in range(30))
    assert any(f"--- loss-{i} (" in prompt for i in range(30))


def test_small_corpora_are_shown_whole(captured) -> None:
    diagnose.propose_capabilities([_trace("a", "win"), _trace("b", "loss")], max_traces=10)
    assert "sampled from" not in captured["user"]


def test_long_traces_are_truncated_per_trace(captured) -> None:
    """One 400-turn trace must not crowd out every other trace in the prompt."""
    diagnose.propose_capabilities(
        [_trace("long", "loss", n_pad=400), _trace("short", "win")],
        max_turns_per_trace=8,
    )
    prompt = captured["user"]

    assert "turns omitted" in prompt
    assert "short" in prompt


def test_sample_evenly_spans_the_list() -> None:
    items = list(range(100))
    picked = diagnose._sample_evenly(items, 5)
    assert len(picked) == 5
    assert picked[0] == 0
    assert picked[-1] >= 80  # reaches the far end, not just the head


def test_sample_evenly_returns_everything_when_small() -> None:
    items = list(range(3))
    assert diagnose._sample_evenly(items, 10) == items
