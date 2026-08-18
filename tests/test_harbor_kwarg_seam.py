"""The agent kwarg harbor actually reads — checked against harbor, not a guess.

`served_to_harbor_kwargs` used to emit `agent_kwargs={"extra_body": ...}`. That
is not a seam Terminus2 has. It declares `llm_call_kwargs` as a named parameter
and splats it into `self._llm.call(...)`; every other keyword falls through to
`BaseAgent.__init__`, which accepts `**kwargs` and never reads them. So the
request went out rendered with the template's own default and nothing said so —
the prompt-seed probe's "Harbor's terminus path dropped them".

Nothing about that failure is visible at runtime: no error, no warning, and the
rollout completes. The only cheap defence is to assert the parameter name
against the installed harbor, so an upstream rename fails here rather than
silently un-pinning the chat template on a GPU run.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("harbor")

from harbor.agents.terminus_2.terminus_2 import Terminus2

from vektori_trace.runtime.serve import (
    LLM_CALL_KWARGS_KEY,
    ServedModel,
    served_to_harbor_kwargs,
)


def _served() -> ServedModel:
    return ServedModel(api_base="http://localhost:8000/v1", model_name="Qwen3-14B-x")


def test_terminus2_declares_the_key_we_send():
    params = inspect.signature(Terminus2.__init__).parameters
    assert LLM_CALL_KWARGS_KEY in params, (
        f"harbor {getattr(__import__('harbor'), '__version__', '?')} no longer "
        f"takes {LLM_CALL_KWARGS_KEY!r}; find the new seam before running anything"
    )


def test_a_bare_extra_body_would_be_swallowed():
    """Why the nesting exists. `extra_body` is not a Terminus2 parameter, and
    `**kwargs` reaches a base class that discards it."""
    params = inspect.signature(Terminus2.__init__).parameters
    assert "extra_body" not in params
    assert any(p.kind is p.VAR_KEYWORD for p in params.values())


def test_chat_template_kwargs_ride_in_the_forwarded_slot():
    ak = served_to_harbor_kwargs(_served())["agent_kwargs"]
    assert ak[LLM_CALL_KWARGS_KEY]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert "extra_body" not in ak, "a top-level extra_body is dropped by harbor"


def test_token_capture_shares_the_slot_instead_of_taking_it():
    """Requesting token ids must not un-pin the chat template."""
    ak = served_to_harbor_kwargs(_served(), capture_tokens=True)["agent_kwargs"]
    body = ak[LLM_CALL_KWARGS_KEY]["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["return_token_ids"] is True


def test_rollout_capture_merge_preserves_the_chat_template():
    """The same collision, on the path collect_rollouts takes."""
    from vektori_trace.runtime.rollout import _merge_capture_agent_kwargs

    merged = _merge_capture_agent_kwargs(
        served_to_harbor_kwargs(_served())["agent_kwargs"], capture_logprobs=False
    )
    body = merged[LLM_CALL_KWARGS_KEY]["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["return_token_ids"] is True


def test_capture_merge_works_from_nothing():
    from vektori_trace.runtime.rollout import _merge_capture_agent_kwargs

    merged = _merge_capture_agent_kwargs(None, capture_logprobs=False)
    assert merged[LLM_CALL_KWARGS_KEY]["extra_body"]["return_token_ids"] is True
