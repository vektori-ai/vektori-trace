"""Unit tests for the live capturing agent.

Everything here runs on CPU with no Tau2 install and no endpoint. The Tau2
seam itself (`CapturingLLMAgent`) imports Tau2 lazily, so the parts that carry
the real risk -- splitting a raw generation and refusing an untrainable capture
-- are testable anywhere.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_agent import (
    LiveCaptureError,
    build_capture,
    canonical_head,
    split_generation,
    tau2_messages_to_canonical,
    verify_ids_reconstruct_text,
)


# --- split_generation: the work vLLM's parsers do server-side --------------


def test_split_plain_message():
    reasoning, content, calls = split_generation("Hello, how can I help?")
    assert reasoning is None
    assert content == "Hello, how can I help?"
    assert calls == []


def test_split_think_and_message():
    raw = "<think>the user wants a return</think>Sure, I can help with that."
    reasoning, content, calls = split_generation(raw)
    assert reasoning == "the user wants a return"
    assert content == "Sure, I can help with that."
    assert calls == []


def test_split_think_and_tool_call():
    raw = (
        "<think>need the order first</think>"
        '<tool_call>\n{"name": "get_order_details", '
        '"arguments": {"order_id": "#W4284542"}}\n</tool_call>'
    )
    reasoning, content, calls = split_generation(raw)
    assert reasoning == "need the order first"
    assert content is None
    assert calls == [
        {"name": "get_order_details", "arguments": {"order_id": "#W4284542"}}
    ]


def test_split_multiple_tool_calls():
    raw = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    _, content, calls = split_generation(raw)
    assert content is None
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == {"y": 2}


def test_split_reasoning_survives_multiline_and_newlines():
    """The 47,338-character loop was reasoning. It must be captured whole."""
    body = "line one\nline two\n\nline three"
    reasoning, content, _ = split_generation(f"<think>{body}</think>done")
    assert reasoning == body
    assert content == "done"


def test_split_string_arguments_are_parsed():
    """Hermes usually emits an object, but a JSON string payload is legal and
    must not become a string-valued argument dict."""
    raw = '<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>'
    _, _, calls = split_generation(raw)
    assert calls[0]["arguments"] == {"x": 1}


def test_split_refuses_malformed_tool_json():
    raw = '<tool_call>{"name": "a", "arguments": {oops}}</tool_call>'
    with pytest.raises(LiveCaptureError, match="not JSON"):
        split_generation(raw)


def test_split_refuses_tool_call_without_name():
    raw = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    with pytest.raises(LiveCaptureError, match="no 'name'"):
        split_generation(raw)


def test_split_refuses_non_object_arguments():
    raw = '<tool_call>{"name": "a", "arguments": [1, 2]}</tool_call>'
    with pytest.raises(LiveCaptureError, match="expected object"):
        split_generation(raw)


# --- a tokenizer stand-in ---------------------------------------------------


class FakeTokenizer:
    """Byte-per-token, so ids and bytes are trivially checkable.

    `convert_ids_to_tokens` is present because `token_bytes_from_ids` prefers
    that path -- it is the one that survives a UTF-8 codepoint split across
    two tokens, which is what commits 1e63b6e/f3a1983 were about.
    """

    def convert_ids_to_tokens(self, ids):
        # ByteLevel BPE maps byte 0x41 -> "A"; mimic just enough of that.
        from vektori_trace.vocab_bridge import _token_str_to_bytes  # noqa: F401

        return [chr(i) for i in ids]


def _ids_for(text: str) -> list[int]:
    return list(text.encode("utf-8"))


# --- verify_ids_reconstruct_text -------------------------------------------


def test_ids_reconstructing_text_pass():
    tok = FakeTokenizer()
    text = "hello"
    out = verify_ids_reconstruct_text(tok, _ids_for(text), text)
    assert b"".join(out) == b"hello"


def test_ids_disagreeing_with_text_raise():
    tok = FakeTokenizer()
    with pytest.raises(LiveCaptureError, match="different prompt"):
        verify_ids_reconstruct_text(tok, _ids_for("hello"), "goodbye")


# --- build_capture: what must be refused ------------------------------------


def _body(text="hi", *, ids=None, logprobs=None, finish="stop"):
    ids = _ids_for(text) if ids is None else ids
    logprobs = [-0.1] * len(ids) if logprobs is None else logprobs
    return {
        "choices": [{
            "text": text,
            "finish_reason": finish,
            "token_ids": ids,
            "logprobs": {"token_logprobs": logprobs},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": len(ids)},
    }


def _capture(body, **kw):
    kw.setdefault("episode_id", "ep1")
    kw.setdefault("task_id", "57")
    kw.setdefault("turn_index", 0)
    kw.setdefault("policy_version", "sha-abc")
    kw.setdefault("prompt_ids", [1, 2, 3])
    kw.setdefault("tokenizer", FakeTokenizer())
    kw.setdefault("max_tokens", 2048)
    return build_capture(body, **kw)


def test_capture_happy_path():
    cap = _capture(_body("all set"))
    assert cap.content == "all set"
    assert cap.sampled_token_ids == _ids_for("all set")
    assert len(cap.behavior_logprobs) == len(cap.sampled_token_ids)
    assert cap.finish_reason == "stop"


def test_capture_refuses_cap_hit():
    """A truncated action is a fragment, not a decision. This is the failure
    that produced the 0/13 OPD run."""
    with pytest.raises(LiveCaptureError, match="finish_reason=length"):
        _capture(_body("truncated mid-th", finish="length"))


def test_capture_refuses_missing_token_ids():
    body = _body("hi")
    body["choices"][0]["token_ids"] = []
    with pytest.raises(LiveCaptureError, match="return_token_ids"):
        _capture(body)


def test_capture_refuses_logprob_length_mismatch():
    with pytest.raises(LiveCaptureError, match="logprobs"):
        _capture(_body("hello", logprobs=[-0.1, -0.2]))


def test_capture_refuses_null_logprob():
    ids = _ids_for("hi")
    lp = [-0.1] * len(ids)
    lp[0] = None
    with pytest.raises(LiveCaptureError, match="null"):
        _capture(_body("hi", logprobs=lp))


def test_capture_json_roundtrip_keeps_bytes_lossless():
    """`to_json` must not `.decode()` -- that is exactly what dropped UTF-8
    characters split across two tokens."""
    cap = _capture(_body("ok"))
    d = cap.to_json()
    assert json.loads(json.dumps(d))  # serialisable
    joined = b"".join(bytes.fromhex(h) for h in d["action_token_bytes_hex"])
    assert joined == b"ok"


# --- canonical rendering shape ---------------------------------------------


def test_canonical_head_carries_tools_on_the_system_turn():
    """`encoding_dsv4._render_system` reads `msg["tools"]` off the system turn.
    Carrying them beside the message leaves the teacher unconditioned on tools
    while the student's ids contain the full schema block -- finite loss,
    clean logs, wrong experiment."""
    head = canonical_head("POLICY", [{"type": "function", "function": {"name": "f"}}])
    assert head["role"] == "system"
    assert head["content"] == "POLICY"
    assert head["tools"][0]["function"]["name"] == "f"


class _Msg:
    def __init__(self, role, content=None, tool_calls=None, id=None):
        self.role, self.content, self.tool_calls, self.id = role, content, tool_calls, id


class _TC:
    def __init__(self, id, name, arguments):
        self.id, self.name, self.arguments = id, name, arguments


def test_canonical_messages_shape():
    msgs = [
        _Msg("user", "I want a refund"),
        _Msg("assistant", None, [_TC("c1", "get_order", {"id": "#W1"})]),
        _Msg("tool", '{"status": "ok"}', id="c1"),
    ]
    out = tau2_messages_to_canonical("POLICY", [], msgs)
    assert out[0]["role"] == "system"
    assert out[1] == {"role": "user", "content": "I want a refund"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "get_order"
    # arguments must be a JSON *string* -- that is what the chat template wants
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"id": "#W1"}
    assert out[3] == {"role": "tool", "content": '{"status": "ok"}',
                      "tool_call_id": "c1"}


def test_canonical_messages_refuse_unknown_role():
    """A silently skipped turn renders a prompt describing a conversation that
    never happened."""
    with pytest.raises(LiveCaptureError, match="unhandled Tau2 message role"):
        tau2_messages_to_canonical("P", [], [_Msg("developer", "x")])


# --- shapes taken from the real corpus --------------------------------------
# Pulled from /data/tau2/artifacts_16384/rows.semantic.jsonl on 2026-08-28.
# The invented cases above all had *either* content *or* a tool call; every
# real DeepSeek target carries both, and often several calls in one turn.


def test_split_content_and_tool_calls_together():
    """The common real shape: a sentence to the user AND tool calls."""
    raw = (
        "I'll help you with that. Let me first authenticate your identity."
        '<tool_call>\n{"name": "find_user_id_by_name_zip", "arguments": '
        '{"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}\n</tool_call>'
        '<tool_call>\n{"name": "list_all_product_types", "arguments": {}}\n</tool_call>'
    )
    reasoning, content, calls = split_generation(raw)
    assert reasoning is None
    assert content == "I'll help you with that. Let me first authenticate your identity."
    assert [c["name"] for c in calls] == [
        "find_user_id_by_name_zip", "list_all_product_types"
    ]
    assert calls[0]["arguments"]["zip"] == "19122"
    # An empty argument object must stay an empty dict, not become None.
    assert calls[1]["arguments"] == {}


def test_split_preserves_markdown_and_newlines_in_content():
    """Real targets carry markdown and blank lines; stripping only the ends
    must not disturb the middle."""
    body = (
        "I've authenticated your identity, Yusuf.\n\n"
        "Regarding the t-shirt question: there are **10 t-shirt options "
        "currently available**.\n\nNow, let me look up your orders."
    )
    raw = body + '<tool_call>{"name": "get_order_details", "arguments": ' \
                 '{"order_id": "#W6247578"}}</tool_call>'
    _, content, calls = split_generation(raw)
    assert content == body
    assert calls[0]["arguments"]["order_id"] == "#W6247578"


def test_split_five_tool_calls_in_one_turn():
    """Task 2 position 3 issues five `get_order_details` calls at once."""
    ids = ["#W6247578", "#W9711842", "#W4776164", "#W6679257", "#W2378156"]
    raw = "Looking up your orders." + "".join(
        '<tool_call>{"name": "get_order_details", "arguments": '
        f'{{"order_id": "{oid}"}}}}</tool_call>' for oid in ids
    )
    _, content, calls = split_generation(raw)
    assert content == "Looking up your orders."
    assert [c["arguments"]["order_id"] for c in calls] == ids


def test_split_conversational_target_yields_no_calls():
    """A plain message that parses as a call is a failure mode the parity
    check exists to catch; the splitter must not invent one."""
    raw = "Your return is all set. You'll get an email shortly."
    reasoning, content, calls = split_generation(raw)
    assert (reasoning, content, calls) == (None, raw, [])


# --- the think-wrapper boundary --------------------------------------------


def test_render_appends_the_think_wrapper():
    """Regression: the live render must end where the frozen prompt ends.

    `add_generation_prompt=True` stops at `<|im_start|>assistant\\n`, but the
    corpus's `prompt_token_ids` cut at the first *unmasked* label -- and
    `mask_think_wrapper=True` masks the four `<think>\\n\\n</think>\\n\\n`
    tokens, putting them on the prompt side of the cut.

    Without the append, live render parity measured **0/289** on the box
    (2026-08-28); with it, 289/289. A live agent that stopped four tokens early
    would ask the model to open a reasoning block the corpus had already
    closed, and the loss would still be finite and the logs still clean.
    """
    from vektori_trace.tau2 import live_agent

    calls = {}

    def fake_encode(tokenizer, messages, template_kwargs, *, add_generation_prompt):
        calls["template_kwargs"] = template_kwargs
        calls["add_generation_prompt"] = add_generation_prompt
        return [10, 11, 12]

    import vektori_trace.dataset as ds

    orig_encode, orig_wrapper = ds._encode_messages, ds._think_wrapper_ids
    ds._encode_messages = fake_encode
    ds._think_wrapper_ids = lambda tok: [901, 902, 903, 904]
    try:
        ids = live_agent.render_prompt_ids(object(), [{"role": "system"}], [])
    finally:
        ds._encode_messages, ds._think_wrapper_ids = orig_encode, orig_wrapper

    assert ids == [10, 11, 12, 901, 902, 903, 904]
    # The serving template settings must be named, never left to defaults.
    assert calls["add_generation_prompt"] is True
    assert calls["template_kwargs"]["enable_thinking"] is True
    assert "tools" in calls["template_kwargs"]


# ---------------------------------------------------------------------------
# Failure archival: every capture refusal is a paid generation
# ---------------------------------------------------------------------------


def _archive_on_failure(body):
    """What the agent's `on_failure` hook does: salvage, then re-raise.

    Mirrors the call site in `_generate_next_message`, so a change to
    `build_capture`'s refusals is caught here rather than silently producing
    an `other`-bucketed record in a real run.
    """
    from vektori_trace.tau2.live_episode import build_failed_turn

    try:
        _capture(body)
    except LiveCaptureError as exc:
        return build_failed_turn(
            body=body, episode_id="ep1", task_id="57", turn_index=0,
            policy_version="sha-abc", prompt_ids=[1, 2, 3],
            semantic_history=[{"role": "user", "content": "hi"}], error=exc,
        )
    raise AssertionError("expected a capture failure")


def test_every_salvageable_capture_refusal_classifies_to_a_named_bucket():
    """`classify_failure` matches on message text, so an unmatched refusal
    would silently land in `other` and vanish from the validity metrics.

    Scope is `build_capture` refusals on a 200 body. A non-200 or a transport
    exception never reaches this hook -- there are no tokens to archive."""
    ids = _ids_for("hi")
    null_lp = [-0.1] * len(ids)
    null_lp[0] = None
    no_ids = _body("hi")
    no_ids["choices"][0]["token_ids"] = []

    cases = {
        "cap_termination": _body("truncated", finish="length"),
        "no_token_ids": no_ids,
        "logprob_mismatch": _body("hello", logprobs=[-0.1, -0.2]),
        "transport": {"choices": []},
    }
    for expected, body in cases.items():
        assert _archive_on_failure(body).failure_kind == expected, expected

    # A null logprob is a logprob problem, not an `other`.
    assert _archive_on_failure(_body("hi", logprobs=null_lp)).failure_kind == (
        "logprob_mismatch"
    )


def test_a_cap_termination_keeps_its_ids_and_logprobs_for_diagnosis():
    ft = _archive_on_failure(_body("truncated mid-th", finish="length"))
    assert ft.finish_reason == "length"
    assert ft.sampled_token_ids == _ids_for("truncated mid-th")
    assert len(ft.behavior_logprobs) == len(ft.sampled_token_ids)
    assert ft.raw_text == "truncated mid-th"
    # Archived for the validity metrics; never trainable as a finished action.
    assert ft.failure_kind == "cap_termination"


def test_on_failure_hook_is_wired_before_the_raise():
    """The hook must fire on the way out, not be swallowed."""
    import inspect

    from vektori_trace.tau2 import live_agent

    src = inspect.getsource(live_agent.CapturingLLMAgent._generate_next_message)
    assert "self._on_failure(" in src
    body = src[src.index("except LiveCaptureError") :]
    assert body.rindex("raise") > body.index("self._on_failure(")
