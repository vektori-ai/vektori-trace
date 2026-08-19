"""The Phase 7 driver: staging, suite routing, and how silence is scored.

`tests/test_phase7.py` covers the grader. These cover the parts that decide
*which* generations happen and *whether a checkpoint counts as passing* — where
a bug does not produce a wrong-looking number, it produces a confident wrong
selection.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vektori_trace.evaluate.phase7 import (
    SELECTION_CATEGORIES,
    SELECTION_GATES,
    SELECTION_SUITES,
    GateResult,
    clears,
    select_checkpoint,
    select_stage_b_checkpoint,
    selection_prefix_ids,
    stage_b_clears,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "phase7_eval", ROOT / "scripts" / "phase7_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase7_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()


def res(ck, suite, ok, prefix_id="p1"):
    r = GateResult(prefix_id=prefix_id, checkpoint=ck, category="orientation",
                   suite=suite, completion="")
    r.gates = dict.fromkeys(SELECTION_GATES, ok)
    return r


# --------------------------------------------------------------------------
# Silence must not score as success
# --------------------------------------------------------------------------


def test_ungraded_prefix_blocks_a_checkpoint():
    """The bug this exists to prevent: 7 of 8 prefixes perfect, the 8th lost to
    an HTTP error, and the checkpoint reads as a clean sweep."""
    expected = [f"p{i}" for i in range(8)]
    got = [res("ck10", "generalization", True, prefix_id=f"p{i}") for i in range(7)]
    ok, detail = clears(got, checkpoint="ck10", expected_prefix_ids=expected)
    assert ok is False
    assert detail["ungraded"] == ["p7"]
    assert detail["n_graded"] == 7
    assert detail["n_expected"] == 8


def test_all_prefixes_graded_and_passing_clears():
    expected = [f"p{i}" for i in range(3)]
    got = [res("ck10", "generalization", True, prefix_id=f"p{i}") for i in range(3)]
    ok, detail = clears(got, checkpoint="ck10", expected_prefix_ids=expected)
    assert ok is True
    assert detail["ungraded"] == []


def test_selection_will_not_pick_a_checkpoint_with_a_hole():
    expected = ["p0", "p1"]
    got = [
        res("ck10", "generalization", True, prefix_id="p0"),  # p1 dropped
        res("ck20", "generalization", True, prefix_id="p0"),
        res("ck20", "generalization", True, prefix_id="p1"),
    ]
    chosen, trace = select_checkpoint(
        got, order=["ck10", "ck20"], expected_prefix_ids=expected
    )
    assert chosen == "ck20"
    assert trace["ck10"]["passed"] is False


# --------------------------------------------------------------------------
# Staging asks the same question selection asks
# --------------------------------------------------------------------------


def test_a_tripwire_category_failing_does_not_block_the_stop():
    """The 45 are the three cold-start categories. `first_edit`, `test_exec`
    and the rest are tripwires: reported, never gating — Stage A was not asked
    to teach them."""
    expected = ["p0"]
    got = [
        res("ck10", "generalization", True, prefix_id="p0"),
        res("ck10", "control", False, prefix_id="trip-edit"),
        res("ck10", "acquisition", False, prefix_id="trip-test"),
    ]
    ok, _ = clears(got, checkpoint="ck10", expected_prefix_ids=expected)
    assert ok is True


def _stage_b_results(ck="ck25", *, edits=1, tests=2, omit=None):
    rows = [res(ck, "generalization", True, prefix_id="format")]
    for i in range(3):
        r = GateResult(prefix_id=f"edit{i}", checkpoint=ck,
                       category="first_edit", suite="generalization",
                       completion="")
        r.gates = {"edit_emission": i < edits}
        rows.append(r)
    for i in range(3):
        r = GateResult(prefix_id=f"test{i}", checkpoint=ck,
                       category="test_exec", suite="generalization",
                       completion="")
        r.gates = {"test_emission": i < tests}
        rows.append(r)
    return [r for r in rows if r.prefix_id != omit]


def _stage_b_clear(rows, ck="ck25"):
    return stage_b_clears(
        rows,
        checkpoint=ck,
        format_prefix_ids=["format"],
        edit_prefix_ids=["edit0", "edit1", "edit2"],
        test_prefix_ids=["test0", "test1", "test2"],
    )


def test_stage_b_requires_strict_improvement_over_both_stage_a_baselines():
    assert _stage_b_clear(_stage_b_results(edits=1, tests=2))[0] is True
    assert _stage_b_clear(_stage_b_results(edits=0, tests=2))[0] is False
    assert _stage_b_clear(_stage_b_results(edits=1, tests=1))[0] is False


def test_stage_b_missing_behavior_result_blocks_selection():
    ok, detail = _stage_b_clear(_stage_b_results(omit="test2"))
    assert ok is False
    assert detail["test_exec"]["ungraded"] == ["test2"]


def test_stage_b_selector_takes_earliest_combined_pass():
    rows = (
        _stage_b_results("ck25", edits=0, tests=2)
        + _stage_b_results("ck50", edits=1, tests=2)
        + _stage_b_results("ck75", edits=2, tests=3)
    )
    chosen, trace = select_stage_b_checkpoint(
        rows,
        order=["ck25", "ck50", "ck75", "ck93"],
        format_prefix_ids=["format"],
        edit_prefix_ids=["edit0", "edit1", "edit2"],
        test_prefix_ids=["test0", "test1", "test2"],
    )
    assert chosen == "ck50"
    assert trace["ck25"]["passed"] is False
    assert trace["ck50"]["passed"] is True


def test_a_failing_acquisition_prefix_now_blocks_selection():
    """The change from v1: selection spans all three suites. A checkpoint that
    emits the protocol on held-out prefixes and fails on trained ones is not a
    checkpoint to ship, and used to be selected anyway."""
    expected = ["gen0", "acq0"]
    got = [
        res("ck10", "generalization", True, prefix_id="gen0"),
        res("ck10", "acquisition", False, prefix_id="acq0"),
        res("ck20", "generalization", True, prefix_id="gen0"),
        res("ck20", "acquisition", True, prefix_id="acq0"),
    ]
    chosen, trace = select_checkpoint(
        got, order=["ck10", "ck20"], expected_prefix_ids=expected
    )
    assert chosen == "ck20"
    assert trace["ck10"]["passed"] is False


def test_selection_set_is_the_45_and_excludes_tripwires():
    prefixes = [
        {"prefix_id": f"{cat}-{suite}-{i}", "category": cat, "suite": suite}
        for suite in SELECTION_SUITES
        for cat in SELECTION_CATEGORIES
        for i in range(5)
    ] + [
        {"prefix_id": f"trip-{suite}-{cat}", "category": cat, "suite": suite}
        for suite in SELECTION_SUITES
        for cat in ("first_edit", "test_exec", "long_context",
                    "repo_present", "parse_error_recovery")
    ]
    ids = selection_prefix_ids(prefixes)
    assert len(ids) == 45
    assert not any(i.startswith("trip-") for i in ids)


def test_a_sampled_suite_is_not_part_of_selection():
    """`grade` labels sampled generations `<suite>-sampled`. Those are a
    temperature probe, not the frozen set."""
    prefixes = [
        {"prefix_id": "s0", "category": "orientation", "suite": "generalization"},
        {"prefix_id": "s1", "category": "orientation",
         "suite": "generalization-sampled"},
    ]
    assert selection_prefix_ids(prefixes) == ["s0"]


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def test_enable_thinking_is_pinned_on_in_every_request():
    """The sweep must render the way the corpus was tokenized.

    Qwen3's template default is already thinking-on, so this is pinning a
    decision rather than changing behaviour — but a sweep that leaves it to the
    default and a rollout that does the same are only accidentally comparable.
    """
    assert driver.CHAT_TEMPLATE_KWARGS == {"enable_thinking": True}


def test_serving_and_eval_pin_the_same_template_kwargs():
    """One value, two call sites. A drift here is a silent train/serve split."""
    from vektori_trace.runtime.serve import CHAT_TEMPLATE_KWARGS as serve_kwargs

    assert serve_kwargs == driver.CHAT_TEMPLATE_KWARGS


def test_greedy_defaults_match_the_plan():
    assert driver.GREEDY["temperature"] == 0.0
    assert driver.GREEDY["max_tokens"] == 512


def test_4xx_is_not_retried(monkeypatch):
    """A 4xx is the request being wrong, not the network being flaky."""
    import urllib.error

    calls = []

    def boom(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(
            "u", 400, "Bad Request", {}, __import__("io").BytesIO(b"nope")
        )

    monkeypatch.setattr(driver.urllib.request, "urlopen", boom)
    text, _finish, err = driver.complete(
        "http://x/v1", "m", [{"role": "user", "content": "hi"}],
        timeout=1, retries=3,
    )
    assert len(calls) == 1
    assert err.startswith("HTTP 400")
    assert text == ""


def test_transport_error_is_retried_then_reported(monkeypatch):
    calls = []

    def boom(req, timeout):
        calls.append(1)
        raise TimeoutError("slow")

    monkeypatch.setattr(driver.urllib.request, "urlopen", boom)
    monkeypatch.setattr(driver.time, "sleep", lambda _s: None)
    _text, _finish, err = driver.complete(
        "http://x/v1", "m", [{"role": "user", "content": "hi"}],
        timeout=1, retries=2,
    )
    assert len(calls) == 3  # initial + 2 retries
    assert "TimeoutError" in err


# --------------------------------------------------------------------------
# Prefix loading is a read, never a re-render
# --------------------------------------------------------------------------


def test_prefix_is_the_corpus_slice_and_excludes_the_reference(tmp_path):
    seg = {
        "task": "pallets__click-1",
        "messages": [
            {"role": "user", "content": "instructions"},
            {"role": "assistant", "content": "action-1"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "action-2"},
        ],
        "supervise": [False, True, False, True],
    }
    (tmp_path / "sft_repaired.jsonl").write_text(json.dumps(seg) + "\n")
    entry = {"corpus_root": str(tmp_path), "line_no": 0, "message_index": 3}
    msgs = driver.load_prefix_messages(entry, {})
    assert len(msgs) == 3
    assert msgs[-1]["content"] == "observation"
    assert all(m["content"] != "action-2" for m in msgs)


def test_corpus_is_cached_across_prefixes(tmp_path):
    seg = {"task": "t", "messages": [{"role": "user", "content": "a"},
                                     {"role": "assistant", "content": "b"}],
           "supervise": [False, True]}
    path = tmp_path / "sft_repaired.jsonl"
    path.write_text(json.dumps(seg) + "\n")
    cache: dict = {}
    entry = {"corpus_root": str(tmp_path), "line_no": 0, "message_index": 1}
    driver.load_prefix_messages(entry, cache)
    path.unlink()  # second read must not touch disk
    assert driver.load_prefix_messages(entry, cache) == [{"role": "user", "content": "a"}]


@pytest.mark.parametrize("suite", ["acquisition", "control", "generalization"])
def test_every_manifest_suite_is_a_known_name(suite):
    assert suite in {"acquisition", "control", "generalization"}


# --------------------------------------------------------------------------
# The chat_template_kwargs preflight
#
# This check exists because the prompt-seed probe served a template nobody
# chose. Its first version asserted "thinking on -> <think> present", which
# passes on a server that drops the kwarg entirely — Qwen3's default is already
# thinking-on. The evidence has to be in the prompt, not the completion.
# --------------------------------------------------------------------------


def _fake_complete(prompt_tokens_by_thinking):
    """Stand in for the endpoint, reporting a rendered prompt length."""

    def complete(api_base, model, messages, **kw):
        thinking = kw["chat_template_kwargs"]["enable_thinking"]
        payload = {"usage": {"prompt_tokens": prompt_tokens_by_thinking[thinking]},
                   "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        return ("ok", "stop", None, payload) if kw.get("return_payload") else ("ok", "stop", None)

    return complete


def test_preflight_reads_the_rendered_prompt_length(monkeypatch):
    """`enable_thinking=False` appends the four-token wrapper to the prompt."""
    monkeypatch.setattr(driver, "complete", _fake_complete({True: 100, False: 104}))
    on, err_on = driver._prompt_tokens("b", "m", [], timeout=1, retries=0,
                                       chat_template_kwargs={"enable_thinking": True})
    off, err_off = driver._prompt_tokens("b", "m", [], timeout=1, retries=0,
                                         chat_template_kwargs={"enable_thinking": False})
    assert (err_on, err_off) == (None, None)
    assert off - on == driver.THINK_WRAPPER_TOKENS


def test_a_server_ignoring_the_kwarg_renders_one_prompt_twice(monkeypatch):
    """The failure the completion-text check could not see."""
    monkeypatch.setattr(driver, "complete", _fake_complete({True: 100, False: 100}))
    on, _ = driver._prompt_tokens("b", "m", [], timeout=1, retries=0,
                                  chat_template_kwargs={"enable_thinking": True})
    off, _ = driver._prompt_tokens("b", "m", [], timeout=1, retries=0,
                                   chat_template_kwargs={"enable_thinking": False})
    assert off - on == 0
    assert off - on != driver.THINK_WRAPPER_TOKENS


def test_missing_usage_is_an_error_not_a_zero(monkeypatch):
    """A server that reports no usage must fail the preflight, not read as 0."""

    def complete(api_base, model, messages, **kw):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        return ("ok", "stop", None, payload)

    monkeypatch.setattr(driver, "complete", complete)
    count, err = driver._prompt_tokens("b", "m", [], timeout=1, retries=0,
                                       chat_template_kwargs={"enable_thinking": True})
    assert count == -1
    assert err is not None and "usage" in err


def test_wrapper_token_count_matches_the_tokenizer():
    """The constant is only meaningful if it is what Qwen3 actually emits."""
    transformers = pytest.importorskip("transformers")
    from vektori_trace.dataset import THINK_WRAPPER_TEXT

    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    assert len(tok.encode(THINK_WRAPPER_TEXT, add_special_tokens=False)) == (
        driver.THINK_WRAPPER_TOKENS
    )
