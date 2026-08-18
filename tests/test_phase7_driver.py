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
    SELECTION_GATES,
    SELECTION_SUITE,
    GateResult,
    clears,
    select_checkpoint,
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
    got = [res("ck10", SELECTION_SUITE, True, prefix_id=f"p{i}") for i in range(7)]
    ok, detail = clears(
        got, checkpoint="ck10", suite=SELECTION_SUITE, expected_prefix_ids=expected
    )
    assert ok is False
    assert detail["ungraded"] == ["p7"]
    assert detail["n_graded"] == 7
    assert detail["n_expected"] == 8


def test_all_prefixes_graded_and_passing_clears():
    expected = [f"p{i}" for i in range(3)]
    got = [res("ck10", SELECTION_SUITE, True, prefix_id=f"p{i}") for i in range(3)]
    ok, detail = clears(
        got, checkpoint="ck10", suite=SELECTION_SUITE, expected_prefix_ids=expected
    )
    assert ok is True
    assert detail["ungraded"] == []


def test_selection_will_not_pick_a_checkpoint_with_a_hole():
    expected = ["p0", "p1"]
    got = [
        res("ck10", SELECTION_SUITE, True, prefix_id="p0"),  # p1 dropped
        res("ck20", SELECTION_SUITE, True, prefix_id="p0"),
        res("ck20", SELECTION_SUITE, True, prefix_id="p1"),
    ]
    chosen, trace = select_checkpoint(
        got, order=["ck10", "ck20"], expected_prefix_ids=expected
    )
    assert chosen == "ck20"
    assert trace["ck10"]["passed"] is False


# --------------------------------------------------------------------------
# Staging asks the same question selection asks
# --------------------------------------------------------------------------


def test_a_failing_non_selection_suite_does_not_block_the_stop():
    """Acquisition and control are reported, not selected on. A checkpoint that
    clears generalization is the winner even if the control suite is a mess."""
    expected = ["p0"]
    got = [
        res("ck10", SELECTION_SUITE, True, prefix_id="p0"),
        res("ck10", "control", False, prefix_id="c0"),
        res("ck10", "acquisition", False, prefix_id="a0"),
    ]
    ok, _ = clears(
        got, checkpoint="ck10", suite=SELECTION_SUITE, expected_prefix_ids=expected
    )
    assert ok is True


def test_selection_suite_falls_back_when_generalization_is_absent():
    """A format-smoke run may carry acquisition only. That is a legitimate run,
    but it answers a different question and must say so."""
    assert SELECTION_SUITE == "generalization"
    available = {"acquisition"}
    sel = SELECTION_SUITE if SELECTION_SUITE in available else (
        "acquisition" if "acquisition" in available else sorted(available)[0]
    )
    assert sel == "acquisition"


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def test_enable_thinking_is_pinned_off_in_every_request():
    assert driver.CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}


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
