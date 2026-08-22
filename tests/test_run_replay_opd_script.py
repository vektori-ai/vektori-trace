"""Failure/recovery seams in the executable replay OPD runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[1] / "scripts" / "run_replay_opd.py"
_SPEC = importlib.util.spec_from_file_location("run_replay_opd_script", _PATH)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_capture_reads_vllm_prompt_ids_from_choice():
    cap = runner._Capture(
        {"prompt_token_ids": [999]},
        {"prompt_token_ids": [1, 2], "token_ids": [3], "logprobs": {}},
    )
    assert cap.prompt_token_ids == [1, 2]


def test_readiness_refuses_the_wrong_advertised_model(monkeypatch):
    monkeypatch.setattr(
        runner,
        "endpoint_models",
        lambda _base: (200, {"data": [{"id": "not-ck75"}]}),
    )
    with pytest.raises(SystemExit, match="does not advertise"):
        runner.require_endpoint_model("https://student.invalid/v1", "ck75")


def test_score_resume_tolerates_partial_tail_but_refuses_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    row = {
        "key": "p@1#0",
        "teacher_token_bytes_b64": ["eA=="],
        "teacher_logprobs": [-0.2],
    }
    path.write_text(json.dumps(row) + "\n{" )
    assert set(runner._load_scores(path)) == {"p@1#0"}

    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate teacher score"):
        runner._load_scores(path)
