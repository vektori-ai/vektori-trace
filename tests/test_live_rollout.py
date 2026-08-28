from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_agent import LiveCaptureError, TurnCapture
from vektori_trace.tau2.live_episode import EpisodeArchive
from vektori_trace.tau2.live_rollout import (
    EpisodePlan,
    EpisodeResult,
    RolloutSettings,
    capture_live_update,
)
from vektori_trace.tau2.reopd_state import ReOPDStateError, UpdateDir, read_jsonl


def _settings() -> RolloutSettings:
    return RolloutSettings(
        domain="retail",
        student_model="qwen",
        api_base="http://student",
        policy_version="policy-1",
        adapter_hash="adapter-1",
        gen_config_hash="gen-1",
        max_tokens=128,
        max_input_tokens=4096,
    )


def _capture(plan: EpisodePlan) -> TurnCapture:
    return TurnCapture(
        episode_id=plan.episode_id,
        task_id=plan.task_id,
        turn_index=0,
        policy_version="policy-1",
        prompt_token_ids=[1, 2],
        sampled_token_ids=[65],
        behavior_logprobs=[-0.2],
        action_token_bytes=[b"A"],
        raw_text="A",
        finish_reason="stop",
        reasoning=None,
        content="A",
        tool_calls=[],
    )


class SuccessfulRunner:
    def run(self, plan, *, on_turn, on_failure, on_observation):
        del on_failure
        cap = _capture(plan)
        history = [{"role": "system", "content": "policy"},
                   {"role": "user", "content": "help"}]
        on_turn(cap, history, {"role": "assistant", "content": "A"})
        on_observation(0, {"role": "user", "content": "thanks"}, "db-after")
        return EpisodeResult(
            termination_reason="user_stop",
            reward=1.0,
            reward_info={"reward": 1.0},
            initial_env_state_hash="db-before",
            final_env_state_hash="db-after",
        )


def test_capture_update_reaches_existing_sampled_boundary(tmp_path):
    update = UpdateDir(tmp_path, 0)
    report = capture_live_update(
        update,
        [EpisodePlan("ep-1", "57", 123)],
        settings=_settings(),
        teacher_context={"model": "deepseek", "tokenizer": "v4", "renderer": "native"},
        runner=SuccessfulRunner(),
    )

    assert report["ok"] is True
    assert update.stage() == "SAMPLED"
    update.validate()
    actions = read_jsonl(update.actions_path)
    assert [row["key"] for row in actions] == ["ep-1@0#0"]
    assert actions[0]["behavior_logprobs"] == [-0.2]
    assert actions[0]["score_fingerprint"]
    rendered = json.loads((update.path / "rendered.json").read_text())
    assert rendered["ep-1@0"][-1]["content"] == "help"

    archive = EpisodeArchive(update.path / "live_archive")
    turn = archive.load_turns("ep-1")[0]
    assert turn["observation"]["content"] == "thanks"
    assert turn["env_state_hash"] == "db-after"


class CaptureFailureRunner:
    def run(self, plan, *, on_turn, on_failure, on_observation):
        del on_turn, on_observation
        error = LiveCaptureError("finish_reason=length")
        on_failure(
            body={
                "choices": [{
                    "text": "partial",
                    "finish_reason": "length",
                    "token_ids": [1],
                    "logprobs": {"token_logprobs": [-0.3]},
                }]
            },
            episode_id=plan.episode_id,
            task_id=plan.task_id,
            turn_index=0,
            policy_version="policy-1",
            prompt_ids=[1, 2],
            semantic_history=[{"role": "user", "content": "help"}],
            error=error,
        )
        raise error


def test_capture_failure_is_counted_and_never_marks_sampled(tmp_path):
    update = UpdateDir(tmp_path, 0)
    with pytest.raises(ReOPDStateError, match="capture failure"):
        capture_live_update(
            update,
            [EpisodePlan("ep-1", "57", 123)],
            settings=_settings(),
            teacher_context={"model": "deepseek", "tokenizer": "v4", "renderer": "native"},
            runner=CaptureFailureRunner(),
        )
    assert update.stage() == "PLANNED"
    archive = EpisodeArchive(update.path / "live_archive")
    assert archive.load_episodes()["ep-1"]["status"] == "failed"
    assert archive.load_failures("ep-1")[0]["failure_kind"] == "cap_termination"


class InfrastructureFailureRunner:
    def run(self, plan, *, on_turn, on_failure, on_observation):
        del plan, on_turn, on_failure, on_observation
        raise ConnectionError("user simulator unavailable")


def test_infrastructure_failure_is_discarded_not_policy_failure(tmp_path):
    update = UpdateDir(tmp_path, 0)
    with pytest.raises(ReOPDStateError, match="discarded"):
        capture_live_update(
            update,
            [EpisodePlan("ep-1", "57", 123)],
            settings=_settings(),
            teacher_context={"model": "deepseek", "tokenizer": "v4", "renderer": "native"},
            runner=InfrastructureFailureRunner(),
        )
    meta = EpisodeArchive(update.path / "live_archive").load_episodes()["ep-1"]
    assert meta["status"] == "discarded"
    assert "infrastructure ConnectionError" in meta["discard_reason"]
