"""Corpus loading and capture adaptation for replay OPD (plan §8).

Two boundaries where a silent failure is expensive:

- a capture without per-token logprobs cannot supply `log pi_old`, and that is
  unrecoverable after sampling — the run has to be repeated;
- a failed trace's late states describe a repo the agent may have corrupted, so
  §8 says start from passing trajectories and this asserts the loader does.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vektori_trace.replay_corpus import (
    CorpusError,
    TraceRecord,
    corpus_report,
    iter_trials,
    load_corpus,
    load_trace,
)
from vektori_trace.replay_sample import (
    CaptureAdaptError,
    sampled_action_from_capture,
    summarize_cap_hits,
    token_bytes_from_ids,
)

# ---------------------------------------------------------------------------
# A minimal Harbor trial tree
# ---------------------------------------------------------------------------


REAL_TRAJECTORY = (
    Path(__file__).parent
    / "fixtures"
    / "atif"
    / "real-terminus2-structlog"
    / "agent"
    / "trajectory.json"
)


def _write_trial(root, task: str, trial: str, *, reward=1.0):
    """A Harbor trial dir wrapping the repo's real ATIF fixture.

    Deliberately not a hand-rolled trajectory: `mining.atif` validates against
    Harbor's own Pydantic models, so a synthetic file tests the fixture-writer
    rather than the loader.
    """
    d = (
        root
        / f"{task}-0"
        / f"{task}-terminus-2"
        / "2026-08-13__00-00-00"
        / f"{task}__{trial}"
    )
    (d / "agent").mkdir(parents=True)
    shutil.copy(REAL_TRAJECTORY, d / "agent" / "trajectory.json")
    if reward is not None:
        # The shape this corpus actually uses: verifier_result.rewards.reward
        (d / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": {"reward": reward}}})
        )
    return d


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "dsv4-corpus60"
    _write_trial(root, "pypa__hatch-2086", "aaa", reward=1.0)
    _write_trial(root, "pallets__click-3466", "bbb", reward=0.0)
    _write_trial(root, "pallets__jinja-1702", "ccc", reward=1.0)
    return root


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_iter_trials_finds_every_agent_trajectory(corpus):
    trials = list(iter_trials(corpus))
    assert len(trials) == 3
    assert all((t / "agent" / "trajectory.json").is_file() for t in trials)


def test_missing_root_is_refused():
    with pytest.raises(CorpusError, match="not a directory"):
        list(iter_trials(Path("/nonexistent/corpus")))


def test_load_trace_reads_task_steps_and_outcome(corpus):
    trials = {p.name: p for p in iter_trials(corpus)}
    rec = load_trace(trials["pypa__hatch-2086__aaa"])

    assert rec.task == "pypa__hatch-2086"
    assert rec.trace_id == "pypa__hatch-2086__aaa"
    assert rec.n_steps > 0
    assert rec.passed is True


def test_failing_trace_is_marked_not_dropped(corpus):
    trials = {p.name: p for p in iter_trials(corpus)}
    rec = load_trace(trials["pallets__click-3466__bbb"])
    assert rec.passed is False


def test_missing_result_json_is_unknown_not_failed(tmp_path):
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "zzz", reward=None)
    rec = load_trace(d)
    assert rec.passed is None, "an unwritten result is not evidence of failure"


# ---------------------------------------------------------------------------
# §8: start from passing trajectories
# ---------------------------------------------------------------------------


def test_load_corpus_keeps_only_passing_by_default(corpus):
    traces = load_corpus(corpus)
    assert {t.task for t in traces} == {"pypa__hatch-2086", "pallets__jinja-1702"}
    assert all(t.passed is True for t in traces)


def test_passing_only_false_keeps_everything(corpus):
    traces = load_corpus(corpus, passing_only=False)
    assert len(traces) == 3


def test_unknown_outcome_excluded_under_passing_only(tmp_path):
    root = tmp_path / "c"
    _write_trial(root, "t__x", "zzz", reward=None)
    assert load_corpus(root) == []
    assert len(load_corpus(root, passing_only=False)) == 1


def test_too_short_traces_are_dropped(tmp_path):
    root = tmp_path / "c"
    _write_trial(root, "t__x", "zzz", reward=1.0)
    assert load_corpus(root, min_steps=10_000) == []
    assert len(load_corpus(root, min_steps=2)) == 1


def test_limit_bounds_the_load(corpus):
    assert len(load_corpus(corpus, passing_only=False, limit=1)) == 1


# ---------------------------------------------------------------------------
# corpus_report — the input to any kappa decision
# ---------------------------------------------------------------------------


def test_corpus_report_describes_the_length_distribution(corpus):
    rep = corpus_report(load_corpus(corpus, passing_only=False))

    assert rep["n_traces"] == 3
    assert rep["n_passed"] == 2
    assert rep["n_failed"] == 1
    assert rep["steps_min"] <= rep["steps_median"] <= rep["steps_max"]
    assert sum(rep["step_count_histogram"].values()) == 3


def test_empty_corpus_report():
    assert corpus_report([]) == {"n_traces": 0}


def test_report_counts_unknown_separately(tmp_path):
    root = tmp_path / "c"
    _write_trial(root, "t__x", "zzz", reward=None)
    rep = corpus_report(load_corpus(root, passing_only=False))
    assert rep["n_unknown"] == 1
    assert rep["n_passed"] == 0


# ---------------------------------------------------------------------------
# Capture -> SampledAction
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One byte per id, so byte reconstruction is checkable by hand."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + int(i)) for i in ids)


class FakeCapture:
    def __init__(self, token_ids, logprobs, finish_reason="stop"):
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.prompt_token_ids = [1, 2, 3]
        self.request_id = "req-1"
        self.model = "ck75"


def test_capture_becomes_a_sampled_action():
    cap = FakeCapture([1, 2, 3], [-0.1, -0.2, -0.3])
    a = sampled_action_from_capture(
        cap, FakeTokenizer(), prefix_id="tr0@2", sample_index=1, policy_version="v0"
    )

    assert a.action_token_ids == [1, 2, 3]
    assert a.behavior_logprobs == [-0.1, -0.2, -0.3]
    assert a.action_bytes == b"abc"
    assert b"".join(a.action_token_bytes) == a.action_bytes
    assert a.key == "tr0@2#1"
    assert a.policy_version == "v0"


def test_capture_without_logprobs_is_refused():
    """log pi_old cannot be recovered after sampling."""
    cap = FakeCapture([1, 2], None)
    with pytest.raises(CaptureAdaptError, match="no per-token logprobs"):
        sampled_action_from_capture(
            cap, FakeTokenizer(), prefix_id="p", sample_index=0, policy_version="v0"
        )


def test_logprob_count_mismatch_is_refused():
    cap = FakeCapture([1, 2, 3], [-0.1])
    with pytest.raises(CaptureAdaptError, match="internally inconsistent"):
        sampled_action_from_capture(
            cap, FakeTokenizer(), prefix_id="p", sample_index=0, policy_version="v0"
        )


def test_empty_capture_is_refused():
    cap = FakeCapture([], [])
    with pytest.raises(CaptureAdaptError, match="no sampled token ids"):
        sampled_action_from_capture(
            cap, FakeTokenizer(), prefix_id="p", sample_index=0, policy_version="v0"
        )


def test_truncated_capture_is_refused_by_default():
    """A cap-truncated action is a fragment the teacher would grade as complete."""
    cap = FakeCapture([1, 2], [-0.1, -0.2], finish_reason="length")
    with pytest.raises(CaptureAdaptError, match="cut this action mid-sequence"):
        sampled_action_from_capture(
            cap, FakeTokenizer(), prefix_id="p", sample_index=0, policy_version="v0"
        )


def test_truncated_capture_can_be_allowed_and_is_marked():
    cap = FakeCapture([1, 2], [-0.1, -0.2], finish_reason="length")
    a = sampled_action_from_capture(
        cap,
        FakeTokenizer(),
        prefix_id="p",
        sample_index=0,
        policy_version="v0",
        allow_truncated=True,
    )
    assert a.meta["truncated"] is True
    assert a.termination_reason == "length"


def test_token_bytes_reconstruct_the_action():
    got = token_bytes_from_ids(FakeTokenizer(), [1, 2, 3])
    assert got == [b"a", b"b", b"c"]
    assert b"".join(got) == b"abc"


def test_token_bytes_survive_utf8_split_across_bytelevel_tokens():
    """Single-id decode may be lossy even though the token sequence is not."""
    class SplitUtf8Tokenizer:
        def convert_ids_to_tokens(self, ids):
            # ByteLevel's unicode alphabet for the UTF-8 bytes E2 82 AC (€).
            return ["\u00e2", "\u0124", "\u00ac"]

        def decode(self, ids, **kwargs):
            # This is what independent decode calls commonly produce.
            return "\ufffd"

    got = token_bytes_from_ids(SplitUtf8Tokenizer(), [1, 2, 3])
    assert got == [b"\xe2", b"\x82", b"\xac"]
    assert b"".join(got).decode() == "\u20ac"


def test_cap_hit_summary():
    tok = FakeTokenizer()
    ok = sampled_action_from_capture(
        FakeCapture([1, 2, 3], [-0.1] * 3),
        tok,
        prefix_id="p",
        sample_index=0,
        policy_version="v0",
    )
    hit = sampled_action_from_capture(
        FakeCapture([1, 2], [-0.1] * 2, finish_reason="length"),
        tok,
        prefix_id="p",
        sample_index=1,
        policy_version="v0",
        allow_truncated=True,
    )

    rep = summarize_cap_hits([ok, hit])
    assert rep["n_actions"] == 2
    assert rep["n_cap_hits"] == 1
    assert rep["cap_hit_rate"] == pytest.approx(0.5)
    assert rep["max_action_tokens"] == 3
    assert rep["min_action_tokens"] == 2


def test_cap_hit_summary_on_empty_batch():
    assert summarize_cap_hits([])["cap_hit_rate"] == 0.0


def test_trace_record_defaults():
    rec = TraceRecord(task="t", trace_id="tr", trial_dir=Path("."), n_steps=3, passed=True)
    assert rec.stored_actions == {}


# ---------------------------------------------------------------------------
# Verdict parsing — the shape the real corpus uses
# ---------------------------------------------------------------------------


def test_reward_is_read_from_verifier_result_rewards(tmp_path):
    """The real layout: verifier_result.rewards.reward, not a top-level key."""
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "pass", reward=1.0)
    assert load_trace(d).passed is True


@pytest.mark.parametrize("reward", [0.5, 0.125, 0.777778, 0.993174])
def test_partial_credit_is_not_a_pass(tmp_path, reward):
    """Fractional rewards are some-F2P-passing, not a fixed bug.

    Counting them would put prefixes from traces that never solved the task
    into a pool §8 says starts from passing runs.
    """
    root = tmp_path / f"c{reward}"
    d = _write_trial(root, "t__x", "partial", reward=reward)
    assert load_trace(d).passed is False


def test_zero_reward_is_a_fail_not_unknown(tmp_path):
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "zero", reward=0.0)
    rec = load_trace(d)
    assert rec.passed is False


def test_result_without_a_reward_is_unknown(tmp_path):
    """AgentTimeoutError trials: the verifier never ran, so there is no verdict."""
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "timeout", reward=None)
    (d / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": None,
                "exception_info": {"exception_type": "AgentTimeoutError"},
            }
        )
    )
    assert load_trace(d).passed is None


def test_legacy_flat_reward_still_read(tmp_path):
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "legacy", reward=None)
    (d / "result.json").write_text(json.dumps({"reward": 1.0}))
    assert load_trace(d).passed is True


def test_nested_shape_wins_over_a_stale_flat_one(tmp_path):
    """A file carrying both must not let the older key decide."""
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "both", reward=None)
    (d / "result.json").write_text(
        json.dumps({"reward": 0.0, "verifier_result": {"rewards": {"reward": 1.0}}})
    )
    assert load_trace(d).passed is True


def test_unreadable_result_is_unknown(tmp_path):
    root = tmp_path / "c"
    d = _write_trial(root, "t__x", "bad", reward=None)
    (d / "result.json").write_text("{not json")
    assert load_trace(d).passed is None


# ---------------------------------------------------------------------------
# The capture -> action -> advantages seam for prompt ids
# ---------------------------------------------------------------------------


def test_prompt_token_ids_survive_capture_to_action():
    """The exact ids, not a count.

    The optimizer recomputes log pi_current over prompt+action. A capture that
    records only `n_prompt_tokens` produces an action the optimizer refuses,
    and the tiny-model tests populate prompt ids by hand so they cannot see it.
    """
    cap = FakeCapture([1, 2, 3], [-0.1, -0.2, -0.3])
    cap.prompt_token_ids = [77, 88, 99, 111]

    a = sampled_action_from_capture(
        cap, FakeTokenizer(), prefix_id="tr0@2", sample_index=0, policy_version="v0"
    )

    assert a.prompt_token_ids == [77, 88, 99, 111], "exact ids, in order"
    assert a.meta["n_prompt_tokens"] == 4


def test_prompt_ids_reach_turn_advantages():
    """End of the seam: what the optimizer actually reads."""
    from vektori_trace.replay_opd import build_replay_batch
    from vektori_trace.replay_select import ReplayPrefix

    cap = FakeCapture([1, 2, 3, 4], [-0.1] * 4)
    cap.prompt_token_ids = [5, 6, 7]
    prefix = ReplayPrefix(task="t", trace_id="tr0", step_index=2, prefix_turns=[])
    action = sampled_action_from_capture(
        cap,
        FakeTokenizer(),
        prefix_id=prefix.prefix_id,
        sample_index=0,
        policy_version="v0",
    )
    # One teacher token per student token keeps the alignment 1:1.
    scored = {action.key: ([b"a", b"b", b"c", b"d"], [-0.2] * 4)}

    # One prefix is 100% of one task by construction; the spread rules target a
    # real batch, so both caps are lifted for this seam check.
    batch = build_replay_batch(
        [prefix], [action], scored, max_trace_share=1.0, max_task_share=1.0
    )

    assert batch.advantages[0].prompt_token_ids == [5, 6, 7]


def test_capture_without_prompt_ids_is_refused():
    cap = FakeCapture([1, 2], [-0.1, -0.2])
    cap.prompt_token_ids = []
    with pytest.raises(CaptureAdaptError, match="no prompt token ids"):
        sampled_action_from_capture(
            cap, FakeTokenizer(), prefix_id="p", sample_index=0, policy_version="v0"
        )
