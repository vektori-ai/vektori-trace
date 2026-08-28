"""The episode archive, its state machine, and failure archival.

No GPU, no teacher, no Tau2 install: every test here is about whether the
schema can be trusted to refuse a batch it should refuse.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.live_agent import TurnCapture
from vektori_trace.tau2.live_episode import (
    EpisodeArchive,
    EpisodeStatus,
    FailedTurn,
    LiveEpisode,
    LiveTurn,
    build_failed_turn,
    classify_failure,
    semantic_hash,
)


def make_capture(episode_id="ep1", turn_index=0, policy_version="v0", n=3,
                 finish_reason="stop"):
    return TurnCapture(
        episode_id=episode_id,
        task_id="57",
        turn_index=turn_index,
        policy_version=policy_version,
        prompt_token_ids=[1, 2, 3],
        sampled_token_ids=list(range(10, 10 + n)),
        behavior_logprobs=[-0.1] * n,
        action_token_bytes=[b"ab", b"cd", b"ef"][:n],
        raw_text="hello",
        finish_reason=finish_reason,
        reasoning=None,
        content="hello",
        tool_calls=[],
    )


def make_turn(episode_id="ep1", turn_index=0, **kw):
    return LiveTurn(
        capture=make_capture(episode_id=episode_id, turn_index=turn_index, **kw),
        semantic_history=[{"role": "user", "content": "hi"}],
        parsed_message={"role": "assistant", "content": "hello"},
        observation={"role": "user", "content": "ok"},
        env_state_hash="deadbeef",
    )


def make_episode(episode_id="ep1", **kw):
    base = dict(
        episode_id=episode_id,
        task_id="57",
        seed=16,
        policy_version="v0",
        adapter_hash="a" * 8,
        gen_config_hash="g" * 8,
    )
    base.update(kw)
    return LiveEpisode(**base)


# -- state machine ---------------------------------------------------------


def test_an_episode_finishes_exactly_once():
    ep = make_episode()
    ep.finish(EpisodeStatus.SAMPLED)
    assert ep.status == EpisodeStatus.SAMPLED
    assert ep.ended_at is not None
    with pytest.raises(ValueError, match="illegal transition"):
        ep.finish(EpisodeStatus.DISCARDED, reason="too late")


def test_scoring_and_training_are_not_episode_states():
    """They belong to the update, and `RunState` already owns them. An episode
    that had to pass through them could never be both terminal and trainable."""
    for absent in ("scoring", "scored", "training", "checkpointed",
                   "serving_verified", "complete"):
        assert not hasattr(EpisodeStatus, absent.upper())
        with pytest.raises(ValueError):
            make_episode().finish(absent)


def test_an_unusable_outcome_must_carry_a_reason():
    for status in (EpisodeStatus.DISCARDED, EpisodeStatus.FAILED):
        ep = make_episode()
        with pytest.raises(ValueError, match="reason"):
            ep.finish(status)
        ep.finish(status, reason="user sim crashed")
        assert ep.discard_reason == "user sim crashed"


def test_sampled_needs_no_reason():
    ep = make_episode()
    ep.finish(EpisodeStatus.SAMPLED)
    assert ep.discard_reason is None


def test_failed_and_discarded_stay_distinct():
    """One is a measurement of the policy, the other an infrastructure event."""
    a, b = make_episode("ep1"), make_episode("ep2")
    a.finish(EpisodeStatus.FAILED, reason="agent error")
    b.finish(EpisodeStatus.DISCARDED, reason="host died")
    assert a.status != b.status
    assert {a.status, b.status} <= EpisodeStatus.UNUSABLE


def test_trainable_is_sampled_only():
    ep = make_episode()
    assert not ep.trainable          # SAMPLING: last turn may not have landed
    ep.finish(EpisodeStatus.SAMPLED)
    assert ep.trainable
    for status in (EpisodeStatus.FAILED, EpisodeStatus.DISCARDED):
        other = make_episode("ep2")
        other.finish(status, reason="x")
        assert not other.trainable


# -- episode ids -----------------------------------------------------------


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".", "..", "/abs",
                                 "x" * 129, "-leading"])
def test_unsafe_episode_ids_are_refused(bad):
    with pytest.raises(ValueError, match="unsafe episode id"):
        make_episode(bad)


def test_the_archive_refuses_to_open_an_unsafe_turns_path(tmp_path):
    arch = EpisodeArchive(tmp_path)
    with pytest.raises(ValueError, match="unsafe episode id"):
        arch.turns_path("../../etc/passwd")


def test_ordinary_generated_ids_are_accepted():
    for good in ("ep1", "u3-t57-s16", "task_57.seed16", "A0"):
        make_episode(good)


# -- archive round-trip ----------------------------------------------------


def test_turn_and_episode_round_trip(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn())
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)

    episodes = arch.load_episodes()
    assert episodes["ep1"]["status"] == EpisodeStatus.SAMPLED
    assert episodes["ep1"]["num_turns"] == 1
    turns = arch.load_turns("ep1")
    assert len(turns) == 1
    assert turns[0]["capture"]["sampled_token_ids"] == [10, 11, 12]
    # The teacher needs this, and it must survive the round trip.
    assert turns[0]["semantic_history"] == [{"role": "user", "content": "hi"}]


def test_episode_file_is_a_state_history_not_an_overwrite(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    lines = arch.episodes_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert arch.load_episodes()["ep1"]["status"] == EpisodeStatus.SAMPLED


def test_episode_identity_cannot_change_across_history(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode("ep1", adapter_hash="original")
    arch.record_episode(ep)
    forged = make_episode("ep1", adapter_hash="replacement")
    forged.started_at = ep.started_at
    forged.termination_reason = "stop"
    forged.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(forged)
    with pytest.raises(ValueError, match="immutable episode identity changed"):
        arch.load_episodes()


def test_no_record_may_follow_a_terminal_episode_state(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode("ep1")
    arch.record_episode(ep)
    ep.termination_reason = "stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    arch.record_episode(ep)
    with pytest.raises(ValueError, match="follows terminal state"):
        arch.load_episodes()


def test_duplicate_sampled_turn_records_are_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    arch.record_turn(make_turn("ep1", 0))
    arch.record_turn(make_turn("ep1", 0))
    with pytest.raises(ValueError, match="duplicate archived capture"):
        arch.load_turns("ep1")


def test_torn_final_line_does_not_destroy_earlier_records(tmp_path):
    arch = EpisodeArchive(tmp_path)
    arch.record_episode(make_episode("ep1"))
    with arch.episodes_path.open("a") as fh:
        fh.write('{"episode_id": "ep2", "sta')  # crash mid-append
    episodes = arch.load_episodes()
    assert list(episodes) == ["ep1"]


def test_a_corrupt_middle_line_still_raises(tmp_path):
    arch = EpisodeArchive(tmp_path)
    arch.record_episode(make_episode("ep1"))
    with arch.episodes_path.open("a") as fh:
        fh.write("not json\n")
    arch.record_episode(make_episode("ep2"))
    with pytest.raises(json.JSONDecodeError):
        arch.load_episodes()


# -- hashing ---------------------------------------------------------------


def test_semantic_hash_is_order_sensitive_and_key_order_stable():
    a = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    b = [{"role": "assistant", "content": "y"}, {"role": "user", "content": "x"}]
    assert semantic_hash(a) != semantic_hash(b)
    c = [{"content": "x", "role": "user"}, {"content": "y", "role": "assistant"}]
    assert semantic_hash(a) == semantic_hash(c)


# -- verification ----------------------------------------------------------


def complete_episode(arch, episode_id="ep1", n_turns=2, **ep_kw):
    ep = make_episode(episode_id, **ep_kw)
    arch.record_episode(ep)
    for i in range(n_turns):
        arch.record_turn(make_turn(episode_id, i, policy_version=ep.policy_version))
    ep.num_turns = n_turns
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    return ep


def test_clean_episode_verifies(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch)
    assert arch.verify_episode("ep1") == []


def test_unfinished_episode_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn())
    problems = arch.verify_episode("ep1")
    assert any("sampling did not finish" in p for p in problems)


def test_discarded_episode_is_refused_and_named(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    ep.finish(EpisodeStatus.DISCARDED, reason="user sim crashed")
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("user sim crashed" in p for p in problems)


def test_mixed_policy_version_within_an_episode_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0, policy_version="v0"))
    arch.record_turn(make_turn("ep1", 1, policy_version="v1"))
    ep.num_turns = 2
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("policy versions" in p for p in problems)


def test_logprob_length_mismatch_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    turn = make_turn()
    turn.capture.behavior_logprobs = [-0.1, -0.2]  # one short
    arch.record_turn(turn)
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("behaviour logprobs" in p for p in problems)


def test_cap_termination_is_refused_as_a_completed_action(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0, finish_reason="length"))
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("fragment, not a completed action" in p for p in problems)


def test_missing_semantic_history_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    turn = make_turn()
    turn.semantic_history = []
    arch.record_turn(turn)
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("teacher cannot be given Qwen prompt ids" in p for p in problems)


def test_a_gap_in_turn_indices_is_refused(tmp_path):
    """A missing generation, however it went missing. Its behaviour logprobs
    cannot be recreated, and the turns after it condition on a state the
    archive cannot describe."""
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0))
    arch.record_turn(make_turn("ep1", 2))  # turn 1 vanished
    ep.num_turns = 2
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    assert any("[0, 2] != [0, 1]" in p for p in arch.verify_episode("ep1"))



# -- batch gating ----------------------------------------------------------


def test_clean_batch_is_ok(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    complete_episode(arch, "ep2")
    rep = arch.batch_report(["ep1", "ep2"])
    assert rep["ok"] is True
    assert rep["trainable"] == 2
    assert rep["trainable_turns"] == 4


def test_batch_spanning_policy_versions_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1", policy_version="v0")
    complete_episode(arch, "ep2", policy_version="v1")
    rep = arch.batch_report(["ep1", "ep2"])
    assert rep["ok"] is False
    assert any("policy versions" in p for p in rep["problems"])


def test_batch_spanning_adapters_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1", adapter_hash="a" * 8)
    complete_episode(arch, "ep2", adapter_hash="b" * 8)
    rep = arch.batch_report(["ep1", "ep2"])
    assert any("adapter hashes" in p for p in rep["problems"])


def test_batch_spanning_generation_configs_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1", gen_config_hash="g" * 8)
    complete_episode(arch, "ep2", gen_config_hash="h" * 8)
    rep = arch.batch_report(["ep1", "ep2"])
    assert any("generation configs" in p for p in rep["problems"])


def test_exact_planned_episode_set_is_required(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    complete_episode(arch, "ep2")
    rep = arch.batch_report(
        ["ep1"], expected_episode_ids=["ep1", "ep2"]
    )
    assert rep["ok"] is False
    assert any("batch is short" in p for p in rep["problems"])


def test_expected_adapter_and_generation_config_are_required(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1", adapter_hash="old", gen_config_hash="oldcfg")
    rep = arch.batch_report(
        ["ep1"], adapter_hash="new", gen_config_hash="newcfg"
    )
    assert rep["ok"] is False
    assert any("planned adapter" in p for p in rep["problems"])
    assert any("planned config" in p for p in rep["problems"])


def test_a_discard_shrinks_the_batch_visibly(tmp_path):
    # The failure this exists to prevent: a discarded episode silently
    # dropping the batch from 8 to 7 while the run reports 8.
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    ep2 = make_episode("ep2")
    arch.record_episode(ep2)
    ep2.finish(EpisodeStatus.DISCARDED, reason="crash")
    arch.record_episode(ep2)
    rep = arch.batch_report(["ep1", "ep2"])
    assert rep["ok"] is False
    assert rep["requested"] == 2
    assert rep["trainable"] == 1
    assert rep["discarded"] == 1
    assert rep["discarded_episode_ids"] == ["ep2"]


def test_missing_episode_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    rep = arch.batch_report(["ep1", "nope"])
    assert rep["ok"] is False
    assert any("no episode record" in p for p in rep["problems"])


# -- failure archival ------------------------------------------------------


def test_classify_failure_buckets():
    assert classify_failure("turn 3 hit the 9216-token cap (finish_reason=length)") == "cap_termination"
    assert classify_failure("tool_call block 0 has no 'name': {}") == "malformed_tool_call"
    assert classify_failure("response carried no token ids. return_token_ids") == "no_token_ids"
    assert classify_failure("3 ids but 2 logprobs") == "logprob_mismatch"
    assert classify_failure("returned token ids do not reconstruct the returned text") == "byte_mismatch"
    assert classify_failure("response has no choices: {}") == "transport"


def test_build_failed_turn_salvages_a_paid_generation():
    body = {
        "choices": [
            {
                "text": "<tool_call>{broken",
                "finish_reason": "stop",
                "token_ids": [7, 8, 9],
                "logprobs": {"token_logprobs": [-0.5, -0.6, None]},
            }
        ],
        "usage": {"completion_tokens": 3},
    }
    ft = build_failed_turn(
        body=body, episode_id="ep1", task_id="57", turn_index=4,
        policy_version="v0", prompt_ids=[1, 2],
        semantic_history=[{"role": "user", "content": "hi"}],
        error=Exception("tool_call block 0 has no 'name'"),
    )
    assert ft.failure_kind == "malformed_tool_call"
    assert ft.sampled_token_ids == [7, 8, 9]
    assert ft.behavior_logprobs == [-0.5, -0.6]  # the null is dropped, not faked
    assert ft.semantic_history  # diagnosis needs the state it acted in
    assert ft.usage == {"completion_tokens": 3}


def test_build_failed_turn_survives_a_body_with_no_shape_at_all():
    # This runs precisely when the response was not what the parser expected,
    # so it must not raise and destroy the record explaining that.
    ft = build_failed_turn(
        body={}, episode_id="ep1", task_id="57", turn_index=0,
        policy_version="v0", prompt_ids=[], semantic_history=[],
        error=Exception("response has no choices: {}"),
    )
    assert ft.failure_kind == "transport"
    assert ft.sampled_token_ids == []


def test_failed_turns_are_readable_and_ordered(tmp_path):
    arch = EpisodeArchive(tmp_path)
    for i in (2, 0):
        arch.record_failure(
            FailedTurn(
                episode_id="ep1", task_id="57", turn_index=i,
                policy_version="v0", raw_text="", finish_reason="length",
                failure_kind="cap_termination", failure_detail="cap",
            )
        )
    failures = arch.load_failures("ep1")
    assert [f["turn_index"] for f in failures] == [0, 2]
    assert arch.load_turns("ep1") == []  # failures never appear as turns


# -- observations arrive after the capture ---------------------------------


def test_capture_is_persisted_before_the_observation_exists(tmp_path):
    """The environment has not run when the turn is written. Delaying that
    write to wait for an observation risks the one thing that cannot be
    recreated: the behaviour logprobs."""
    arch = EpisodeArchive(tmp_path)
    turn = make_turn()
    turn.observation, turn.env_state_hash = None, None
    arch.record_turn(turn)
    rows = arch.load_turns("ep1")
    assert rows[0]["capture"]["behavior_logprobs"] == [-0.1, -0.1, -0.1]
    assert rows[0]["observation"] is None


def test_observation_merges_onto_its_turn_without_rewriting_it(tmp_path):
    arch = EpisodeArchive(tmp_path)
    turn = make_turn()
    turn.observation, turn.env_state_hash = None, None
    arch.record_turn(turn)
    arch.record_observation("ep1", 0, {"role": "tool", "content": "ok"}, "cafe")

    rows = arch.load_turns("ep1")
    assert len(rows) == 1
    assert rows[0]["observation"] == {"role": "tool", "content": "ok"}
    assert rows[0]["env_state_hash"] == "cafe"
    assert rows[0]["observation_hash"]
    # Append-only: the original capture line is still the first record.
    raw = [json.loads(l) for l in arch.turns_path("ep1").read_text().splitlines()]
    assert [r["kind"] for r in raw] == ["turn", "turn_observed"]


def test_a_later_observation_wins(tmp_path):
    arch = EpisodeArchive(tmp_path)
    arch.record_turn(make_turn())
    arch.record_observation("ep1", 0, {"v": 1}, "h1")
    arch.record_observation("ep1", 0, {"v": 2}, "h2")
    row = arch.load_turns("ep1")[0]
    assert row["observation"] == {"v": 2}
    assert row["env_state_hash"] == "h2"


def test_a_null_observation_is_representable(tmp_path):
    # tau2's `get_db_hash()` returns None when the domain has no database.
    arch = EpisodeArchive(tmp_path)
    arch.record_turn(make_turn())
    arch.record_observation("ep1", 0, None, None)
    row = arch.load_turns("ep1")[0]
    assert row["observation"] is None and row["observation_hash"] is None


def test_an_observation_with_no_capture_is_reported(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0))
    arch.record_observation("ep1", 4, {"x": 1}, "h")  # no such capture
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    assert any("no archived capture" in p for p in arch.verify_episode("ep1"))


# -- declared counts must match what is on disk ----------------------------


def test_a_declared_turn_count_that_overstates_the_archive_is_refused(tmp_path):
    """A batch one turn smaller than the run reports changes the global
    denominator invisibly."""
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0))
    ep.num_turns = 5  # claims five, archived one
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")
    assert any("declares 5 turns but 1 are archived" in p for p in problems)


def test_a_declared_failure_count_mismatch_is_refused(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0))
    ep.num_turns, ep.num_failed_turns = 1, 3
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    assert any("declares 3 failed turns" in p for p in arch.verify_episode("ep1"))


# -- failed vs discarded stay separate in a batch report -------------------


def test_batch_report_separates_failures_from_discards(tmp_path):
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    for eid, status in (("ep2", EpisodeStatus.FAILED),
                        ("ep3", EpisodeStatus.DISCARDED)):
        ep = make_episode(eid)
        arch.record_episode(ep)
        ep.finish(status, reason="x")
        arch.record_episode(ep)

    rep = arch.batch_report(["ep1", "ep2", "ep3"])
    assert rep["ok"] is False
    assert rep["requested"] == 3 and rep["trainable"] == 1
    assert rep["failed_episode_ids"] == ["ep2"]
    assert rep["discarded_episode_ids"] == ["ep3"]


# -- batch composition hazards ---------------------------------------------


def test_the_same_episode_listed_twice_is_refused(tmp_path):
    """A batch of 8 that is really 4 episodes listed twice would double every
    one of their turns in the denominator while halving actual diversity --
    and every other check would pass."""
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1")
    rep = arch.batch_report(["ep1", "ep1"])
    assert rep["ok"] is False
    assert rep["duplicates"] == ["ep1"]
    assert rep["requested"] == 1          # deduplicated, not double counted
    assert rep["trainable_turns"] == 2


def test_an_empty_batch_is_refused(tmp_path):
    """Not vacuously fine: it would step the optimizer over zero tokens."""
    arch = EpisodeArchive(tmp_path)
    rep = arch.batch_report([])
    assert rep["ok"] is False
    assert any("empty batch" in p for p in rep["problems"])


def test_a_batch_sampled_under_the_previous_policy_is_refused(tmp_path):
    """Self-consistent but stale: every episode agrees with every other, and
    all of them predate the reload."""
    arch = EpisodeArchive(tmp_path)
    complete_episode(arch, "ep1", policy_version="v0")
    complete_episode(arch, "ep2", policy_version="v0")
    assert arch.batch_report(["ep1", "ep2"], policy_version="v0")["ok"] is True
    rep = arch.batch_report(["ep1", "ep2"], policy_version="v1")
    assert rep["ok"] is False
    assert any("must be sampled from the newly reloaded" in p
               for p in rep["problems"])


# -- episode-record integrity ----------------------------------------------


def test_a_sampled_episode_with_no_termination_reason_is_refused(tmp_path):
    """It is what separates a finished conversation from a loop that hit
    max_steps; a batch cannot report validity without it."""
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    arch.record_episode(ep)
    arch.record_turn(make_turn("ep1", 0))
    ep.num_turns = 1
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    assert any("no termination_reason" in p for p in arch.verify_episode("ep1"))


def test_verify_does_not_crash_on_an_episode_with_no_turns_file(tmp_path):
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()
    ep.termination_reason = "agent_stop"
    arch.record_episode(ep)
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    problems = arch.verify_episode("ep1")   # must report, not raise
    assert any("no parsed turns" in p for p in problems)


def test_an_episode_record_without_an_id_is_refused(tmp_path):
    from vektori_trace.tau2.reopd_state import append_jsonl

    arch = EpisodeArchive(tmp_path)
    append_jsonl(arch.episodes_path, {"kind": "episode", "status": "sampled"})
    with pytest.raises(ValueError, match="no 'episode_id'"):
        arch.load_episodes()


def test_a_turn_mislabelled_with_another_task_is_refused(tmp_path):
    """`LivePrefix.task` is read off the turn and `max_task_share` limits how
    much of a batch one task may supply, so a mislabelled turn skews exactly
    the balance that check enforces."""
    arch = EpisodeArchive(tmp_path)
    ep = make_episode()                      # task 57
    arch.record_episode(ep)
    turn = make_turn("ep1", 0)
    turn.capture.task_id = "99"
    arch.record_turn(turn)
    ep.num_turns = 1
    ep.termination_reason = "user_stop"
    ep.finish(EpisodeStatus.SAMPLED)
    arch.record_episode(ep)
    assert any("per-task batch shares" in p for p in arch.verify_episode("ep1"))
