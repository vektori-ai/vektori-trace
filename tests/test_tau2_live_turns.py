"""Live turns entering the existing ReOPD backend.

The point of these tests is that a live update reaches `score_replay_batch`
and `run_replay_chunk_opd` in exactly the shapes a replay update does -- and
that the episode-level rules replay does not need are enforced before it gets
there.
"""

from __future__ import annotations

import pytest

from vektori_trace.tau2.live_episode import semantic_hash
from vektori_trace.tau2.live_turns import (
    LivePrefix,
    LiveTurnError,
    live_prefixes,
    live_score_fingerprint,
    score_row_provenance,
    capture_row_from_turn as _capture_row_from_turn,
    flatten_live_turns as _flatten_live_turns,
    live_turn_key,
    stale_score_keys,
    teacher_context_hash,
    verify_prompt_parity,
)
from vektori_trace.tau2.reopd_sample import capture_to_sampled_action
from tests.test_tau2_live_episode import make_turn


TEACHER_CONTEXT = teacher_context_hash({
    "model": "deepseek-v4-flash",
    "tokenizer": "encoding_dsv4",
    "renderer": "chat-v1",
})


def capture_row_from_turn(turn):
    return _capture_row_from_turn(turn, teacher_context=TEACHER_CONTEXT)


def flatten_live_turns(turns, *, policy_version):
    return _flatten_live_turns(
        turns, policy_version=policy_version, teacher_context=TEACHER_CONTEXT
    )


def episode(episode_id="ep1", n=2, policy_version="v0"):
    return [
        make_turn(episode_id, i, policy_version=policy_version).to_json()
        for i in range(n)
    ]


# -- translation into the existing row format ------------------------------


def test_capture_row_is_readable_by_the_existing_replay_converter():
    """The whole reuse argument rests on this: no forked converter."""
    row = capture_row_from_turn(make_turn().to_json())
    action = capture_to_sampled_action(row)
    assert action.prefix_id == live_turn_key("ep1", 0)
    assert action.action_token_ids == [10, 11, 12]
    assert action.action_bytes == b"abcdef"
    assert action.behavior_logprobs == [-0.1, -0.1, -0.1]
    # Required by the optimizer: log pi_current must be recomputed under the
    # same conditioning log pi_old was captured in.
    assert action.prompt_token_ids == [1, 2, 3]
    assert action.policy_version == "v0"


def test_action_key_matches_the_existing_convention():
    row = capture_row_from_turn(make_turn("ep7", 3).to_json())
    action = capture_to_sampled_action(row)
    assert row["key"] == "ep7@3#0"
    assert f"{action.prefix_id}#{action.sample_index}" == row["key"]


def test_utf8_split_across_tokens_survives_hex_to_base64():
    turn = make_turn()
    turn.capture.action_token_bytes = [b"\xe2\x9c", b"\x93"]
    turn.capture.sampled_token_ids = [10, 11]
    turn.capture.behavior_logprobs = [-0.1, -0.2]
    action = capture_to_sampled_action(capture_row_from_turn(turn.to_json()))
    assert action.action_bytes.decode("utf-8") == "✓"


# -- flattening ------------------------------------------------------------


def test_flatten_produces_the_two_arguments_the_scorer_takes():
    rows, rendered = flatten_live_turns(
        {"ep1": episode("ep1", 2), "ep2": episode("ep2", 3)}, policy_version="v0"
    )
    assert len(rows) == 5
    assert set(rendered) == {r["prefix_id"] for r in rows}
    # `rendered` is prefix_id -> canonical messages, exactly as the replay
    # driver builds it from `p.canonical_messages`.
    assert rendered[live_turn_key("ep1", 0)] == [{"role": "user", "content": "hi"}]


def test_flatten_orders_deterministically():
    a, _ = flatten_live_turns({"ep2": episode("ep2"), "ep1": episode("ep1")},
                              policy_version="v0")
    b, _ = flatten_live_turns({"ep1": episode("ep1"), "ep2": episode("ep2")},
                              policy_version="v0")
    assert [r["key"] for r in a] == [r["key"] for r in b]


def test_a_turn_from_another_policy_version_is_refused():
    turns = episode("ep1", 2)
    turns[1]["capture"]["policy_version"] = "v1"
    with pytest.raises(LiveTurnError, match="fixed"):
        flatten_live_turns({"ep1": turns}, policy_version="v0")


def test_whole_batch_must_match_the_update_policy_version():
    with pytest.raises(LiveTurnError, match="this update is"):
        flatten_live_turns({"ep1": episode("ep1", 2, "v0")}, policy_version="v1")


def test_a_turn_without_semantic_history_is_refused():
    turns = episode("ep1", 2)
    turns[0]["semantic_history"] = []
    with pytest.raises(LiveTurnError, match="not a teacher context"):
        flatten_live_turns({"ep1": turns}, policy_version="v0")


def test_a_gap_in_turn_indices_is_refused():
    turns = [make_turn("ep1", i).to_json() for i in (0, 2)]
    with pytest.raises(LiveTurnError, match="not contiguous"):
        flatten_live_turns({"ep1": turns}, policy_version="v0")


def test_an_empty_episode_is_refused():
    with pytest.raises(LiveTurnError, match="no turns"):
        flatten_live_turns({"ep1": []}, policy_version="v0")


def test_a_repeated_turn_index_within_an_episode_is_refused():
    # Caught by the contiguity rule -- [0, 0] is not range(2).
    with pytest.raises(LiveTurnError, match="not contiguous"):
        flatten_live_turns({"ep1": episode("ep1", 1) * 2}, policy_version="v0")


def test_a_turn_claiming_another_episodes_id_is_refused():
    # The duplicate guard, which contiguity cannot catch: two episodes whose
    # turns carry the same episode_id collide on one key, and one would
    # silently overwrite the other in `rendered`.
    stray = make_turn("ep1", 0).to_json()
    with pytest.raises(LiveTurnError, match="collides with a real turn key"):
        flatten_live_turns({"ep1": episode("ep1", 1), "ep2": [stray]},
                           policy_version="v0")


# -- the stale-score gap the byte check does not close ---------------------


def test_a_score_bought_under_a_different_history_is_dropped():
    """`score_replay_batch` reuses a cached score when its bytes reconstruct
    the action. For a frozen prefix that is sufficient. A live turn key can be
    reused after a discard-and-resample, and the same action bytes can follow
    a different history -- reusing that score trains on a number bought for a
    trajectory that no longer exists, with every assertion still passing."""
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    paid = {rows[0]["key"]: {"semantic_history_hash": semantic_hash(
        [{"role": "user", "content": "a different conversation"}])}}
    assert stale_score_keys(rows, paid) == [rows[0]["key"]]


def test_a_score_bought_under_the_same_history_is_kept():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    paid = {rows[0]["key"]: {
        "semantic_history_hash": rows[0]["semantic_history_hash"]}}
    assert stale_score_keys(rows, paid) == []


def test_a_live_score_row_without_provenance_fails_closed():
    """Replay rows legitimately lack the hash -- a frozen prefix id names a
    fixed history. A key in the live namespace without provenance cannot be
    shown to belong to this trajectory, so it is rescored, not reused."""
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    assert stale_score_keys(rows, {rows[0]["key"]: {"teacher_logprobs": []}}) == [
        rows[0]["key"]
    ]


def test_scores_for_keys_not_in_this_batch_are_ignored():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    assert stale_score_keys(rows, {"ep9@0#0": {"semantic_history_hash": "x"}}) == []


# -- the prefix objects the training path reads ----------------------------


def test_live_prefixes_carry_the_four_fields_the_training_path_reads():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 2)}, policy_version="v0")
    prefixes = live_prefixes(rows)
    assert [p.prefix_id for p in prefixes] == [r["prefix_id"] for r in rows]
    assert [p.task for p in prefixes] == ["57", "57"]
    assert [p.trace_id for p in prefixes] == ["ep1", "ep1"]
    assert [p.step_index for p in prefixes] == [0, 1]


def test_live_prefix_id_matches_the_replay_convention():
    """`ReplayPrefix.prefix_id` is f"{trace_id}@{step_index}"; the live key
    convention is the same, so the two arguments line up."""
    from vektori_trace.replay_select import ReplayPrefix

    live = LivePrefix(task="57", trace_id="ep1", step_index=3)
    replay = ReplayPrefix(task="57", trace_id="ep1", step_index=3,
                          prefix_turns=[])
    assert live.prefix_id == replay.prefix_id == "ep1@3"


def test_prefixes_and_rows_stay_aligned_across_episodes():
    rows, rendered = flatten_live_turns(
        {"ep1": episode("ep1", 2), "ep2": episode("ep2", 1)}, policy_version="v0"
    )
    prefixes = live_prefixes(rows)
    assert len(prefixes) == len(rows) == 3
    # Every prefix has a rendered teacher context, which is what the scorer
    # looks up by prefix_id.
    assert all(p.prefix_id in rendered for p in prefixes)


def test_one_episode_is_one_trace_which_the_replay_share_limit_rejects():
    """A live batch of few episodes concentrates every supervised token in a
    handful of traces. `max_trace_share=0.35` would reject it, so a live update
    must derive its share limit from the episode count rather than inherit the
    replay default."""
    from collections import Counter

    rows, _ = flatten_live_turns({"ep1": episode("ep1", 4)}, policy_version="v0")
    shares = Counter(p.trace_id for p in live_prefixes(rows))
    assert max(shares.values()) / len(rows) == 1.0 > 0.35


# -- provenance a live score row must carry --------------------------------


def test_score_row_provenance_pins_the_history_the_score_was_bought_for():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")

    class Own:
        key = rows[0]["key"]

    prov = score_row_provenance(Own(), rows[0])
    assert prov["semantic_history_hash"] == rows[0]["semantic_history_hash"]
    assert prov["policy_version"] == "v0"
    assert (prov["episode_id"], prov["turn_index"]) == ("ep1", 0)
    # A row persisted with this is exactly what stale_score_keys accepts.
    assert stale_score_keys(rows, {rows[0]["key"]: prov}) == []


def test_flattening_no_episodes_is_refused():
    """An empty batch would take an optimizer step over zero supervised
    tokens."""
    with pytest.raises(LiveTurnError, match="no episodes to flatten"):
        flatten_live_turns({}, policy_version="v0")


def test_provenance_refuses_a_score_from_another_turn():
    """Attaching episode A's history to episode B's score would forge exactly
    the provenance `stale_score_keys` exists to check."""
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")

    class Foreign:
        key = "ep9@0#0"

    with pytest.raises(LiveTurnError, match="does not belong to capture row"):
        score_row_provenance(Foreign(), rows[0])


def test_provenance_refuses_an_object_with_no_key():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    with pytest.raises(LiveTurnError, match="does not belong"):
        score_row_provenance(object(), rows[0])


def test_provenance_accepts_its_own_score():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")

    class Own:
        key = rows[0]["key"]

    assert score_row_provenance(Own(), rows[0])["episode_id"] == "ep1"


# -- binding into RunState.validate() --------------------------------------


def test_action_row_carries_the_fingerprint_runstate_compares():
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")
    assert rows[0]["score_fingerprint"] == live_score_fingerprint(rows[0])


def test_teacher_context_changes_the_score_fingerprint():
    turn = make_turn().to_json()
    a = _capture_row_from_turn(turn, teacher_context="teacher-a")
    b = _capture_row_from_turn(turn, teacher_context="teacher-b")
    assert a["score_fingerprint"] != b["score_fingerprint"]


def test_archived_history_must_reproduce_captured_prompt_ids():
    turns = {"ep1": episode("ep1", 1)}
    assert verify_prompt_parity(
        turns, lambda _messages: [1, 2, 3]
    ) == {"checked": 1, "ok": True}
    with pytest.raises(LiveTurnError, match="does not reproduce"):
        verify_prompt_parity(turns, lambda _messages: [9, 9])


def test_provenance_fingerprint_matches_the_action_rows():
    """The two halves `RunState.validate()` compares: the action row's
    `score_fingerprint` and the score row's `fingerprint`."""
    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")

    class Own:
        key = rows[0]["key"]

    assert score_row_provenance(Own(), rows[0])["fingerprint"] == (
        rows[0]["score_fingerprint"]
    )


def test_runstate_validate_rejects_a_score_bought_for_another_history(tmp_path):
    """End to end through the real validator: two turns sharing a key, an
    action and a policy version, differing only in the conversation they
    followed. `capture_fingerprint` would not separate them."""
    from vektori_trace.tau2.reopd_state import ReOPDStateError, RunState, append_jsonl

    turn = make_turn("ep1", 0)
    rows_a, _ = flatten_live_turns({"ep1": [turn.to_json()]}, policy_version="v0")

    other = make_turn("ep1", 0)
    other.semantic_history = [{"role": "user", "content": "a different call"}]
    rows_b, _ = flatten_live_turns({"ep1": [other.to_json()]}, policy_version="v0")

    assert rows_a[0]["key"] == rows_b[0]["key"]
    assert rows_a[0]["score_fingerprint"] != rows_b[0]["score_fingerprint"]

    run = RunState(tmp_path / "run", n_updates=1)
    u = run.update(0)
    u.path.mkdir(parents=True, exist_ok=True)
    append_jsonl(u.actions_path, rows_a[0])
    # A score bought against the *other* history, reaching this update on resume.
    append_jsonl(u.scores_path, {"key": rows_b[0]["key"],
                                 "fingerprint": rows_b[0]["score_fingerprint"]})
    u.mark("PLANNED", {}); u.mark("SAMPLED", {}); u.mark("SCORED", {})
    with pytest.raises(ReOPDStateError, match="bought for a different action"):
        u.validate()


def test_runstate_validate_accepts_a_matching_live_score(tmp_path):
    from vektori_trace.tau2.reopd_state import RunState, append_jsonl

    rows, _ = flatten_live_turns({"ep1": episode("ep1", 1)}, policy_version="v0")

    class Own:
        key = rows[0]["key"]

    run = RunState(tmp_path / "run", n_updates=1)
    u = run.update(0)
    u.path.mkdir(parents=True, exist_ok=True)
    append_jsonl(u.actions_path, rows[0])
    append_jsonl(u.scores_path,
                 {"key": rows[0]["key"], **score_row_provenance(Own(), rows[0])})
    u.mark("PLANNED", {}); u.mark("SAMPLED", {}); u.mark("SCORED", {})
    u.validate()  # must not raise
