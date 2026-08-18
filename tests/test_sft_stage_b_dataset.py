"""Stage B's slicer: the rules step 8 states, each as a refusal or a count.

The expensive failures here are silent ones — a recovery quietly dropped, a
Stage A action quietly reused, a cold share quietly at zero. Every one of those
is a build that looks fine and trains the wrong thing, so each gets a test that
fails loudly on CPU rather than a comment saying it should not happen.
"""

from __future__ import annotations

import collections
import json
import sys

import pytest

from scripts import sft_stage_b_dataset as sb

ACTION = '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": %s, "duration": 0.1}]}'
READ = ACTION % json.dumps("ls -la\n")
EDIT = ACTION % json.dumps("sed -i s/a/b/ f.py\n")
TEST = ACTION % json.dumps("pytest -x\n")


def _segment(task, seg, actions, *, parse_error_at=None, rollout=0):
    """A (row, manifest) pair with `actions` supervised assistant turns."""
    messages = [{"role": "user", "content": "spec"}]
    meta = [{"kind": "prompt"}]
    supervise = [False]
    for i, content in enumerate(actions):
        if parse_error_at is not None and i == parse_error_at:
            messages.append({"role": "assistant", "content": "not json"})
            meta.append({"kind": "parse_error"})
            supervise.append(False)
            messages.append({"role": "user", "content": "parse error"})
            meta.append({"kind": "parse_error_reply"})
            supervise.append(False)
        messages.append({"role": "assistant", "content": content})
        meta.append({"kind": "action"})
        supervise.append(True)
        messages.append({"role": "user", "content": f"obs {i}"})
        meta.append({"kind": "observation"})
        supervise.append(False)
    row = {"messages": messages, "supervise": supervise}
    man = {"task": task, "segment_index": seg, "turn_meta": meta,
           "rollout_index": rollout}
    return row, man


def _first_ids(rows, manifest):
    """The keys Stage A would have taken: each segment's first action."""
    out = set()
    for row, man in zip(rows, manifest, strict=True):
        first = next(i for i, s in enumerate(row["supervise"]) if s)
        out.add(sb.row_key(
            man["task"], man.get("rollout_index"), man["segment_index"], first
        ))
    return out


# --------------------------------------------------------------------------
# Stage A's action is never reused
# --------------------------------------------------------------------------


def test_stage_a_first_action_is_never_reused():
    row, man = _segment("pallets__click-1", 0, [READ, EDIT, TEST, READ, READ])
    excl = _first_ids([row], [man])
    out = sb.slice_rows([row], [man], exclude=excl)
    assert out, "expected later rows"
    assert not ({r["source_id"] for r in out} & excl)


def test_exclusion_is_by_source_id_not_by_content():
    """Two segments of the same task can hold byte-identical actions. Excluding
    on rendered text would drop the wrong one; excluding on task|seg|msg cannot."""
    a = _segment("pallets__click-1", 0, [READ, READ, READ])
    b = _segment("pallets__click-1", 1, [READ, READ, READ])
    rows, mans = [a[0], b[0]], [a[1], b[1]]
    out = sb.slice_rows(rows, mans, exclude=_first_ids(rows, mans))
    segs = {(r["segment_index"], r["source_id"]) for r in out}
    assert {s for s, _ in segs} == {0, 1}


# --------------------------------------------------------------------------
# The <=4 rule, and the read-only case
# --------------------------------------------------------------------------


def test_at_most_four_later_actions_per_segment():
    row, man = _segment("pallets__click-1", 0, [READ] * 3 + [EDIT, TEST] + [READ] * 8)
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    assert len(out) <= sb.MAX_LATER_PER_SEGMENT


def test_edit_and_test_and_last_are_all_taken():
    row, man = _segment("pallets__click-1", 0, [READ, READ, EDIT, READ, TEST, READ])
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    kinds = {r["kind"] for r in out}
    assert sb.LATER_FIRST_EDIT in kinds
    assert sb.LATER_FIRST_TEST in kinds
    assert sb.LATER_LAST in kinds
    # the last action really is the segment's final one
    last = next(r for r in out if r["kind"] == sb.LATER_LAST)
    assert last["messages"][-1]["content"] == READ
    assert last["turn_ordinal"] == 5


def test_a_read_only_segment_takes_four_evenly_spaced_including_last():
    row, man = _segment("pallets__click-1", 0, [READ] * 9)
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    assert [r["kind"] for r in out] == [sb.LATER_READ_ONLY] * 4
    ordinals = [r["turn_ordinal"] for r in out]
    assert ordinals[-1] == 8, "the last action must always be drawn"
    assert len(set(ordinals)) == 4


def test_evenly_spaced_always_includes_the_last_even_when_it_is_odd():
    """The even-*numbered* reading of step 8 is unsatisfiable here; the
    evenly-*spaced* reading is what the builder implements."""
    picked = sb.evenly_spaced([1, 2, 3, 4, 5, 6, 7], 4)
    assert len(picked) == 4
    assert picked[-1] == 7


def test_a_short_read_only_segment_takes_what_it_has():
    row, man = _segment("pallets__click-1", 0, [READ, READ])
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    assert len(out) == 1
    assert out[0]["turn_ordinal"] == 1


# --------------------------------------------------------------------------
# The 18 recoveries
# --------------------------------------------------------------------------


def test_a_recovery_is_taken_and_labelled_as_one():
    row, man = _segment("pallets__click-1", 0, [READ, READ, EDIT], parse_error_at=2)
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    kinds = [r["kind"] for r in out]
    assert sb.PARSE_ERROR_RECOVERY in kinds
    rec = next(r for r in out if r["kind"] == sb.PARSE_ERROR_RECOVERY)
    assert rec["messages"][-1]["content"] == EDIT


def test_two_rejections_in_a_row_are_one_recovery_row():
    messages = [{"role": "user", "content": "spec"}]
    meta = [{"kind": "prompt"}]
    supervise = [False]
    for _ in range(2):
        messages.append({"role": "assistant", "content": "not json"})
        meta.append({"kind": "parse_error"})
        supervise.append(False)
        messages.append({"role": "user", "content": "parse error"})
        meta.append({"kind": "parse_error_reply"})
        supervise.append(False)
    messages.append({"role": "assistant", "content": READ})
    meta.append({"kind": "action"})
    supervise.append(True)
    assert sb.recovery_targets(meta, supervise) == [len(messages) - 1]


def test_the_recovery_label_wins_over_a_later_kind():
    """A recovery that is also the segment's first edit must still count as a
    recovery, or the exact-18 check silently goes short."""
    row, man = _segment("pallets__click-1", 0, [READ, EDIT, READ], parse_error_at=1)
    out = sb.slice_rows([row], [man], exclude=_first_ids([row], [man]))
    edits = [r for r in out if r["messages"][-1]["content"] == EDIT]
    assert edits and all(r["kind"] == sb.PARSE_ERROR_RECOVERY for r in edits)


def test_a_short_recovery_count_fails_the_build():
    rep = {
        "pallets_mass": 0.4,
        "by_repo": {"anyio": {"mass": 0.25, "upsample": 1.0},
                    "hatch": {"mass": 0.15, "upsample": 1.0},
                    "prefect": {"mass": 0.15, "upsample": 1.0}},
        "by_kind": {sb.LATER_LAST: 600, sb.PARSE_ERROR_RECOVERY: 17,
                    sb.COLD_REPLAY: 165},
        "cold": {"token_share": 0.30},
    }
    bad = sb.check_mix(rep)
    assert any("17 rows, expected exactly 18" in b for b in bad)

    rep["by_kind"][sb.PARSE_ERROR_RECOVERY] = 18
    assert not [b for b in sb.check_mix(rep) if "parse_error_recovery" in b]


# --------------------------------------------------------------------------
# The cold-token floor
# --------------------------------------------------------------------------


def test_the_cold_token_floor_refuses_a_short_mix():
    rep = {
        "pallets_mass": 0.4,
        "by_repo": {"anyio": {"mass": 0.25, "upsample": 1.0},
                    "hatch": {"mass": 0.15, "upsample": 1.0},
                    "prefect": {"mass": 0.15, "upsample": 1.0}},
        "by_kind": {sb.LATER_LAST: 600, sb.PARSE_ERROR_RECOVERY: 18},
        "cold": {"token_share": 0.19},
    }
    assert any("cold supervised-token share" in b for b in sb.check_mix(rep))

    rep["cold"]["token_share"] = sb.COLD_TOKEN_FLOOR
    assert not [b for b in sb.check_mix(rep) if "cold supervised-token" in b]


def test_cold_mass_is_solved_for_the_token_share_not_the_row_share():
    """Later actions carry ~1.5x the supervised tokens of a first action, so a
    mass chosen to look like 30% of rows lands well under 30% of tokens."""
    cold_per_draw, new_per_draw = 187.0, 286.0
    m = sb.solve_cold_mass(cold_per_draw, new_per_draw, 0.30)
    share = m * cold_per_draw / (m * cold_per_draw + (1 - m) * new_per_draw)
    assert share == pytest.approx(0.30, abs=1e-9)
    assert m > 0.30, "the cold component must outweigh its token target in mass"


def test_no_cold_component_means_a_zero_share_and_a_failed_build():
    rep = {
        "pallets_mass": 0.4,
        "by_repo": {"anyio": {"mass": 0.25, "upsample": 1.0},
                    "hatch": {"mass": 0.15, "upsample": 1.0},
                    "prefect": {"mass": 0.15, "upsample": 1.0}},
        "by_kind": {sb.LATER_LAST: 600, sb.PARSE_ERROR_RECOVERY: 18},
        "cold": {"token_share": 0.0},
    }
    assert any("cold supervised-token share is 0.0%" in b for b in sb.check_mix(rep))


# --------------------------------------------------------------------------
# Source pin
# --------------------------------------------------------------------------


def test_a_source_sha_mismatch_refuses_to_slice(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "sft_repaired.jsonl").write_text('{"messages": [], "supervise": []}\n')
    (src / "repair_manifest.jsonl").write_text("")
    monkeypatch.setattr(
        sys, "argv",
        ["sft_stage_b_dataset.py", "--src", str(src), "--stage-a", str(tmp_path),
         "--out", str(tmp_path / "out")],
    )
    assert sb.main() == 2
    assert "refusing to slice" in capsys.readouterr().err


def test_the_expected_pin_is_the_repaired_corpus():
    assert sb.EXPECTED_SRC_SHA == "7ecfee31"
    assert sb.MAX_LENGTH == 40960
    assert sb.TEMPLATE_KWARGS == {"enable_thinking": True}


# --------------------------------------------------------------------------
# Row identity. Stage A's `source_id` omits the rollout and collides.
# --------------------------------------------------------------------------


def test_the_same_task_and_segment_in_two_rollouts_are_two_rows():
    """Stage A's `task|segN|msgI` collapsed its 165 rows onto 60 keys, so every
    sampler weight and per-row token count landed on one entry per collision."""
    a = _segment("pallets__click-1", 0, [READ, READ, READ], rollout=0)
    b = _segment("pallets__click-1", 0, [READ, READ, READ], rollout=1)
    rows, mans = [a[0], b[0]], [a[1], b[1]]
    out = sb.slice_rows(rows, mans, exclude=_first_ids(rows, mans))
    ids = [r["source_id"] for r in out]
    assert len(ids) == len(set(ids)), "row_key must separate rollouts"
    assert {r["rollout_index"] for r in out} == {0, 1}


def test_row_key_separates_every_coordinate():
    base = sb.row_key("t", 0, 0, 1)
    assert base != sb.row_key("t", 1, 0, 1)
    assert base != sb.row_key("t", 0, 1, 1)
    assert base != sb.row_key("t", 0, 0, 3)
    assert base != sb.row_key("u", 0, 0, 1)


def test_stage_a_key_reconstructs_the_rollout_aware_key():
    row = {
        "source_id": "pallets__click-1|seg2|msg5",
        "task": "pallets__click-1",
        "segment_index": 2,
        "rollout_index": 3,
    }
    assert sb.stage_a_key(row) == sb.row_key("pallets__click-1", 3, 2, 5)


def test_exclusion_only_removes_the_matching_rollout():
    """Keyed on the colliding id, excluding rollout 0's action at msg 3 would
    also have removed rollout 1's action at msg 3 — a different conversation.

    Each segment's first action is dropped structurally (`supervised[1:]`), so
    what the exclusion set governs is the *later* pool, and that is what this
    pins.
    """
    a = _segment("pallets__click-1", 0, [READ, READ, READ], rollout=0)
    b = _segment("pallets__click-1", 0, [READ, READ, READ], rollout=1)
    rows, mans = [a[0], b[0]], [a[1], b[1]]
    later_of_r0 = {sb.row_key("pallets__click-1", 0, 0, 3)}
    out = sb.slice_rows(rows, mans, exclude=later_of_r0)
    per_rollout = collections.Counter(r["rollout_index"] for r in out)
    assert per_rollout[0] == 1, "rollout 0 loses the excluded later action"
    assert per_rollout[1] == 2, "rollout 1 keeps both of its later actions"
    assert sb.row_key("pallets__click-1", 1, 0, 3) in {
        r["source_id"] for r in out
    }
