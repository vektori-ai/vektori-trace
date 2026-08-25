"""Split invariants. These are the experiment's contamination boundary."""
import pytest

from vektori_trace.tau2.split import (
    CONTAMINATED, SplitInvariantError, TaskMeta, assert_invariants, build_split,
)

FAMS = ["readonly_lookup", "cancel", "modify", "return", "exchange", "modify+return__multi"]
DIFFS = ["easy", "med", "hard", "proxy_short", "proxy_long"]


def metas(n=114, n_eligible=72):
    out = {}
    for i in range(n):
        t = str(i)
        out[t] = TaskMeta(
            task_id=t, family=FAMS[i % len(FAMS)], difficulty=DIFFS[i % len(DIFFS)],
            eligible=False, has_mutation=(i % 3 != 0), has_trace=(i % 4 != 0),
        )
    # make a deterministic subset eligible, always including the contaminated ids
    picked = 0
    for i in range(n):
        t = str(i)
        if picked < n_eligible:
            out[t].eligible = True
            picked += 1
    return out


def test_split_is_exactly_30_30_16_38():
    s = build_split(metas())
    assert (len(s.W30), len(s.C30), len(s.S16), len(s.F38)) == (30, 30, 16, 38)
    assert sum(len(v) for v in s.as_dict().values()) == 114


def test_partitions_are_disjoint():
    s = build_split(metas())
    ids = s.W30 + s.C30 + s.S16 + s.F38
    assert len(ids) == len(set(ids))


def test_w30_and_c30_never_overlap():
    s = build_split(metas())
    assert not (set(s.W30) & set(s.C30))


def test_contaminated_diagnostics_land_in_s16():
    s = build_split(metas())
    for t in CONTAMINATED:
        assert t in s.S16, f"{t} escaped S16"
        assert t not in s.F38 and t not in s.W30 and t not in s.C30


def test_training_tasks_are_all_eligible():
    m = metas()
    s = build_split(m)
    assert all(m[t].eligible for t in s.W30 + s.C30)


def test_training_never_leaks_into_evaluation():
    s = build_split(metas())
    assert not ((set(s.W30) | set(s.C30)) & (set(s.S16) | set(s.F38)))


def test_same_seed_gives_identical_split():
    a, b = build_split(metas(), seed=7), build_split(metas(), seed=7)
    assert a.manifest_hash() == b.manifest_hash()
    assert a.as_dict() == b.as_dict()


def test_different_seed_gives_a_different_split():
    a, b = build_split(metas(), seed=1), build_split(metas(), seed=2)
    assert a.manifest_hash() != b.manifest_hash()


def test_too_few_eligible_tasks_fails_loudly():
    with pytest.raises(SplitInvariantError, match="eligible"):
        build_split(metas(n_eligible=40))


def test_wrong_task_count_fails():
    with pytest.raises(SplitInvariantError, match="exactly 114"):
        build_split(metas(n=100))


def test_invariant_checker_catches_a_planted_overlap():
    s = build_split(metas())
    s.C30[0] = s.W30[0]
    with pytest.raises(SplitInvariantError):
        assert_invariants(s, metas())


def test_invariant_checker_catches_contaminated_in_f38():
    m = metas()
    s = build_split(m)
    victim = CONTAMINATED[0]
    s.S16.remove(victim)
    s.S16.append(s.F38.pop())
    s.F38.append(victim)
    with pytest.raises(SplitInvariantError, match="contaminated"):
        assert_invariants(s, m)


def test_families_are_spread_not_clumped():
    m = metas()
    s = build_split(m)
    for part in (s.W30, s.C30, s.S16):
        fams = {m[t].family for t in part}
        assert len(fams) >= 3, f"only {fams} in a partition of {len(part)}"
