"""Replay prefix selection (plan §8.3, §8.4).

Two of §8.4's pass conditions are properties of *selection*, not of the loss:
every prefix must correspond to an actually observed trace state, and no task or
single trace may dominate the supervised-token count. The previous OPD run put
74% of its examples on one task; these tests are the guard against repeating it.
"""

from __future__ import annotations

import pytest

from vektori_trace.replay_select import (
    ReplayPrefix,
    ReplaySelectionError,
    assert_no_source_dominates,
    enumerate_prefixes,
    select_replay_prefixes,
)
from vektori_trace.schema import ToolCall, Turn


def _trace(n_steps: int) -> list[Turn]:
    """A parent trajectory with `n_steps` assistant tool calls and results."""
    turns: list[Turn] = [Turn(index=0, role="system", content="you are an agent")]
    idx = 1
    for s in range(n_steps):
        turns.append(
            Turn(
                index=idx,
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id=f"c{s}", name="bash", args={"cmd": f"ls {s}"})
                ],
            )
        )
        idx += 1
        turns.append(
            Turn(index=idx, role="tool", content=f"output {s}", tool_call_id=f"c{s}")
        )
        idx += 1
    return turns


def _prefix(task: str, trace: str, step: int, *, pc: bool = False) -> ReplayPrefix:
    return ReplayPrefix(
        task=task, trace_id=trace, step_index=step, prefix_turns=[], post_compaction=pc
    )


# ---------------------------------------------------------------------------
# enumerate_prefixes — states come from the trace, never guessed
# ---------------------------------------------------------------------------


def test_enumerate_yields_one_prefix_per_reachable_step():
    turns = _trace(5)
    got = enumerate_prefixes("taskA", "trace-1", turns)

    assert got, "expected reachable replay states"
    assert all(p.task == "taskA" and p.trace_id == "trace-1" for p in got)
    assert [p.step_index for p in got] == sorted(p.step_index for p in got)
    # Step 0 is excluded by default: its prefix is a cold start, not the
    # long-horizon state a trace is supposed to contribute.
    assert all(p.step_index >= 1 for p in got)


def test_each_prefix_grows_with_its_step():
    got = enumerate_prefixes("taskA", "trace-1", _trace(5))
    lengths = [p.n_prefix_turns for p in got]
    assert lengths == sorted(lengths), "later steps must carry more history"
    assert all(n > 0 for n in lengths)


def test_prefix_id_is_the_reproducibility_key():
    p = enumerate_prefixes("taskA", "trace-1", _trace(3))[0]
    assert p.prefix_id == f"trace-1@{p.step_index}"


def test_trace_with_no_tool_steps_yields_nothing():
    turns = [Turn(index=0, role="system", content="hi")]
    assert enumerate_prefixes("t", "tr", turns) == []


def test_compaction_steps_are_marked():
    got = enumerate_prefixes("taskA", "trace-1", _trace(6), compaction_steps={2, 4})
    marked = {p.step_index for p in got if p.post_compaction}
    assert marked == {2, 4}


def test_max_step_bounds_enumeration():
    got = enumerate_prefixes("taskA", "trace-1", _trace(8), max_step=4)
    assert all(p.step_index < 4 for p in got)


# ---------------------------------------------------------------------------
# select_replay_prefixes — §8.3 spread
# ---------------------------------------------------------------------------


def _pool(n_tasks: int, per_task: int = 3, pc_on: set[str] | None = None):
    pc_on = pc_on or set()
    out = []
    for t in range(n_tasks):
        for k in range(per_task):
            trace = f"tr{t}-{k}"
            out.append(
                _prefix(f"task{t}", trace, step=2 + k, pc=trace in pc_on)
            )
    return out


def test_selects_eight_across_distinct_tasks():
    chosen = select_replay_prefixes(_pool(10), require_post_compaction=0)

    assert len(chosen) == 8
    assert len({c.task for c in chosen}) == 8, "one prefix per task"
    assert len({c.trace_id for c in chosen}) == 8


def test_one_prefix_per_trace_by_default():
    """Two states from one trace share most of their history."""
    chosen = select_replay_prefixes(_pool(10), require_post_compaction=0)
    traces = [c.trace_id for c in chosen]
    assert len(traces) == len(set(traces))


def test_post_compaction_prefixes_are_taken_when_available():
    """§8.3: at least two authentic post-compaction prefixes if available."""
    pool = _pool(10, pc_on={"tr0-0", "tr1-0", "tr2-0"})
    chosen = select_replay_prefixes(pool, require_post_compaction=2)

    assert sum(1 for c in chosen if c.post_compaction) >= 2


def test_no_post_compaction_available_is_not_a_failure():
    """"if available" — a corpus without them still selects."""
    chosen = select_replay_prefixes(_pool(10), require_post_compaction=2)
    assert len(chosen) == 8
    assert not any(c.post_compaction for c in chosen)


def test_too_few_distinct_tasks_is_refused():
    """A batch concentrated on one task repeats the previous run's failure."""
    pool = _pool(2, per_task=10)
    with pytest.raises(ReplaySelectionError, match="distinct tasks"):
        select_replay_prefixes(pool, require_post_compaction=0)


def test_too_small_a_pool_is_refused_rather_than_padded():
    with pytest.raises(ReplaySelectionError, match="satisfy the spread constraints"):
        select_replay_prefixes(_pool(3, per_task=1), require_post_compaction=0)


def test_empty_pool_is_refused():
    with pytest.raises(ReplaySelectionError, match="no candidate prefixes"):
        select_replay_prefixes([], require_post_compaction=0)


def test_selection_is_deterministic():
    pool = _pool(12)
    a = select_replay_prefixes(pool, require_post_compaction=0)
    b = select_replay_prefixes(list(reversed(pool)), require_post_compaction=0)
    assert [c.prefix_id for c in a] == [c.prefix_id for c in b]


def test_relaxing_distinct_tasks_allows_a_narrower_corpus():
    pool = _pool(4, per_task=4)
    chosen = select_replay_prefixes(
        pool, min_distinct_tasks=4, max_per_trace=1, require_post_compaction=0
    )
    assert len(chosen) == 8
    assert len({c.task for c in chosen}) == 4


# ---------------------------------------------------------------------------
# assert_no_source_dominates — §8.4, on tokens not examples
# ---------------------------------------------------------------------------


def test_even_spread_passes():
    prefixes = [_prefix(f"task{i}", f"tr{i}", 2) for i in range(8)]
    counts = {p.prefix_id: 100 for p in prefixes}

    rep = assert_no_source_dominates(counts, prefixes)

    assert rep["total_supervised_tokens"] == 800
    assert rep["max_task_share"] == pytest.approx(0.125)


def test_one_task_dominating_tokens_is_refused():
    """Eight prefixes evenly spread by count can still be one task by tokens."""
    prefixes = [_prefix("hot", f"tr{i}", 2) for i in range(4)] + [
        _prefix(f"task{i}", f"other{i}", 2) for i in range(4)
    ]
    counts = {p.prefix_id: (1000 if p.task == "hot" else 10) for p in prefixes}

    with pytest.raises(ReplaySelectionError, match="of supervised tokens"):
        # Trace cap relaxed so this asserts the *task* limit specifically.
        assert_no_source_dominates(counts, prefixes, max_trace_share=1.0)


def test_one_trace_dominating_tokens_is_refused():
    """Two traces under one task: the task share is fine, the trace share is not."""
    prefixes = [_prefix("shared", f"tr{i}", 2) for i in range(2)]
    counts = {prefixes[0].prefix_id: 900, prefixes[1].prefix_id: 100}

    with pytest.raises(ReplaySelectionError, match=r"trace .* holds"):
        # Task cap relaxed so this asserts the *trace* limit specifically.
        assert_no_source_dominates(counts, prefixes, max_task_share=1.0)


def test_share_is_measured_on_tokens_not_example_count():
    """The loss is normalised by tokens (§7.3), so shares must be too."""
    prefixes = [_prefix("a", "tr0", 2), _prefix("b", "tr1", 2)]
    counts = {"tr0@2": 60, "tr1@2": 40}

    rep = assert_no_source_dominates(
        counts, prefixes, max_task_share=0.7, max_trace_share=0.7
    )

    assert rep["task_share"]["a"] == pytest.approx(0.6)
    assert rep["task_share"]["b"] == pytest.approx(0.4)


def test_unknown_prefix_in_counts_is_refused():
    prefixes = [_prefix("a", "tr0", 2)]
    with pytest.raises(ReplaySelectionError, match="unknown prefixes"):
        assert_no_source_dominates({"ghost@1": 10}, prefixes)


def test_zero_tokens_is_refused():
    prefixes = [_prefix("a", "tr0", 2)]
    with pytest.raises(ReplaySelectionError, match="no supervised tokens"):
        assert_no_source_dominates({"tr0@2": 0}, prefixes)


def test_selection_spreads_across_trace_stages():
    """§8.3 asks for distinct tasks AND trace stages.

    Sorting by step_index alone made the one-per-task rule pick every task's
    earliest state — a kappa-decay batch by accident, skipping the long-horizon
    states a diagnostic run exists to stress.
    """
    # The real corpus's shape, which a uniform fixture does not reproduce: many
    # more eligible tasks than slots, several traces per task, and trace lengths
    # spanning 9..238 states. Under a task-first ordering this produced eight
    # step-1 prefixes.
    import random

    rng = random.Random(0)
    cands = []
    for t in range(34):
        for r in range(3):
            n = rng.choice([9, 20, 37, 60, 120, 238])
            for step in range(1, n + 1):
                cands.append(_prefix(f"task{t}", f"tr{t}_{r}", step))

    chosen = select_replay_prefixes(cands, require_post_compaction=0, max_per_trace=1)

    steps = sorted(c.step_index for c in chosen)
    assert len(chosen) == 8
    assert len({c.task for c in chosen}) == 8
    assert len({c.trace_id for c in chosen}) == 8
    assert max(steps) > 20, f"no late-trace state selected: {steps}"
    assert min(steps) < 20, f"no early-trace state selected: {steps}"


def test_stage_spread_is_reproducible_across_processes():
    """Uses a stable digest, not hash(), which is randomised per process."""
    from vektori_trace.replay_select import _task_offset

    assert _task_offset("pallets__click-3152") == _task_offset("pallets__click-3152")
    assert 0 <= _task_offset("anything") < 3
