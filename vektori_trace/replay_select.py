"""Selecting replay prefixes from the stored DeepSeek corpus (plan §8.3).

`docs/OPD-MULTITURN-PLAN.md` §8 uses stored DeepSeek trajectories as *states*,
never as targets: a trace contributes a task, an authentic Harbor prefix, and
nothing else. ck75 samples the next action at that state and DeepSeek scores
ck75's action. The stored DeepSeek continuation must not appear in the loss.

This module picks which prefixes to use. Selection is not a detail — §8.4 makes
two of its properties hard pass conditions:

- *"every replay prefix corresponds to an actually observed trace state"* —
  so a prefix is identified by (trace, step) and reconstructed with
  `reopd.prefix_turns_through_step`, never by concatenating a whole trace and
  hoping;
- *"no task or single trace dominates the global supervised-token count"* —
  the previous OPD run put 74% of its examples on one task, and this is the
  guard against repeating that.

What it deliberately does **not** do: read `trajectory.json` files itself. The
repo already parses Harbor/ATIF trajectories into `schema.Turn`, and a second
parser would drift from the first.

Two selection policies, and which one a run used is not a detail
---------------------------------------------------------------
ReOPD (arXiv:2607.04763) samples replay prefixes under a decaying weight
`w(t, kappa) = kappa ** t`, kappa=0.6 by default, deliberately favouring *early*
trace steps because those carry the least distribution shift from the student's
own states. That is the paper's recipe, and `reopd_step_weights` implements it.

This first 32-action run does **not** use it. It uses `select_replay_prefixes`,
a stratified diagnostic sample spread across distinct tasks and trace stages,
including post-compaction states, because the run's purpose is to exercise the
mechanics — long prefixes, compaction boundaries, ragged alignment — rather than
to maximise learning signal. Under a kappa=0.6 schedule the late and
post-compaction states this run most wants to test are precisely the ones that
would almost never be drawn.

That is a deliberate deviation, not the ReOPD recipe, and a run report must say
which policy it used. Once mechanics are proven, a learning-oriented run should
switch to `reopd_step_weights`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .evaluate.resume import assistant_tool_steps
from .reopd import prefix_turns_through_step
from .schema import Turn


def _task_offset(task: str) -> int:
    """Stable per-task offset in [0, 3). `hash()` is randomised per process."""
    import hashlib

    return int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) % 3


class ReplaySelectionError(ValueError):
    """The chosen prefixes cannot support a defensible replay update."""


@dataclass(frozen=True)
class ReplayPrefix:
    """One replay state: a trace, a step inside it, and the turns before it.

    `trace_id` and `step_index` together are the reproducibility key §8.4
    requires. `prefix_turns` is derived from them, not stored independently, so
    the two cannot disagree.
    """

    task: str
    trace_id: str
    step_index: int
    prefix_turns: list[Turn]
    #: True when this state follows a recorded compaction boundary. §8.3 wants
    #: at least two such prefixes "if available" — they are the long-horizon
    #: cases most likely to break the rendering contract.
    post_compaction: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prefix_id(self) -> str:
        return f"{self.trace_id}@{self.step_index}"

    @property
    def n_prefix_turns(self) -> int:
        return len(self.prefix_turns)


def enumerate_prefixes(
    task: str,
    trace_id: str,
    turns: list[Turn],
    *,
    min_step: int = 1,
    max_step: int | None = None,
    compaction_steps: set[int] | None = None,
) -> list[ReplayPrefix]:
    """Every usable replay state inside one stored trajectory.

    `min_step` defaults to 1 rather than 0: at step 0 the prefix is just the
    system and task messages, which is a cold start rather than the
    "authentic long-horizon state" §8.1 says a trace contributes. Step 0 is
    reachable by passing `min_step=0` explicitly.

    A step whose prefix cannot be reconstructed is skipped rather than
    guessed — §8.4 forbids a "guessed concatenation of the whole trace".
    """
    steps = assistant_tool_steps(turns)
    if not steps:
        return []
    upper = len(steps) if max_step is None else min(max_step, len(steps))
    comp = compaction_steps or set()

    out: list[ReplayPrefix] = []
    for t in range(min_step, upper):
        try:
            prefix = prefix_turns_through_step(turns, t - 1) if t > 0 else []
        except (IndexError, ValueError, StopIteration):
            continue
        if t > 0 and not prefix:
            continue
        out.append(
            ReplayPrefix(
                task=task,
                trace_id=trace_id,
                step_index=t,
                prefix_turns=prefix,
                post_compaction=t in comp,
            )
        )
    return out


def select_replay_prefixes(
    candidates: list[ReplayPrefix],
    *,
    n_prefixes: int = 8,
    min_distinct_tasks: int | None = None,
    require_post_compaction: int = 2,
    max_per_task: int | None = None,
    max_per_trace: int = 1,
) -> list[ReplayPrefix]:
    """Pick §8.3's eight prefixes across distinct tasks and trace stages.

    Spread is the point, so the defaults are strict:

    - `max_per_trace=1` — one state per trajectory, because two states from the
      same trace share most of their history and are close to one sample;
    - `min_distinct_tasks` defaults to `n_prefixes` (all distinct), which is
      what "across distinct tasks" means; loosen it only when the corpus cannot
      supply that many;
    - `require_post_compaction=2` — §8.3's "at least two authentic
      post-compaction prefixes **if available**". Availability is checked
      against the candidate pool, so a corpus without them is not a failure,
      but silently dropping them when they exist would be.

    Selection is deterministic: candidates are ordered by task, then trace, then
    step, and post-compaction states are taken first. A reproducible batch
    matters more here than a random one, because §8.4 requires every example to
    be reproducible from its ids.
    """
    if n_prefixes <= 0:
        raise ReplaySelectionError(f"n_prefixes must be > 0, got {n_prefixes}")
    if not candidates:
        raise ReplaySelectionError("no candidate prefixes to select from")

    want_tasks = n_prefixes if min_distinct_tasks is None else min_distinct_tasks
    available_pc = [c for c in candidates if c.post_compaction]
    if require_post_compaction > len(available_pc):
        # Refuse rather than degrade. `min(require, available)` quietly asked
        # for zero when the pool had none, and the run then reported a
        # stratified batch with post-compaction coverage it never had. Nothing
        # downstream could tell the difference — the batch is well-formed
        # either way.
        #
        # As of 2026-08-21 the pool is *always* empty: `replay_corpus` never
        # derives boundary steps, and §15 records why a parser is not yet
        # justified. So this fires on the real corpus by design, and callers
        # that genuinely want no post-compaction coverage must say
        # `require_post_compaction=0` explicitly.
        raise ReplaySelectionError(
            f"asked for {require_post_compaction} post-compaction prefixes but "
            f"only {len(available_pc)} candidates are marked. Compaction "
            "boundaries are not currently derived from the corpus "
            "(docs/OPD-MULTITURN-PLAN.md §15), so this pool is empty by "
            "construction. Pass require_post_compaction=0 to run without that "
            "coverage — and do not report the batch as having it."
        )
    want_pc = require_post_compaction

    # Spread across trace *stages*, not just tasks. §8.3 asks for both, and
    # with 34 eligible tasks for 8 slots a task-first ordering fills every slot
    # from whichever stage sorts first — the real corpus produced eight step-1
    # prefixes, a kappa-decay batch by accident.
    #
    # So stage is assigned to the *slot*, not derived from the sort: slot k
    # wants stage k % 3, and each pass takes the best available candidate in
    # that stage. Deterministic, and it cannot collapse to one end.
    by_trace_len: dict[str, int] = {}
    for c in candidates:
        by_trace_len[c.trace_id] = max(by_trace_len.get(c.trace_id, 0), c.step_index)

    def _stage(c: ReplayPrefix) -> int:
        """0=early, 1=middle, 2=late within this candidate's own trace."""
        span = by_trace_len.get(c.trace_id, 0)
        if span <= 0:
            return 0
        frac = c.step_index / span
        return 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)

    ordered = sorted(
        candidates,
        key=lambda c: (not c.post_compaction, c.task, c.trace_id, c.step_index),
    )
    by_stage: dict[int, list[ReplayPrefix]] = {0: [], 1: [], 2: []}
    for c in ordered:
        by_stage[_stage(c)].append(c)

    chosen: list[ReplayPrefix] = []
    per_task: Counter[str] = Counter()
    per_trace: Counter[str] = Counter()
    pc_taken = 0

    def _fits(c: ReplayPrefix, *, allow_task_repeat: bool) -> bool:
        if per_trace[c.trace_id] >= max_per_trace:
            return False
        if max_per_task is not None and per_task[c.task] >= max_per_task:
            return False
        return allow_task_repeat or per_task[c.task] < 1

    # Pass 1: post-compaction quota first, then round-robin the stages so each
    # slot draws from early / middle / late in turn.
    def _take(pool, *, allow_repeat: bool, want_pc: bool) -> bool:
        nonlocal pc_taken
        for c in pool:
            if c in chosen:
                continue
            if want_pc and not c.post_compaction:
                continue
            if not _fits(c, allow_task_repeat=allow_repeat):
                continue
            chosen.append(c)
            per_task[c.task] += 1
            per_trace[c.trace_id] += 1
            if c.post_compaction:
                pc_taken += 1
            return True
        return False

    while pc_taken < want_pc and len(chosen) < n_prefixes:
        if not _take(ordered, allow_repeat=False, want_pc=True):
            break

    for allow_repeat in (False, True):
        slot = len(chosen)
        while len(chosen) < n_prefixes:
            target = slot % 3
            # Try the target stage, then the others, so a corpus without late
            # states still fills the batch rather than failing.
            if not any(
                _take(by_stage[(target + off) % 3], allow_repeat=allow_repeat,
                      want_pc=False)
                for off in (0, 1, 2)
            ):
                break
            slot += 1
        if len(chosen) >= n_prefixes:
            break

    if len(chosen) < n_prefixes:
        raise ReplaySelectionError(
            f"only {len(chosen)} prefixes satisfy the spread constraints "
            f"(wanted {n_prefixes}; max_per_trace={max_per_trace}, "
            f"max_per_task={max_per_task}). Widen the candidate pool rather than "
            "relaxing the constraints silently."
        )

    n_tasks = len({c.task for c in chosen})
    if n_tasks < want_tasks:
        raise ReplaySelectionError(
            f"selection covers {n_tasks} distinct tasks, wanted {want_tasks}. "
            "§8.3 asks for prefixes across distinct tasks; a batch concentrated "
            "on one task repeats the previous run's 74%-on-one-task failure."
        )
    if pc_taken < want_pc:
        raise ReplaySelectionError(
            f"only {pc_taken} post-compaction prefixes selected but {want_pc} were "
            "available — §8.3 asks for them when they exist"
        )
    return chosen


def assert_no_source_dominates(
    supervised_tokens_by_prefix: dict[str, int],
    prefixes: list[ReplayPrefix],
    *,
    max_task_share: float = 0.5,
    max_trace_share: float = 0.35,
) -> dict[str, Any]:
    """§8.4: no task or single trace dominates the supervised-token count.

    Checked on *token* counts rather than example counts, because that is what
    the loss is normalised by (§7.3): eight prefixes spread evenly across tasks
    still concentrate the gradient if one task's actions are ten times longer.

    Returns the computed shares so a run report can state them.
    """
    by_id = {p.prefix_id: p for p in prefixes}
    missing = set(supervised_tokens_by_prefix) - set(by_id)
    if missing:
        raise ReplaySelectionError(
            f"token counts reference unknown prefixes: {sorted(missing)[:4]}"
        )

    total = sum(supervised_tokens_by_prefix.values())
    if total <= 0:
        raise ReplaySelectionError("no supervised tokens in this batch")

    task_tokens: Counter[str] = Counter()
    trace_tokens: Counter[str] = Counter()
    for pid, n in supervised_tokens_by_prefix.items():
        p = by_id[pid]
        task_tokens[p.task] += n
        trace_tokens[p.trace_id] += n

    task_share = {k: v / total for k, v in task_tokens.items()}
    trace_share = {k: v / total for k, v in trace_tokens.items()}

    worst_task, worst_task_share = max(task_share.items(), key=lambda kv: kv[1])
    worst_trace, worst_trace_share = max(trace_share.items(), key=lambda kv: kv[1])

    if worst_task_share > max_task_share:
        raise ReplaySelectionError(
            f"task {worst_task!r} holds {worst_task_share:.1%} of supervised "
            f"tokens (limit {max_task_share:.0%}) — §8.4 forbids one task "
            "dominating the update"
        )
    if worst_trace_share > max_trace_share:
        raise ReplaySelectionError(
            f"trace {worst_trace!r} holds {worst_trace_share:.1%} of supervised "
            f"tokens (limit {max_trace_share:.0%}) — §8.4 forbids one trace "
            "dominating the update"
        )

    return {
        "total_supervised_tokens": total,
        "task_share": dict(sorted(task_share.items())),
        "trace_share": dict(sorted(trace_share.items())),
        "max_task_share": worst_task_share,
        "max_trace_share": worst_trace_share,
    }


#: ReOPD's default steepness (arXiv:2607.04763). kappa=1.0 is uniform.
REOPD_DEFAULT_KAPPA = 0.6


def reopd_step_weights(
    candidates: list[ReplayPrefix], *, kappa: float = REOPD_DEFAULT_KAPPA
) -> dict[str, float]:
    """ReOPD's `w(t, kappa) = kappa ** t` prefix weights, normalised to sum to 1.

    The paper's argument is that a stored teacher prefix is off-policy for the
    student, and the further into the trace it sits the further the state has
    drifted from anything the student would have reached itself. Early steps are
    therefore the reliable ones, and the schedule is a single steepness knob
    rather than a hard cutoff.

    Returned as `{prefix_id: weight}` for the caller to sample with — this
    module does not own the RNG, because a run has to be reproducible from its
    recorded seed and prefix ids.

    **This is the pooled form**, and that is a choice, not a detail. Two
    distributions are easy to conflate:

    - *uniform over trajectories, then decay within a trace* — every trace
      contributes equally regardless of length;
    - *pooled globally over all eligible prefixes, then decay* — what this
      function does: weights are normalised across the whole candidate list, so
      a long trace contributes more prefixes and carries more total mass before
      decay is even applied.

    Pooled sampling is therefore one of the ways a single long trace comes to
    dominate a batch, which §8.4 forbids — `assert_no_source_dominates` is the
    backstop, but the sampling policy is where it originates. A caller wanting
    the uniform-over-traces form must group candidates by `trace_id` and call
    this once per trace. Whichever is used belongs in the run manifest: two runs
    differing only here are not comparable.

    Calibrating kappa: 0.6 is the paper's default, and on our corpus it is very
    steep. Over a 25-step trace (indices 0..24) it puts 99.4% of the mass in the
    first ten steps, so states at step >= 10 are drawn about once per five
    32-action batches. kappa=0.95 over the same range leaves 44.5% at steps
    >= 10. Pick it from the corpus's measured length distribution rather than
    inheriting 0.6, and report the realized step histogram alongside the choice.

    Not used by `select_replay_prefixes`; see the module docstring for why this
    first run samples stratified instead, and record which policy was used.
    """
    if not 0.0 < kappa <= 1.0:
        raise ReplaySelectionError(
            f"kappa must be in (0, 1], got {kappa!r} (1.0 = uniform)"
        )
    if not candidates:
        raise ReplaySelectionError("no candidates to weight")

    raw = {c.prefix_id: kappa ** c.step_index for c in candidates}
    total = sum(raw.values())
    if total <= 0:
        raise ReplaySelectionError(
            f"all weights underflowed at kappa={kappa} — steps are too deep "
            "for this schedule; raise kappa or restrict max_step"
        )
    return {k: v / total for k, v in raw.items()}


__all__ = [
    "REOPD_DEFAULT_KAPPA",
    "ReplayPrefix",
    "ReplaySelectionError",
    "assert_no_source_dominates",
    "enumerate_prefixes",
    "reopd_step_weights",
    "select_replay_prefixes",
]
