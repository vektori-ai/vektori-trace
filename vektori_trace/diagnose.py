"""Capability-deficit diagnosis: label each trace against candidate capabilities,
then score how much each capability's absence actually separates wins from losses."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .llm import call_json
from .schema import Trace

Label = str  # "NA" | "PRESENT" | "LACKING"


@dataclass
class Capability:
    id: str
    name: str
    description: str


@dataclass
class TraceLabels:
    trace: Trace
    labels: dict[str, Label]  # capability id -> label
    evidence: dict[str, str]  # capability id -> one-line evidence


@dataclass
class DeficitScore:
    capability: Capability
    baseline_rate: float | None
    incident_rate: float | None
    gap: float | None
    prevalence: float
    priority: float
    n_relevant_wins: int = 0
    n_relevant_losses: int = 0
    lacking_loss_traces: list[Trace] = field(default_factory=list)
    # run_id -> one-line labeller evidence for this capability on that loss.
    # Persisted into diagnosis.json so A1 can template a prompt without
    # re-labelling (previously thrown away after scoring).
    lacking_loss_evidence: dict[str, str] = field(default_factory=dict)


# Uncalibrated. Our labeller is a blurry ruler and blur shrinks effects toward
# zero, so a true gap of 0.5 can surface here as 0.18. Step 5 of the v0 plan
# hand-labels ~50 traces to measure the blur and set this from data; until then
# we do not know whether 0.20 is strict or loose.
DEFAULT_MIN_GAP = 0.20
# A gap computed from two traces is not a gap. This floor is deliberately low —
# it exists to reject the degenerate case, not to establish significance.
DEFAULT_MIN_SUPPORT = 3
# Exact McNemar on n discordant pairs can never go below 2 * 0.5**n, so under 6
# pairs nothing reaches p<0.05 whatever the data; under 9, only a *perfectly*
# one-sided split does (n=8 with one pair the other way gives p=0.07). The v0
# plan's "~8" is that floor: below it the test has essentially no power.
# Reported as an advisory, never a gate — "the test couldn't have detected this"
# is a different statement from "there is no difference".
MIN_DISCORDANT_PAIRS = 8


_CAPABILITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["capabilities"],
    "additionalProperties": False,
}


def _sample_evenly(traces: list[Trace], k: int) -> list[Trace]:
    """Up to k traces, evenly spaced. Deterministic, and because manifests tend
    to arrive grouped by outcome, a stride covers both groups where taking the
    first k would show the proposer only wins."""
    if len(traces) <= k:
        return list(traces)
    stride = len(traces) / k
    return [traces[int(i * stride)] for i in range(k)]


def propose_capabilities(
    traces: list[Trace],
    model: str | None = None,
    *,
    max_traces: int = 20,
    max_turns_per_trace: int = 24,
    max_field_chars: int = 240,
) -> list[Capability]:
    """One LLM call: read trajectory excerpts, propose 4-8 candidate capabilities.

    Blind to outcome, like the labeller. Told which runs failed, the model
    proposes capabilities that describe the *failures it was shown* rather than
    the task distribution — the separation then comes out of the prompt instead
    of the data, and every gap downstream is measured against a rigged list.

    It sees actual turns, not one-line summaries. Given only the opening
    request and a turn count there is nothing behavioural in the prompt to
    reason about, and the model does the only thing the input supports: it
    groups traces by subject matter and proposes task domains — "Cloud Object
    Storage Upload", "Release Branch Push". Those aren't capabilities, they
    can't be lacking, and the planted-deficit self-test recovers nothing
    against them. Capabilities are behaviours, so proposing them requires
    seeing behaviour.
    """
    sample = _sample_evenly(traces, max_traces)
    excerpts = "\n\n".join(
        f"--- {t.run_id} ({len(t.turns)} turns) ---\n"
        + t.condensed(max_turns=max_turns_per_trace, max_field_chars=max_field_chars)
        for t in sample
    )
    omitted = len(traces) - len(sample)
    header = f"Here are {len(sample)} agent trajectories"
    header += f" (sampled from {len(traces)}):" if omitted else ":"
    user = (
        f"{header}\n\n{excerpts}\n\n"
        "Propose 4-8 distinct agent capabilities on which these trajectories "
        "differ or could differ — concrete behaviours the agent either exercises "
        "competently or fails to.\n\n"
        "A capability is something an agent DOES, not a topic it does it to. "
        "'Recovers from a failed tool call by changing its arguments' is a "
        "capability. 'Cloud storage uploads', 'database queries', 'deployment' "
        "are task domains, not capabilities — do not propose those. The test: if "
        "you cannot sensibly say an agent LACKS it in one trajectory and "
        "DEMONSTRATES it in another, it is not a capability.\n\n"
        "Each id should be a short snake_case slug."
    )
    result = call_json(
        system=(
            "You are an ML engineer diagnosing capability deficits in an LLM agent "
            "from its tool-use trajectories."
        ),
        user=user,
        schema_name="capabilities",
        json_schema=_CAPABILITIES_SCHEMA,
        model=model,
    )
    return [Capability(**c) for c in result["capabilities"]]


def _label_schema(capability_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "capability_id": {"type": "string", "enum": capability_ids},
                        "label": {"type": "string", "enum": ["NA", "PRESENT", "LACKING"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["capability_id", "label", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def label_trace(trace: Trace, capabilities: list[Capability], model: str | None = None) -> TraceLabels:
    """One LLM call per trace: for each capability, label NA / PRESENT / LACKING.

    The label must come from the trajectory alone. Handed the outcome, the
    model reasons backwards from it — a trace it is told is a loss gets its
    capabilities marked LACKING to justify the ending — and the resulting gap
    measures the prompt, not the agent."""
    cap_list = "\n".join(f"- {c.id}: {c.name} — {c.description}" for c in capabilities)
    user = (
        f"Trajectory:\n\n{trace.condensed()}\n\n"
        f"Capabilities to assess:\n{cap_list}\n\n"
        "For each capability: NA if it wasn't relevant/needed for this trajectory; "
        "PRESENT if it was needed and the agent demonstrated it; LACKING if it was "
        "needed but the agent failed to exercise it competently. One capability_id "
        "entry per capability listed above."
    )
    result = call_json(
        system="You are labeling whether an agent exercised specific capabilities in a trajectory.",
        user=user,
        schema_name="trace_labels",
        json_schema=_label_schema([c.id for c in capabilities]),
        model=model,
    )
    labels = {r["capability_id"]: r["label"] for r in result["labels"]}
    evidence = {r["capability_id"]: r["evidence"] for r in result["labels"]}
    return TraceLabels(trace=trace, labels=labels, evidence=evidence)


def score_deficits(
    capabilities: list[Capability], trace_labels: list[TraceLabels]
) -> list[DeficitScore]:
    """Score every candidate capability and return them ranked, highest priority first."""
    wins = [tl for tl in trace_labels if tl.trace.outcome == "win"]
    losses = [tl for tl in trace_labels if tl.trace.outcome == "loss"]
    scores = []
    for cap in capabilities:
        # Default NA, not PRESENT. A capability the labeller simply didn't
        # return a verdict on is unmeasured; counting it as relevant-and-not-
        # LACKING reads silence as demonstrated competence, which pushes the
        # baseline rate down and manufactures a gap out of missing data.
        relevant_wins = [tl for tl in wins if tl.labels.get(cap.id, "NA") != "NA"]
        relevant_losses = [tl for tl in losses if tl.labels.get(cap.id, "NA") != "NA"]
        lacking_wins = [tl for tl in relevant_wins if tl.labels.get(cap.id) == "LACKING"]
        lacking_losses = [tl for tl in relevant_losses if tl.labels.get(cap.id) == "LACKING"]

        baseline_rate = len(lacking_wins) / len(relevant_wins) if relevant_wins else None
        incident_rate = len(lacking_losses) / len(relevant_losses) if relevant_losses else None
        gap = (
            (incident_rate - baseline_rate)
            if (baseline_rate is not None and incident_rate is not None)
            else None
        )
        prevalence = (len(lacking_losses) / len(losses)) if losses else 0.0
        priority = (gap or 0.0) * prevalence

        scores.append(
            DeficitScore(
                capability=cap,
                baseline_rate=baseline_rate,
                incident_rate=incident_rate,
                gap=gap,
                prevalence=prevalence,
                priority=priority,
                n_relevant_wins=len(relevant_wins),
                n_relevant_losses=len(relevant_losses),
                lacking_loss_traces=[tl.trace for tl in lacking_losses],
                lacking_loss_evidence={
                    tl.trace.run_id: tl.evidence.get(cap.id, "")
                    for tl in lacking_losses
                    if tl.evidence.get(cap.id)
                },
            )
        )
    return sorted(scores, key=lambda s: s.priority, reverse=True)


def select_deficit(
    scores: list[DeficitScore],
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> DeficitScore | None:
    """The top-ranked capability that actually clears the bar, or None.

    "No deficit found" is a real answer and has to be reachable. Taking
    scores[0] unconditionally means an inverted result (gap = -1.0: the
    capability was *more* present in losses) still comes back as a confident
    diagnosis, and a capability seen in two traces outranks one seen in forty.
    """
    for s in scores:
        if s.gap is None or s.gap < min_gap:
            continue
        if min(s.n_relevant_wins, s.n_relevant_losses) < min_support:
            continue
        return s
    return None


def cross_model_trace_labels(
    trace_labels: list[TraceLabels], frontier_model: str, candidate_model: str
) -> list[TraceLabels]:
    """The frontier's wins and the candidate's losses, and nothing else.

    `score_deficits` splits whatever it's given on `outcome`, so pre-filtering
    the input this way makes its own win/loss split *be* the cross-model
    contrast — "what's worth fixing" — with no change to its scoring at all.
    A replay manifest fed in unfiltered mixes both models into one win/loss set
    and averages exactly the contrast away.
    """
    return [
        tl
        for tl in trace_labels
        if (tl.trace.model == frontier_model and tl.trace.outcome == "win")
        or (tl.trace.model == candidate_model and tl.trace.outcome == "loss")
    ]


def within_model_trace_labels(trace_labels: list[TraceLabels], model: str) -> list[TraceLabels]:
    """One model's traces only, so `score_deficits`'s split becomes that model's
    wins vs its own losses — "is there anything here to train from"."""
    return [tl for tl in trace_labels if tl.trace.model == model]


@dataclass
class McNemarResult:
    capability_id: str
    frontier_only: int  # b: frontier PRESENT, candidate not — same task
    candidate_only: int  # c: candidate PRESENT, frontier not — same task
    concordant: int  # both PRESENT or neither — same task
    discordant_n: int  # b + c: the pairs the test is actually computed from
    p_value: float | None  # None when there are no discordant pairs to test

    @property
    def underpowered(self) -> bool:
        return self.discordant_n < MIN_DISCORDANT_PAIRS


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar: under H0 each discordant pair is a coin flip, so
    the count is Binomial(n, 0.5) and the p-value is twice the smaller tail.

    Exact rather than the chi-square approximation because the discordant counts
    here are single digits, which is precisely where the approximation is wrong.
    Pure stdlib — this project has no scipy/numpy dependency and needs none.
    """
    n = b + c
    if n == 0:
        raise ValueError("no discordant pairs — there is nothing to test")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return min(1.0, 2 * tail)


def mcnemar_test(
    trace_labels: list[TraceLabels],
    capability_id: str,
    frontier_model: str,
    candidate_model: str,
) -> McNemarResult:
    """Compare the two models on the SAME task, not on average.

    Frontier wins come from easier tasks and candidate losses from harder ones,
    so an averaged rate mixes capability with difficulty. Pairing by task (the
    same key `gap.compute_gap` pairs on) cancels difficulty: each pair asks
    whether one model demonstrated the capability where the other didn't.

    A pair is dropped when either side is NA or unlabelled — not relevant on one
    side means the two sides aren't comparable, and counting that as agreement
    would inflate `concordant` with pairs that were never measured. Tasks only
    one model has a trace for are dropped for the same reason.
    """
    frontier = {
        tl.trace.task: tl
        for tl in trace_labels
        if tl.trace.model == frontier_model and tl.trace.task is not None
    }
    candidate = {
        tl.trace.task: tl
        for tl in trace_labels
        if tl.trace.model == candidate_model and tl.trace.task is not None
    }

    b = c = concordant = 0
    for task in sorted(frontier.keys() & candidate.keys()):
        f_label = frontier[task].labels.get(capability_id, "NA")
        c_label = candidate[task].labels.get(capability_id, "NA")
        if f_label == "NA" or c_label == "NA":
            continue
        f_present = f_label == "PRESENT"
        c_present = c_label == "PRESENT"
        if f_present == c_present:
            concordant += 1
        elif f_present:
            b += 1
        else:
            c += 1

    return McNemarResult(
        capability_id=capability_id,
        frontier_only=b,
        candidate_only=c,
        concordant=concordant,
        discordant_n=b + c,
        p_value=_exact_mcnemar_p(b, c) if b + c else None,
    )


@dataclass
class ReplayDiagnosis:
    frontier_model: str
    candidate_model: str
    cross_model_scores: list[DeficitScore]  # ranked, same shape as the plain path
    chosen: DeficitScore | None  # top cross-model deficit clearing the thresholds
    within_model_score: DeficitScore | None  # the same capability, candidate-only
    mcnemar: McNemarResult | None
    trainable: bool | None  # None exactly when `chosen` is None


def diagnose_replay(
    trace_labels: list[TraceLabels],
    capabilities: list[Capability],
    frontier_model: str,
    candidate_model: str,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> ReplayDiagnosis:
    """The two contrasts out of one replay run: cross-model says what is worth
    fixing, within-model says whether there is anything to train from.

    A cross-model deficit with no within-model signal is "identified, not
    trainable": rejection sampling keeps only rollouts that pass, so if the
    candidate has never once demonstrated the capability there is nothing for it
    to keep, and generating a task against that deficit produces an empty
    dataset rather than a bad one.
    """
    cross = score_deficits(
        capabilities, cross_model_trace_labels(trace_labels, frontier_model, candidate_model)
    )
    chosen = select_deficit(cross, min_gap=min_gap, min_support=min_support)
    if chosen is None:
        return ReplayDiagnosis(
            frontier_model=frontier_model,
            candidate_model=candidate_model,
            cross_model_scores=cross,
            chosen=None,
            within_model_score=None,
            mcnemar=None,
            trainable=None,
        )

    within = score_deficits(capabilities, within_model_trace_labels(trace_labels, candidate_model))
    within_score = next(s for s in within if s.capability.id == chosen.capability.id)
    # `baseline_rate` is the LACKING rate among the candidate's own relevant
    # wins, so < 1.0 means it did demonstrate the capability at least once —
    # and `n_relevant_wins` says on how many traces that verdict rests.
    trainable = (
        within_score.baseline_rate is not None
        and within_score.baseline_rate < 1.0
        and within_score.n_relevant_wins >= min_support
    )
    return ReplayDiagnosis(
        frontier_model=frontier_model,
        candidate_model=candidate_model,
        cross_model_scores=cross,
        chosen=chosen,
        within_model_score=within_score,
        mcnemar=mcnemar_test(trace_labels, chosen.capability.id, frontier_model, candidate_model),
        trainable=trainable,
    )


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"
