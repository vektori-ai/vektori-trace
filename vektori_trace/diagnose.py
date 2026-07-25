"""Capability-deficit diagnosis: label each trace against candidate capabilities,
then score how much each capability's absence actually separates wins from losses."""

from __future__ import annotations

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


# Uncalibrated. Our labeller is a blurry ruler and blur shrinks effects toward
# zero, so a true gap of 0.5 can surface here as 0.18. Step 5 of the v0 plan
# hand-labels ~50 traces to measure the blur and set this from data; until then
# we do not know whether 0.20 is strict or loose.
DEFAULT_MIN_GAP = 0.20
# A gap computed from two traces is not a gap. This floor is deliberately low —
# it exists to reject the degenerate case, not to establish significance.
DEFAULT_MIN_SUPPORT = 3


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


def propose_capabilities(traces: list[Trace], model: str | None = None) -> list[Capability]:
    """One LLM call: read short summaries of every trace, propose 4-8 candidate
    capabilities the task distribution requires.

    Blind to outcome, like the labeller. Told which runs failed, the model
    proposes capabilities that describe the *failures it was shown* rather than
    the task distribution — the separation then comes out of the prompt instead
    of the data, and every gap downstream is measured against a rigged list."""
    summaries = []
    for t in traces:
        first_user = next((turn.content for turn in t.turns if turn.role == "user"), "")
        summaries.append(f"- run {t.run_id}: task={_truncate(first_user, 300)!r}, {len(t.turns)} turns")
    user = (
        "Here are summaries of agent trajectories:\n\n" + "\n".join(summaries) + "\n\n"
        "Propose 4-8 distinct agent capabilities (skills/behaviors, not vague traits) "
        "that these tasks require, and on which agents plausibly differ from one "
        "another. Each id should be a short snake_case slug."
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


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"
