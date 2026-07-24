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
    lacking_loss_traces: list[Trace] = field(default_factory=list)


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
    capabilities that plausibly separate wins from losses."""
    summaries = []
    for t in traces:
        first_user = next((turn.content for turn in t.turns if turn.role == "user"), "")
        summaries.append(
            f"- run {t.run_id} [{t.outcome}]: task={_truncate(first_user, 300)!r}, "
            f"{len(t.turns)} turns"
        )
    user = (
        "Here are summaries of agent trajectories, each labeled win (task succeeded) "
        "or loss (task failed):\n\n" + "\n".join(summaries) + "\n\n"
        "Propose 4-8 distinct agent capabilities (skills/behaviors, not vague traits) "
        "that could plausibly explain the difference between wins and losses here. "
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
    """One LLM call per trace: for each capability, label NA / PRESENT / LACKING."""
    cap_list = "\n".join(f"- {c.id}: {c.name} — {c.description}" for c in capabilities)
    user = (
        f"Trajectory (outcome: {trace.outcome}):\n\n{trace.condensed()}\n\n"
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
        relevant_wins = [tl for tl in wins if tl.labels.get(cap.id) != "NA"]
        relevant_losses = [tl for tl in losses if tl.labels.get(cap.id) != "NA"]
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
                lacking_loss_traces=[tl.trace for tl in lacking_losses],
            )
        )
    return sorted(scores, key=lambda s: s.priority, reverse=True)


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"
