"""Plant a capability deficit in synthetic traces and check the ranker finds it.

This is the cheapest experiment that can kill the idea. Everything downstream —
task selection, training, the A3-vs-A2 comparison — assumes the diagnosis
identifies a real capability. If the ranker cannot recover a deficit *we put
there ourselves*, on data with no noise beyond the LLM's own, none of it
matters. So it runs early and on synthetic data, where ground truth is known
per trace rather than argued about.

What makes this a test rather than a demo:

  * **Distractor failure modes.** If every loss failed for the planted reason,
    any capability that correlates with failure "wins" and recovery is
    trivially guaranteed. Some losses here fail for unrelated reasons, so the
    ranker has to pick the right capability out of several real competitors.
  * **Controlled confounds.** Wins and losses are drawn from the same scenario
    pool, and losing traces reason about their situation just as much as
    winning ones — they just don't engage with the error text. Without that,
    the ranker can separate the two on "has a thinking block" and look right
    for the wrong reason.
  * **Wins that never exercise the capability.** Not every success involves a
    tool error, so the capability is genuinely NA on some traces and the
    relevant-trace denominators get exercised rather than assumed.
  * **Per-trace ground truth.** We know what every label *should* be, so the
    labeller's blur is measured directly instead of inferred from the ranking.

The corpus is seeded and written to disk as ordinary trace JSON + a manifest,
so the recovery run goes through exactly the code path a real diagnosis takes.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .diagnose import (
    DEFAULT_MIN_GAP,
    DEFAULT_MIN_SUPPORT,
    Capability,
    DeficitScore,
    TraceLabels,
    label_trace,
    propose_capabilities,
    score_deficits,
    select_deficit,
)
from .schema import Trace, load_manifest

# ---------------------------------------------------------------------------
# The planted capability
# ---------------------------------------------------------------------------

PLANTED_NAME = "Reads a failed tool call's error and adjusts the arguments"
PLANTED_DESCRIPTION = (
    "When a tool call fails with an informative error, the agent reads the error "
    "text and issues a corrected call, instead of repeating the identical call."
)

# Matching an LLM-authored capability to the one we planted is itself a blurry
# ruler, so the verdict is three-way rather than boolean. STRICT requires the
# model to have named both halves of the idea (the error signal AND doing
# something with it); NEAR accepts naming only the error signal — "error
# handling" is probably the right concept, vaguely stated, and calling that a
# miss would understate the ranker just as calling it a hit would overstate it.
# Both rates are reported; neither is the number on its own.
_ERROR_TERMS = (
    "error",
    "failure",
    "failed",
    "failing",
    "traceback",
    "exception",
    "diagnostic",
    "feedback",
    "stderr",
    "message",
)
_ACT_TERMS = (
    "adjust",
    "adapt",
    "correct",
    "revise",
    "modify",
    "amend",
    "act on",
    "acts on",
    "respond",
    "incorporate",
    "interpret",
    "parse",
    "read",
    "reads",
    "recover",
    "self-correct",
    "iterate",
)
_RETRY_TERMS = ("retry", "retries", "repeat", "verbatim", "unchanged", "identical", "same call")

MATCH_STRICT = "strict"
MATCH_NEAR = "near"
MATCH_NONE = "none"


def match_planted(cap: Capability) -> str:
    """How well an LLM-proposed capability corresponds to the planted one."""
    text = f"{cap.id} {cap.name} {cap.description}".lower().replace("_", " ")
    has_error = any(t in text for t in _ERROR_TERMS)
    has_act = any(t in text for t in _ACT_TERMS)
    has_retry = any(t in text for t in _RETRY_TERMS)
    if has_error and (has_act or has_retry):
        return MATCH_STRICT
    if has_error or has_retry:
        return MATCH_NEAR
    return MATCH_NONE


# ---------------------------------------------------------------------------
# Scenario pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    task: str
    tool: str
    bad_args: dict[str, Any]
    error: str
    fixed_args: dict[str, Any]
    success: str
    # What a competent agent notices in the error text.
    insight: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        task="Upload report.csv to the 'analytics' bucket.",
        tool="s3_upload",
        bad_args={"bucket": "analytics", "key": "report.csv"},
        error='{"error": "AccessDenied: bucket policy requires key prefix \'incoming/\'"}',
        fixed_args={"bucket": "analytics", "key": "incoming/report.csv"},
        success='{"status": "ok", "etag": "d41d8cd9"}',
        insight="the bucket policy requires an 'incoming/' key prefix",
    ),
    Scenario(
        task="Create a user record for jane@example.com via the accounts API.",
        tool="http_post",
        bad_args={"url": "/v2/accounts", "body": {"email": "jane@example.com"}},
        error='{"status": 422, "error": "missing required field: display_name"}',
        fixed_args={
            "url": "/v2/accounts",
            "body": {"email": "jane@example.com", "display_name": "jane"},
        },
        success='{"status": 201, "id": "acct_8812"}',
        insight="display_name is a required field on this endpoint",
    ),
    Scenario(
        task="Report how many orders were placed yesterday.",
        tool="sql_query",
        bad_args={"sql": "SELECT count(*) FROM orders WHERE created = current_date - 1"},
        error='{"error": "column \\"created\\" does not exist; did you mean \\"created_at\\"?"}',
        fixed_args={"sql": "SELECT count(*) FROM orders WHERE created_at = current_date - 1"},
        success='{"rows": [[412]]}',
        insight="the column is named created_at, not created",
    ),
    Scenario(
        task="Save the parsed output to build/out/summary.json.",
        tool="write_file",
        bad_args={"path": "build/out/summary.json", "content": "{}"},
        error='{"error": "ENOENT: no such file or directory, open \'build/out/summary.json\'"}',
        fixed_args={"path": "build/out/summary.json", "content": "{}", "mkdirs": True},
        success='{"status": "written", "bytes": 2}',
        insight="the parent directory does not exist yet",
    ),
    Scenario(
        task="Install the pinned analytics dependency.",
        tool="shell",
        bad_args={"cmd": "pip install pandas==2.2.0"},
        error=(
            '{"exit_code": 1, "stderr": "ERROR: Cannot install pandas==2.2.0 because '
            'numpy==1.21.0 is pinned; pandas 2.2.0 requires numpy>=1.26"}'
        ),
        fixed_args={"cmd": "pip install 'numpy>=1.26' pandas==2.2.0"},
        success='{"exit_code": 0, "stdout": "Successfully installed pandas-2.2.0"}',
        insight="numpy is pinned too low for this pandas version",
    ),
    Scenario(
        task="Push the release branch to origin.",
        tool="shell",
        bad_args={"cmd": "git push origin release"},
        error=(
            '{"exit_code": 1, "stderr": "! [rejected] release -> release (non-fast-forward); '
            'fetch and integrate remote changes first"}'
        ),
        fixed_args={"cmd": "git pull --rebase origin release && git push origin release"},
        success='{"exit_code": 0, "stdout": "release -> release"}',
        insight="the remote has commits that need integrating first",
    ),
    Scenario(
        task="Deploy the worker manifest to the staging cluster.",
        tool="kubectl_apply",
        bad_args={"manifest": "worker.yaml", "namespace": "staging"},
        error='{"error": "namespaces \\"staging\\" not found"}',
        fixed_args={"manifest": "worker.yaml", "namespace": "staging", "create_namespace": True},
        success='{"status": "deployment.apps/worker created"}',
        insight="the staging namespace does not exist yet",
    ),
    Scenario(
        task="Resize the uploaded avatar to 128px and store it.",
        tool="image_resize",
        bad_args={"src": "avatar.heic", "width": 128},
        error='{"error": "UnsupportedFormat: heic; supported: png, jpeg, webp"}',
        fixed_args={"src": "avatar.heic", "width": 128, "convert_to": "png"},
        success='{"status": "ok", "path": "avatar-128.png"}',
        insight="heic is not a supported input format",
    ),
)

# Failure modes that are NOT the planted deficit. Their presence is what makes
# recovery a real result: the ranker has to choose between genuine competing
# explanations rather than the only one on offer.
DISTRACTOR_MODES = ("stops_to_ask", "wrong_output_shape", "drops_second_requirement")


# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------


def _turn(
    index: int,
    role: str,
    *,
    content: str | None = None,
    thinking: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> dict:
    return {
        "index": index,
        "role": role,
        "thinking": thinking,
        "content": content,
        "toolCalls": tool_calls or [],
        "toolCallId": tool_call_id,
    }


def _call(call_id: str, name: str, args: dict) -> list[dict]:
    return [{"id": call_id, "name": name, "args": args}]


def _win_recovering(run_id: str, s: Scenario) -> dict:
    """Hits the error, reads it, corrects the call. Capability PRESENT."""
    return {
        "runId": run_id,
        "status": "success",
        "turns": [
            _turn(0, "user", content=s.task),
            _turn(
                1,
                "assistant",
                thinking="Straightforward request; I'll make the call.",
                content="Working on it.",
                tool_calls=_call("c1", s.tool, s.bad_args),
            ),
            _turn(2, "tool", content=s.error, tool_call_id="c1"),
            _turn(
                3,
                "assistant",
                thinking=f"The error says {s.insight}. I'll correct the call accordingly.",
                content="Adjusting and retrying.",
                tool_calls=_call("c2", s.tool, s.fixed_args),
            ),
            _turn(4, "tool", content=s.success, tool_call_id="c2"),
            _turn(5, "assistant", content="Done — the operation completed."),
        ],
    }


def _win_clean(run_id: str, s: Scenario) -> dict:
    """Succeeds first try. The capability is genuinely NA here — nothing in the
    trajectory says anything about how this agent handles tool errors."""
    return {
        "runId": run_id,
        "status": "success",
        "turns": [
            _turn(0, "user", content=s.task),
            _turn(
                1,
                "assistant",
                thinking="Straightforward request; I'll make the call.",
                content="Working on it.",
                tool_calls=_call("c1", s.tool, s.fixed_args),
            ),
            _turn(2, "tool", content=s.success, tool_call_id="c1"),
            _turn(3, "assistant", content="Done — the operation completed."),
        ],
    }


def _loss_planted(run_id: str, s: Scenario) -> dict:
    """Hits the same error and repeats the identical call until it gives up.

    Note the thinking blocks: this agent reasons about its situation as much as
    the winning one does, it just never engages with what the error said. That
    keeps 'produced a thinking block' from separating wins from losses on its
    own.
    """
    return {
        "runId": run_id,
        "status": "failure",
        "turns": [
            _turn(0, "user", content=s.task),
            _turn(
                1,
                "assistant",
                thinking="Straightforward request; I'll make the call.",
                content="Working on it.",
                tool_calls=_call("c1", s.tool, s.bad_args),
            ),
            _turn(2, "tool", content=s.error, tool_call_id="c1"),
            _turn(
                3,
                "assistant",
                thinking="That didn't go through. It may be transient, so I'll try again.",
                content="Retrying.",
                tool_calls=_call("c2", s.tool, s.bad_args),
            ),
            _turn(4, "tool", content=s.error, tool_call_id="c2"),
            _turn(
                5,
                "assistant",
                thinking="Still not working. I'll give it one more attempt.",
                content="Retrying once more.",
                tool_calls=_call("c3", s.tool, s.bad_args),
            ),
            _turn(6, "tool", content=s.error, tool_call_id="c3"),
            _turn(
                7,
                "assistant",
                content="I wasn't able to complete this after several attempts.",
            ),
        ],
    }


def _loss_stops_to_ask(run_id: str, s: Scenario) -> dict:
    """Never fails a call — never makes one. Bounces the task back to the user."""
    return {
        "runId": run_id,
        "status": "failure",
        "turns": [
            _turn(0, "user", content=s.task),
            _turn(
                1,
                "assistant",
                thinking="I could do this directly, but I'd rather confirm the details first.",
                content=(
                    "Before I proceed, could you confirm the exact target and whether you "
                    "want me to overwrite anything that already exists?"
                ),
            ),
        ],
    }


def _loss_wrong_output_shape(run_id: str, s: Scenario) -> dict:
    """Does the work correctly, then reports it in a shape the task didn't ask
    for. The tool calls all succeed."""
    return {
        "runId": run_id,
        "status": "failure",
        "turns": [
            _turn(0, "user", content=s.task + " Reply with only a JSON object."),
            _turn(
                1,
                "assistant",
                thinking="I'll perform the operation and then summarize.",
                content="Working on it.",
                tool_calls=_call("c1", s.tool, s.fixed_args),
            ),
            _turn(2, "tool", content=s.success, tool_call_id="c1"),
            _turn(
                3,
                "assistant",
                content=(
                    "All set! I went ahead and handled that for you. Let me know if "
                    "there's anything else you need."
                ),
            ),
        ],
    }


def _loss_drops_second_requirement(run_id: str, s: Scenario) -> dict:
    """Two-part task, one part done. Again, no tool call ever fails."""
    return {
        "runId": run_id,
        "status": "failure",
        "turns": [
            _turn(
                0,
                "user",
                content=s.task + " Then record the result in the audit log.",
            ),
            _turn(
                1,
                "assistant",
                thinking="I'll perform the main operation.",
                content="Working on it.",
                tool_calls=_call("c1", s.tool, s.fixed_args),
            ),
            _turn(2, "tool", content=s.success, tool_call_id="c1"),
            _turn(3, "assistant", content="Done — the operation completed."),
        ],
    }


_DISTRACTOR_BUILDERS = {
    "stops_to_ask": _loss_stops_to_ask,
    "wrong_output_shape": _loss_wrong_output_shape,
    "drops_second_requirement": _loss_drops_second_requirement,
}

# What the labeller *should* say about the planted capability for each trace
# kind, if it were a perfect instrument.
TRUTH = {
    "win_recovering": "PRESENT",
    "win_clean": "NA",
    "loss_planted": "LACKING",
    "loss_stops_to_ask": "NA",
    "loss_wrong_output_shape": "NA",
    "loss_drops_second_requirement": "NA",
}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass
class PlantedCorpus:
    traces: list[dict]
    manifest: list[dict]
    truth: dict[str, str]  # run_id -> expected label for the planted capability
    kinds: dict[str, str]  # run_id -> trace kind
    n_wins: int
    n_losses: int
    prevalence: float  # actual share of losses carrying the planted deficit

    @property
    def n_planted(self) -> int:
        return sum(1 for k in self.kinds.values() if k == "loss_planted")


def build_corpus(
    *,
    n_wins: int,
    n_losses: int,
    prevalence: float = 1.0,
    clean_win_share: float = 0.3,
    seed: int = 0,
) -> PlantedCorpus:
    """Build a seeded corpus with the deficit planted in `prevalence` of losses.

    Run ids are deliberately opaque (`run-0007`). They appear verbatim in the
    proposer's prompt, so naming them `loss-planted-3` would hand the model the
    answer key and the whole experiment would measure nothing.
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {prevalence}")
    if n_wins < 1 or n_losses < 1:
        raise ValueError("need at least one win and one loss")

    rng = random.Random(seed)
    n_planted = round(prevalence * n_losses)
    n_clean_wins = min(n_wins - 1, round(clean_win_share * n_wins)) if n_wins > 1 else 0

    kinds: list[str] = (
        ["win_clean"] * n_clean_wins
        + ["win_recovering"] * (n_wins - n_clean_wins)
        + ["loss_planted"] * n_planted
        + [
            f"loss_{DISTRACTOR_MODES[i % len(DISTRACTOR_MODES)]}"
            for i in range(n_losses - n_planted)
        ]
    )

    traces: list[dict] = []
    manifest: list[dict] = []
    truth: dict[str, str] = {}
    kind_by_id: dict[str, str] = {}

    for i, kind in enumerate(kinds):
        run_id = f"run-{i:04d}"
        scenario = SCENARIOS[rng.randrange(len(SCENARIOS))]
        if kind == "win_recovering":
            payload = _win_recovering(run_id, scenario)
        elif kind == "win_clean":
            payload = _win_clean(run_id, scenario)
        elif kind == "loss_planted":
            payload = _loss_planted(run_id, scenario)
        else:
            payload = _DISTRACTOR_BUILDERS[kind.removeprefix("loss_")](run_id, scenario)

        traces.append(payload)
        manifest.append(
            {
                "path": f"traces/{run_id}.json",
                "outcome": "win" if kind.startswith("win") else "loss",
            }
        )
        truth[run_id] = TRUTH[kind]
        kind_by_id[run_id] = kind

    return PlantedCorpus(
        traces=traces,
        manifest=manifest,
        truth=truth,
        kinds=kind_by_id,
        n_wins=n_wins,
        n_losses=n_losses,
        prevalence=n_planted / n_losses,
    )


def write_corpus(corpus: PlantedCorpus, out_dir: Path) -> Path:
    """Write the corpus as ordinary traces + manifest. Returns the manifest path.

    Deliberately the same on-disk format a real run consumes, so recovery is
    measured through the real load path rather than an in-memory shortcut.
    """
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    for payload in corpus.traces:
        (traces_dir / f"{payload['runId']}.json").write_text(json.dumps(payload, indent=2))
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(corpus.manifest, indent=2))
    # Ground truth alongside, never inside the traces themselves.
    (out_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "planted_capability": {
                    "name": PLANTED_NAME,
                    "description": PLANTED_DESCRIPTION,
                },
                "prevalence": corpus.prevalence,
                "kinds": corpus.kinds,
                "expected_labels": corpus.truth,
            },
            indent=2,
        )
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


@dataclass
class LabelAccuracy:
    """How well the labeller reproduced the labels we know to be correct.

    Blur measured directly, on data where the right answer isn't a judgement
    call — the same quantity step 5 of the plan proposes to estimate by hand-
    labelling ~50 real traces.
    """

    n: int = 0
    exact: int = 0
    confusion: dict[str, int] = field(default_factory=dict)  # "TRUTH->PREDICTED" -> count

    @property
    def accuracy(self) -> float | None:
        return (self.exact / self.n) if self.n else None


@dataclass
class RecoveryResult:
    """One end-to-end attempt at recovering the planted deficit."""

    proposed: list[Capability]
    scores: list[DeficitScore]
    selected: DeficitScore | None
    # The best-matching proposed capability, if any, and how it matched.
    matched: Capability | None
    match_kind: str
    matched_rank: int | None  # position in the priority-ranked list
    recovered: bool  # the selected deficit IS the planted one
    label_accuracy: LabelAccuracy
    verdict: str

    def to_dict(self) -> dict:
        return {
            "recovered": self.recovered,
            "verdict": self.verdict,
            "match_kind": self.match_kind,
            "matched_capability": (
                {"id": self.matched.id, "name": self.matched.name} if self.matched else None
            ),
            "matched_rank": self.matched_rank,
            "selected_capability": (
                self.selected.capability.name if self.selected else None
            ),
            "label_accuracy": self.label_accuracy.accuracy,
            "label_confusion": self.label_accuracy.confusion,
            "ranked": [
                {
                    "name": s.capability.name,
                    "id": s.capability.id,
                    "gap": s.gap,
                    "prevalence": s.prevalence,
                    "priority": s.priority,
                    "n_wins": s.n_relevant_wins,
                    "n_losses": s.n_relevant_losses,
                    "match": match_planted(s.capability),
                }
                for s in self.scores
            ],
        }


# Why a run failed to recover. Kept separate because they call for completely
# different responses: the proposer never naming the capability is a prompt
# problem, a distractor outranking it is a labeller problem, and a top-ranked
# match rejected by the threshold is a calibration problem — and only the last
# one means the pipeline is basically working.
VERDICT_RECOVERED = "recovered"
VERDICT_NOT_PROPOSED = "not_proposed"
VERDICT_OUTRANKED = "outranked_by_distractor"
VERDICT_BELOW_THRESHOLD = "top_ranked_but_below_threshold"


def run_recovery(
    manifest_path: Path,
    truth: dict[str, str],
    *,
    model: str | None = None,
    min_gap: float = DEFAULT_MIN_GAP,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_workers: int = 8,
) -> RecoveryResult:
    """Run the real diagnosis over a planted corpus and score the recovery."""
    entries = load_manifest(manifest_path)
    traces = [Trace.load(e.path, outcome=e.outcome) for e in entries]

    proposed = propose_capabilities(traces, model=model)
    # executor.map preserves input order, so labels line up with traces and the
    # run stays reproducible under a fixed seed.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        trace_labels = list(pool.map(lambda t: label_trace(t, proposed, model=model), traces))
    return _assemble(proposed, trace_labels, truth, min_gap=min_gap, min_support=min_support)


def _assemble(
    proposed: list[Capability],
    trace_labels: list[TraceLabels],
    truth: dict[str, str],
    *,
    min_gap: float,
    min_support: int,
) -> RecoveryResult:
    """Score a set of labels and decide whether the planted deficit was found.

    Shared by the live run and the perfect-labeller ceiling so the two are
    scored by identical code — a ceiling computed a different way would not
    bound anything.
    """
    scores = score_deficits(proposed, trace_labels)
    selected = select_deficit(scores, min_gap=min_gap, min_support=min_support)

    # Best match wins on strictness, then on rank.
    ranked_ids = {s.capability.id: i for i, s in enumerate(scores)}
    candidates = [
        (match_planted(c), ranked_ids.get(c.id, len(scores)), c)
        for c in proposed
        if match_planted(c) != MATCH_NONE
    ]
    candidates.sort(key=lambda t: (0 if t[0] == MATCH_STRICT else 1, t[1]))

    matched = candidates[0][2] if candidates else None
    match_kind = candidates[0][0] if candidates else MATCH_NONE
    matched_rank = ranked_ids.get(matched.id) if matched else None

    recovered = bool(selected and matched and selected.capability.id == matched.id)
    if recovered:
        verdict = VERDICT_RECOVERED
    elif matched is None:
        verdict = VERDICT_NOT_PROPOSED
    elif matched_rank == 0:
        verdict = VERDICT_BELOW_THRESHOLD
    else:
        verdict = VERDICT_OUTRANKED

    return RecoveryResult(
        proposed=proposed,
        scores=scores,
        selected=selected,
        matched=matched,
        match_kind=match_kind,
        matched_rank=matched_rank,
        recovered=recovered,
        label_accuracy=_label_accuracy(matched, trace_labels, truth),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# The ceiling: what a perfect instrument would recover
# ---------------------------------------------------------------------------

# The capabilities a flawless proposer would name: the planted one, plus one
# per distractor failure mode.
ORACLE_PLANTED = Capability(
    id="error_adaptation",
    name="Adapts after a failed tool call",
    description=(
        "Reads the error message from a failed call and issues a corrected one, "
        "rather than repeating the identical call."
    ),
)
ORACLE_DISTRACTORS = {
    "stops_to_ask": Capability(
        id="proceeds_without_confirmation",
        name="Proceeds without asking for confirmation",
        description="Carries out a well-specified task instead of bouncing it back.",
    ),
    "wrong_output_shape": Capability(
        id="follows_output_format",
        name="Follows the requested output format",
        description="Returns results in the shape the task asked for.",
    ),
    "drops_second_requirement": Capability(
        id="completes_all_requirements",
        name="Completes every part of the task",
        description="Satisfies all requirements, not just the first one.",
    ),
}

# What a perfect labeller says about each distractor capability, by trace kind.
# Two of the three are only *relevant* on the traces that raise them — a task
# with no stated output format says nothing about whether the agent follows
# one — so they are NA elsewhere rather than PRESENT.
_DISTRACTOR_ELSEWHERE = {
    "stops_to_ask": "PRESENT",  # every trace either bounced the task or didn't
    "wrong_output_shape": "NA",
    "drops_second_requirement": "NA",
}


def oracle_labels(kind: str) -> dict[str, str]:
    """The labels a flawless labeller would produce for a trace of this kind."""
    labels = {ORACLE_PLANTED.id: TRUTH[kind]}
    for mode, cap in ORACLE_DISTRACTORS.items():
        labels[cap.id] = "LACKING" if kind == f"loss_{mode}" else _DISTRACTOR_ELSEWHERE[mode]
    return labels


def run_ceiling(
    corpus: PlantedCorpus,
    *,
    min_gap: float = DEFAULT_MIN_GAP,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> RecoveryResult:
    """Recovery under a perfect proposer and a perfect labeller. No LLM.

    This is the upper bound on any real run of the same corpus, and it is free,
    so there is no reason to spend money finding out that a config was
    unrecoverable by construction. Where the ceiling is 0%, the corpus is too
    small or too diluted for the thresholds to ever pass it, and a live run
    there measures the thresholds rather than the ranker.
    """
    proposed = [ORACLE_PLANTED, *ORACLE_DISTRACTORS.values()]
    trace_labels = [
        TraceLabels(
            trace=Trace(
                run_id=run_id,
                status="",
                turns=[],
                outcome="win" if kind.startswith("win") else "loss",
                source_path=Path(run_id),
            ),
            labels=oracle_labels(kind),
            evidence={},
        )
        for run_id, kind in corpus.kinds.items()
    ]
    return _assemble(
        proposed, trace_labels, corpus.truth, min_gap=min_gap, min_support=min_support
    )


def _label_accuracy(
    matched: Capability | None,
    trace_labels: list[TraceLabels],
    truth: dict[str, str],
) -> LabelAccuracy:
    acc = LabelAccuracy()
    if matched is None:
        return acc
    for tl in trace_labels:
        expected = truth.get(tl.trace.run_id)
        if expected is None:
            continue
        # Absent label reads as NA — the same convention scoring uses.
        predicted = tl.labels.get(matched.id, "NA")
        acc.n += 1
        acc.exact += int(predicted == expected)
        key = f"{expected}->{predicted}"
        acc.confusion[key] = acc.confusion.get(key, 0) + 1
    return acc


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    n_wins: int
    n_losses: int
    prevalence: float

    @property
    def label(self) -> str:
        return f"{self.n_wins}w/{self.n_losses}l @ p={self.prevalence:g}"

    @property
    def slug(self) -> str:
        return f"w{self.n_wins}-l{self.n_losses}-p{self.prevalence:g}"


# Trace count and prevalence, the two axes the plan calls for. Counts start
# below what anyone would call adequate on purpose: knowing where recovery
# breaks down is the point, and "how many traces do we need" is the number
# v1.1 has to quote at a customer when it asks them to send more.
DEFAULT_SWEEP: tuple[SweepConfig, ...] = tuple(
    SweepConfig(n_wins=w, n_losses=w, prevalence=p)
    for w in (3, 6, 12)
    for p in (1.0, 0.6, 0.3)
)


@dataclass
class SweepCell:
    config: SweepConfig
    results: list[RecoveryResult]
    # Perfect-proposer, perfect-labeller runs over the same corpora. Free, so
    # always computed; the live rate can never exceed it.
    ceilings: list[RecoveryResult] = field(default_factory=list)

    @property
    def ceiling_rate(self) -> float | None:
        if not self.ceilings:
            return None
        return sum(r.recovered for r in self.ceilings) / len(self.ceilings)

    @property
    def recovery_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recovered for r in self.results) / len(self.results)

    @property
    def proposed_rate(self) -> float | None:
        """How often the capability was named at all, whether or not it won."""
        if not self.results:
            return None
        return sum(r.matched is not None for r in self.results) / len(self.results)

    @property
    def mean_label_accuracy(self) -> float | None:
        vals = [r.label_accuracy.accuracy for r in self.results]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def verdicts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.verdict] = out.get(r.verdict, 0) + 1
        return out

    @property
    def ceiling_verdicts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.ceilings:
            out[r.verdict] = out.get(r.verdict, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "config": {
                "n_wins": self.config.n_wins,
                "n_losses": self.config.n_losses,
                "prevalence": self.config.prevalence,
            },
            "repeats": len(self.results) or len(self.ceilings),
            "ceiling_rate": self.ceiling_rate,
            "ceiling_verdicts": self.ceiling_verdicts,
            "recovery_rate": self.recovery_rate if self.results else None,
            "proposed_rate": self.proposed_rate,
            "mean_label_accuracy": self.mean_label_accuracy,
            "verdicts": self.verdicts,
            "runs": [r.to_dict() for r in self.results],
            "ceiling_runs": [r.to_dict() for r in self.ceilings],
        }


def estimate_calls(configs: list[SweepConfig] | tuple[SweepConfig, ...], repeats: int) -> int:
    """LLM calls a sweep will make: one proposer call plus one per trace."""
    return repeats * sum(1 + c.n_wins + c.n_losses for c in configs)


def run_sweep(
    configs: list[SweepConfig] | tuple[SweepConfig, ...],
    out_dir: Path,
    *,
    repeats: int = 3,
    model: str | None = None,
    min_gap: float = DEFAULT_MIN_GAP,
    min_support: int = DEFAULT_MIN_SUPPORT,
    seed: int = 0,
    max_workers: int = 8,
    ceiling_only: bool = False,
    on_cell=None,
) -> list[SweepCell]:
    """Run every config `repeats` times and return the cells.

    Repeats are not optional decoration. The proposer and labeller are both
    sampled, so a single run at a single config is one draw from a distribution
    — it can recover a deficit that usually gets missed, or miss one that
    usually lands. The reported number is a rate, and with 3 repeats it is a
    very coarse one.
    """
    cells: list[SweepCell] = []
    for config in configs:
        results: list[RecoveryResult] = []
        ceilings: list[RecoveryResult] = []
        for rep in range(repeats):
            # Distinct seed per (config, repeat) so corpora differ across
            # repeats and we measure the ranker, not one lucky corpus.
            run_seed = seed + rep
            corpus = build_corpus(
                n_wins=config.n_wins,
                n_losses=config.n_losses,
                prevalence=config.prevalence,
                seed=run_seed,
            )
            run_dir = out_dir / "corpora" / f"{config.slug}-seed{run_seed}"
            manifest_path = write_corpus(corpus, run_dir)

            ceiling = run_ceiling(corpus, min_gap=min_gap, min_support=min_support)
            (run_dir / "ceiling.json").write_text(json.dumps(ceiling.to_dict(), indent=2))
            ceilings.append(ceiling)

            if ceiling_only:
                continue
            result = run_recovery(
                manifest_path,
                corpus.truth,
                model=model,
                min_gap=min_gap,
                min_support=min_support,
                max_workers=max_workers,
            )
            (run_dir / "recovery.json").write_text(json.dumps(result.to_dict(), indent=2))
            results.append(result)
        cell = SweepCell(config=config, results=results, ceilings=ceilings)
        cells.append(cell)
        if on_cell:
            on_cell(cell)
    return cells


def write_sweep_report(
    cells: list[SweepCell],
    out_dir: Path,
    *,
    model: str | None,
    min_gap: float,
    min_support: int,
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "planted_capability": {"name": PLANTED_NAME, "description": PLANTED_DESCRIPTION},
        "settings": {
            "model": model,
            "min_gap": min_gap,
            "min_support": min_support,
            "seed": seed,
            "distractor_modes": list(DISTRACTOR_MODES),
        },
        "cells": [c.to_dict() for c in cells],
    }
    (out_dir / "selftest.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Planted-deficit recovery\n",
        f"**Planted capability:** {PLANTED_NAME}\n",
        f"> {PLANTED_DESCRIPTION}\n",
        (
            f"\nModel `{model or 'default'}`, min_gap={min_gap}, min_support={min_support}, "
            f"seed={seed}, {len(cells[0].results) if cells else 0} repeat(s) per cell.\n"
        ),
        (
            "\nLosses that do *not* carry the planted deficit fail for unrelated "
            "reasons ("
            + ", ".join(DISTRACTOR_MODES)
            + "), so the ranker has to choose between real competing explanations.\n"
        ),
        "\n## Recovery rate\n",
        "| corpus | prevalence | ceiling | recovered | proposed | label acc. | verdicts |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        acc = c.mean_label_accuracy
        verdicts = ", ".join(f"{k}×{v}" for k, v in sorted(c.verdicts.items()))
        ceiling = c.ceiling_rate
        if not c.results:
            verdicts = ", ".join(f"{k}×{v}" for k, v in sorted(c.ceiling_verdicts.items()))
        lines.append(
            f"| {c.config.n_wins}w/{c.config.n_losses}l | {c.config.prevalence:g} | "
            f"{'n/a' if ceiling is None else f'{ceiling:.0%}'} | "
            f"{'—' if not c.results else f'{c.recovery_rate:.0%}'} | "
            f"{'—' if c.proposed_rate is None else f'{c.proposed_rate:.0%}'} | "
            f"{'n/a' if acc is None else f'{acc:.0%}'} | {verdicts} |"
        )

    lines += [
        "\n## Reading this\n",
        "- **ceiling** — recovery under a perfect proposer and a perfect labeller "
        "on the same corpora, computed without an LLM. The live rate cannot exceed "
        "it. A 0% ceiling means the config is unrecoverable by construction at "
        "these thresholds, and a live run there measures the thresholds, not the "
        "ranker.",
        "- **recovered** — the planted capability was selected as *the* deficit.",
        "- **proposed** — it was named at all. A gap between the two columns is a "
        "labelling or threshold problem, not a proposer problem.",
        "- **label acc.** — how often the labeller reproduced the label we know to "
        "be correct, per trace. This is the labeller blur that shrinks every gap "
        "downstream toward zero.",
        "- **verdicts** — `not_proposed` is a proposer problem, "
        "`outranked_by_distractor` a labeller problem, "
        "`top_ranked_but_below_threshold` a calibration problem.\n",
        "\nEvery corpus, its ground truth, and its per-run recovery detail are "
        "under `corpora/` — nothing above is derived from anything not on disk.\n",
    ]

    md_path = out_dir / "selftest.md"
    md_path.write_text("\n".join(lines))
    return md_path
