"""Capture-only Tau2 rollout driver for live cross-tokenizer OPD.

This module supplies the missing PLANNED -> SAMPLED boundary.  It deliberately
stops there: scoring and optimization continue through the existing ReOPD
backend once ``actions.jsonl`` and the native semantic histories are durable.

Tau2 is imported lazily so archive and driver tests do not require a benchmark
checkout.  The live machine must provide the pinned Tau2 v0.2 package.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from vektori_trace.tau2.live_agent import (
    CapturingLLMAgent,
    LiveCaptureError,
    render_prompt_ids,
)
from vektori_trace.tau2.live_episode import (
    EpisodeArchive,
    EpisodeStatus,
    LiveEpisode,
    LiveTurn,
    build_failed_turn,
)
from vektori_trace.tau2.live_turns import flatten_live_turns, teacher_context_hash
from vektori_trace.tau2.reopd_state import (
    ReOPDStateError,
    UpdateDir,
    atomic_write_json,
    atomic_write_jsonl,
)

__all__ = [
    "EpisodePlan",
    "EpisodeResult",
    "RolloutSettings",
    "Tau2EpisodeRunner",
    "capture_live_update",
]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: str
    task_id: str
    seed: int


@dataclass(frozen=True)
class EpisodeResult:
    termination_reason: str
    reward: float
    reward_info: dict[str, Any]
    initial_env_state_hash: str | None
    final_env_state_hash: str | None
    simulation: dict[str, Any] | None = None
    viewer_results: dict[str, Any] | None = None


@dataclass(frozen=True)
class RolloutSettings:
    domain: str
    student_model: str
    api_base: str
    policy_version: str
    adapter_hash: str
    gen_config_hash: str
    max_tokens: int
    max_input_tokens: int
    temperature: float = 0.0
    timeout: float = 600.0
    max_steps: int = 100
    max_errors: int = 10
    user: str = "user_simulator"
    user_model: str = "gpt-4o-mini"
    user_model_args: dict[str, Any] = field(default_factory=dict)
    require_reasoning: bool = True


class EpisodeRunner(Protocol):
    def run(
        self,
        plan: EpisodePlan,
        *,
        on_turn: Callable[[Any, list[dict[str, Any]], dict[str, Any]], None],
        on_failure: Callable[..., None],
        on_observation: Callable[[int, dict[str, Any] | None, str | None], None],
    ) -> EpisodeResult: ...


class Tau2EpisodeRunner:
    """Run one official Tau2 v0.2 episode with ``CapturingLLMAgent``."""

    def __init__(self, settings: RolloutSettings, tokenizer: Any) -> None:
        self.settings = settings
        self.tokenizer = tokenizer

    def run(
        self,
        plan: EpisodePlan,
        *,
        on_turn: Callable[[Any, list[dict[str, Any]], dict[str, Any]], None],
        on_failure: Callable[..., None],
        on_observation: Callable[[int, dict[str, Any] | None, str | None], None],
    ) -> EpisodeResult:
        try:
            from tau2.data_model.simulation import Results
            from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
            from tau2.orchestrator.orchestrator import Orchestrator, Role
            from tau2.registry import registry
            from tau2.run import get_info, get_tasks
        except ImportError as exc:  # pragma: no cover - live-machine dependency
            raise RuntimeError(
                "Tau2 is not installed. Install the pinned v0.2 checkout on the "
                "rollout machine before running live capture."
            ) from exc

        s = self.settings
        tasks = get_tasks(s.domain, [plan.task_id])
        task = tasks[0]
        environment = registry.get_env_constructor(s.domain)()
        tools = environment.get_tools()
        schemas = [tool.openai_schema for tool in tools]
        pending: list[int] = []

        def archive_turn(cap: Any, history: list[dict[str, Any]], parsed: dict[str, Any]):
            rerendered = render_prompt_ids(self.tokenizer, history, schemas)
            if rerendered != cap.prompt_token_ids:
                raise LiveCaptureError(
                    f"turn {cap.turn_index}: archived semantic history does not "
                    "re-render to the prompt ids sent to the student"
                )
            on_turn(cap, history, parsed)
            pending.append(int(cap.turn_index))

        agent = CapturingLLMAgent(
            tools=tools,
            domain_policy=environment.get_policy(),
            llm=s.student_model,
            api_base=s.api_base,
            tokenizer=self.tokenizer,
            tool_schemas=schemas,
            system_content=environment.get_policy(),
            policy_version=s.policy_version,
            max_tokens=s.max_tokens,
            max_input_tokens=s.max_input_tokens,
            temperature=s.temperature,
            timeout=s.timeout,
            require_reasoning=s.require_reasoning,
            on_turn=archive_turn,
            on_failure=on_failure,
        )
        agent.set_episode(plan.episode_id, plan.task_id)

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        user_cls = registry.get_user_constructor(s.user)
        user = user_cls(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=s.user_model,
            llm_args=dict(s.user_model_args),
        )
        initial_hash: str | None = None

        class ArchivingOrchestrator(Orchestrator):
            def initialize(self):
                nonlocal initial_hash
                super().initialize()
                initial_hash = self.environment.get_db_hash()

            def step(self):
                was_agent_action = self.from_role == Role.AGENT and pending
                super().step()
                if was_agent_action:
                    turn_index = pending.pop(0)
                    on_observation(
                        turn_index,
                        _jsonable(self.message),
                        self.environment.get_db_hash(),
                    )

        orchestrator = ArchivingOrchestrator(
            domain=s.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=s.max_steps,
            max_errors=s.max_errors,
            seed=plan.seed,
            solo_mode=False,
        )
        simulation = orchestrator.run()
        reward_info = evaluate_simulation(
            domain=s.domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        simulation.reward_info = reward_info
        simulation.trial = 0
        info = get_info(
            domain=s.domain,
            agent="llm_agent",
            user=s.user,
            llm_agent=s.student_model,
            llm_args_agent={"temperature": s.temperature},
            llm_user=s.user_model,
            llm_args_user=dict(s.user_model_args),
            num_trials=1,
            max_steps=s.max_steps,
            max_errors=s.max_errors,
            seed=plan.seed,
        )
        viewer_results = Results(info=info, tasks=[task], simulations=[simulation])
        return EpisodeResult(
            termination_reason=str(simulation.termination_reason),
            reward=float(reward_info.reward),
            reward_info=_jsonable(reward_info),
            initial_env_state_hash=initial_hash,
            final_env_state_hash=environment.get_db_hash(),
            simulation=_jsonable(simulation),
            viewer_results=_jsonable(viewer_results),
        )


def capture_live_update(
    update: UpdateDir,
    plans: list[EpisodePlan],
    *,
    settings: RolloutSettings,
    teacher_context: dict[str, Any],
    runner: EpisodeRunner,
) -> dict[str, Any]:
    """Run and durably assemble one capture-only live update.

    Every generation is appended immediately.  The update is marked SAMPLED
    only after the exact planned episode set passes archive validation and its
    turns have been translated to the existing ``actions.jsonl`` format.
    """
    if not plans:
        raise ValueError("a live update must plan at least one episode")
    ids = [p.episode_id for p in plans]
    if len(ids) != len(set(ids)):
        raise ValueError("episode plan contains duplicate episode ids")
    if update.reached("SAMPLED"):
        update.validate()
        return json.loads(update.report_path.read_text())

    update.path.mkdir(parents=True, exist_ok=True)
    archive = EpisodeArchive(update.path / "live_archive")
    planned_payload = {
        "stage": "PLANNED",
        "episode_ids": ids,
        "plans": [p.__dict__ for p in plans],
        "policy_version": settings.policy_version,
        "adapter_hash": settings.adapter_hash,
        "gen_config_hash": settings.gen_config_hash,
        "require_reasoning": settings.require_reasoning,
    }
    planned_marker = update.marker("PLANNED")
    if planned_marker.exists():
        frozen = json.loads(planned_marker.read_text())
        if frozen != planned_payload:
            raise ReOPDStateError(
                "live update plan changed after sampling began; use a new "
                "update instead of mixing two recipes"
            )
    else:
        update.mark("PLANNED", planned_payload)

    existing = archive.load_episodes()
    # A process death can leave an environment half-mutated. Tau2 cannot
    # restore that state, so make the loss explicit before validating resume.
    for episode_id, row in existing.items():
        if episode_id in ids and row["status"] == EpisodeStatus.SAMPLING:
            interrupted = LiveEpisode(**{
                key: row[key]
                for key in LiveEpisode.__dataclass_fields__
                if key in row
            })
            interrupted.finish(
                EpisodeStatus.DISCARDED,
                reason="infrastructure interruption during sampling; state cannot rewind",
            )
            interrupted.num_turns = len(archive.load_turns(episode_id))
            interrupted.num_failed_turns = len(archive.load_failures(episode_id))
            archive.record_episode(interrupted)
    existing = archive.load_episodes()
    for plan in plans:
        if plan.episode_id in existing:
            # Episodes are stateful and cannot be rewound. Never append a
            # second sampling history under the same identity on resume.
            continue
        episode = LiveEpisode(
            episode_id=plan.episode_id,
            task_id=plan.task_id,
            seed=plan.seed,
            policy_version=settings.policy_version,
            adapter_hash=settings.adapter_hash,
            gen_config_hash=settings.gen_config_hash,
            require_reasoning=settings.require_reasoning,
        )
        archive.record_episode(episode)

        def on_turn(cap: Any, history: list[dict[str, Any]], parsed: dict[str, Any]):
            archive.record_turn(LiveTurn(cap, history, parsed))

        def on_failure(**kwargs: Any):
            archive.record_failure(build_failed_turn(**kwargs))

        def on_observation(
            index: int,
            observation: dict[str, Any] | None,
            state: str | None,
            episode_id: str = plan.episode_id,
        ):
            archive.record_observation(episode_id, index, observation, state)

        try:
            result = runner.run(
                plan,
                on_turn=on_turn,
                on_failure=on_failure,
                on_observation=on_observation,
            )
            episode.initial_env_state_hash = result.initial_env_state_hash
            episode.final_env_state_hash = result.final_env_state_hash
            episode.termination_reason = result.termination_reason
            episode.reward = result.reward
            episode.reward_info = result.reward_info
            if result.simulation is not None:
                archive.record_simulation(
                    plan.episode_id,
                    result.simulation,
                    viewer_results=result.viewer_results,
                )
            episode.num_turns = len(archive.load_turns(plan.episode_id))
            episode.num_failed_turns = len(archive.load_failures(plan.episode_id))
            if episode.num_failed_turns:
                episode.finish(EpisodeStatus.FAILED, reason="capture failure")
            else:
                episode.finish(EpisodeStatus.SAMPLED)
        except LiveCaptureError as exc:
            episode.num_turns = len(archive.load_turns(plan.episode_id))
            episode.num_failed_turns = len(archive.load_failures(plan.episode_id))
            status = (
                EpisodeStatus.FAILED
                if episode.num_failed_turns
                else EpisodeStatus.DISCARDED
            )
            episode.finish(status, reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            episode.num_turns = len(archive.load_turns(plan.episode_id))
            episode.num_failed_turns = len(archive.load_failures(plan.episode_id))
            episode.finish(
                EpisodeStatus.DISCARDED,
                reason=f"infrastructure {type(exc).__name__}: {exc}",
            )
        archive.record_episode(episode)

    report = archive.batch_report(
        ids,
        expected_episode_ids=ids,
        policy_version=settings.policy_version,
        adapter_hash=settings.adapter_hash,
        gen_config_hash=settings.gen_config_hash,
    )
    atomic_write_json(update.report_path, report)
    if not report["ok"]:
        raise ReOPDStateError(
            "live update did not produce its exact planned batch: "
            + "; ".join(report["problems"])
        )

    turns = {episode_id: archive.load_turns(episode_id) for episode_id in ids}
    context_hash = teacher_context_hash(teacher_context)
    rows, rendered = flatten_live_turns(
        turns,
        policy_version=settings.policy_version,
        teacher_context=context_hash,
    )
    atomic_write_jsonl(update.actions_path, rows)
    atomic_write_json(update.path / "rendered.json", rendered)
    update.mark(
        "SAMPLED",
        {
            "stage": "SAMPLED",
            "episodes": len(ids),
            "actions": len(rows),
            "policy_version": settings.policy_version,
            "adapter_hash": settings.adapter_hash,
            "gen_config_hash": settings.gen_config_hash,
            "teacher_context_hash": context_hash,
        },
    )
    update.validate()
    return report
