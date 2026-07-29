"""A0–A4 arm orchestrator for V0 Step 6.

Reads `selection.json` from `select`, drives rollout→train→serve→measure for the
trained arms, and writes `arms.json`/`arms.md`. Does **not** touch `gap.py` —
arms produce `PassRate` aggregates, a different shape from `Trace` pairing.

Pre-declared rules (before any number is looked at):
- McNemar binarization: `PassRate.majority_pass` (strict majority of rollouts).
- IFEval non-regression tolerance: 5-point absolute drop (`nonregression.py`).
- Candidate default placeholder: `Qwen/Qwen3-8B` (swap once Step 4 gap exists).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset import tokenize_sft_example
from .diagnose import MIN_DISCORDANT_PAIRS, _exact_mcnemar_p
from .nonregression import (
    IFEVAL_TOLERANCE,
    NonRegressionResult,
    evaluate_ifeval,
    load_ifeval_subset,
)
from .passrate import DEFAULT_ROLLOUTS, PassRate, measure_pass_rates
from .rollout import CollectedRollout, collect_rollouts
from .routing import Route, RoutingDecision
from .serve import (
    ServedModel,
    dump_serve_record,
    litellm_generate,
    serve_model,
    served_to_harbor_kwargs,
)
from .train import TrainConfig, TrainResult, run_training, write_train_report

DEFAULT_CANDIDATE_MODEL = "Qwen/Qwen3-8B"
PILOT_TASK_CAP = 10
MCNEMAR_BINARIZATION = "majority_of_rollouts_pass"  # PassRate.majority_pass


@dataclass
class ArmResult:
    arm: str
    description: str
    pass_rates: dict[str, PassRate]
    mean_rate: float | None
    n_tasks: int
    adapter_dir: str | None = None
    volume_adapter_path: str | None = None
    serve: dict[str, Any] | None = None
    train: dict[str, Any] | None = None
    cost_usd: float | None = None
    cost_per_solved: float | None = None
    error: str | None = None


def _mean_rate(rates: dict[str, PassRate]) -> float | None:
    vals = [pr.rate for pr in rates.values() if pr.rate is not None]
    return (sum(vals) / len(vals)) if vals else None


def _cost_per_solved(cost_usd: float | None, rates: dict[str, PassRate]) -> float | None:
    if cost_usd is None:
        return None
    solved = sum(1 for pr in rates.values() if pr.majority_pass())
    return (cost_usd / solved) if solved else None


def compare_arms_mcnemar(
    a3: dict[str, PassRate],
    a2: dict[str, PassRate],
) -> dict[str, Any]:
    """Paired A3 vs A2 over the shared held-out task set.

    Binarization (pre-declared): majority_of_rollouts_pass.
    Reuses diagnose._exact_mcnemar_p — do not invent a second test.
    """
    shared = sorted(set(a3) & set(a2))
    b = c = concordant = 0
    for task in shared:
        p3 = a3[task].majority_pass()
        p2 = a2[task].majority_pass()
        if p3 and not p2:
            b += 1
        elif p2 and not p3:
            c += 1
        else:
            concordant += 1
    p_value = None
    if b + c > 0:
        p_value = _exact_mcnemar_p(b, c)
    return {
        "binarization": MCNEMAR_BINARIZATION,
        "shared_tasks": len(shared),
        "a3_only": b,
        "a2_only": c,
        "concordant": concordant,
        "discordant_n": b + c,
        "p_value": p_value,
    }


def build_a1_prompt(diagnosis: dict[str, Any]) -> str:
    """Template A1's deficit-targeted instruction from stored diagnosis evidence."""
    chosen = diagnosis.get("chosen_deficit") or {}
    cap = (chosen.get("capability") or {}).get("name") or "the diagnosed deficit"
    evidence = chosen.get("evidence_summary") or ""
    return (
        f"You have previously struggled with: {cap}. {evidence} "
        "Pay particular attention to this."
    ).strip()


def _resolve_task_dirs(tasks_dir: Path, task_ids: list[str]) -> list[Path]:
    return [tasks_dir / t for t in task_ids if (tasks_dir / t).is_dir()]


def _cap_pilot(task_ids: list[str], *, pilot: bool, seed: int) -> list[str]:
    if not pilot or len(task_ids) <= PILOT_TASK_CAP:
        return list(task_ids)
    rng = random.Random(seed)
    picked = list(task_ids)
    rng.shuffle(picked)
    return sorted(picked[:PILOT_TASK_CAP])


def _tokenize_rollouts(rollouts: list[CollectedRollout], tokenizer: Any) -> list:
    from .dataset import TokenizedExample

    out: list[TokenizedExample] = []
    for r in rollouts:
        if not r.passed:
            continue
        ex = tokenize_sft_example(r.turns, tokenizer)
        if ex is not None:
            out.append(ex)
    return out


def _load_tokenizer(base_model: str):
    from .dataset import _require_train

    _require_train()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def select_a2_tasks(
    all_task_ids: list[str],
    *,
    exclude: set[str],
    pass_rates: dict[str, PassRate],
    n: int,
    seed: int,
    band: tuple[float, float],
) -> list[str]:
    """In-band random control, same count as A3 train.

    `exclude` must cover A3 train ids **and** the held-out set — training on
    holdout would contaminate the primary metric.
    """
    pool = [
        t
        for t in all_task_ids
        if t not in exclude and t in pass_rates and pass_rates[t].in_band(band)
    ]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:n])


@dataclass
class ArmsConfig:
    selection_path: Path
    diagnosis_path: Path
    tasks_dir: Path
    out_dir: Path
    agent: str
    candidate_model: str = DEFAULT_CANDIDATE_MODEL
    frontier_model: str | None = None
    rollouts: int = DEFAULT_ROLLOUTS
    seed: int = 0
    pilot: bool = False
    use_modal: bool = True
    modal_gpu: str = "A10G"
    max_train_steps: int = 50
    skip_nonregression: bool = False
    # Injectables for unit tests (monkeypatch-friendly).
    measure_fn: Callable[..., dict[str, PassRate]] | None = None
    collect_fn: Callable[..., list[CollectedRollout]] | None = None
    train_fn: Callable[..., TrainResult] | None = None
    serve_cm: Any = None
    nonregression_fn: Callable[..., NonRegressionResult] | None = None


def run_arms(cfg: ArmsConfig) -> dict[str, Any]:
    """Full A0–A4 orchestrator. Returns the arms report dict (also written to disk)."""
    selection = json.loads(cfg.selection_path.read_text())
    diagnosis = json.loads(cfg.diagnosis_path.read_text())

    holdout = list(selection.get("holdout") or [])
    train_ids = list(selection.get("train") or [])
    band = (
        float((selection.get("band") or {}).get("min", 0.10)),
        float((selection.get("band") or {}).get("max", 0.40)),
    )
    frontier = cfg.frontier_model or selection.get("frontier_model")
    candidate = cfg.candidate_model or selection.get("candidate_model") or DEFAULT_CANDIDATE_MODEL
    if not frontier:
        raise ValueError("frontier_model missing from selection.json and flags")

    holdout = _cap_pilot(holdout, pilot=cfg.pilot, seed=cfg.seed)
    train_ids = _cap_pilot(train_ids, pilot=cfg.pilot, seed=cfg.seed + 1)

    measure = cfg.measure_fn or measure_pass_rates
    collect = cfg.collect_fn or collect_rollouts
    train = cfg.train_fn or run_training
    serve_cm = cfg.serve_cm or serve_model

    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    holdout_dirs = _resolve_task_dirs(cfg.tasks_dir, holdout)
    train_dirs = _resolve_task_dirs(cfg.tasks_dir, train_ids)

    arms: dict[str, ArmResult] = {}
    nonregression_note: str | None = None

    # --- A0 + A1: open candidate must be Modal-served (not a named API model) ---
    prompt_text = build_a1_prompt(diagnosis)
    prompt_path = out / "a1_extra_instruction.md"
    prompt_path.write_text(prompt_text + "\n")

    with serve_cm(candidate, gpu=cfg.modal_gpu) as base_served:
        assert isinstance(base_served, ServedModel)
        hk = served_to_harbor_kwargs(base_served)

        a0_rates = measure(
            holdout_dirs,
            agent=cfg.agent,
            jobs_dir=out / "jobs" / "A0",
            rollouts=cfg.rollouts,
            **hk,
        )
        arms["A0"] = ArmResult(
            arm="A0",
            description="candidate, untouched",
            pass_rates=a0_rates,
            mean_rate=_mean_rate(a0_rates),
            n_tasks=len(a0_rates),
            serve=dump_serve_record(base_served),
        )

        a1_rates = measure(
            holdout_dirs,
            agent=cfg.agent,
            jobs_dir=out / "jobs" / "A1",
            rollouts=cfg.rollouts,
            extra_instruction_path=prompt_path,
            **hk,
        )
        arms["A1"] = ArmResult(
            arm="A1",
            description="candidate + deficit-targeted prompt",
            pass_rates=a1_rates,
            mean_rate=_mean_rate(a1_rates),
            n_tasks=len(a1_rates),
            serve=dump_serve_record(base_served),
        )

    # --- A4: frontier (API model — never Modal) ---
    a4_rates = measure(
        holdout_dirs,
        agent=cfg.agent,
        model=frontier,
        jobs_dir=out / "jobs" / "A4",
        rollouts=cfg.rollouts,
    )
    arms["A4"] = ArmResult(
        arm="A4",
        description="frontier",
        pass_rates=a4_rates,
        mean_rate=_mean_rate(a4_rates),
        n_tasks=len(a4_rates),
    )

    # --- A2 pre-measurement: need in-band rates on a pool that excludes
    #     A3 train AND holdout (holdout contamination would invalidate A3 vs A2). ---
    all_task_ids = sorted(
        p.name for p in cfg.tasks_dir.iterdir() if p.is_dir() and (p / "task.toml").exists()
    )
    exclude_a2 = set(train_ids) | set(holdout)
    a2_pool_ids = [t for t in all_task_ids if t not in exclude_a2]
    a2_pool_ids = _cap_pilot(a2_pool_ids, pilot=cfg.pilot, seed=cfg.seed + 2)
    a2_pool_dirs = _resolve_task_dirs(cfg.tasks_dir, a2_pool_ids)

    with serve_cm(candidate, gpu=cfg.modal_gpu) as pool_served:
        assert isinstance(pool_served, ServedModel)
        hk_pool = served_to_harbor_kwargs(pool_served)
        a2_pool_rates = measure(
            a2_pool_dirs,
            agent=cfg.agent,
            jobs_dir=out / "jobs" / "A2_pool",
            rollouts=cfg.rollouts,
            **hk_pool,
        )

    a2_train_ids = select_a2_tasks(
        a2_pool_ids,
        exclude=exclude_a2,
        pass_rates=a2_pool_rates,
        n=len(train_ids),
        seed=cfg.seed,
        band=band,
    )
    if len(a2_train_ids) < len(train_ids):
        raise RuntimeError(
            f"A2 in-band pool too small: need {len(train_ids)} tasks, found "
            f"{len(a2_train_ids)} after excluding A3-train+holdout. Mine more "
            "tasks or widen the band before claiming a control arm."
        )
    a2_train_dirs = _resolve_task_dirs(cfg.tasks_dir, a2_train_ids)

    def _train_arm(
        arm_name: str,
        task_dirs: list[Path],
        task_ids: list[str],
    ) -> ArmResult:
        with serve_cm(candidate, gpu=cfg.modal_gpu) as served:
            assert isinstance(served, ServedModel)
            hk = served_to_harbor_kwargs(served)
            rollouts = collect(
                task_dirs,
                agent=cfg.agent,
                jobs_dir=out / "jobs" / f"{arm_name}_rollout",
                rollouts=cfg.rollouts,
                **hk,
            )
            serve_rec = dump_serve_record(served)

        if not any(r.passed for r in rollouts):
            return ArmResult(
                arm=arm_name,
                description=(
                    "trained on random mined tasks"
                    if arm_name == "A2"
                    else "trained on deficit-selected mined tasks"
                ),
                pass_rates={},
                mean_rate=None,
                n_tasks=0,
                train={"task_ids": task_ids, "n_passing_rollouts": 0},
                serve=serve_rec,
                error="rejection sampling kept nothing — no adapter trained",
            )

        tokenizer = _load_tokenizer(candidate)
        examples = _tokenize_rollouts(rollouts, tokenizer)
        if not examples:
            return ArmResult(
                arm=arm_name,
                description=(
                    "trained on random mined tasks"
                    if arm_name == "A2"
                    else "trained on deficit-selected mined tasks"
                ),
                pass_rates={},
                mean_rate=None,
                n_tasks=0,
                train={
                    "task_ids": task_ids,
                    "n_passing_rollouts": sum(1 for r in rollouts if r.passed),
                },
                serve=serve_rec,
                error="passing rollouts had no parent-assistant turns to train on",
            )

        train_cfg = TrainConfig(
            base_model=candidate,
            output_dir=out / arm_name,
            task_ids=task_ids,
            max_steps=cfg.max_train_steps,
            seed=cfg.seed,
            use_modal=cfg.use_modal,
            modal_gpu=cfg.modal_gpu,
            arm=arm_name,
        )
        result = train(examples, train_cfg, tokenizer=tokenizer)
        write_train_report(result, out / arm_name)

        # Serve the *Volume* adapter path — local stub dirs only have a pointer.
        adapter_for_serve = result.volume_adapter_path or str(result.adapter_dir)
        with serve_cm(
            candidate, adapter_path=adapter_for_serve, gpu=cfg.modal_gpu
        ) as served2:
            assert isinstance(served2, ServedModel)
            hk2 = served_to_harbor_kwargs(served2)
            rates = measure(
                holdout_dirs,
                agent=cfg.agent,
                jobs_dir=out / "jobs" / arm_name,
                rollouts=cfg.rollouts,
                **hk2,
            )
            serve_rec2 = dump_serve_record(served2)

        return ArmResult(
            arm=arm_name,
            description=(
                "trained on random mined tasks"
                if arm_name == "A2"
                else "trained on deficit-selected mined tasks"
            ),
            pass_rates=rates,
            mean_rate=_mean_rate(rates),
            n_tasks=len(rates),
            adapter_dir=str(result.adapter_dir),
            volume_adapter_path=result.volume_adapter_path,
            serve=serve_rec2,
            train={
                "task_ids": task_ids,
                "n_passing_rollouts": sum(1 for r in rollouts if r.passed),
                "n_sft_examples": len(examples),
                "steps": result.steps,
                "final_loss": result.final_loss,
                "rollout_serve": serve_rec,
                "volume_adapter_path": result.volume_adapter_path,
            },
        )

    arms["A2"] = _train_arm("A2", a2_train_dirs, a2_train_ids)
    arms["A3"] = _train_arm("A3", train_dirs, train_ids)

    comparison = compare_arms_mcnemar(arms["A3"].pass_rates, arms["A2"].pass_rates)

    # --- Non-regression (IFEval): re-serve base + each trained arm (URLs die
    #     when the earlier serve_cm contexts exit). ---
    nr_results: list[NonRegressionResult] = []
    if cfg.nonregression_fn is not None:
        for arm_name in ("A2", "A3"):
            nr_results.append(cfg.nonregression_fn(arm=arm_name))  # type: ignore[call-arg]
    elif not cfg.skip_nonregression:
        trained = [
            arms[n] for n in ("A2", "A3") if arms[n].volume_adapter_path and not arms[n].error
        ]
        if not trained:
            nonregression_note = "non-regression skipped: no trained adapters"
        else:
            try:
                subset = load_ifeval_subset()
                prompts = [r.get("prompt") or r.get("instruction") or "" for r in subset]
                with serve_cm(candidate, gpu=cfg.modal_gpu) as base_nr:
                    assert isinstance(base_nr, ServedModel)
                    baseline_responses = [litellm_generate(base_nr, p) for p in prompts]
                # Serve contexts must not nest — base is torn down before arms.
                for ar in trained:
                    with serve_cm(
                        candidate,
                        adapter_path=ar.volume_adapter_path,
                        gpu=cfg.modal_gpu,
                    ) as arm_served:
                        assert isinstance(arm_served, ServedModel)
                        arm_responses = [litellm_generate(arm_served, p) for p in prompts]
                    # evaluate_ifeval wants a generate_fn; feed cached responses.
                    def _mk(resps: list[str]):
                        it = iter(resps)
                        return lambda _p: next(it)

                    baseline_acc = evaluate_ifeval(subset, _mk(baseline_responses))
                    arm_acc = evaluate_ifeval(subset, _mk(arm_responses))
                    delta = arm_acc - baseline_acc
                    nr_results.append(
                        NonRegressionResult(
                            subset_size=len(subset),
                            seed=0,
                            tolerance=IFEVAL_TOLERANCE,
                            baseline_strict_acc=baseline_acc,
                            arm_strict_acc=arm_acc,
                            delta=delta,
                            passed=delta >= -IFEVAL_TOLERANCE,
                            arm=ar.arm,
                        )
                    )
            except Exception as e:
                nr_results = []
                nonregression_note = f"non-regression skipped: {e}"
    else:
        nonregression_note = "non-regression skipped (--skip / test mode)"

    for a in arms.values():
        a.cost_per_solved = _cost_per_solved(a.cost_usd, a.pass_rates)

    report = write_arms_report(
        out,
        arms=arms,
        comparison=comparison,
        non_regression=nr_results,
        meta={
            "candidate_model": candidate,
            "frontier_model": frontier,
            "agent": cfg.agent,
            "seed": cfg.seed,
            "pilot": cfg.pilot,
            "pilot_cap": PILOT_TASK_CAP if cfg.pilot else None,
            "rollouts": cfg.rollouts,
            "holdout": holdout,
            "a3_train": train_ids,
            "a2_train": a2_train_ids,
            "a2_excluded": sorted(exclude_a2),
            "mcnemar_binarization": MCNEMAR_BINARIZATION,
            "ifeval_tolerance": IFEVAL_TOLERANCE,
            "modal_gpu": cfg.modal_gpu,
            "band": {"min": band[0], "max": band[1]},
            "nonregression_note": nonregression_note,
        },
    )
    return report


def write_arms_report(
    out_dir: Path,
    *,
    arms: dict[str, ArmResult],
    comparison: dict[str, Any],
    non_regression: list[NonRegressionResult],
    meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _arm_json(a: ArmResult) -> dict[str, Any]:
        return {
            "arm": a.arm,
            "description": a.description,
            "mean_rate": a.mean_rate,
            "n_tasks": a.n_tasks,
            "pass_rates": {
                t: {
                    "passed": pr.passed,
                    "n": pr.n,
                    "rate": pr.rate,
                    "majority": pr.majority_pass(),
                }
                for t, pr in sorted(a.pass_rates.items())
            },
            "adapter_dir": a.adapter_dir,
            "volume_adapter_path": a.volume_adapter_path,
            "serve": a.serve,
            "train": a.train,
            "cost_usd": a.cost_usd,
            "cost_per_solved": a.cost_per_solved,
            "error": a.error,
        }

    report = {
        **meta,
        "arms": {k: _arm_json(v) for k, v in arms.items()},
        "comparison": {
            "a3_vs_a2": comparison,
            "non_regression": [asdict(r) for r in non_regression],
        },
    }
    json_path = out_dir / "arms.json"
    json_path.write_text(json.dumps(report, indent=2))

    lines = ["# Vektori-trace A0–A4 arms\n"]
    lines.append(
        f"Scaffold: `{meta.get('agent')}`  ·  candidate: `{meta.get('candidate_model')}`  ·  "
        f"frontier: `{meta.get('frontier_model')}`  ·  seed: {meta.get('seed')}\n"
    )
    if meta.get("pilot"):
        lines.append(f"**Pilot mode** — task cap {meta.get('pilot_cap')}.\n")
    lines.append(
        f"McNemar binarization (pre-declared): `{meta.get('mcnemar_binarization')}`.\n"
    )
    lines.append(f"Environment: Modal GPU `{meta.get('modal_gpu')}`.\n")
    lines.append("\n## Pass rates (held-out)\n")
    for key in ("A0", "A1", "A2", "A3", "A4"):
        a = arms[key]
        rate = "n/a" if a.mean_rate is None else f"{a.mean_rate:.2%}"
        err = f"  ⚠ {a.error}" if a.error else ""
        lines.append(
            f"- **{key}** ({a.description}): {rate} over N={a.n_tasks} tasks{err}"
        )
        if a.cost_per_solved is not None:
            lines.append(f"  - cost/solved: ${a.cost_per_solved:.4f}")
    lines.append("\n## A3 vs A2 (paired McNemar)\n")
    lines.append(
        f"- shared tasks: {comparison['shared_tasks']}\n"
        f"- A3-only passes: {comparison['a3_only']}\n"
        f"- A2-only passes: {comparison['a2_only']}\n"
        f"- concordant: {comparison['concordant']}\n"
        f"- discordant N: {comparison['discordant_n']}\n"
        f"- p-value: {comparison['p_value']}\n"
    )
    lines.append("\n## Non-regression (IFEval)\n")
    if meta.get("nonregression_note"):
        lines.append(f"{meta['nonregression_note']}\n")
    if not non_regression:
        lines.append(
            f"Tolerance pre-declared at {IFEVAL_TOLERANCE:.0%} absolute drop; "
            "not measured this run.\n"
        )
    for r in non_regression:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"- `{r.arm}`: {r.arm_strict_acc:.2%} (baseline {r.baseline_strict_acc:.2%}, "
            f"delta={r.delta:+.2%}) [{status}]"
        )
    md_path = out_dir / "arms.md"
    md_path.write_text("\n".join(lines) + "\n")
    report["_paths"] = {"json": str(json_path), "md": str(md_path)}
    return report


# --- B1–B4 (PLAN.md Stage 7) -------------------------------------------------
# B1 routed: RL on in-support, OPD on out-of-support
# B2 anti-routed: identical tasks/compute/method mix, assignments inverted
# B3 RL on everything
# B4 OPD on everything


CELL_SEP = "::"


def cell_key(task: str, capability: str) -> str:
    """Stable JSON-safe key for a (task × capability) routing cell."""
    return f"{task}{CELL_SEP}{capability}"


def split_cell_key(key: str) -> tuple[str, str]:
    task, _, capability = key.partition(CELL_SEP)
    return task, capability


@dataclass
class BArmPlan:
    """Per-arm cell→method assignment. Resolvable effect size recorded before train.

    The assignment unit is the (task × capability) cell, not the task: PLAN.md
    routes per cell, and a task can be RL for one capability and OPD for another.
    Keying by task collapses those into whichever decision was seen last, which
    silently drops rows from the very method mix B2 has to hold identical.
    """

    arm: str
    description: str
    assignments: dict[str, Route]  # cell_key(task, capability) → RL|OPD
    task_ids: list[str]
    method_mix: dict[str, int]
    resolvable_effect_size: float | None = None
    cells: list[tuple[str, str]] = field(default_factory=list)


def _invert_route(route: Route) -> Route:
    if route == "RL":
        return "OPD"
    if route == "OPD":
        return "RL"
    return route


def plan_b_arms(
    decisions: list[RoutingDecision],
    *,
    holdout: list[str] | None = None,
    resolvable_effect_size: float | None = None,
    pilot: bool = False,
    seed: int = 0,
    exclude_not_preregistered: bool = False,
) -> dict[str, BArmPlan]:
    """Build B1–B4 assignment plans from routing decisions.

    Only RL/OPD cells are trainable. Quarantine/NONE excluded from all B arms.
    B2 inverts RL↔OPD on the same *cell* set (identical method mix counts).

    Assignments are keyed by `cell_key(task, capability)`, because the routing
    unit is the (task × capability) cell: one task may be RL for one capability
    and OPD for another, and a task-keyed map keeps only whichever decision came
    last. `task_ids` remains the deduplicated task set (what pilot-capping and
    holdout removal operate on); `cells` lists the (task, capability) pairs.

    `holdout` is the *evaluation* slice and is therefore **removed** from every
    training arm — training on it is the contamination the split exists to
    prevent. `exclude_not_preregistered` drops cells decided by a rule outside
    the pre-registration (currently the mid-band extension), so the B1-vs-B2
    comparison can be re-run on registered cells only.
    """
    trainable = [d for d in decisions if d.route in ("RL", "OPD")]
    if exclude_not_preregistered:
        trainable = [d for d in trainable if d.preregistered]
    if holdout is not None:
        hold = set(holdout)
        trainable = [d for d in trainable if d.task not in hold]
    task_ids = _cap_pilot(
        sorted({d.task for d in trainable}),
        pilot=pilot,
        seed=seed,
    )
    kept = set(task_ids)
    # Ordered by cell, deduplicated on (task, capability). Two decisions for the
    # same cell would be a routing bug upstream; the first is kept and the shape
    # here makes the collision visible rather than absorbing it as it did when the
    # map was keyed by task alone.
    by_cell: dict[str, RoutingDecision] = {}
    for d in trainable:
        if d.task not in kept:
            continue
        by_cell.setdefault(cell_key(d.task, d.capability), d)
    cell_keys = sorted(by_cell)
    cells = [split_cell_key(k) for k in cell_keys]

    b1_assign = {k: by_cell[k].route for k in cell_keys}
    b2_assign = {k: _invert_route(by_cell[k].route) for k in cell_keys}
    b3_assign = {k: "RL" for k in cell_keys}  # type: ignore[dict-item]
    b4_assign = {k: "OPD" for k in cell_keys}  # type: ignore[dict-item]

    def _mix(assign: dict[str, Route]) -> dict[str, int]:
        return {
            "RL": sum(1 for r in assign.values() if r == "RL"),
            "OPD": sum(1 for r in assign.values() if r == "OPD"),
        }

    # B2 holds task count and method-mix identical and permutes only the
    # assignment — that is what isolates the routing decision. True by
    # construction here (`_invert_route` is a bijection on the trainable routes),
    # so it is stated as an invariant rather than tested as if it could vary.

    return {
        "B1": BArmPlan(
            arm="B1",
            description="routed: RL on in-support, OPD on out-of-support",
            assignments=b1_assign,
            task_ids=task_ids,
            method_mix=_mix(b1_assign),
            resolvable_effect_size=resolvable_effect_size,
            cells=cells,
        ),
        "B2": BArmPlan(
            arm="B2",
            description="anti-routed: identical tasks/compute/mix, assignments inverted",
            assignments=b2_assign,
            task_ids=task_ids,
            method_mix=_mix(b2_assign),
            resolvable_effect_size=resolvable_effect_size,
            cells=cells,
        ),
        "B3": BArmPlan(
            arm="B3",
            description="RL on everything",
            assignments=b3_assign,
            task_ids=task_ids,
            method_mix=_mix(b3_assign),
            resolvable_effect_size=resolvable_effect_size,
            cells=cells,
        ),
        "B4": BArmPlan(
            arm="B4",
            description="OPD on everything",
            assignments=b4_assign,
            task_ids=task_ids,
            method_mix=_mix(b4_assign),
            resolvable_effect_size=resolvable_effect_size,
            cells=cells,
        ),
    }


def compare_b1_b2_paired(
    b1_rates: dict[str, PassRate],
    b2_rates: dict[str, PassRate],
    *,
    resolvable_effect_size: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Primary metric: pass@1 on held-out slice, B1 vs B2, paired task-by-task.

    Reports an exact paired test, not just a mean delta. Discordant tasks (one
    arm solves it, the other does not) are the paired evidence; under H0 each is
    a coin flip, so the same exact McNemar `diagnose._exact_mcnemar_p` used for
    the cross-model contrast applies here. Do not invent a second test.

    `resolvable_effect_size` is PLAN.md's pre-registered smallest effect the
    slice can resolve (AC #8). When the observed delta is smaller than it, the
    result is reported as **unresolvable** regardless of p — a slice sized for a
    5-point effect saying nothing about a 1-point one is a fact about the design,
    not a finding. When it is absent, that absence is itself flagged: the number
    was supposed to be recorded before training.
    """
    shared = sorted(set(b1_rates) & set(b2_rates))
    deltas = []
    b1_wins = b2_wins = ties = 0
    for task in shared:
        r1 = b1_rates[task].rate
        r2 = b2_rates[task].rate
        if r1 is None or r2 is None:
            continue
        deltas.append(r1 - r2)
        if r1 > r2:
            b1_wins += 1
        elif r2 > r1:
            b2_wins += 1
        else:
            ties += 1
    mean_delta = (sum(deltas) / len(deltas)) if deltas else None

    # Paired exact test on the discordant tasks — same binarization the A-arms
    # use, so the two comparisons stay commensurable.
    b = c = 0
    for task in shared:
        p1, p2 = b1_rates[task].majority_pass(), b2_rates[task].majority_pass()
        if p1 and not p2:
            b += 1
        elif p2 and not p1:
            c += 1
    discordant = b + c
    p_value = _exact_mcnemar_p(b, c) if discordant > 0 else None

    resolvable = None
    interpretation = ""
    if resolvable_effect_size is None:
        interpretation = (
            "no pre-registered resolvable effect size — PLAN.md AC #8 requires it "
            "to be recorded before training, so this comparison cannot be "
            "interpreted as designed"
        )
    elif mean_delta is not None:
        resolvable = abs(mean_delta) >= resolvable_effect_size
        if not resolvable:
            interpretation = (
                f"observed |delta|={abs(mean_delta):.4f} is below the "
                f"pre-registered resolvable effect {resolvable_effect_size} — "
                "underpowered, no claim either way"
            )
    if discordant < MIN_DISCORDANT_PAIRS:
        floor_note = (
            f"only {discordant} discordant pair(s) — below the "
            f"{MIN_DISCORDANT_PAIRS} floor at which any paired test can reach "
            "significance, whatever the data"
        )
        interpretation = f"{interpretation}; {floor_note}" if interpretation else floor_note

    return {
        "shared_tasks": len(shared),
        "paired_n": len(deltas),
        "mean_pass1_delta_b1_minus_b2": mean_delta,
        "b1_wins": b1_wins,
        "b2_wins": b2_wins,
        "ties": ties,
        "binarization": MCNEMAR_BINARIZATION,
        "b1_only": b,
        "b2_only": c,
        "discordant_n": discordant,
        "p_value": p_value,
        "alpha": alpha,
        "significant": (p_value is not None and p_value < alpha),
        "resolvable_effect_size": resolvable_effect_size,
        "effect_is_resolvable": resolvable,
        "interpretation": interpretation,
        "per_task": {
            t: {
                "b1": b1_rates[t].rate,
                "b2": b2_rates[t].rate,
                "delta": (
                    None
                    if b1_rates[t].rate is None or b2_rates[t].rate is None
                    else b1_rates[t].rate - b2_rates[t].rate
                ),
            }
            for t in shared
        },
    }


__all__ = [
    "DEFAULT_CANDIDATE_MODEL",
    "MCNEMAR_BINARIZATION",
    "PILOT_TASK_CAP",
    "ArmResult",
    "ArmsConfig",
    "BArmPlan",
    "build_a1_prompt",
    "cell_key",
    "compare_arms_mcnemar",
    "compare_b1_b2_paired",
    "plan_b_arms",
    "run_arms",
    "select_a2_tasks",
    "split_cell_key",
    "write_arms_report",
]
