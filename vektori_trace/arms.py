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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import tokenize_sft_example
from .diagnose import _exact_mcnemar_p
from .nonregression import (
    IFEVAL_TOLERANCE,
    NonRegressionResult,
    evaluate_ifeval,
    load_ifeval_subset,
)
from .passrate import DEFAULT_ROLLOUTS, PassRate, measure_pass_rates
from .rollout import CollectedRollout, collect_rollouts
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


__all__ = [
    "DEFAULT_CANDIDATE_MODEL",
    "MCNEMAR_BINARIZATION",
    "PILOT_TASK_CAP",
    "ArmResult",
    "ArmsConfig",
    "build_a1_prompt",
    "compare_arms_mcnemar",
    "run_arms",
    "select_a2_tasks",
    "write_arms_report",
]
