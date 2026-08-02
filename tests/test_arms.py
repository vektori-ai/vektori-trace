"""A0–A4 orchestration tests — monkeypatched I/O, mirroring test_cli_replay.py.

Covers A2's pre-measurement step, holdout exclusion from A2 pool, majority
McNemar binarization, A1 prompt templating, and --pilot capping.
No Docker/Modal/GPU.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from vektori_trace.arms import (
    MCNEMAR_BINARIZATION,
    PILOT_TASK_CAP,
    ArmsConfig,
    build_a1_prompt,
    compare_arms_mcnemar,
    run_arms,
    select_a2_tasks,
)
from vektori_trace.passrate import PassRate
from vektori_trace.rollout import CollectedRollout
from vektori_trace.schema import Turn
from vektori_trace.serve import ServedModel
from vektori_trace.train import TrainResult


def _pr(task: str, passed: int, n: int) -> PassRate:
    return PassRate(task=task, passed=passed, n=n)


def test_mcnemar_binarization_is_majority_and_predeclared() -> None:
    assert MCNEMAR_BINARIZATION == "majority_of_rollouts_pass"
    assert _pr("a", 5, 8).majority_pass()
    assert not _pr("a", 4, 8).majority_pass()
    assert not _pr("a", 3, 8).majority_pass()


def test_compare_arms_mcnemar_pairs_by_task() -> None:
    a3 = {"t1": _pr("t1", 7, 8), "t2": _pr("t2", 1, 8), "t3": _pr("t3", 5, 8)}
    a2 = {"t1": _pr("t1", 1, 8), "t2": _pr("t2", 7, 8), "t3": _pr("t3", 5, 8)}
    out = compare_arms_mcnemar(a3, a2)
    assert out["binarization"] == MCNEMAR_BINARIZATION
    assert out["a3_only"] == 1
    assert out["a2_only"] == 1
    assert out["concordant"] == 1
    assert out["p_value"] is not None


def test_build_a1_prompt_templates_from_diagnosis_evidence() -> None:
    diagnosis = {
        "chosen_deficit": {
            "capability": {"name": "prefix-aware retries"},
            "evidence_summary": "Ignored the AccessDenied prefix hint.",
        }
    }
    text = build_a1_prompt(diagnosis)
    assert "prefix-aware retries" in text
    assert "AccessDenied" in text
    assert "Pay particular attention" in text


def test_select_a2_tasks_excludes_holdout_and_a3_train() -> None:
    rates = {
        "a": _pr("a", 2, 10),
        "b": _pr("b", 3, 10),
        "c": _pr("c", 2, 10),
        "hold": _pr("hold", 2, 10),  # in band but held out — must not train
        "d": _pr("d", 9, 10),  # out of band
    }
    picked = select_a2_tasks(
        ["a", "b", "c", "hold", "d"],
        exclude={"a", "hold"},
        pass_rates=rates,
        n=2,
        seed=0,
        band=(0.10, 0.40),
    )
    assert "a" not in picked
    assert "hold" not in picked
    assert "d" not in picked
    assert len(picked) == 2


def test_run_arms_orchestration_with_mocks(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    for name in ("h1", "h2", "tr1", "rand1", "rand2"):
        d = tasks_dir / name
        d.mkdir(parents=True)
        (d / "task.toml").write_text("x = 1")

    selection = {
        "frontier_model": "gpt-5",
        "candidate_model": "Qwen/Qwen3-8B",
        "band": {"min": 0.10, "max": 0.40},
        "train": ["tr1"],
        "holdout": ["h1", "h2"],
    }
    diagnosis = {
        "chosen_deficit": {
            "capability": {"name": "cap", "id": "c", "description": "d"},
            "evidence_summary": "evidence line",
            "lacking_loss_run_ids": [],
        }
    }
    sel_path = tmp_path / "selection.json"
    diag_path = tmp_path / "diagnosis.json"
    sel_path.write_text(json.dumps(selection))
    diag_path.write_text(json.dumps(diagnosis))

    measured_jobs: list[str] = []

    def fake_measure(task_dirs, agent, model=None, jobs_dir=None, rollouts=8, **kw):
        measured_jobs.append(str(jobs_dir))
        return {p.name: _pr(p.name, 2, 8) for p in task_dirs}

    def fake_collect(task_dirs, agent, model=None, jobs_dir=None, rollouts=8, **kw):
        return [
            CollectedRollout(
                task=p.name,
                passed=True,
                reward=1.0,
                turns=[
                    Turn(index=0, role="user", content="go"),
                    Turn(index=1, role="assistant", content="done"),
                ],
            )
            for p in task_dirs
        ]

    def fake_train(examples, config, **kw):
        adapter = config.output_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}")
        return TrainResult(
            adapter_dir=adapter,
            base_model=config.base_model,
            task_ids=config.task_ids,
            steps=1,
            final_loss=0.1,
            lora={},
            seed=0,
            volume_adapter_path=f"/adapters/fake/{config.arm or 'arm'}/adapter",
        )

    @contextmanager
    def fake_serve(base_model, adapter_path=None, gpu="A10", model_info=None):
        yield ServedModel(
            api_base="http://example.test/v1",
            model_name="Qwen3-8B",
            base_model=base_model,
            adapter_path=adapter_path,
            gpu=gpu,
        )

    import vektori_trace.arms as arms_mod

    original_load = arms_mod._load_tokenizer
    original_tok = arms_mod.tokenize_sft_example
    arms_mod._load_tokenizer = lambda model: object()  # type: ignore

    def fake_tokenize(turns, tokenizer, max_length=4096):
        from vektori_trace.dataset import TokenizedExample

        return TokenizedExample([1, 2, 3], [-100, 2, 3], [1, 1, 1])

    arms_mod.tokenize_sft_example = fake_tokenize  # type: ignore
    try:
        report = run_arms(
            ArmsConfig(
                selection_path=sel_path,
                diagnosis_path=diag_path,
                tasks_dir=tasks_dir,
                out_dir=tmp_path / "out",
                agent="terminus-2",
                candidate_model="Qwen/Qwen3-8B",
                frontier_model="gpt-5",
                rollouts=2,
                seed=0,
                pilot=False,
                use_modal=False,
                skip_nonregression=True,
                measure_fn=fake_measure,
                collect_fn=fake_collect,
                train_fn=fake_train,
                serve_cm=fake_serve,
            )
        )
    finally:
        arms_mod._load_tokenizer = original_load
        arms_mod.tokenize_sft_example = original_tok

    assert (tmp_path / "out" / "arms.json").is_file()
    assert (tmp_path / "out" / "arms.md").is_file()
    assert set(report["arms"]) == {"A0", "A1", "A2", "A3", "A4"}
    assert any("A2_pool" in j for j in measured_jobs)
    # Holdout must be in the A2 exclude list.
    assert "h1" in report["a2_excluded"] and "h2" in report["a2_excluded"]
    assert "tr1" in report["a2_excluded"]
    a1_prompt = (tmp_path / "out" / "a1_extra_instruction.md").read_text()
    assert "evidence line" in a1_prompt
    assert report["mcnemar_binarization"] == MCNEMAR_BINARIZATION
    assert report["comparison"]["a3_vs_a2"]["binarization"] == MCNEMAR_BINARIZATION


def test_pilot_caps_task_count() -> None:
    from vektori_trace.arms import _cap_pilot

    ids = [f"t{i}" for i in range(30)]
    capped = _cap_pilot(ids, pilot=True, seed=0)
    assert len(capped) == PILOT_TASK_CAP
    assert _cap_pilot(ids, pilot=False, seed=0) == ids


def test_api_base_forces_use_modal_off(tmp_path) -> None:
    """`api_base` and `use_modal` must not be able to disagree.

    `run_arms` already keys `stage_to_volume` off `api_base`. If a programmatic
    caller supplies only `api_base` and leaves `use_modal` at its `True` default,
    the arm trains on Modal *and* skips the Volume upload — `volume_adapter_path`
    comes back None, the serve falls back to a local stub dir, and the endpoint is
    handed a path it cannot read, after the training has been paid for. The CLI
    derives this correctly; the coupling belongs where the field lives.
    """
    cfg = ArmsConfig(
        selection_path=tmp_path / "sel.json",
        diagnosis_path=tmp_path / "diag.json",
        tasks_dir=tmp_path / "tasks",
        out_dir=tmp_path / "out",
        agent="terminus-2",
        api_base="http://10.0.0.5:8000/v1",
        # Left at its default on purpose — this is the disagreement under test.
    )
    assert cfg.use_modal is False

    # No endpoint means Modal stays the default, unchanged.
    assert ArmsConfig(
        selection_path=tmp_path / "sel.json",
        diagnosis_path=tmp_path / "diag.json",
        tasks_dir=tmp_path / "tasks",
        out_dir=tmp_path / "out",
        agent="terminus-2",
    ).use_modal is True
