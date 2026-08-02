"""Integration: replay → pass@k → bisection → route → report (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from vektori_trace.evaluate.intervene import bisect_forking_step
from vektori_trace.evaluate.passk import RolloutOutcome, compute_task_passk
from vektori_trace.evaluate.resume import replay_prefix
from vektori_trace.routing import CurveSummary, decision_to_dict, route_cell
from vektori_trace.schema import ToolCall, Turn


class _SB:
    def exec(self, cmd: str, *, timeout: int = 120):
        class R:
            ok = True
            exit_code = 0
            stdout = "diff --git a/f b/f\n+x"
            stderr = ""

        return R()


def test_end_to_end_fixture_pipeline(tmp_path: Path) -> None:
    turns = [
        Turn(
            0,
            "assistant",
            tool_calls=[ToolCall("1", "bash", {"cmd": "echo 1"})],
        ),
        Turn(
            1,
            "assistant",
            tool_calls=[ToolCall("2", "bash", {"cmd": "echo 2"})],
        ),
        Turn(
            2,
            "assistant",
            tool_calls=[ToolCall("3", "bash", {"cmd": "echo 3"})],
        ),
    ]
    # Resume
    expected = "diff --git a/f b/f\n+x"
    resume = replay_prefix(turns, T=1, sandbox=_SB(), expected_diff=expected)
    assert resume.consistent

    # pass@k from synthetic rollouts
    outcomes = [RolloutOutcome("taskA", False, "stage1") for _ in range(8)]
    curves = compute_task_passk(outcomes)
    pr = curves[("taskA", "stage1")]
    assert pr.c == 0

    # Bisection
    def cont(_t, T: int) -> bool:
        return T < 1

    bisect = bisect_forking_step(turns, continue_with_teacher=cont, samples_per_probe=1)
    assert bisect.forking_step == 1

    # Route
    decision = route_cell(
        "taskA",
        "cap",
        CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
        CurveSummary(pass1=0.9, pass32=0.95, n32=32, c32=30),
    )
    assert decision.route == "OPD"

    report = {
        "resume": {"T": resume.T, "consistent": resume.consistent},
        "passk": {"n": pr.n, "c": pr.c, "curves": {str(k): v for k, v in pr.curves.items()}},
        "bisection": {
            "forking_step": bisect.forking_step,
            "monotone": bisect.monotone,
            "teacher_continuations": bisect.teacher_continuations,
        },
        "routing": decision_to_dict(decision),
    }
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report, indent=2))
    loaded = json.loads(out.read_text())
    assert loaded["routing"]["route"] == "OPD"


def _passk_report(model: str, rows: dict[str, tuple[int, int]]) -> dict:
    from vektori_trace.evaluate.passk import TaskPassK, _task_to_dict

    return {
        "model": model,
        "agent": "a",
        "stage1_n": 8,
        "stage2_n": 32,
        "stage1": {},
        "stage2": {
            t: _task_to_dict(TaskPassK.from_counts(t, n, c, stratum="stage2"))
            for t, (n, c) in rows.items()
        },
        "escalated": sorted(rows),
        "luck_quarantine": [],
    }


def test_route_cli_builds_task_by_capability_cells(tmp_path: Path, monkeypatch) -> None:
    """PLAN.md routes (task × capability). One blanket --capability label makes
    the per-capability counts meaningless, so the cells come from diagnose.py."""
    from vektori_trace.cli import build_parser, cmd_route

    (tmp_path / "s.json").write_text(
        json.dumps(_passk_report("student", {"task1": (32, 0), "task2": (32, 4)}))
    )
    (tmp_path / "t.json").write_text(
        json.dumps(_passk_report("teacher", {"task1": (32, 28), "task2": (32, 30)}))
    )
    (tmp_path / "diag.json").write_text(
        json.dumps(
            {
                "all_deficits_ranked": [
                    {
                        "capability": {"id": "cap_a", "name": "A"},
                        "lacking_loss_run_ids": ["r1"],
                    },
                    {
                        "capability": {"id": "cap_b", "name": "B"},
                        "lacking_loss_run_ids": ["r1", "r2"],
                    },
                ]
            }
        )
    )

    class _T:
        def __init__(self, run_id: str, task: str):
            self.run_id, self.task = run_id, task

    monkeypatch.setattr(
        # Patch where it is read, not where it is defined: `cmd_route` binds
        # `_load_traces` into its own module namespace at import time, so
        # patching the `vektori_trace.cli` re-export would silently do nothing.
        "vektori_trace.cli.commands.route._load_traces",
        lambda _p: [_T("r1", "task1"), _T("r2", "task2")],
    )

    args = build_parser().parse_args(
        [
            "route",
            "--student-passk",
            str(tmp_path / "s.json"),
            "--teacher-passk",
            str(tmp_path / "t.json"),
            "--diagnosis",
            str(tmp_path / "diag.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert cmd_route(args) == 0

    report = json.loads((tmp_path / "out" / "routing.json").read_text())
    cells = {(d["task"], d["capability"]): d["route"] for d in report["decisions"]}
    # task1 is LACKING in both capabilities → two cells; task2 in one.
    assert cells == {
        ("task1", "cap_a"): "OPD",
        ("task1", "cap_b"): "OPD",
        ("task2", "cap_b"): "RL",
    }
    assert report["counts"]["by_capability"]["cap_a"]["OPD"] == 1
    assert report["counts"]["by_capability"]["cap_b"] == {
        "RL": 1,
        "OPD": 1,
        "QUARANTINE": 0,
        "NONE": 0,
    }
    assert "R2_outside_student_support" in report["rules"]


def test_route_cli_rejects_diagnosis_without_manifest(tmp_path: Path) -> None:
    from vektori_trace.cli import build_parser, cmd_route

    (tmp_path / "s.json").write_text(json.dumps(_passk_report("s", {"t1": (32, 0)})))
    (tmp_path / "t.json").write_text(json.dumps(_passk_report("t", {"t1": (32, 30)})))
    (tmp_path / "diag.json").write_text(json.dumps({"all_deficits_ranked": []}))
    args = build_parser().parse_args(
        [
            "route",
            "--student-passk",
            str(tmp_path / "s.json"),
            "--teacher-passk",
            str(tmp_path / "t.json"),
            "--diagnosis",
            str(tmp_path / "diag.json"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert cmd_route(args) == 2


def test_two_stage_sweep_reports_support_capabilities_and_infra(tmp_path: Path, monkeypatch) -> None:
    """AC #2: every task carries a support classification; PLAN.md's aggregation
    unit is (capability, model); infra failures are reported, not vanished."""
    from vektori_trace.evaluate import passk as passk_mod

    plan = {
        "in_support": [True] + [False] * 7,   # passes at stage 1
        "outside": [False] * 8,               # escalates, still 0
        "infra": [None] * 8,                  # never gradeable
    }
    calls: dict[str, int] = {}

    # The real TrialResult, not a `.passed`-only stub: the sweep now records the
    # job dir, reward and timing behind every rollout, and a stub that omits them
    # would let a missing field pass here and fail on the box.
    from vektori_trace.evaluate.validity import TrialResult

    def fake_run_trial(task_dir, agent="a", jobs_dir=None, **kw):
        name = task_dir.name
        i = calls.get(name, 0)
        calls[name] = i + 1
        passed = plan[name][i % len(plan[name])]
        return TrialResult(
            agent=agent,
            passed=passed,
            reward=None if passed is None else float(passed),
            jobs_dir=Path(jobs_dir) if jobs_dir else tmp_path / "jobs" / name,
            raw_stdout="",
            elapsed_sec=0.1,
            started_at=1_700_000_000.0,
        )

    monkeypatch.setattr(passk_mod, "run_trial", fake_run_trial)
    dirs = []
    for name in plan:
        d = tmp_path / name
        d.mkdir()
        dirs.append(d)

    report = passk_mod.two_stage_sweep(
        dirs,
        agent="a",
        model="m",
        jobs_dir=tmp_path / "jobs",
        stage1_n=8,
        stage2_n=32,
        task_to_capability={"in_support": "capA", "outside": "capA"},
    )

    assert report["support"] == {
        "in_support": "in_support",
        "outside": "outside_support",
        "infra": "no_rollouts",
    }
    assert report["escalated"] == ["outside"]
    assert report["no_gradeable_rollouts"] == ["infra"]
    assert report["infra_failures"]["infra"] == 8
    agg = report["by_capability"]["stage1"]["capA"]
    assert agg["N"] == 2 and agg["model"] == "m"
    assert agg["mean_curves"]["1"] is not None


def test_plan_b_arms_cli_preserves_the_rule_tag(tmp_path: Path) -> None:
    """Reloading routing.json must not launder a non-pre-registered cell into a
    pre-registered one."""
    from vektori_trace.cli import build_parser, cmd_plan_b_arms
    from vektori_trace.routing import CurveSummary, decision_to_dict, route_cell

    decisions = [
        route_cell(
            "midband",
            "cap",
            CurveSummary(pass1=0.5, pass32=0.8, n32=32, c32=16),
            CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        ),
        route_cell(
            "registered",
            "cap",
            CurveSummary(pass1=0.0, pass32=0.0, n32=32, c32=0),
            CurveSummary(pass1=0.9, pass32=1.0, n32=32, c32=30),
        ),
    ]
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps({"decisions": [decision_to_dict(d) for d in decisions]}))

    args = build_parser().parse_args(
        ["plan-b-arms", "--routing", str(routing), "--out", str(tmp_path / "out")]
    )
    assert cmd_plan_b_arms(args) == 0

    from vektori_trace.arms import plan_b_arms
    from vektori_trace.cli import _reload_routing_decisions

    reloaded = _reload_routing_decisions(json.loads(routing.read_text()))
    by_task = {d.task: d for d in reloaded}
    assert by_task["midband"].rule == "R7_mid_band_extension"
    assert by_task["midband"].preregistered is False
    assert plan_b_arms(reloaded, exclude_not_preregistered=True)["B1"].task_ids == [
        "registered"
    ]
