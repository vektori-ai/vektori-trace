"""Step E grounding, Step G ReOPD, Step H OPD/GRPO, Step I B-arms."""

from __future__ import annotations

import pytest

from vektori_trace.arms import compare_b1_b2_paired, plan_b_arms
from vektori_trace.evaluate.passrate import PassRate
from vektori_trace.grounding import GroundingPair, ground_diagnosis
from vektori_trace.opd import (
    OPDConfig,
    blended_loss,
    build_branch_spec,
    grpo_advantage,
    mean_log_ratio,
)
from vektori_trace.reopd import (
    InMemoryTeacherPool,
    build_reopd_example,
    iter_reopd_examples,
)
from vektori_trace.routing import CurveSummary, route_cell
from vektori_trace.schema import Turn
from vektori_trace.tokenizer_check import TokenizerMismatchError


def test_grounding_suggests_min_gap() -> None:
    pairs = [
        GroundingPair("t1", "cap", 3, True, diagnose_gap=0.4),
        GroundingPair("t2", "cap", 5, False, diagnose_gap=0.55),
        GroundingPair("t3", "cap", 2, True, diagnose_gap=0.3),
    ]
    report = ground_diagnosis(pairs, current_min_gap=0.2, agreement_floor=0.9)
    assert report.agreement_rate == pytest.approx(2 / 3)
    assert report.suggested_min_gap == pytest.approx(0.55)


def test_reopd_prefix_split() -> None:
    turns = [
        Turn(0, "user", content="fix"),
        Turn(1, "assistant", content="step1"),
        Turn(2, "tool", content="ok"),
        Turn(3, "assistant", content="step2"),
    ]
    ex = build_reopd_example(turns, task="t", action_index=1)
    assert ex.teacher_action_turn.content == "step2"
    assert ex.prefix_turns[-1].role == "tool"
    assert ex.step_index == 1
    assert ex.later_teacher_turns == []


def test_iter_reopd_examples() -> None:
    turns = [
        Turn(0, "user", content="q"),
        Turn(1, "assistant", content="a1"),
        Turn(2, "assistant", content="a2"),
    ]
    xs = list(iter_reopd_examples([("t", turns)]))
    assert len(xs) == 2


def test_teacher_pool_prompt_logprobs() -> None:
    pool = InMemoryTeacherPool(logprob=-0.2)
    assert pool.prompt_logprobs("p", [1, 2, 3]) == [-0.2, -0.2, -0.2]


def test_reverse_kl_and_grpo() -> None:
    assert mean_log_ratio([-1.0, -2.0], [-0.5, -1.0]) == pytest.approx(-0.75)
    adv = grpo_advantage([0.0, 1.0, 0.0, 1.0])
    assert sum(adv) == pytest.approx(0.0)
    assert blended_loss(1.0, 0.5, distillation_loss_coef=2.0) == 2.0


def test_build_opd_branch_checks_tokenizer(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise TokenizerMismatchError("nope")

    monkeypatch.setattr("vektori_trace.opd.check_tokenizers", boom)
    with pytest.raises(TokenizerMismatchError):
        build_branch_spec("OPD", opd=OPDConfig(), verify_tokenizers=True)


def test_opd_spec_requires_a_prompt_logprobs_endpoint() -> None:
    """PLAN.md C1: OPD scores supplied tokens, so the teacher must be
    self-hosted. A spec with no endpoint declares a loss it cannot compute."""
    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        build_branch_spec("OPD", opd=OPDConfig(), verify_tokenizers=False)

    spec = build_branch_spec(
        "OPD",
        opd=OPDConfig(teacher_endpoint="http://vllm:8000/v1"),
        verify_tokenizers=False,
    )
    assert spec.teacher_endpoint == "http://vllm:8000/v1"


def test_build_rl_branch_no_teacher() -> None:
    spec = build_branch_spec("RL", verify_tokenizers=False)
    assert spec.branch == "RL"
    assert spec.extra["loss"] == "grpo"


def test_plan_b_arms_invert_mix() -> None:
    decisions = [
        route_cell(
            "a",
            "cap",
            CurveSummary(0.1, 0.4, 32, 10),
            CurveSummary(0.9, 0.9, 32, 28),
        ),
        route_cell(
            "b",
            "cap",
            CurveSummary(0.0, 0.0, 32, 0),
            CurveSummary(0.9, 0.9, 32, 28),
        ),
        route_cell(
            "c",
            "cap",
            CurveSummary(0.0, 0.0, 32, 0),
            CurveSummary(0.0, 0.0, 32, 0),
        ),
    ]
    plans = plan_b_arms(decisions, resolvable_effect_size=0.05)
    assert set(plans) == {"B1", "B2", "B3", "B4"}
    assert "c" not in plans["B1"].task_ids  # quarantined
    assert plans["B1"].method_mix["RL"] == plans["B2"].method_mix["OPD"]
    assert plans["B1"].method_mix["OPD"] == plans["B2"].method_mix["RL"]
    assert plans["B1"].resolvable_effect_size == 0.05
    assert all(r == "RL" for r in plans["B3"].assignments.values())
    assert all(r == "OPD" for r in plans["B4"].assignments.values())


def test_compare_b1_b2_paired() -> None:
    b1 = {
        "a": PassRate("a", 3, 4),
        "b": PassRate("b", 1, 4),
    }
    b2 = {
        "a": PassRate("a", 1, 4),
        "b": PassRate("b", 2, 4),
    }
    cmp_ = compare_b1_b2_paired(b1, b2)
    assert cmp_["paired_n"] == 2
    assert cmp_["b1_wins"] == 1
    assert cmp_["b2_wins"] == 1


def test_reverse_kl_surrogate_gradient_matches_the_stated_formula() -> None:
    """PLAN.md / docs/OPD.md: OPD's gradient is Σ (log π_s − log π_t) ∇log π_s.
    A plain mean of logprob differences is not that objective and gives the
    wrong gradient, so check the analytic form directly."""
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import reverse_kl_surrogate

    logits = torch.randn(2, 5, 7, requires_grad=True)
    tokens = torch.randint(0, 7, (2, 5))
    teacher = torch.randn(2, 5)

    from vektori_trace.opd import token_logprobs

    student_lp = token_logprobs(logits, tokens)
    loss = reverse_kl_surrogate(student_lp, teacher)
    loss.backward()
    got = logits.grad.clone()

    # Analytic: d/dθ of mean[(log π_s − log π_t).detach() * log π_s]
    logits2 = logits.detach().clone().requires_grad_(True)
    lp2 = token_logprobs(logits2, tokens)
    coef = (lp2.detach() - teacher)
    (coef * lp2).mean().backward()
    assert torch.allclose(got, logits2.grad, atol=1e-6)
    # Nonzero even though no outcome/reward is involved — the property that
    # distinguishes OPD from RL when every rollout fails.
    assert got.abs().sum() > 0


def test_reverse_kl_surrogate_respects_the_prefix_mask() -> None:
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import mask_from_labels, reverse_kl_surrogate, token_logprobs

    logits = torch.randn(1, 4, 6, requires_grad=True)
    tokens = torch.randint(0, 6, (1, 4))
    teacher = torch.randn(1, 4)
    # First two positions are the student prefix — masked out of the loss.
    labels = tokens.clone()
    labels[0, :2] = -100

    student_lp = token_logprobs(logits, tokens)
    reverse_kl_surrogate(student_lp, teacher, mask_from_labels(labels)).backward()
    grad = logits.grad
    assert torch.count_nonzero(grad[0, :2]) == 0, "masked prefix must not carry gradient"
    assert torch.count_nonzero(grad[0, 2:]) > 0


def test_reverse_kl_surrogate_shape_mismatch_is_loud() -> None:
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import reverse_kl_surrogate

    with pytest.raises(ValueError, match="shape mismatch"):
        reverse_kl_surrogate(torch.zeros(1, 4), torch.zeros(1, 5))


def test_fully_masked_batch_is_a_no_op_not_a_crash() -> None:
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import reverse_kl_surrogate

    student = torch.zeros(1, 3, requires_grad=True)
    loss = reverse_kl_surrogate(student, torch.zeros(1, 3), torch.zeros(1, 3))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.count_nonzero(student.grad) == 0


def test_b_arms_holdout_is_excluded_from_training() -> None:
    """The held-out slice is the evaluation set — training the B arms on it is
    the contamination the split exists to prevent."""
    from vektori_trace.routing import CurveSummary as _CS
    from vektori_trace.routing import route_cell as _rc

    decisions = [
        _rc("train1", "cap", _CS(0.0, 0.0, 32, 0), _CS(0.9, 1.0, 32, 30)),
        _rc("hold1", "cap", _CS(0.0, 0.0, 32, 0), _CS(0.9, 1.0, 32, 30)),
    ]
    plans = plan_b_arms(decisions, holdout=["hold1"])
    assert plans["B1"].task_ids == ["train1"]


def test_b_arms_can_drop_non_preregistered_cells() -> None:
    from vektori_trace.routing import CurveSummary as _CS
    from vektori_trace.routing import route_cell as _rc

    decisions = [
        _rc("registered", "cap", _CS(0.0, 0.0, 32, 0), _CS(0.9, 1.0, 32, 30)),
        _rc("midband", "cap", _CS(0.5, 0.8, 32, 16), _CS(0.9, 1.0, 32, 30)),
    ]
    assert sorted(plan_b_arms(decisions)["B1"].task_ids) == ["midband", "registered"]
    strict = plan_b_arms(decisions, exclude_not_preregistered=True)
    assert strict["B1"].task_ids == ["registered"]


def test_preregistered_flag_is_derived_not_hardcoded() -> None:
    from vektori_trace.routing import PREREGISTERED_RULES, CurveSummary, RoutingDecision

    d = RoutingDecision(
        task="t",
        capability="c",
        route="RL",
        student=CurveSummary(None, None),
        teacher=CurveSummary(None, None),
        thresholds={},
        rule="R7_mid_band_extension",
    )
    assert d.preregistered is False
    d2 = RoutingDecision(
        task="t",
        capability="c",
        route="RL",
        student=CurveSummary(None, None),
        teacher=CurveSummary(None, None),
        thresholds={},
        rule="R1_in_support_unreliable",
    )
    assert d2.preregistered is True
    assert "R1_in_support_unreliable" in PREREGISTERED_RULES


def test_b_arms_assign_per_cell_not_per_task() -> None:
    """PLAN.md routes per (task × capability). One task can be RL for capability
    A and OPD for capability B; a task-keyed assignment map keeps only whichever
    decision came last, so B2's "identical method mix, assignments inverted" is
    computed off dropped rows."""
    from vektori_trace.arms import cell_key

    decisions = [
        # Same task, two capabilities, opposite routes.
        route_cell(  # in support, unreliable → RL
            "shared",
            "capA",
            CurveSummary(0.1, 0.4, 32, 10),
            CurveSummary(0.9, 0.9, 32, 28),
        ),
        route_cell(  # outside student support → OPD
            "shared",
            "capB",
            CurveSummary(0.0, 0.0, 32, 0),
            CurveSummary(0.9, 0.9, 32, 28),
        ),
    ]
    assert [d.route for d in decisions] == ["RL", "OPD"]

    plans = plan_b_arms(decisions)
    b1, b2 = plans["B1"], plans["B2"]

    # Both cells survive, keyed by cell rather than task.
    assert b1.task_ids == ["shared"]
    assert sorted(b1.cells) == [("shared", "capA"), ("shared", "capB")]
    assert b1.assignments == {
        cell_key("shared", "capA"): "RL",
        cell_key("shared", "capB"): "OPD",
    }
    assert b2.assignments == {
        cell_key("shared", "capA"): "OPD",
        cell_key("shared", "capB"): "RL",
    }

    # The method mix counts both cells — one of each, inverted, not one row.
    assert b1.method_mix == {"RL": 1, "OPD": 1}
    assert b2.method_mix == {"RL": 1, "OPD": 1}
    assert sum(b1.method_mix.values()) == len(decisions)
    assert plans["B3"].assignments == dict.fromkeys(b1.assignments, "RL")
    assert plans["B4"].assignments == dict.fromkeys(b1.assignments, "OPD")


def test_b_arm_plan_survives_json_round_trip() -> None:
    """cmd_plan_b_arms writes `assignments` straight to JSON — the cell key has
    to be a string, and it has to decode back to (task, capability)."""
    import json

    from vektori_trace.arms import split_cell_key

    decisions = [
        route_cell("t1", "cap x", CurveSummary(0.0, 0.0, 32, 0), CurveSummary(0.9, 0.9, 32, 28)),
        route_cell("t2", "cap y", CurveSummary(0.1, 0.4, 32, 10), CurveSummary(0.9, 0.9, 32, 28)),
    ]
    plan = plan_b_arms(decisions)["B1"]
    decoded = json.loads(json.dumps(plan.assignments))
    assert {split_cell_key(k) for k in decoded} == {("t1", "cap x"), ("t2", "cap y")}
    assert set(decoded.values()) == {"OPD", "RL"}


def _rates(**kw) -> dict:
    from vektori_trace.evaluate.passrate import PassRate

    return {t: PassRate(task=t, passed=p, n=n) for t, (p, n) in kw.items()}


def test_b1_b2_comparison_reports_a_paired_test_not_just_a_mean() -> None:
    """PLAN.md AC #8: B1 vs B2 paired task-by-task. A mean delta with no test
    cannot distinguish a real effect from noise on a 50-task slice."""
    b1 = _rates(t1=(8, 8), t2=(8, 8), t3=(8, 8), t4=(8, 8), t5=(0, 8), t6=(8, 8))
    b2 = _rates(t1=(0, 8), t2=(0, 8), t3=(0, 8), t4=(0, 8), t5=(8, 8), t6=(0, 8))
    out = compare_b1_b2_paired(b1, b2, resolvable_effect_size=0.1)
    assert out["b1_only"] == 5
    assert out["b2_only"] == 1
    assert out["discordant_n"] == 6
    assert out["p_value"] is not None
    assert out["binarization"] == "majority_of_rollouts_pass"


def test_effect_below_the_preregistered_size_is_unresolvable() -> None:
    b1 = _rates(t1=(5, 8), t2=(4, 8))
    b2 = _rates(t1=(4, 8), t2=(4, 8))
    out = compare_b1_b2_paired(b1, b2, resolvable_effect_size=0.25)
    assert out["effect_is_resolvable"] is False
    assert "below the pre-registered resolvable effect" in out["interpretation"]

    big = compare_b1_b2_paired(b1, b2, resolvable_effect_size=0.01)
    assert big["effect_is_resolvable"] is True


def test_missing_resolvable_effect_size_is_flagged_not_ignored() -> None:
    out = compare_b1_b2_paired(_rates(t1=(8, 8)), _rates(t1=(0, 8)))
    assert out["resolvable_effect_size"] is None
    assert out["effect_is_resolvable"] is None
    assert "AC #8" in out["interpretation"] or "before training" in out["interpretation"]


def test_too_few_discordant_pairs_is_reported_as_a_floor() -> None:
    out = compare_b1_b2_paired(
        _rates(t1=(8, 8), t2=(8, 8)),
        _rates(t1=(0, 8), t2=(8, 8)),
        resolvable_effect_size=0.01,
    )
    assert out["discordant_n"] == 1
    assert "floor at which any paired test" in out["interpretation"]
