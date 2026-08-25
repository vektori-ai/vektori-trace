"""Guardrails on the live Tau2 evaluation runner.

These test the refusals, not the rollouts. A rollout needs a GPU and a
simulator; a refusal is what stands between a typo and either a burned blind
test or an unapproved GPU bill, and it is cheap to prove.

`scripts/` is not a package, so the module is loaded by path.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# `serve_model` pulls in modal at import time. It is a real dependency of the
# script but not of these tests, so skip rather than fail if it is absent.
pytest.importorskip("modal")

_spec = importlib.util.spec_from_file_location(
    "tau2_eval_modal", REPO / "scripts" / "tau2_eval_modal.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tau2_eval_modal"] = _mod
_spec.loader.exec_module(_mod)


@pytest.fixture
def artifacts(tmp_path):
    """A manifest shaped like the frozen one, with disjoint partitions."""
    manifest = {
        "manifest_hash": "b741bfceb1f3d027",
        "partitions": {
            # Disjoint by construction, with the four contaminated diagnostics
            # held out of the trainable halves the way the real split does.
            "W30": [str(i) for i in range(1, 31)],
            "C30": [str(i) for i in range(31, 57)] + ["58", "59", "60", "74"],
            "S16": ["57", "73", "75", "93"] + [str(i) for i in range(61, 73)],
            "F38": [str(i) for i in range(76, 114)],
        },
    }
    (tmp_path / "task_split_manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_f38_is_refused(artifacts):
    """The blind final test is never reachable from a selection script.

    This is the one mistake here that rerunning cannot undo: once F38 has been
    looked at, it is no longer the frozen comparison V2 §11 reports.
    """
    with pytest.raises(SystemExit) as e:
        _mod._resolve_tasks(artifacts, "F38")
    assert "frozen final test" in str(e.value)


def test_f38_refused_even_though_the_manifest_has_it(artifacts):
    """The refusal is by name, not by absence -- F38 ids are right there."""
    manifest = json.load(open(Path(artifacts) / "task_split_manifest.json"))
    assert len(manifest["partitions"]["F38"]) == 38
    with pytest.raises(SystemExit):
        _mod._resolve_tasks(artifacts, "F38")


@pytest.mark.parametrize("partition,n", [("S16", 16), ("C30", 30), ("W30", 30)])
def test_evaluable_partitions_resolve(artifacts, partition, n):
    ids, man_hash = _mod._resolve_tasks(artifacts, partition)
    assert len(ids) == n
    assert man_hash == "b741bfceb1f3d027"


def test_unknown_partition_is_refused(artifacts):
    with pytest.raises(SystemExit) as e:
        _mod._resolve_tasks(artifacts, "Z9")
    assert "unknown partition" in str(e.value)


def test_missing_manifest_does_not_get_regenerated(tmp_path):
    """A manifest rebuilt under an evaluation is a different experiment."""
    with pytest.raises(SystemExit) as e:
        _mod._resolve_tasks(str(tmp_path), "S16")
    assert "task_split_manifest.json" in str(e.value)


def test_selection_and_test_partitions_stay_disjoint(artifacts):
    """S16 selects and C30 trains; an overlap would leak one into the other."""
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    c30, _ = _mod._resolve_tasks(artifacts, "C30")
    w30, _ = _mod._resolve_tasks(artifacts, "W30")
    assert not set(s16) & set(c30)
    assert not set(s16) & set(w30)
    assert not set(c30) & set(w30)


def test_contaminated_diagnostics_are_in_s16(artifacts):
    """57/73/75/93 influenced design and may never be in the blind test."""
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    assert {"57", "73", "75", "93"} <= set(s16)


def test_prompt_budget_clears_the_measured_p90():
    """The pinned budget, not the default arithmetic.

    `serve_model`'s default would clamp a 16,384 window to 8,192 in / 8,192
    out. Retail final-turn prompts reach p90 ~11,942 (handoff §4.3), so the
    default starves the prompt and the resulting context refusals are graded as
    model failures. 14,336 clears that p90 with room.
    """
    info = _mod._model_info()
    assert info["max_input_tokens"] == 14336
    assert info["max_output_tokens"] == 2048
    # input + output must fit the window together -- they share it.
    assert info["max_input_tokens"] + info["max_output_tokens"] == _mod.MAX_MODEL_LEN
    assert info["max_input_tokens"] > 11942


def test_generation_reserve_clears_the_measured_p99():
    """Measured generation p99 <= 344 across all domains."""
    assert _mod.GENERATION_RESERVE > 344


def test_tool_call_parser_is_hermes():
    """qwen3_coder extracted nothing from a Qwen3 dense model: 0 calls parsed.

    Without a working parser every episode fails on format rather than
    capability, which reads as a model result and is not one.
    """
    assert "--tool-call-parser" in _mod.VLLM_ARGS
    assert _mod.VLLM_ARGS[_mod.VLLM_ARGS.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-auto-tool-choice" in _mod.VLLM_ARGS


def test_arms_map_suffixes_to_checkpoint_dirs():
    arms = _mod._arms("/vol/tau2/runs/a_warm", ["35", "70", "105"])
    assert arms == {
        "ck35": "/vol/tau2/runs/a_warm/checkpoint-35",
        "ck70": "/vol/tau2/runs/a_warm/checkpoint-70",
        "ck105": "/vol/tau2/runs/a_warm/checkpoint-105",
    }


def test_arms_tolerates_a_trailing_slash():
    arms = _mod._arms("/vol/tau2/runs/a_warm/", ["35"])
    assert arms["ck35"] == "/vol/tau2/runs/a_warm/checkpoint-35"


def test_no_arm_collides_with_the_base_served_name():
    """A suffix equal to the base name resolves every request to base weights.

    `serve_model` raises on the collision; the point here is that the suffixes
    this script generates cannot trigger it.
    """
    arms = _mod._arms("/vol/tau2/runs/a_warm", ["35", "70", "105"])
    assert all(s.startswith("ck") for s in arms)
    assert _mod._canonical_base_name() not in arms


def test_spend_tolerates_a_partial_write(tmp_path):
    """tau2 rewrites results as it goes; a poll can land mid-write."""
    p = tmp_path / "r.json"
    p.write_text('{"simulations": [{"agent_cost": 0.1, "user_c')
    assert _mod._spend_so_far(p) is None


def test_spend_sums_both_sides(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"simulations": [
        {"agent_cost": 0.01, "user_cost": 0.02},
        {"agent_cost": 0.03, "user_cost": None},
    ]}))
    assert _mod._spend_so_far(p) == pytest.approx(0.06)


def test_spend_of_a_missing_file_is_none(tmp_path):
    assert _mod._spend_so_far(tmp_path / "nope.json") is None


def test_named_tasks_must_belong_to_the_partition(artifacts):
    """An id from another partition is refused, not silently run.

    A two-task probe is only safe if the two tasks are still selection tasks.
    A typo reaching a test task is the one mistake here that rerunning cannot
    undo.
    """
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    with pytest.raises(SystemExit) as e:
        _mod._subset(s16, "S16", ["73", "31"], None)   # 31 is C30
    assert "not in S16" in str(e.value)


def test_a_test_task_cannot_be_smuggled_in_by_name(artifacts):
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    with pytest.raises(SystemExit):
        _mod._subset(s16, "S16", ["99"], None)         # 99 is F38


def test_named_tasks_keep_caller_order_and_dedupe(artifacts):
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    assert _mod._subset(s16, "S16", ["75", "73", "75"], None) == ["75", "73"]


def test_max_tasks_takes_a_manifest_order_prefix(artifacts):
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    assert _mod._subset(s16, "S16", None, 2) == s16[:2]


def test_max_tasks_rejects_a_nonsense_count(artifacts):
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    with pytest.raises(SystemExit):
        _mod._subset(s16, "S16", None, 0)


def test_no_subset_flags_runs_the_whole_partition(artifacts):
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    assert _mod._subset(s16, "S16", None, None) == s16


def test_baseline_covered_tasks_are_available_for_a_probe(artifacts):
    """57/73/75/93 have audited 4B/8B trajectories from 2026-08-24.

    Probing over these means the old baseline stays usable as context instead
    of being re-paid for as an A0 arm.
    """
    s16, _ = _mod._resolve_tasks(artifacts, "S16")
    assert _mod._subset(s16, "S16", ["73", "75"], None) == ["73", "75"]


def test_default_run_is_under_the_real_volume_mount():
    """A wrong prefix fails only at serve time, after the run is committed.

    `_resolve_volume_adapter` accepts a path solely if it starts with
    VOLUME_MOUNT; anything else it tries to stage as a local directory and
    raises "neither a Volume path nor a local adapter dir". This was written as
    "/vol/..." from prose and got all the way to a launch before failing.
    """
    from vektori_trace.runtime.modal_env import VOLUME_MOUNT

    assert _mod.DEFAULT_RUN.startswith(VOLUME_MOUNT + "/")


def test_every_generated_adapter_path_is_a_volume_path():
    from vektori_trace.runtime.modal_env import VOLUME_MOUNT

    arms = _mod._arms(_mod.DEFAULT_RUN, ["35", "70", "105"])
    assert all(p.startswith(VOLUME_MOUNT + "/") for p in arms.values())


def test_generated_paths_survive_the_resolver_prefix_check():
    """The exact condition `_resolve_volume_adapter` branches on."""
    from vektori_trace.runtime.modal_env import VOLUME_MOUNT

    for path in _mod._arms(_mod.DEFAULT_RUN, ["35"]).values():
        assert path.startswith(VOLUME_MOUNT)
