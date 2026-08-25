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


# --- checkpoint ordering -------------------------------------------------

def test_checkpoints_run_in_numeric_not_string_order():
    """sorted() puts ck105 first because "1" < "3" as strings.

    Arms run sequentially, so on 2026-08-25 a hung ck105 consumed the whole
    session and ck35/ck70 -- the checkpoints selection cared about -- never
    started.
    """
    assert _mod._ck_order(["ck35", "ck70", "ck105"]) == ["ck35", "ck70", "ck105"]
    assert sorted(["ck35", "ck70", "ck105"]) == ["ck105", "ck35", "ck70"]


def test_ck_order_is_stable_regardless_of_input_order():
    for perm in (["ck105", "ck35", "ck70"], ["ck70", "ck105", "ck35"]):
        assert _mod._ck_order(perm) == ["ck35", "ck70", "ck105"]


def test_ck_order_tolerates_a_non_numeric_suffix():
    assert _mod._ck_order(["ck70", "final", "ck35"]) == ["ck35", "ck70", "final"]


# --- simulator role gate -------------------------------------------------

def test_real_role_confused_openings_are_caught():
    """Verbatim from stored task-57 runs (Qwen3-14B and Qwen3-8B trial 1)."""
    for text in (
        "Hello! I'd be happy to help you with your order. Could you please "
        "provide me with your account details or order number",
        "Hi! I'd be happy to help you with your order. Could you please "
        "provide your account email or order details",
        "Hi! I'd be happy to help you with your order. Could you please "
        "provide your order number",
    ):
        assert _mod._role_confused(text) is not None


def test_real_valid_openings_are_not_flagged():
    """Verbatim customer openings from the same stored runs."""
    for text in (
        "Hi, I'm calling about my order W4284542. Can you tell me when it's "
        "supposed to arrive?",
        "Hi, I'd like to check on my order W4284542. Can you tell me when "
        "it's supposed to arrive, and if it's already been shipped?",
        "Hi, I need to exchange the laptop I just received. I want to swap "
        "it for one with better specs.",
    ):
        assert _mod._role_confused(text) is None


def test_role_gate_flags_a_confused_sim_despite_reward_one(tmp_path):
    """The Qwen3-14B case: role-confused opening, official reward 1.0.

    Reward cannot filter these -- the simulator drifts into the wrong role and
    still walks the agent to the expected DB state.
    """
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"simulations": [{
        "task_id": "57", "trial": 0,
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user",
             "content": "Hello! I'd be happy to help you with your order."},
        ],
    }]}))
    bad = _mod._audit_simulator_roles(p)
    assert len(bad) == 1
    assert bad[0]["reward"] == 1.0
    assert bad[0]["task_id"] == "57"


def test_role_gate_ignores_the_scripted_agent_greeting(tmp_path):
    """Index 0 is always the agent's own greeting; only the USER turn counts."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"simulations": [{
        "task_id": "93", "trial": 0,
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Hi, I need to exchange a laptop."},
        ],
    }]}))
    assert _mod._audit_simulator_roles(p) == []


def test_role_gate_tolerates_a_partial_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"simulations": [{"task_id": "5')
    assert _mod._audit_simulator_roles(p) == []


def test_stall_timeout_exceeds_the_slowest_observed_episode():
    """The watchdog must not be stricter than one healthy episode.

    Run 93c (2026-08-25) killed two good arms because the detector watched the
    *results* file, which tau2 only writes when a simulation COMPLETES. A
    task-93 episode runs 400+ s, so a healthy conversation looked idle. ck70 was
    at "Step 12" and still progressing when it was shot.

    Measured episode durations: task 73/75 ~200 s, task 93 ~427 s (ck105),
    task 57 up to ~225 s. The default must clear the slowest of those, since a
    per-episode signal cannot distinguish "slow" from "hung".
    """
    ap = _mod.argparse.ArgumentParser()
    # Mirror the real parser's default rather than trusting a literal here.
    src = (REPO / "scripts" / "tau2_eval_modal.py").read_text()
    assert '"--stall-timeout", type=float, default=' in src
    default = float(src.split('"--stall-timeout", type=float, default=')[1]
                    .split(",")[0].split(")")[0])
    # DeepSeek does task 93 in 43 s; ck105 took 427 s for the same task -- a 4B
    # student is ~10x slower per turn, so the margin has to be generous or a
    # slower checkpoint gets killed mid-conversation.
    assert default > 2 * 427, (
        f"--stall-timeout default {default}s leaves too little margin over the "
        f"427s task-93 episode measured for ck105; a slower checkpoint would be "
        f"killed mid-conversation"
    )


def test_progress_is_measured_per_turn_not_per_episode():
    """The watchdog must poll tau2's own output, not the results file.

    Results move once per completed episode -- coarser than any timeout worth
    setting. tau2's stderr moves every orchestrator step, which is what
    separates a slow episode from a hung provider call.
    """
    src = (REPO / "scripts" / "tau2_eval_modal.py").read_text()
    assert "progress_log.stat().st_size" in src
    assert "results.stat().st_size" not in src, (
        "stall detection is back on the results file; that only updates once "
        "per completed simulation and kills healthy slow arms"
    )


def test_user_simulator_timeout_is_passed_through():
    """tau2 passes no timeout to litellm, which then defaults to 600s x 3 retries.

    One dropped Fireworks connection therefore stalls a whole run for ~40 min
    with no log output (litellm retries internally; tau2 logs only on the final
    exception). `--user-llm-args` is json.loads'd straight into
    litellm.completion, so these land as real kwargs.
    """
    src = (REPO / "scripts" / "tau2_eval_modal.py").read_text()
    assert '"timeout": a.user_timeout' in src
    assert '"num_retries": a.user_retries' in src
    # It REPLACES tau2's defaults, so temperature must be restated or user
    # sampling silently changes.
    assert '"temperature": 0.0,' in src


def test_progress_log_is_defined_before_it_is_opened():
    """Catch the UnboundLocalError class of bug without paying for a boot.

    2026-08-25: `progress_fh = open(progress_log, ...)` sat ABOVE
    `progress_log = ...`. Every CPU test passed, the dry-run passed, and the arm
    died with UnboundLocalError *after* a 106 s vLLM boot -- the failure lives
    inside `with serve_model(...)`, which no CPU test reaches.

    A line-order check is crude but exactly matches the defect: the name must be
    bound before the open() that consumes it.
    """
    src = (REPO / "scripts" / "tau2_eval_modal.py").read_text().splitlines()
    define = next(i for i, l in enumerate(src)
                  if "progress_log = results_dir" in l)
    open_at = next(i for i, l in enumerate(src) if "progress_fh = open(" in l)
    assert define < open_at, (
        f"progress_log is opened on line {open_at + 1} but only defined on line "
        f"{define + 1}; this raises UnboundLocalError after the GPU has booted"
    )


def test_progress_file_is_wired_into_the_subprocess():
    """The watchdog's signal only exists if tau2's output actually goes there."""
    src = (REPO / "scripts" / "tau2_eval_modal.py").read_text()
    assert "stdout=progress_fh" in src
    assert "stderr=subprocess.STDOUT" in src


# --- runtime: actually execute the paid arm-launch path ----------------------
#
# The tests above inspect source strings. That is why 39 of them passed while
# the program still died with UnboundLocalError *after* a 132 s boot: no test
# had ever executed the block inside `with serve_model(...)`.
#
# These run main() for real with serve_model, Popen and the endpoint check
# mocked, so the arm loop executes on CPU in milliseconds.

class _FakeProc:
    """A tau2 subprocess that writes progress and exits after N polls."""

    def __init__(self, progress_path, polls_before_exit=1, rc=0, write=True):
        self._left = polls_before_exit
        self._rc = rc
        self._path = Path(progress_path)
        self._write = write
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._left <= 0:
            self.returncode = self._rc
            return self._rc
        self._left -= 1
        if self._write:
            # Simulate tau2 emitting an orchestrator step.
            with open(self._path, "ab") as fh:
                fh.write(b"Step. Sending message from Role.AGENT to Role.USER\n")
        return None

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else -15
        return self.returncode


class _FakeServed:
    api_base = "http://fake/v1"
    model_name = "Qwen3-4B"
    adapter_models = {"Qwen3-4B-ck35": "/adapters/x/checkpoint-35",
                      "Qwen3-4B-ck70": "/adapters/x/checkpoint-70"}


@pytest.fixture
def runtime_env(monkeypatch, artifacts, tmp_path):
    """Run main()'s paid path on CPU: no Modal, no GPU, no tau2, no network."""
    import contextlib

    tau2_dir = tmp_path / "tau2"
    (tau2_dir / "data" / "simulations").mkdir(parents=True)
    (tau2_dir / ".env").write_text("FIREWORKS_API_KEY=x\n")
    # main() checks the tau2 entry point exists on a real run.
    bin_dir = Path(sys.executable).parent
    monkeypatch.setattr(_mod.shutil, "which", lambda _n: str(bin_dir / "python"))

    @contextlib.contextmanager
    def _fake_serve(*_args, **_kw):
        yield _FakeServed()

    monkeypatch.setattr(_mod, "serve_model", _fake_serve)
    monkeypatch.setattr(_mod, "require_endpoint_model", lambda *_a, **_k: None)
    monkeypatch.setattr(_mod.time, "sleep", lambda _s: None)
    return {"artifacts": artifacts, "tau2_dir": str(tau2_dir)}


def _argv(env, *extra):
    return ["tau2_eval_modal.py", "--artifacts", env["artifacts"],
            "--tau2-dir", env["tau2_dir"], "--tasks", "93",
            "--checkpoints", "35", "70", "--no-base", "--yes", *extra]


def test_arm_launch_path_executes_without_unbound_locals(runtime_env, monkeypatch):
    """The regression test for run 93d.

    Executes the real arm loop -- the code that only runs after a paid boot.
    Before the fix this raised UnboundLocalError on `open(progress_log)`.
    """
    made = []

    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        # stdout must be the progress file handle, or the watchdog is blind.
        assert stdout is not None and hasattr(stdout, "write")
        p = _FakeProc(stdout.name, polls_before_exit=1)
        made.append(p)
        return p

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(sys, "argv", _argv(runtime_env))
    rc = _mod.main()
    assert rc == 0
    assert len(made) == 2, "both checkpoint arms should have launched"


def test_arms_launch_in_numeric_order_at_runtime(runtime_env, monkeypatch):
    """ck35 before ck70 -- not sorted() order, which puts ck105 first."""
    order = []

    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        order.append([c for c in cmd if c.startswith("hosted_vllm/")][0])
        return _FakeProc(stdout.name, polls_before_exit=1)

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(sys, "argv", _argv(runtime_env))
    _mod.main()
    assert order == ["hosted_vllm/Qwen3-4B-ck35", "hosted_vllm/Qwen3-4B-ck70"]


def test_a_progressing_arm_is_not_killed_by_the_watchdog(runtime_env, monkeypatch):
    """A slow-but-progressing arm must survive.

    This is run 93c's failure: ck35 and ck70 were at "Step 12" and still
    progressing when a results-file-based watchdog shot them.
    """
    procs = []

    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        # Writes on every poll -- i.e. healthy -- for many polls.
        p = _FakeProc(stdout.name, polls_before_exit=50)
        procs.append(p)
        return p

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(sys, "argv", _argv(runtime_env, "--stall-timeout", "900"))
    _mod.main()
    assert all(not p.terminated for p in procs), (
        "a progressing arm was killed; the watchdog is measuring the wrong thing"
    )


def test_infrastructure_failure_aborts_the_remaining_arms(runtime_env, monkeypatch):
    """A stall is a session property, not a checkpoint property.

    Run 93c burned two arms back to back on the same wall. The arms share one
    endpoint and one user-simulator provider, so continuing spends GPU minutes
    to collect another ungradeable result.
    """
    procs = []

    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        # write=False -> the progress file never grows -> looks stalled.
        p = _FakeProc(stdout.name, polls_before_exit=50, write=False)
        procs.append(p)
        return p

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    # A tiny stall budget so the fake stalls immediately.
    monkeypatch.setattr(sys, "argv", _argv(runtime_env, "--stall-timeout", "0"))
    _mod.main()
    assert len(procs) == 1, (
        f"{len(procs)} arms launched; the second should have been skipped after "
        f"the first failed for infrastructure reasons"
    )
    assert procs[0].terminated


def test_continue_after_infra_overrides_the_abort(runtime_env, monkeypatch):
    """The override exists for when you deliberately want every arm attempted."""
    procs = []

    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        p = _FakeProc(stdout.name, polls_before_exit=50, write=False)
        procs.append(p)
        return p

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(sys, "argv", _argv(runtime_env, "--stall-timeout", "0",
                                           "--continue-after-infra"))
    _mod.main()
    assert len(procs) == 2


def test_heartbeat_reports_progress_while_an_arm_runs(runtime_env, monkeypatch,
                                                      capsys):
    """The caller's log must show liveness during an arm, not only after it.

    tau2's output goes to progress_log now, so without a heartbeat a healthy
    400 s episode and a hung one look identical from outside.
    """
    def _popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None):
        return _FakeProc(stdout.name, polls_before_exit=3)

    monkeypatch.setattr(_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(sys, "argv", _argv(runtime_env))
    _mod.main()
    out = capsys.readouterr().out
    assert "elapsed, last progress" in out
    assert "kill at" in out, "the heartbeat should name the stall budget"
