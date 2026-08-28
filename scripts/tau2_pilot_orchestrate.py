#!/usr/bin/env python3
"""Drive a multi-update live-OPD pilot as separate, resumable GPU stages.

The monolithic `train` runs rollout -> score -> train inside ONE container that
declares `gpu=`. That card stays allocated while another GPU serves the
rollout, while DeepSeek scores over the network, and while the user simulator
thinks -- for a 10-update pilot, hours of an idle training GPU. It is also
all-or-nothing: a failure in the last stage costs the whole update.

This orchestrator runs on CPU (the box, in tmux) and dispatches each stage to
its own Modal function:

    rollout_only   serving GPU (already up)   -> SAMPLED
    rescore        CPU + DeepSeek, NO GPU     -> SCORED
    one_step       training GPU ~2 min        -> TRAINED
    refresh        serving GPU (already up)   -> next policy

Two rules it exists to enforce:

**Retries are stage-local.** It never deletes an update directory and never
restarts the pilot because a later stage failed. Every stage reads the markers
on the volume and skips what is already done, so a resumed run resamples
nothing and re-buys no scores. The artifacts ARE the recovery mechanism.

**Teardown is scoped.** It stops only the app ids it started, recorded before
dispatch. "Stop every running app" would kill unrelated work, and an
orchestrator that crashes before recording an id leaves a card billing.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODAL_SCRIPT = REPO / "scripts" / "tau2_live_opd_modal.py"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- scoped teardown -------------------------------------------------------

class OwnedApps:
    """App ids this pilot started, and only those.

    Persisted before the process can die, so a resumed orchestrator adopts the
    endpoint the previous one left running instead of starting a second card.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.ids: list[str] = []
        if path.exists():
            try:
                self.ids = json.loads(path.read_text()).get("app_ids", [])
            except (OSError, ValueError):
                self.ids = []

    def add(self, app_id: str) -> None:
        if app_id and app_id not in self.ids:
            self.ids.append(app_id)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"app_ids": self.ids}, indent=1))

    def stop_all(self) -> None:
        for app_id in list(self.ids):
            log(f"teardown: stopping {app_id}")
            try:
                subprocess.run(
                    [str(REPO / ".venv/bin/modal"), "app", "stop", app_id, "-y"],
                    timeout=180, capture_output=True,
                )
            except Exception as exc:  # noqa: BLE001
                log(f"  WARNING could not stop {app_id}: {exc}")
        self.ids = []
        if self.path.exists():
            self.path.write_text(json.dumps({"app_ids": []}, indent=1))


# --- modal dispatch --------------------------------------------------------

def modal_run(fn: str, args: list[str], *, log_path: Path,
              timeout: int = 7200) -> int:
    """Invoke one Modal function, streaming to a per-stage log."""
    cmd = [str(REPO / ".venv/bin/modal"), "run",
           f"{MODAL_SCRIPT}::{fn}", *args]
    log(f"dispatch {fn}: {' '.join(args[:6])}...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(f"\n=== {fn} {time.strftime('%F %T')} ===\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout)
    return p.returncode


def stage_done(run_dir: Path, update: int, marker: str) -> bool:
    return (run_dir / f"update-{update:03d}" / marker).exists()


def read_marker(run_dir: Path, update: int, marker: str) -> dict:
    p = run_dir / f"update-{update:03d}" / marker
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def preflight_checkpoint(run_dir: Path, update: int) -> None:
    """Refuse to dispatch a training GPU that is going to fail on arrival.

    Update k>0 must resume update k-1's optimizer, so a missing `optimizer.pt`
    is fatal -- but discovering that inside the GPU container costs a card
    allocation and a model load to learn something a stat() answers for free.
    Fresh Adam every update would also be silently WRONG rather than an error,
    which is why this refuses instead of falling back.
    """
    if update == 0:
        return
    cp = run_dir / f"update-{update - 1:03d}" / "checkpoint"
    required = ("optimizer.pt", "state.json", "adapter_config.json")
    missing = [f for f in required if not (cp / f).exists()]
    if missing:
        raise SystemExit(
            f"update {update} cannot train: {cp} is missing {missing}. "
            "Update k>0 resumes update k-1's optimizer, scheduler and RNG; "
            "training with a fresh optimizer would change the recipe at every "
            "update while every reported metric looked normal. Refusing before "
            "allocating a GPU."
        )


def served_adapter_hash(api_base: str, name: str, timeout: float = 60.0
                        ) -> str | None:
    """Which adapter the endpoint currently serves under `name`, if any."""
    sys.path.insert(0, str(REPO))
    try:
        from vektori_trace.tau2.reopd_refresh import served_models
        return name if name in served_models(api_base, timeout=timeout) else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--reload-url", default="")
    ap.add_argument("--student-model", required=True,
                    help="served adapter name for update 0")
    ap.add_argument("--serve-app-id", default="",
                    help="app id of an already-running endpoint to USE. Not "
                         "stopped on exit unless --own-endpoint is passed: an "
                         "endpoint this pilot did not start may be serving "
                         "something else.")
    ap.add_argument("--own-endpoint", action="store_true",
                    help="take ownership of --serve-app-id, so teardown stops "
                         "it. Pass this only when the endpoint exists solely "
                         "for this pilot.")
    ap.add_argument("--run-dir", default="",
                    help="the pilot's run directory, readable from here, so "
                         "stage markers can be inspected without dispatching")
    ap.add_argument("--parent-adapter-hash", default="3869b147ab7ce5d2",
                    help="hash of the untouched parent update 0 samples from")
    ap.add_argument("--state-dir", default="",
                    help="where owned app ids and stage logs live")
    ap.add_argument("--start-at", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the stage plan and exit; dispatches nothing")
    a = ap.parse_args()

    state_dir = Path(a.state_dir or (REPO / f".pilot-{a.run_id}"))
    state_dir.mkdir(parents=True, exist_ok=True)
    owned = OwnedApps(state_dir / "owned_apps.json")
    # Ownership is explicit. "Using" an endpoint and "being responsible for
    # stopping it" are different claims: an endpoint started outside this
    # pilot may be serving other work, and stopping it on exit would kill
    # something this process never started.
    if a.serve_app_id and a.own_endpoint:
        owned.add(a.serve_app_id)
        log(f"owning endpoint {a.serve_app_id}: it WILL be stopped on exit")
    elif a.serve_app_id:
        log(f"using endpoint {a.serve_app_id} (not owned; left running on exit)")

    # Teardown on every exit path a signal can reach. A crash between dispatch
    # and recording is the one gap; recording happens first for that reason.
    def _teardown(*_: object) -> None:
        owned.stop_all()

    atexit.register(_teardown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    if a.dry_run:
        log(f"DRY RUN: {a.n_updates} updates, run {a.run_id}")
        for i in range(a.start_at, a.n_updates):
            log(f"  update {i}: rollout_only -> rescore -> one_step -> refresh")
        log("nothing dispatched, no GPU, no teacher call")
        return 0

    log(f"pilot {a.run_id}: updates {a.start_at}..{a.n_updates - 1}")
    log(f"  state dir : {state_dir}")
    log(f"  endpoint  : {a.api_base}")

    run_dir = Path(a.run_dir) if a.run_dir else None
    if run_dir is None or not run_dir.exists():
        raise SystemExit(
            "--run-dir must point at the pilot's run directory (the same path "
            "the Modal volume exposes), so stage markers can be read WITHOUT "
            "dispatching a container to ask"
        )

    # Endpoint recovery: after an orchestrator restart the served policy is
    # whatever the last completed refresh left. Derive it from the markers
    # rather than assuming update 0's name, or a resumed run would sample the
    # SFT checkpoint while claiming update k's policy.
    served_state = state_dir / "served.json"
    served_name = a.student_model
    if served_state.exists():
        try:
            served_name = json.loads(served_state.read_text()).get(
                "served_name", a.student_model)
            log(f"resumed served name from state: {served_name}")
        except (OSError, ValueError):
            pass

    def _record_served(name: str) -> None:
        served_state.write_text(json.dumps(
            {"served_name": name, "at": time.strftime("%F %T")}, indent=1))

    for idx in range(a.start_at, a.n_updates):
        log(f"=== update {idx} ===")
        stage_log = state_dir / f"update-{idx:03d}.log"
        trained = stage_done(run_dir, idx, ".TRAINED")

        # --- 0. REFRESH (serving GPU, already up) -------------------------
        # Serve update k-1's checkpoint BEFORE sampling update k. Skipping this
        # is invisible: update k would sample update k-2's weights while every
        # marker, hash and loss still looked right, and every importance ratio
        # would compare two distributions that never met.
        #
        # Skipped when this update is already SAMPLED: its episodes exist and
        # re-pointing the endpoint now would serve a policy nothing here needs.
        need_rollout = not stage_done(run_dir, idx, ".SAMPLED")
        if idx > 0 and need_rollout:
            expected = f"{a.student_model}-u{idx - 1:03d}"
            if served_adapter_hash(a.api_base, expected) == expected:
                log(f"endpoint already serves {expected}; refresh skipped")
                served_name = expected
                _record_served(served_name)
            else:
                rc = modal_run("refresh_only", [
                    "--run-id", a.run_id, "--update", str(idx),
                    "--api-base", a.api_base,
                    "--reload-url", a.reload_url,
                    "--base-served-name", a.student_model,
                    "--previous-served-name", served_name,
                ], log_path=stage_log)
                if rc != 0:
                    log(f"refresh failed (rc={rc}) -- the child checkpoint is "
                        f"intact; restart/refresh serving and resume "
                        f"--start-at {idx}. Nothing retrained.")
                    return rc
                served_name = expected  # refresh_live_policy's naming
                _record_served(served_name)
                log(f"endpoint now serves {served_name}")

        # --- 1. ROLLOUT (serving GPU) -------------------------------------
        if need_rollout:
            # Provenance for every archived episode: the hash of the adapter
            # actually being served. Update 0 is the frozen parent; later
            # updates are update k-1's child, read off its checkpoint state.
            if idx == 0:
                expect_hash = a.parent_adapter_hash
            else:
                st = read_marker(run_dir, idx - 1, "checkpoint/state.json")
                expect_hash = st.get("adapter_hash", "")
            if not expect_hash:
                raise SystemExit(
                    f"cannot determine the adapter hash update {idx} will "
                    "sample from; every archived episode is stamped with it "
                    "and batch_report checks the batch against it"
                )
            rc = modal_run("rollout_only", [
                "--run-id", a.run_id, "--update", str(idx),
                "--api-base", a.api_base,
                "--student-model", served_name,
                "--adapter-hash-expect", expect_hash,
            ], log_path=stage_log)
            if rc != 0:
                log(f"rollout failed (rc={rc}) -- archived turns kept, nothing "
                    f"deleted. Resume --start-at {idx}. See {stage_log}")
                return rc
        else:
            log(f"update {idx} already SAMPLED; not resampling")

        # --- 2. SCORE (CPU + DeepSeek, no GPU) ----------------------------
        if not stage_done(run_dir, idx, ".SCORED"):
            rc = modal_run("rescore", [
                "--run-id", a.run_id, "--update", str(idx),
            ], log_path=stage_log)
            if rc != 0:
                log(f"scoring failed (rc={rc}) -- paid scores persist per "
                    f"action; a retry buys only the missing ones. "
                    f"Resume --start-at {idx}")
                return rc
        else:
            log(f"update {idx} already SCORED; buying nothing")

        # --- 3. TRAIN (JIT training GPU, ~2 min) --------------------------
        if not trained:
            # Everything checkable is checked before this line: this is the
            # first stage that allocates a card.
            preflight_checkpoint(run_dir, idx)
            rc = modal_run("one_step", [
                "--run-id", a.run_id, "--update", str(idx),
            ], log_path=stage_log)
            if rc != 0:
                log(f"training failed (rc={rc}) -- TRAINED absent, rollout and "
                    f"scores intact. Retry the step: --start-at {idx}")
                return rc
        else:
            log(f"update {idx} already TRAINED; not retraining")

        log(f"update {idx} complete")

    log("pilot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
