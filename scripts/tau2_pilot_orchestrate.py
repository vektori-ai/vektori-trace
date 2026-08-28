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
        """Stop owned apps, and KEEP any id whose stop did not succeed.

        Clearing the list unconditionally would discard the only handle to an
        endpoint that is still billing. A failed teardown must stay visible --
        in the file and in the log -- so it can be finished by hand.
        """
        still: list[str] = []
        for app_id in list(self.ids):
            log(f"teardown: stopping {app_id}")
            try:
                p = subprocess.run(
                    [str(REPO / ".venv/bin/modal"), "app", "stop", app_id, "-y"],
                    timeout=180, capture_output=True, text=True,
                )
                out = (p.stdout or "") + (p.stderr or "")
                if p.returncode != 0 and "already stopped" not in out.lower():
                    log(f"  WARNING stop returned {p.returncode} for {app_id}; "
                        "KEEPING the id -- it may still be billing")
                    still.append(app_id)
                else:
                    log(f"  stopped {app_id}")
            except Exception as exc:  # noqa: BLE001
                log(f"  WARNING could not stop {app_id}: {exc}; KEEPING the id")
                still.append(app_id)
        self.ids = still
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"app_ids": still}, indent=1))
        if still:
            log(f"!! {len(still)} app id(s) NOT confirmed stopped: {still}")
            log(f"!! check `modal app list` and stop them by hand")


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


def fetch_status(run_id: str, state_dir: Path) -> dict:
    """Stage markers from the VOLUME, not from a local path that cannot exist.

    `/adapters` lives inside Modal containers. Testing it from the box always
    reports "absent", so a resumed orchestrator would conclude nothing had run
    and resample updates it already holds -- paying twice and overwriting
    evidence.
    """
    out = state_dir / "status.json"
    cmd = [str(REPO / ".venv/bin/modal"), "run",
           f"{MODAL_SCRIPT}::pilot_status", "--run-id", run_id]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SystemExit(
            f"cannot read pilot status from the volume:\n{p.stdout[-2000:]}\n"
            f"{p.stderr[-2000:]}"
        )
    # Parse ONLY the sentinel line. Scanning from the first "{" to the last
    # "}" silently mis-parses as soon as anything else prints an object.
    line = next((ln for ln in reversed(p.stdout.splitlines())
                 if ln.startswith("PILOT_STATUS_JSON=")), None)
    if line is None:
        raise SystemExit(
            f"pilot_status printed no PILOT_STATUS_JSON line:\n"
            f"{p.stdout[-2000:]}"
        )
    status = json.loads(line[len("PILOT_STATUS_JSON="):])
    out.write_text(json.dumps(status, indent=1))
    return status


#: L40S, https://modal.com/pricing. Used only for the running estimate the
#: ceiling is checked against -- it is not a bill.
GPU_USD_PER_HOUR = 1.95
#: DeepSeek V4 Flash input, per million tokens.
TEACHER_USD_PER_MTOK = 0.22


class Ledger:
    """A running cost estimate, and the hard stop that uses it.

    Nobody is watching an unattended run, so the update-5 judgement call has to
    be a rule the process can apply itself. Wall-clock endpoint uptime is the
    dominant term and the one that runs away when a stage hangs.

    These are ESTIMATES from measured rates, not billing data. The ceiling is
    deliberately conservative for that reason.
    """

    def __init__(self, path: Path, *, max_usd: float, max_hours: float) -> None:
        self.path = path
        self.max_usd = max_usd
        self.max_hours = max_hours
        self.started = time.time()
        self.teacher_tokens = 0
        self.updates_done = 0
        if path.exists():
            try:
                d = json.loads(path.read_text())
                self.started = d.get("started", self.started)
                self.teacher_tokens = d.get("teacher_tokens", 0)
                self.updates_done = d.get("updates_done", 0)
            except (OSError, ValueError):
                pass

    @property
    def hours(self) -> float:
        return (time.time() - self.started) / 3600.0

    def estimate_usd(self) -> float:
        # One serving card up for the whole run, plus ~2 min of training GPU
        # per completed update, plus the teacher.
        serving = self.hours * GPU_USD_PER_HOUR
        training = self.updates_done * (2.0 / 60.0) * GPU_USD_PER_HOUR
        teacher = self.teacher_tokens * TEACHER_USD_PER_MTOK / 1e6
        return serving + training + teacher

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "started": self.started,
            "hours": round(self.hours, 3),
            "updates_done": self.updates_done,
            "teacher_tokens": self.teacher_tokens,
            "estimate_usd": round(self.estimate_usd(), 2),
            "max_usd": self.max_usd,
            "max_hours": self.max_hours,
        }, indent=1))

    def check(self) -> str | None:
        """The reason to stop, or None. Checked before each paid stage."""
        if self.hours >= self.max_hours:
            return (f"time ceiling: {self.hours:.2f}h >= {self.max_hours}h")
        est = self.estimate_usd()
        if est >= self.max_usd:
            return (f"budget ceiling: estimated ${est:.2f} >= "
                    f"${self.max_usd:.2f}")
        return None


def probe_now(api_base: str, model: str, probe_ids: list[int]) -> list[float]:
    sys.path.insert(0, str(REPO))
    from vektori_trace.tau2.reopd_refresh import probe_logprobs
    return probe_logprobs(api_base, model, probe_ids, timeout=300.0)


def refresh_already_done(a, state_dir: Path, idx: int, expected_name: str,
                         expected_hash: str) -> bool:
    """Is the endpoint ALREADY serving update idx-1's weights?

    A name check is not enough: vLLM will happily advertise
    `...-u000` while serving whatever was loaded under that name, and after an
    endpoint restart the name may be absent entirely. So this compares the live
    probe logprobs against the ones the refresh recorded when it verified the
    swap -- weights, not labels.

    Returns False on any doubt, because re-refreshing is cheap and sampling
    from a stale policy is not.
    """
    rec_path = state_dir / f"refresh-{idx:03d}.json"
    if not rec_path.exists():
        return False
    try:
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError):
        return False
    if rec.get("adapter_hash") != expected_hash:
        return False
    probe_ids = rec.get("probe_ids")
    recorded = rec.get("probe_logprobs")
    if not probe_ids or not recorded:
        return False
    try:
        live = probe_now(a.api_base, expected_name, probe_ids)
    except Exception as exc:  # noqa: BLE001
        log(f"  probe failed ({exc}); will refresh rather than assume")
        return False
    if len(live) != len(recorded):
        return False
    drift = max(abs(x - y) for x, y in zip(live, recorded))
    if drift > 1e-6:
        log(f"  endpoint logprobs drifted {drift:.3g} from the recorded "
            "refresh; treating the served policy as unverified")
        return False
    log(f"  endpoint already serves {expected_name} "
        f"(adapter {expected_hash}, probe matches to {drift:.3g})")
    return True


def update_row(status: dict, idx: int) -> dict:
    for row in status.get("updates", []):
        if row.get("update") == idx:
            return row
    return {}


def preflight_checkpoint(status: dict, update: int) -> None:
    """Refuse to dispatch a training GPU that will fail on arrival.

    Update k>0 resumes update k-1's optimizer, so a missing `optimizer.pt` is
    fatal -- and discovering that inside the GPU container costs a card
    allocation and a model load to learn something the status call already
    answered. Fresh Adam would also be silently WRONG rather than an error,
    which is why this refuses instead of falling back.
    """
    if update == 0:
        return
    prev = update_row(status, update - 1)
    if not prev.get("trained"):
        raise SystemExit(
            f"update {update} cannot train: update {update - 1} is not TRAINED"
        )
    if not prev.get("checkpoint_complete"):
        raise SystemExit(
            f"update {update} cannot train: update {update - 1}'s checkpoint "
            "is missing optimizer.pt/state.json/adapter_config.json. Training "
            "with a fresh optimizer would change the recipe at every update "
            "while every reported metric looked normal. Refusing before "
            "allocating a GPU."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--reload-url", default="",
                    help="REQUIRED: serving-volume reload url. Without it the "
                         "endpoint may never see the new checkpoint.")
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
    ap.add_argument("--parent-adapter-hash", default="3869b147ab7ce5d2",
                    help="hash of the untouched parent update 0 samples from")
    ap.add_argument("--state-dir", default="",
                    help="where owned app ids and stage logs live")
    ap.add_argument("--start-at", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=30.0,
                    help="hard stop when the ESTIMATED spend reaches this. "
                         "Checked before every paid stage, so an unattended "
                         "run cannot spend past it.")
    ap.add_argument("--max-hours", type=float, default=7.0,
                    help="hard stop on wall clock. A hung stage otherwise "
                         "bills a serving card indefinitely.")
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

    ledger = Ledger(state_dir / "ledger.json",
                    max_usd=a.max_usd, max_hours=a.max_hours)
    ledger.save()
    log(f"budget: hard stop at ${a.max_usd:.2f} estimated or "
        f"{a.max_hours}h wall clock "
        f"(already {ledger.hours:.2f}h, ~${ledger.estimate_usd():.2f})")

    log(f"pilot {a.run_id}: updates {a.start_at}..{a.n_updates - 1}")
    log(f"  state dir : {state_dir}")
    log(f"  endpoint  : {a.api_base}")

    if not a.reload_url:
        raise SystemExit(
            "--reload-url is required. Without a serving-volume reload the "
            "endpoint may never see the newly committed checkpoint, so update "
            "k+1 would resample update k-1's policy while every marker, hash "
            "and loss still looked correct -- the on-policy claim would be "
            "false with nothing in the logs to show it."
        )

    status = fetch_status(a.run_id, state_dir)
    if not status.get("exists"):
        raise SystemExit(
            f"no run {a.run_id} on the volume. Freeze a manifest with "
            "scripts/tau2_pilot_manifest.py and stage it first."
        )
    log(f"status: {len(status.get('trained_updates', []))}/"
        f"{status.get('n_updates')} updates trained, "
        f"next={status.get('next_update')}, plan={status.get('plan_hash')}")

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
        stop = ledger.check()
        if stop is not None:
            log(f"STOPPING before update {idx}: {stop}")
            log("artifacts are intact; resume with a raised ceiling if that is "
                "the deliberate choice")
            return 2
        row = update_row(status, idx)
        trained = bool(row.get("trained"))

        # --- 0. REFRESH (serving GPU, already up) -------------------------
        # Serve update k-1's checkpoint BEFORE sampling update k. Skipping this
        # is invisible: update k would sample update k-2's weights while every
        # marker, hash and loss still looked right, and every importance ratio
        # would compare two distributions that never met.
        #
        # Skipped when this update is already SAMPLED: its episodes exist and
        # re-pointing the endpoint now would serve a policy nothing here needs.
        need_rollout = not row.get("sampled")
        if idx > 0 and need_rollout:
            expected = f"{a.student_model}-u{idx - 1:03d}"
            expected_hash = update_row(status, idx - 1).get(
                "trained_adapter_hash", "")
            if not expected_hash:
                raise SystemExit(
                    f"update {idx - 1} reports no trained adapter_hash; the "
                    "endpoint cannot be verified against the weights it should "
                    "now serve"
                )
            if refresh_already_done(a, state_dir, idx, expected, expected_hash):
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
                # Record the verified fingerprint so a resumed run can tell
                # "already serving this" from "serving something else under
                # the same name".
                try:
                    sys.path.insert(0, str(REPO))
                    from transformers import AutoTokenizer
                    tok = AutoTokenizer.from_pretrained(
                        "Qwen/Qwen3-4B", trust_remote_code=True)
                    pids = tok("You are a retail agent.",
                               add_special_tokens=False)["input_ids"][:256]
                    (state_dir / f"refresh-{idx:03d}.json").write_text(
                        json.dumps({
                            "update": idx, "served_name": served_name,
                            "adapter_hash": expected_hash,
                            "probe_ids": pids,
                            "probe_logprobs": probe_now(
                                a.api_base, served_name, pids),
                            "at": time.strftime("%F %T"),
                        }, indent=1))
                except Exception as exc:  # noqa: BLE001
                    log(f"  WARNING could not record refresh fingerprint: "
                        f"{exc} (a resume will simply refresh again)")
                log(f"endpoint now serves {served_name}")

        # --- 1. ROLLOUT (serving GPU) -------------------------------------
        if need_rollout:
            # Provenance for every archived episode: the hash of the adapter
            # actually being served. Update 0 is the frozen parent; later
            # updates are update k-1's child, read off its checkpoint state.
            if idx == 0:
                expect_hash = status.get("parent_adapter_hash") or \
                    a.parent_adapter_hash
            else:
                expect_hash = update_row(status, idx - 1).get(
                    "trained_adapter_hash", "")
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
        if not row.get("scored"):
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
            preflight_checkpoint(status, idx)
            rc = modal_run("one_step", [
                "--run-id", a.run_id, "--update", str(idx),
            ], log_path=stage_log)
            if rc != 0:
                log(f"training failed (rc={rc}) -- TRAINED absent, rollout and "
                    f"scores intact. Retry the step: --start-at {idx}")
                return rc
        else:
            log(f"update {idx} already TRAINED; not retraining")

        status = fetch_status(a.run_id, state_dir)
        ledger.updates_done = len([u for u in status.get("updates", [])
                                   if u.get("trained")])
        ledger.save()
        log(f"update {idx} complete "
            f"({ledger.hours:.2f}h, ~${ledger.estimate_usd():.2f} of "
            f"${a.max_usd:.2f})")

    log("pilot complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
