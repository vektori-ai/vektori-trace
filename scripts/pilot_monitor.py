#!/usr/bin/env python3
"""Read-only monitor for the live tau2 OPD pilot.

Observation only. This process never stops, kills, or restarts anything —
not tmux, not Modal, not the orchestrator. Teardown belongs to the runner.

    python3 scripts/pilot_monitor.py poll     # 60s compact status line
    python3 scripts/pilot_monitor.py alert    # tail serve.log + pilot_run.log
    python3 scripts/pilot_monitor.py health   # one-shot

State dir defaults to /data/tau2/pilot-state. Override with --state-dir or
PILOT_STATE. Never tails update-NNN.log for alerts: those files hold Tau2
retail dialogue, where "refuse"/"refusal" are normal successful turns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Lockstep with scripts/tau2_pilot_orchestrate.py Ledger. Do not import that
# file: it registers atexit teardown.
GPU_USD_PER_HOUR = 1.95
TEACHER_USD_PER_MTOK = 0.22
TRAIN_HOURS_PER_UPDATE = 2.0 / 60.0

DEFAULT_STATE = Path("/data/tau2/pilot-state")
STALL_S = 20 * 60
WARN_FRAC = 0.80
POLL_S = 60
MODAL_LIST_TIMEOUT_S = 30

# serve.log only. `(?i:refus)` catches "refusing to report an endpoint" and
# "Refusing to guess one"; it is never applied to update-NNN.log.
SERVE_ALERT_RE = re.compile(
    r"(?i:refus)"
    r"|Traceback"
    r"|RuntimeError"
    r"|ValueError"
    r"|SystemExit"
    r"|LiveBatchError"
    r"|LiveTrainError"
    r"|OOM"
    r"|Killed"
)

# Orchestrator stdout. Exact-ish tokens, not bare "Error" / "refus".
RUN_ALERT_RE = re.compile(
    r"STOPPING"
    r"|failed \(rc="
    r"|!!"
    r"|Traceback"
    r"|RuntimeError"
    r"|ValueError"
    r"|SystemExit"
    r"|LiveBatchError"
    r"|LiveTrainError"
    r"|OOM"
    r"|Killed"
)

STAGE_RE = re.compile(
    r"=== update \d+ ==="
    r"|dispatch \S+"
    r"|rollout took \S+"
    r"|scored \d+ new"
    r"|trained in \S+"
    r"|endpoint now serves \S+"
    r"|update \d+ complete"
    r"|pilot complete"
    r"|STOPPING\b"
    r"|already SAMPLED"
    r"|already SCORED"
    r"|already TRAINED"
)

UPDATE_RE = re.compile(r"=== update (\d+) ===")
UP_RE = re.compile(r"UP in (\d+)s")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
APP_ID_RE = re.compile(r"\b(ap-[A-Za-z0-9]+)\b")

WANTED_TMUX = ("serve", "pilot")


# --- views -----------------------------------------------------------------

@dataclass(frozen=True)
class ModalView:
    billing: int | None  # None = could not ask
    ephemeral: int
    ids: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class LedgerView:
    hours: float | None
    estimate_usd: float | None
    max_usd: float
    max_hours: float
    updates_done: int
    started: float | None
    error: str = ""


@dataclass(frozen=True)
class StatusView:
    trained: int
    sampled: int
    scored: int
    n_updates: int
    complete: bool
    next_update: int | None
    exists: bool
    error: str = ""


@dataclass
class Snapshot:
    now: float
    state: Path
    modal: ModalView
    tmux: tuple[str, ...]
    ledger: LedgerView
    status: StatusView
    current_update: int | None
    last_stage: str
    serve_up_s: int | None
    env_present: bool
    run_log_mtime: float | None
    status_mtime: float | None
    serve_alerts: tuple[str, ...] = ()
    run_alerts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str  # critical | bad | warn
    message: str


# --- parsers ---------------------------------------------------------------

def parse_app_list(text: str) -> ModalView:
    """Count ephemeral rows; Tasks >= 1 is billing. Fail closed on parse doubt."""
    text = ANSI_RE.sub("", text)
    ephemeral = 0
    billing = 0
    ids: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[\u2500-\u257F]+", " ", raw).strip()
        line = re.sub(r" +", " ", line)
        if not line:
            continue
        if "ephemeral" not in line:
            continue
        if re.search(r"\bState\b", line) and re.search(r"\bTasks\b", line):
            continue
        ephemeral += 1
        mid = APP_ID_RE.search(line)
        if mid:
            ids.append(mid.group(1))
        tm = re.search(r"ephemeral\s+(\d+)", line, re.I)
        if tm:
            tasks = int(tm.group(1))
        else:
            tasks = 1  # unknown task count on an ephemeral row: treat as billing
        if tasks >= 1:
            billing += 1
    return ModalView(billing=billing, ephemeral=ephemeral, ids=tuple(ids))


def parse_tmux_ls(text: str, wanted: tuple[str, ...] = WANTED_TMUX) -> tuple[str, ...]:
    found: list[str] = []
    wanted_set = set(wanted)
    for line in text.splitlines():
        name = line.split(":", 1)[0].strip()
        if name in wanted_set and name not in found:
            found.append(name)
    return tuple(n for n in wanted if n in found)


def live_ledger(data: dict, now: float) -> LedgerView:
    """Hours and spend from `started` + now, not the stale saved `hours` field."""
    max_usd = float(data.get("max_usd", 30.0))
    max_hours = float(data.get("max_hours", 7.0))
    updates_done = int(data.get("updates_done", 0))
    teacher_tokens = int(data.get("teacher_tokens", 0) or 0)
    started = data.get("started")
    if started is None:
        hours = data.get("hours")
        est = data.get("estimate_usd")
        return LedgerView(
            hours=float(hours) if hours is not None else None,
            estimate_usd=float(est) if est is not None else None,
            max_usd=max_usd,
            max_hours=max_hours,
            updates_done=updates_done,
            started=None,
            error="no started; using saved hours/estimate",
        )
    started_f = float(started)
    hours = (now - started_f) / 3600.0
    serving = hours * GPU_USD_PER_HOUR
    training = updates_done * TRAIN_HOURS_PER_UPDATE * GPU_USD_PER_HOUR
    teacher = teacher_tokens * TEACHER_USD_PER_MTOK / 1e6
    return LedgerView(
        hours=hours,
        estimate_usd=serving + training + teacher,
        max_usd=max_usd,
        max_hours=max_hours,
        updates_done=updates_done,
        started=started_f,
    )


def parse_status(data: dict) -> StatusView:
    updates = data.get("updates") or []
    trained_list = data.get("trained_updates")
    if isinstance(trained_list, list):
        trained = len(trained_list)
    else:
        trained = sum(1 for u in updates if u.get("trained"))
    sampled = sum(1 for u in updates if u.get("sampled"))
    scored = sum(1 for u in updates if u.get("scored"))
    n = int(data.get("n_updates") or 10)
    complete = bool(data.get("complete")) or (n > 0 and trained >= n)
    nxt = data.get("next_update")
    return StatusView(
        trained=trained,
        sampled=sampled,
        scored=scored,
        n_updates=n,
        complete=complete,
        next_update=int(nxt) if nxt is not None else None,
        exists=bool(data.get("exists", True)),
    )


def last_match(text: str, cre: re.Pattern[str]) -> str:
    found = ""
    for line in text.splitlines():
        m = cre.search(line)
        if m:
            found = m.group(0).strip()
    return found


def last_update_idx(text: str) -> int | None:
    found: int | None = None
    for m in UPDATE_RE.finditer(text):
        found = int(m.group(1))
    return found


def serve_up_seconds(text: str) -> int | None:
    found: int | None = None
    for m in UP_RE.finditer(text):
        found = int(m.group(1))
    return found


def matching_lines(text: str, cre: re.Pattern[str], *, limit: int = 20) -> tuple[str, ...]:
    hits = [ln.rstrip() for ln in text.splitlines() if cre.search(ln)]
    return tuple(hits[-limit:])


def alert_re_for(filename: str) -> re.Pattern[str] | None:
    if filename == "serve.log":
        return SERVE_ALERT_RE
    if filename == "pilot_run.log":
        return RUN_ALERT_RE
    return None


def stall_from_mtimes(
    *,
    run_mtime: float | None,
    status_mtime: float | None,
    complete: bool,
    now: float,
    stall_s: float = STALL_S,
) -> bool:
    """Both the run log and the trained-count file are quiet for stall_s.

    A stage can be legitimately silent mid-rollout; 20 min is past that.
    Missing status.json means the orchestrator has not written markers yet,
    so this is not a trained-count stall.
    """
    if complete or run_mtime is None or status_mtime is None:
        return False
    return (now - run_mtime) >= stall_s and (now - status_mtime) >= stall_s


# --- live probes (read-only) -----------------------------------------------

def _read_json(path: Path) -> tuple[dict | None, str]:
    if not path.is_file():
        return None, "absent"
    try:
        return json.loads(path.read_text()), ""
    except (OSError, ValueError) as exc:
        return None, str(exc)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def probe_modal(modal_bin: Path) -> ModalView:
    cmd = [str(modal_bin), "app", "list"]
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=MODAL_LIST_TIMEOUT_S,
        )
    except FileNotFoundError:
        return ModalView(billing=None, ephemeral=0, ids=(), error="modal binary missing")
    except subprocess.TimeoutExpired:
        return ModalView(billing=None, ephemeral=0, ids=(), error="modal app list timed out")
    except OSError as exc:
        return ModalView(billing=None, ephemeral=0, ids=(), error=str(exc))
    text = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0 and "ephemeral" not in text:
        err = (p.stderr or p.stdout or f"exit {p.returncode}").strip().splitlines()
        return ModalView(
            billing=None, ephemeral=0, ids=(),
            error=(err[-1] if err else f"exit {p.returncode}"),
        )
    return parse_app_list(text)


def probe_tmux() -> tuple[str, ...]:
    for args in (
        ["tmux", "ls", "-F", "#{session_name}"],
        ["tmux", "ls"],
    ):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return ()
        text = p.stdout or ""
        if p.returncode != 0 and not text.strip():
            # "no server running" → no sessions. Fall through to the next form
            # only when -F itself is rejected.
            if "unknown option" in (p.stderr or "").lower() or "invalid option" in (p.stderr or "").lower():
                continue
            return ()
        return parse_tmux_ls(text)
    return ()


def take_snapshot(
    state: Path,
    *,
    now: float | None = None,
    modal: ModalView | None = None,
    tmux: tuple[str, ...] | None = None,
    modal_bin: Path | None = None,
) -> Snapshot:
    now = time.time() if now is None else now
    if modal is None:
        if modal_bin is None:
            raise ValueError("modal_bin required when modal view is not injected")
        modal = probe_modal(modal_bin)
    if tmux is None:
        tmux = probe_tmux()

    led_raw, led_err = _read_json(state / "ledger.json")
    if led_raw is None:
        ledger = LedgerView(
            hours=None, estimate_usd=None, max_usd=30.0, max_hours=7.0,
            updates_done=0, started=None, error=led_err or "absent",
        )
    else:
        ledger = live_ledger(led_raw, now)
        if led_err:
            ledger = LedgerView(**{**ledger.__dict__, "error": led_err})

    st_raw, st_err = _read_json(state / "status.json")
    if st_raw is None:
        status = StatusView(
            trained=0, sampled=0, scored=0, n_updates=10, complete=False,
            next_update=None, exists=False, error=st_err or "absent",
        )
    else:
        status = parse_status(st_raw)
        if st_err:
            status = StatusView(**{**status.__dict__, "error": st_err})

    run_text = _read_text(state / "pilot_run.log")
    serve_text = _read_text(state / "serve.log")
    complete = status.complete or ("pilot complete" in run_text)

    current = last_update_idx(run_text)
    last_stage = last_match(run_text, STAGE_RE)
    # Status-only peek at the current update log: "rollout took" lives there,
    # not in the orchestrator stdout. Never used for alerts.
    if current is not None:
        ulog = state / f"update-{current:03d}.log"
        extra = last_match(_read_text(ulog), STAGE_RE)
        if extra and (_mtime(ulog) or 0) >= (_mtime(state / "pilot_run.log") or 0):
            last_stage = extra

    if complete and not last_stage:
        last_stage = "pilot complete"

    return Snapshot(
        now=now,
        state=state,
        modal=modal,
        tmux=tmux,
        ledger=ledger,
        status=StatusView(**{**status.__dict__, "complete": complete}),
        current_update=current,
        last_stage=last_stage or "—",
        serve_up_s=serve_up_seconds(serve_text),
        env_present=(state / "pilot_env.sh").is_file(),
        run_log_mtime=_mtime(state / "pilot_run.log"),
        status_mtime=_mtime(state / "status.json"),
        serve_alerts=matching_lines(serve_text, SERVE_ALERT_RE),
        run_alerts=matching_lines(run_text, RUN_ALERT_RE),
    )


def classify(snap: Snapshot, *, stall_s: float = STALL_S) -> list[Flag]:
    flags: list[Flag] = []
    billing = snap.modal.billing
    tmux_ok = bool(snap.tmux)

    if billing is None:
        flags.append(Flag(
            "MODAL_UNREACHABLE", "warn",
            f"modal app list failed ({snap.modal.error or 'unknown'}) — "
            "cannot tell if a GPU is billing",
        ))
    elif billing >= 1 and not tmux_ok:
        ids = ",".join(snap.modal.ids) or "?"
        flags.append(Flag(
            "GPU_NO_TMUX", "critical",
            f"GPU BILLING WITH NO TMUX — {billing} ephemeral app(s) still "
            f"up ({ids}) and neither `serve` nor `pilot` exists. The "
            "orchestrator is gone without teardown; this is the idle-GPU leak. "
            "Observation only: not stopping anything.",
        ))
    elif billing >= 1 and "serve" not in snap.tmux and not snap.status.complete:
        flags.append(Flag(
            "SERVE_TMUX_MISSING", "warn",
            "GPUs billing but tmux session `serve` is absent",
        ))
    elif billing >= 1 and "pilot" not in snap.tmux and not snap.status.complete:
        flags.append(Flag(
            "PILOT_TMUX_MISSING", "warn",
            "GPUs billing but tmux session `pilot` is absent",
        ))

    if stall_from_mtimes(
        run_mtime=snap.run_log_mtime,
        status_mtime=snap.status_mtime,
        complete=snap.status.complete,
        now=snap.now,
        stall_s=stall_s,
    ):
        run_ago = int(snap.now - (snap.run_log_mtime or snap.now))
        st_ago = int(snap.now - (snap.status_mtime or snap.now))
        flags.append(Flag(
            "STALL", "warn",
            f"no progress >{int(stall_s // 60)} min "
            f"(pilot_run.log {run_ago // 60}m ago, "
            f"status.json trained count {st_ago // 60}m ago, "
            f"trained={snap.status.trained})",
        ))

    led = snap.ledger
    if led.hours is not None and led.max_hours > 0:
        if led.hours >= led.max_hours:
            flags.append(Flag(
                "HOURS_CEILING", "warn",
                f"{led.hours:.2f}h >= ceiling {led.max_hours:.1f}h",
            ))
        elif led.hours + 1e-9 >= WARN_FRAC * led.max_hours:
            flags.append(Flag(
                "HOURS_WARN", "warn",
                f"{led.hours:.2f}h approaching {led.max_hours:.1f}h "
                f"(warn at {WARN_FRAC:.0%})",
            ))
    if led.estimate_usd is not None and led.max_usd > 0:
        if led.estimate_usd >= led.max_usd:
            flags.append(Flag(
                "BUDGET_CEILING", "warn",
                f"estimated ${led.estimate_usd:.2f} >= ceiling ${led.max_usd:.2f}",
            ))
        elif led.estimate_usd + 1e-9 >= WARN_FRAC * led.max_usd:
            flags.append(Flag(
                "BUDGET_WARN", "warn",
                f"estimated ${led.estimate_usd:.2f} approaching "
                f"${led.max_usd:.2f} (warn at {WARN_FRAC:.0%})",
            ))

    if snap.serve_alerts:
        flags.append(Flag(
            "SERVE_LOG", "bad",
            f"{len(snap.serve_alerts)} alert line(s) in serve.log: "
            f"{snap.serve_alerts[-1][:160]}",
        ))
    if snap.run_alerts:
        flags.append(Flag(
            "RUN_LOG", "bad",
            f"{len(snap.run_alerts)} alert line(s) in pilot_run.log: "
            f"{snap.run_alerts[-1][:160]}",
        ))
    return flags


def worst(flags: list[Flag]) -> str:
    order = {"critical": 3, "bad": 2, "warn": 1}
    if not flags:
        return "ok"
    return max(flags, key=lambda f: order.get(f.severity, 0)).severity


# --- rendering -------------------------------------------------------------

def _ts(now: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(now))


def compact_line(snap: Snapshot, flags: list[Flag]) -> str:
    gpus = "?" if snap.modal.billing is None else str(snap.modal.billing)
    tmux = ",".join(snap.tmux) if snap.tmux else "none"
    n_u = snap.status.n_updates
    cur = snap.current_update
    u = f"{cur if cur is not None else '-'}/{n_u}"
    hours = f"{snap.ledger.hours:.2f}h" if snap.ledger.hours is not None else "?h"
    if snap.ledger.estimate_usd is not None:
        spend = f"~${snap.ledger.estimate_usd:.2f}/${snap.ledger.max_usd:.0f}"
    else:
        spend = f"~?/${snap.ledger.max_usd:.0f}"
    stage = snap.last_stage
    if len(stage) > 48:
        stage = stage[:45] + "..."
    bits = [
        _ts(snap.now),
        f"GPUs={gpus}",
        f"tmux={tmux}",
        f"u={u}",
        f"stage={stage}",
        hours,
        spend,
        f"trained={snap.status.trained}",
    ]
    codes = [f.code for f in flags]
    if "GPU_NO_TMUX" in codes:
        bits.append("**** GPU-NO-TMUX ****")
    elif codes:
        bits.append("**" + ",".join(codes) + "**")
    return "  ".join(bits)


def loud(msg: str, *, bell: bool = False) -> None:
    bar = "#" * 72
    if bell:
        sys.stdout.write("\a")
    print(bar, flush=True)
    for line in msg.splitlines() or [msg]:
        print(f"# {line}", flush=True)
    print(bar, flush=True)


def health_report(snap: Snapshot, flags: list[Flag]) -> str:
    gpus = (
        "unreachable: " + (snap.modal.error or "?")
        if snap.modal.billing is None
        else f"{snap.modal.billing} billing"
        + (f"  ids={','.join(snap.modal.ids)}" if snap.modal.ids else "")
    )
    tmux = ",".join(snap.tmux) if snap.tmux else "none"
    if snap.serve_up_s is None:
        serve = "no UP line yet"
    else:
        serve = f"UP in {snap.serve_up_s}s"
    serve += ", env=" + ("yes" if snap.env_present else "NO")
    hours = (
        f"{snap.ledger.hours:.2f}h / {snap.ledger.max_hours:.1f}h"
        if snap.ledger.hours is not None else f"? / {snap.ledger.max_hours:.1f}h"
    )
    spend = (
        f"~${snap.ledger.estimate_usd:.2f} / ${snap.ledger.max_usd:.2f}  "
        f"(warn at ${WARN_FRAC * snap.ledger.max_usd:.2f})"
        if snap.ledger.estimate_usd is not None
        else f"? / ${snap.ledger.max_usd:.2f}"
    )
    if snap.run_log_mtime is None:
        stall = "n/a (no pilot_run.log)"
    elif snap.status_mtime is None:
        stall = "n/a (no status.json)"
    else:
        stalled = stall_from_mtimes(
            run_mtime=snap.run_log_mtime, status_mtime=snap.status_mtime,
            complete=snap.status.complete, now=snap.now,
        )
        run_ago = int(snap.now - snap.run_log_mtime)
        st_ago = int(snap.now - snap.status_mtime)
        stall = (
            f"{'YES' if stalled else 'no'}  "
            f"(log {run_ago // 60}m{run_ago % 60:02d}s ago, "
            f"status {st_ago // 60}m{st_ago % 60:02d}s ago)"
        )
    u = snap.current_update
    lines = [
        f"pilot health  {time.strftime('%F %T', time.localtime(snap.now))}",
        f"  state    {snap.state}",
        f"  tmux     {tmux}",
        f"  GPUs     {gpus}",
        f"  serve    {serve}",
        f"  update   {u if u is not None else '—'} / {snap.status.n_updates}"
        f"  stage={snap.last_stage}",
        f"  trained  {snap.status.trained}  sampled={snap.status.sampled}"
        f"  scored={snap.status.scored}"
        + ("  COMPLETE" if snap.status.complete else ""),
        f"  elapsed  {hours}",
        f"  spend    {spend}",
        f"  stall    {stall}",
    ]
    if snap.ledger.error and snap.ledger.error != "absent":
        lines.append(f"  ledger   {snap.ledger.error}")
    if snap.status.error and snap.status.error != "absent":
        lines.append(f"  status   {snap.status.error}")
    if flags:
        lines.append("  flags")
        for f in flags:
            lines.append(f"    [{f.severity}] {f.code}: {f.message}")
    sev = worst(flags)
    label = {"ok": "HEALTHY", "warn": "WARN", "bad": "BAD", "critical": "BAD"}.get(sev, sev)
    if sev == "critical":
        label = "BAD  GPU BILLING WITH NO TMUX"
    lines.append("")
    lines.append(label)
    return "\n".join(lines)


def exit_code(flags: list[Flag]) -> int:
    sev = worst(flags)
    if sev == "ok":
        return 0
    if sev == "warn":
        return 1
    return 2


# --- commands --------------------------------------------------------------

def repo_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def modal_bin(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "modal"


def cmd_health(args: argparse.Namespace) -> int:
    snap = take_snapshot(args.state_dir, modal_bin=modal_bin(args.repo))
    flags = classify(snap, stall_s=args.stall_s)
    print(health_report(snap, flags), flush=True)
    for f in flags:
        if f.code == "GPU_NO_TMUX":
            loud(f.message, bell=True)
    return exit_code(flags)


def cmd_poll(args: argparse.Namespace) -> int:
    interval = args.interval
    rc = 0
    while True:
        t0 = time.time()
        try:
            snap = take_snapshot(args.state_dir, modal_bin=modal_bin(args.repo))
            flags = classify(snap, stall_s=args.stall_s)
            print(compact_line(snap, flags), flush=True)
            # GPU-without-tmux is the only poller banner: it is the leak that
            # bills for hours doing nothing. Log failures belong to `alert`.
            for f in flags:
                if f.code == "GPU_NO_TMUX":
                    loud(f.message, bell=True)
            rc = exit_code(flags)
        except Exception as exc:  # noqa: BLE001 — a watcher must not die on one tick
            print(f"{_ts(time.time())}  MONITOR-ERROR {exc!r}", flush=True)
            rc = 2
        if args.once:
            return rc
        slept = time.time() - t0
        time.sleep(max(0.0, interval - slept))


class _Tail:
    """In-memory follow. No offset file, no writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = path.stat().st_size if path.is_file() else 0
        self.pending = ""

    def read_new(self) -> list[str]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.pending = ""
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
            self.offset = fh.tell()
        if not chunk and not self.pending:
            return []
        text = self.pending + chunk.decode("utf-8", "replace")
        if text.endswith("\n"):
            self.pending = ""
            return text.splitlines()
        parts = text.split("\n")
        self.pending = parts[-1]
        return parts[:-1]


def cmd_alert(args: argparse.Namespace) -> int:
    state: Path = args.state_dir
    targets = (
        (state / "serve.log", SERVE_ALERT_RE),
        (state / "pilot_run.log", RUN_ALERT_RE),
    )
    print(
        f"alerter  state={state}  (serve.log + pilot_run.log only; "
        "never update-NNN.log)",
        flush=True,
    )
    # Existing hits first, then follow from EOF so a mid-run attach still
    # surfaces a Traceback that already happened.
    for path, cre in targets:
        for line in matching_lines(_read_text(path), cre, limit=50):
            loud(f"ALREADY  {path.name}\n{line}")
    if args.once:
        n = sum(
            1 for path, cre in targets
            for _ in matching_lines(_read_text(path), cre)
        )
        return 2 if n else 0
    tails = {path: _Tail(path) for path, _cre in targets}
    while True:
        for path, cre in targets:
            try:
                lines = tails[path].read_new()
            except OSError as exc:
                print(f"{_ts(time.time())}  MONITOR-ERROR {path.name}: {exc}", flush=True)
                continue
            for line in lines:
                if cre.search(line):
                    loud(f"{_ts(time.time())}  {path.name}\n{line}")
        time.sleep(1.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("PILOT_STATE", str(DEFAULT_STATE))),
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("PILOT_REPO", str(repo_from_script()))),
    )
    ap.add_argument("--stall-s", type=float, default=STALL_S)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_poll = sub.add_parser("poll", help="compact status line every --interval s")
    p_poll.add_argument("--interval", type=float, default=POLL_S)
    p_poll.add_argument("--once", action="store_true")
    p_poll.set_defaults(func=cmd_poll)

    p_alert = sub.add_parser("alert", help="print loudly on failure strings")
    p_alert.add_argument("--once", action="store_true",
                         help="scan existing logs and exit")
    p_alert.set_defaults(func=cmd_alert)

    p_health = sub.add_parser("health", help="one-shot is-everything-healthy")
    p_health.set_defaults(func=cmd_health)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
