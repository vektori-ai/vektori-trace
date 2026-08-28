"""The OPD stages, extracted here so the two drivers need not reimplement them.

**Status, 2026-08-28: only the live driver imports this.**
`scripts/tau2_reopd_train.py` still carries its own copy of the scoring and
training stages, so today this is a shared *destination*, not shared execution.
Migrating replay onto it is deliberately a separate change -- the replay arm has
a paid run behind it, and re-pointing it at new code is not something to do in
passing. Until that migration lands, do not describe these stages as
drift-proof: one caller plus a copy is exactly the arrangement drift comes from.

`scripts/tau2_reopd_train.py` and `scripts/tau2_live_opd_train.py` run the same
state machine::

    PLANNED -> SAMPLED -> SCORED -> TRAINED -> checkpoint -> reload -> next

and differ in exactly one place: where `(actions, rendered)` come from. Replay
fills them from frozen C30 prefixes; live fills them from complete Tau2
episodes. `TAU2-OPD-DEEP-DIVE.md` calls that substitution "the delta", and the
whole point of naming it is that *everything else must be the same code*.

Two near-identical 800-line drivers is how the second one quietly stops
enforcing something the first one does. The score cache, the fingerprint
binding, the commit discipline and the checkpoint/reload proof are each a
correctness gate that was paid for in a wrong turn; a copy that omits one still
produces a finite loss and clean logs. So the stages live here once and both
drivers call them.

What stays in the drivers is what genuinely differs: how a batch is planned,
how it is sampled, and the run manifest.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Callable

from vektori_trace.tau2.reopd_state import (
    UpdateDir,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
)

__all__ = [
    "CommitFailed",
    "ScoreStageResult",
    "commit_scores",
    "gpu_snapshot",
    "log",
    "run_score_stage",
    "run_train_stage",
    "set_commit_fn",
    "telemetry",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


#: Called after every durable write. On Modal, fsync only reaches the
#: container's local disk -- a Volume publishes nothing to shared storage until
#: `vol.commit()`. A container killed by OOM, timeout or an infrastructure fault
#: never runs a `finally`, so an uncommitted score is a score that was paid for
#: and lost.
_COMMIT: Any = None


def set_commit_fn(fn: Any) -> None:
    """Install the volume-commit callback (Modal), or leave it None locally."""
    global _COMMIT
    _COMMIT = fn


class CommitFailed(RuntimeError):
    """A paid artifact could not be published to shared storage."""


def commit_scores(why: str = "", *, required: bool = True, attempts: int = 3) -> None:
    """Publish everything written so far, and stop the run if it will not.

    A swallowed commit failure is the worst available outcome: the run keeps
    dispatching paid requests while nothing it produces reaches the volume, and
    it can still end green.
    """
    if _COMMIT is None:
        return
    last: Exception | None = None
    for i in range(attempts):
        try:
            _COMMIT()
            return
        except Exception as e:  # noqa: PERF203 - retry is the point
            last = e
            log(f"  commit failed ({why}, attempt {i + 1}/{attempts}): {e}")
            time.sleep(2**i)
    if required:
        raise CommitFailed(
            f"could not publish {why} to the volume after {attempts} attempts: "
            f"{last}. Stopping before another paid request -- a run that keeps "
            "spending while nothing reaches shared storage can still end green."
        )
    log(f"  WARNING dropping non-essential commit ({why}): {last}")


def telemetry(run_dir: str | Path, row: dict[str, Any]) -> None:
    """One durable line per timed event, fsync'd as it happens."""
    row = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    append_jsonl(Path(run_dir) / "telemetry.jsonl", row)


def gpu_snapshot() -> dict[str, Any]:
    """Training-GPU telemetry, best effort.

    Never raises: a missing nvidia-smi or a CPU-only box must not stop a run,
    and an absent metric is more honest than a fabricated zero.
    """
    out: dict[str, Any] = {}
    try:
        import subprocess

        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        for i, line in enumerate(raw.splitlines()):
            util, used, total, temp = [x.strip() for x in line.split(",")]
            out[f"gpu{i}"] = {
                "util_pct": int(float(util)),
                "mem_used_mib": int(float(used)),
                "mem_total_mib": int(float(total)),
                "temp_c": int(float(temp)),
            }
    except Exception as e:
        out["nvidia_smi_error"] = str(e)[:120]
    try:
        import torch

        if torch.cuda.is_available():
            out["torch_peak_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
            out["torch_reserved_gib"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 2
            )
    except Exception:
        pass
    return out


class ScoreStageResult:
    """What the SCORED stage produced, for the train stage to consume."""

    def __init__(
        self,
        scored: dict[str, tuple[list[bytes], list[float]]],
        ledger: dict[str, Any],
        seconds: float,
    ) -> None:
        self.scored = scored
        self.ledger = ledger
        self.seconds = seconds


def run_score_stage(
    update: UpdateDir,
    actions: list[Any],
    rendered: dict[str, list[dict[str, Any]]],
    *,
    teacher_tok: Any,
    pool: Any,
    run_dir: str | Path,
    paid: dict[str, Any],
    drop_keys: set[str] | None = None,
    provenance: Callable[[Any], dict[str, Any]] | None = None,
) -> ScoreStageResult:
    """Score every action in this update, reusing what was already paid for.

    Shared verbatim between the two drivers, including the part that matters
    most: cached bytes are re-validated even after the SCORED marker exists. An
    earlier tokenizer bug can leave a paid row whose logprobs are valid for the
    token ids but whose reconstructed bytes contain U+FFFD; blindly trusting the
    marker makes every resume fail forever in the optimizer.

    `drop_keys` is the live path's addition. `score_replay_batch` reuses a paid
    score when its bytes reconstruct the action, which is sufficient for a
    frozen prefix id but not for a live turn key that can recur within one
    update after a discard-and-resample. The driver computes the stale set with
    `live_turns.stale_score_keys`; replay passes nothing and behaves as before.

    `provenance` lets the live driver attach the semantic-history binding to
    each persisted score row, which is what makes the staleness check possible
    on the *next* resume.
    """
    from vektori_trace.replay_score import score_replay_batch

    drop = set(drop_keys or ())
    if drop:
        log(f"  dropping {len(drop)} stale cached score(s): {sorted(drop)[:4]}")

    if paid:
        log(f"  checking {len(paid)} cached teacher scores")

    already = {
        k: (
            [base64.b64decode(b) for b in r["teacher_token_bytes_b64"]],
            [float(x) for x in r["teacher_logprobs"]],
        )
        for k, r in paid.items()
        if r.get("teacher_token_bytes_b64") and k not in drop
    }

    def _persist(sc: Any) -> None:
        # Every one of these was billed. Committing per score is the difference
        # between a crash costing seconds and costing the batch.
        row = {
            "key": sc.key,
            "teacher_token_bytes_b64": [
                base64.b64encode(b).decode() for b in sc.teacher_token_bytes
            ],
            "teacher_logprobs": list(sc.teacher_logprobs),
            "n_prefix_tokens": sc.n_prefix_tokens,
            "n_trailing_dropped": sc.n_trailing_dropped,
        }
        if provenance is not None:
            row.update(provenance(sc))
        existing = read_jsonl(update.scores_path)
        if any(r.get("key") == sc.key for r in existing):
            atomic_write_jsonl(
                update.scores_path,
                [row if r.get("key") == sc.key else r for r in existing],
            )
        else:
            append_jsonl(update.scores_path, row)
        commit_scores("score")

    t_score = time.time()
    scored, ledger = score_replay_batch(
        actions,
        rendered,
        teacher_tok,
        pool,
        on_scored=_persist,
        already_scored=already,
    )
    score_s = round(time.time() - t_score, 1)

    tin = ledger.get("teacher_input_tokens", 0)
    n_new = int(ledger.get("n_newly_scored", 0))
    if not update.reached("SCORED"):
        telemetry(
            run_dir,
            {
                "event": "stage",
                "stage": "SCORED",
                "update": update.index,
                "seconds": score_s,
                "n_scores": len(scored),
                "teacher_input_tokens": tin,
                "repeated_prefix_tokens": ledger.get("repeated_prefix_tokens"),
                # $0.22/M uncached (Fireworks deepseek-v4-flash). Upper bound:
                # assumes nothing cached.
                "est_cost_usd_uncached": round(tin * 0.22 / 1e6, 4),
            },
        )
        update.mark(
            "SCORED",
            {
                "n_scores": len(scored),
                "seconds": score_s,
                "teacher_input_tokens": tin,
            },
        )
        commit_scores("scored marker")
        log(
            f"  scored {len(scored)} in {score_s}s; {tin:,} teacher input "
            f"tokens (<= ${tin * 0.22 / 1e6:.4f})"
        )
    elif n_new:
        telemetry(
            run_dir,
            {
                "event": "score_cache_repair",
                "update": update.index,
                "n_repaired": n_new,
                "seconds": score_s,
                "teacher_input_tokens": tin,
            },
        )
        commit_scores("score cache repair")
        log(
            f"  repaired {n_new} cached teacher score(s) in {score_s}s; "
            f"reused {ledger.get('n_reused_from_disk', 0)}"
        )
    else:
        log(
            f"  reused all {ledger.get('n_reused_from_disk', 0)} cached "
            "teacher scores"
        )
    return ScoreStageResult(scored, ledger, score_s)


def run_train_stage(
    update: UpdateDir,
    prefixes: list[Any],
    actions: list[Any],
    scored: dict[str, tuple[list[bytes], list[float]]],
    trainer: Any,
    *,
    run_dir: str | Path,
    policy_version: str,
    max_new_tokens: int,
    max_trace_share: float,
    selection_policy: str,
    n_samples_per_prefix: int = 1,
) -> dict[str, Any]:
    """One optimizer step, checkpointed and committed. Shared by both drivers.

    Returns the trainer's checkpoint state. Idempotent across a resume: a
    TRAINED marker short-circuits to the state already on disk, because the
    optimizer step is the one stage that must never run twice for one batch.
    """
    from vektori_trace.replay_opd import run_replay_chunk_opd

    if update.reached("TRAINED"):
        log("  already trained; skipping")
        return json.loads((update.checkpoint_path / "state.json").read_text())

    t_train = time.time()
    telemetry(
        run_dir,
        {
            "event": "gpu",
            "when": "pre_train",
            "update": update.index,
            **gpu_snapshot(),
        },
    )
    report = run_replay_chunk_opd(
        prefixes,
        actions,
        scored,
        trainer.step,
        max_new_tokens=max_new_tokens,
        n_samples_per_prefix=n_samples_per_prefix,
        max_trace_share=max_trace_share,
        selection_policy=selection_policy,
    )
    atomic_write_json(update.report_path, report)

    train_s = round(time.time() - t_train, 1)
    gpu = gpu_snapshot()
    # run_replay_chunk_opd nests the optimizer's own report under "optimizer";
    # reading `loss` off the top level silently logs null for the one number
    # the canary exists to produce.
    opt = report.get("optimizer") or {}
    # Advantage spread is the collapse signal: if the student samples the same
    # action every time, the teacher scores it identically and every advantage
    # is ~0.
    advs = [
        a
        for st in (report.get("per_action_stats") or [])
        for a in ([st.get("mean_advantage")] if isinstance(st, dict) else [])
        if a is not None
    ]
    telemetry(
        run_dir,
        {
            "event": "stage",
            "stage": "TRAINED",
            "update": update.index,
            "seconds": train_s,
            "loss": opt.get("loss"),
            "clip_fraction": opt.get("clip_fraction"),
            "n_examples": opt.get("n_examples"),
            "grad_norm": opt.get("grad_norm"),
            "supervised_tokens": report.get("global_supervised_tokens"),
            "advantage_spread": (
                {
                    "min": min(advs),
                    "max": max(advs),
                    "n_distinct": len({round(a, 6) for a in advs}),
                }
                if advs
                else None
            ),
            "spread": report.get("spread"),
            "realized_step_histogram": report.get("realized_step_histogram"),
            **gpu,
        },
    )

    state = trainer.checkpoint(
        update.checkpoint_path,
        update_index=update.index,
        policy_version=policy_version,
    )
    # The adapter is the run's entire output. Commit it before anything else
    # can fail, and again after the marker, so a container killed between the
    # two still leaves reloadable weights on the volume.
    commit_scores("checkpoint")
    update.mark(
        "TRAINED",
        {
            "loss": opt.get("loss"),
            "seconds": train_s,
            "adapter_hash": state.get("adapter_hash"),
        },
    )
    commit_scores("trained marker")
    log(
        f"  trained in {train_s}s; loss={opt.get('loss')} "
        f"adapter={state.get('adapter_hash')} "
        f"peak={gpu.get('torch_peak_gib')}GiB "
        f"util={gpu.get('gpu0', {}).get('util_pct')}%"
    )
    return state
