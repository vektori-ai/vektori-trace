"""Durable run logging for Tau2 SFT.

Terminal output is not a record. A run that OOMs at step 40 on a remote GPU
leaves nothing behind unless each step was flushed to disk as it happened, and
"the loss looked fine" is not reconstructable afterwards from a scrollback that
died with the container.

Files, each independently useful:

    run_config.json        written BEFORE training, so a crashed run still says
                           exactly what it was trying to do
    train_steps.jsonl      one flushed line per optimizer step
    checkpoints.jsonl      path, step, and hash of every saved adapter
    probe_generations.jsonl  raw prompt/output/parser evidence, kept apart from
                           metrics because transcripts are orders of magnitude
                           larger
    run_summary.json       final aggregate
    failure.json           written on any exception, OOM or gate failure

Gradient norms come from TRL's own `on_log` payload rather than from reading
`p.grad` after training: `Trainer` zeroes gradients after each optimizer step,
so a post-hoc `p.grad` inspection reads zeros and would report "no gradient"
for a perfectly healthy run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from typing import Any


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_environment(model: str) -> dict[str, Any]:
    """Versions and revisions that change results if they drift."""
    env: dict[str, Any] = {"model": model}
    for mod in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:
            env[mod] = f"unavailable: {type(e).__name__}"
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else "cpu")
        env["gpu_total_gib"] = (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            if torch.cuda.is_available() else None)
    except Exception:
        pass
    try:
        from transformers import AutoConfig
        env["base_model_revision"] = getattr(
            AutoConfig.from_pretrained(model, trust_remote_code=True),
            "_commit_hash", None)
    except Exception:
        pass
    return env


class RunLog:
    """Owns the run's directory. Every write flushes."""

    #: Files whose presence means a previous run already used this directory.
    RUN_ARTIFACTS = ("train_steps.jsonl", "run_config.json", "run_summary.json",
                     "checkpoints.jsonl", "failure.json")

    def __init__(self, out_dir: str, *, allow_existing: bool = False):
        """Own `out_dir` exclusively.

        The step files are opened in append mode so a crash cannot truncate what
        was already flushed. That makes reusing a directory dangerous: a probe's
        three steps would sit in front of a full run's hundred, and the gradient
        check -- which reads `train_steps.jsonl` back -- would be judging a
        mixture of two runs. So a directory that already holds run artifacts is
        refused rather than appended to.
        """
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        stale = [f for f in self.RUN_ARTIFACTS
                 if os.path.exists(os.path.join(out_dir, f))]
        if stale and not allow_existing:
            raise FileExistsError(
                f"{out_dir} already contains run artifacts {stale}. Appending "
                "would interleave two runs in one log; use a fresh output "
                "directory."
            )
        self.t0 = time.time()
        self._steps = open(os.path.join(out_dir, "train_steps.jsonl"), "a")
        self._ckpts = open(os.path.join(out_dir, "checkpoints.jsonl"), "a")
        self._gens = None

    # ---- config -------------------------------------------------------
    def write_config(self, config: dict[str, Any]) -> None:
        config = dict(config)
        config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(self.dir, "run_config.json"), "w") as fh:
            json.dump(config, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"run_config.json written to {self.dir}", flush=True)

    # ---- per step -----------------------------------------------------
    def step(self, rec: dict[str, Any]) -> None:
        rec = {"timestamp": time.strftime("%H:%M:%S"),
               "elapsed_seconds": round(time.time() - self.t0, 1), **rec}
        self._steps.write(json.dumps(rec) + "\n")
        self._steps.flush()
        os.fsync(self._steps.fileno())
        print(
            f"  step {rec.get('optimizer_step', '?'):>4} "
            f"loss {rec.get('loss', float('nan')):.4f} "
            f"gnorm {rec.get('grad_norm') if rec.get('grad_norm') is not None else 'na'} "
            f"lr {rec.get('learning_rate', 0):.2e} "
            f"vram {rec.get('allocated_vram_gib', 0):.1f}G "
            f"{rec.get('elapsed_seconds', 0):.0f}s",
            flush=True,
        )

    # ---- checkpoints --------------------------------------------------
    def checkpoint(self, path: str, step: int, extra: dict[str, Any] | None = None,
                   commit: Any = None) -> None:
        """Record a checkpoint, and persist it if a volume commit is supplied.

        `commit` is called after the record is flushed: on Modal the volume is
        otherwise only committed when the wrapper returns, so an interruption
        mid-run would lose every intermediate checkpoint it had written.
        """
        # Every regular file, not only weights and config: `optimizer.pt`,
        # `scheduler.pt` and `rng_state.pth` are what make a checkpoint
        # resumable, and a record that omits them cannot prove the checkpoint
        # was written whole.
        files = {}
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                fp = os.path.join(path, fn)
                if os.path.isfile(fp):
                    files[fn] = sha256_file(fp)[:16]
        rec = {"path": path, "optimizer_step": step, "files": files,
               "elapsed_seconds": round(time.time() - self.t0, 1), **(extra or {})}
        # Commit first, then record: writing the line before the commit
        # returns means `committed` can never be true in the file, and the run
        # report would claim every checkpoint was uncommitted.
        err = None
        if commit is not None:
            try:
                commit()
                rec["committed"] = True
            except Exception as e:
                rec["committed"] = False
                rec["commit_error"] = f"{type(e).__name__}: {e}"
                err = e

        self._ckpts.write(json.dumps(rec) + "\n")
        self._ckpts.flush()
        os.fsync(self._ckpts.fileno())
        print(f"  checkpoint step {step}: {len(files)} files -> {path}"
              f"{' [committed]' if rec.get('committed') else ''}", flush=True)

        # A failed commit means the checkpoint is not durable. Printing and
        # continuing would leave the run asserting a guarantee it no longer has.
        if err is not None:
            raise RuntimeError(
                f"volume commit failed after checkpoint at step {step}: "
                f"{type(err).__name__}: {err}. The checkpoint is not durable; "
                "refusing to continue as if it were."
            ) from err

    # ---- generations --------------------------------------------------
    def generation(self, rec: dict[str, Any]) -> None:
        if self._gens is None:
            self._gens = open(os.path.join(self.dir, "probe_generations.jsonl"), "a")
        self._gens.write(json.dumps(rec, default=str) + "\n")
        self._gens.flush()
        os.fsync(self._gens.fileno())

    # ---- terminal states ----------------------------------------------
    def summary(self, rec: dict[str, Any]) -> None:
        rec = dict(rec)
        rec["total_elapsed_seconds"] = round(time.time() - self.t0, 1)
        with open(os.path.join(self.dir, "run_summary.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"run_summary.json written", flush=True)

    def failure(self, exc: BaseException, context: dict[str, Any] | None = None) -> None:
        rec = {
            "error": type(exc).__name__,
            "message": str(exc)[:4000],
            "traceback": traceback.format_exc()[:8000],
            "elapsed_seconds": round(time.time() - self.t0, 1),
            "context": context or {},
        }
        with open(os.path.join(self.dir, "failure.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"FAILURE recorded: {type(exc).__name__}: {exc}", flush=True)

    def close(self) -> None:
        for fh in (self._steps, self._ckpts, self._gens):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass


def make_checkpoint_callback(runlog: RunLog, commit=None):
    """Record and persist each epoch checkpoint as TRL writes it.

    Without this, `checkpoints.jsonl` holds only the final save and the Modal
    volume is committed once at the end -- so an interruption at epoch 2 loses
    all evidence that epoch 1's checkpoint ever existed, even though the bytes
    were written.
    """
    from transformers import TrainerCallback

    class _Ck(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            path = os.path.join(args.output_dir,
                                f"checkpoint-{state.global_step}")
            if os.path.isdir(path):
                runlog.checkpoint(path, state.global_step,
                                  extra={"epoch": round(float(state.epoch or 0), 3)},
                                  commit=commit)

    return _Ck()


def make_callback(runlog: RunLog, supervised_per_step: int):
    """A TrainerCallback that records every `on_log` payload.

    TRL reports `loss` and `grad_norm` here while they are still live; reading
    them off the model afterwards does not work, because Trainer zeroes
    gradients once the optimizer has stepped.
    """
    from transformers import TrainerCallback

    class _Cb(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            alloc = reserved = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 2**30
                    reserved = torch.cuda.memory_reserved() / 2**30
            except Exception:
                pass
            elapsed = max(1e-6, time.time() - runlog.t0)
            runlog.step({
                "optimizer_step": state.global_step,
                "epoch": round(float(state.epoch or 0), 3),
                "loss": float(logs["loss"]),
                "grad_norm": (float(logs["grad_norm"])
                              if logs.get("grad_norm") is not None else None),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
                "supervised_tokens_cumulative": supervised_per_step * state.global_step,
                "tokens_per_second": round(
                    supervised_per_step * state.global_step / elapsed, 1),
                "allocated_vram_gib": round(alloc, 2),
                "reserved_vram_gib": round(reserved, 2),
            })

    return _Cb()
