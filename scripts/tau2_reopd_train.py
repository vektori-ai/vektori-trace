#!/usr/bin/env python3
"""Run the Tau2 ReOPD arm: 32 updates x 16 C30 states, resumably.

The driver. Everything it needs already exists as tested modules; this is the
loop that wires them together and the durable bookkeeping that makes a crash
survivable.

    for update in 0..31:
        16 prefixes from the FROZEN schedule (not re-derived here)
        student samples ONE action each, from frozen prompt token ids
        DeepSeek scores those exact bytes under its own render
        chunk-OPD batch -> one optimizer step
        save adapter + optimizer + scheduler + RNG + update index
        reload and prove the adapter changed the logits
        the next update samples from that checkpoint

Why the bookkeeping is not optional
-----------------------------------
Every update buys teacher calls. A crash at update 20 must resume at update 20
with the ~320 scores already paid for, the optimizer's Adam moments intact, and
the policy version advanced correctly. Restarting from CK35 with a fresh
optimizer would silently produce a different experiment that reports success at
every step.

Staged, and each stage is separately authorized:

    --dry-run       no endpoint, no teacher, no GPU: plan and validate only
    --canary N      N prefixes, one update, then stop (small paid spend)
    --yes           the full 32-update run

The environment is never queried after a student action. This is an offline
replay experiment; the history is a teacher replay and only the action is
on-policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.tau2.c30_loader import (  # noqa: E402
    assert_render_parity,
    load_c30_prefixes,
    recover_system_policy,
)
from vektori_trace.tau2.reopd_schedule import (  # noqa: E402
    N_PER_UPDATE,
    N_UPDATES,
    batch_for,
    build_schedule,
    describe,
    freeze_schedule,
)
from vektori_trace.tau2.reopd_state import (  # noqa: E402
    RunState,
    append_jsonl,
    atomic_write_json,
)

#: Stored max action is 592 tokens; 2048 is ~3.5x that. A cap hit invalidates
#: the sample rather than being scored -- a fragment is not a decision.
MAX_ACTION_TOKENS = 2048

#: The longest C30 prefix is 12,880 tokens. With the action cap that needs
#: 14,928, so a 12,288-token server cannot serve this corpus and would drop
#: tokens from the FRONT of the prompt, sampling a state the run cannot
#: describe while every downstream assertion still passes.
REQUIRED_CONTEXT = 16384
LONGEST_PREFIX_TOKENS = 12_880


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_snapshot() -> dict[str, Any]:
    """Training-GPU telemetry, best effort.

    Never raises: a missing nvidia-smi or a CPU-only box must not stop a run,
    and an absent metric is more honest than a fabricated zero.
    """
    out: dict[str, Any] = {}
    try:
        import subprocess
        raw = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        for i, line in enumerate(raw.splitlines()):
            util, used, total, temp = [x.strip() for x in line.split(",")]
            out[f"gpu{i}"] = {"util_pct": int(float(util)),
                              "mem_used_mib": int(float(used)),
                              "mem_total_mib": int(float(total)),
                              "temp_c": int(float(temp))}
    except Exception as e:
        out["nvidia_smi_error"] = str(e)[:120]
    try:
        import torch
        if torch.cuda.is_available():
            out["torch_peak_gib"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            out["torch_reserved_gib"] = round(
                torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
    except Exception:
        pass
    return out


#: Called after every durable write. On Modal, fsync only reaches the
#: container's local disk -- a Volume publishes nothing to shared storage until
#: `vol.commit()`. A container killed by OOM, timeout, `app stop` or an
#: infrastructure fault never runs a `finally`, so an uncommitted score is a
#: score that was paid for and lost.
_COMMIT: Any = None


def set_commit_fn(fn: Any) -> None:
    """Install the volume-commit callback (Modal), or leave it None locally."""
    global _COMMIT
    _COMMIT = fn


class CommitFailed(RuntimeError):
    """A paid artifact could not be published to shared storage."""


def commit(why: str = "", *, required: bool = True, attempts: int = 3) -> None:
    """Publish everything written so far, and stop the run if it will not.

    A swallowed commit failure is the worst available outcome: the run keeps
    dispatching paid requests while nothing it produces reaches the volume, and
    it can still end green. So a required commit retries briefly and then
    raises -- better to stop before the next paid call than to finish a run
    whose adapter was never published.

    `required=False` is for telemetry-only commits, where losing a log line is
    not worth killing a run over.
    """
    if _COMMIT is None:
        return
    last: Exception | None = None
    for i in range(attempts):
        try:
            _COMMIT()
            return
        except Exception as e:               # noqa: PERF203 - retry is the point
            last = e
            log(f"  commit failed ({why}, attempt {i + 1}/{attempts}): {e}")
            time.sleep(2 ** i)
    if required:
        raise CommitFailed(
            f"could not publish {why} to the volume after {attempts} attempts: "
            f"{last}. Stopping before another paid request -- a run that keeps "
            "spending while nothing reaches shared storage can still end green."
        )
    log(f"  WARNING dropping non-essential commit ({why}): {last}")


def telemetry(run_dir: str | Path, row: dict[str, Any]) -> None:
    """One durable line per timed event, fsync'd as it happens.

    Separate from train_progress.jsonl, which replay_train owns and which only
    covers the optimizer step. This covers sampling and scoring too -- the two
    stages that dominate wall clock and therefore dominate GPU cost.
    """
    row = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    append_jsonl(Path(run_dir) / "telemetry.jsonl", row)


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


def verify_endpoint(api_base: str, model: str, *, timeout: float = 30.0) -> dict:
    """Refuse an endpoint that cannot serve this corpus.

    Reads the server's *reported* context rather than trusting the launch flag:
    a `--max-model-len` typo is invisible otherwise, and the failure it causes
    is a silently truncated prompt rather than an error.
    """
    import urllib.request

    from vektori_trace.tau2.reopd_sample import ReOPDSampleError, post_json

    url = api_base.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = json.loads(r.read().decode())

    served = [m.get("id") for m in (body.get("data") or [])]
    if model not in served:
        raise ReOPDSampleError(
            f"endpoint does not advertise {model!r}; it serves {served}. vLLM "
            "resolves an unknown name against the base model, so the adapter "
            "would silently do nothing."
        )

    entry = next(m for m in body["data"] if m.get("id") == model)
    ctx = (entry.get("max_model_len") or entry.get("context_length")
           or entry.get("max_context_length"))
    need = LONGEST_PREFIX_TOKENS + MAX_ACTION_TOKENS

    if ctx is None:
        # vLLM does not expose a per-model context in every version, and a LoRA
        # entry inherits the base model's window rather than carrying its own.
        # Rather than refuse a healthy endpoint over a missing metadata field,
        # probe the thing that actually matters: does the server accept a prompt
        # as long as the longest prefix this corpus will send?
        probe = [1] * need
        st, body2 = post_json(
            api_base.rstrip("/") + "/completions",
            {"model": model, "prompt": probe, "max_tokens": 1,
             "temperature": 0.0}, timeout)
        if st != 200:
            raise ReOPDSampleError(
                f"endpoint reports no context length AND rejected a "
                f"{need}-token probe (HTTP {st}): {str(body2)[:200]}. The "
                f"longest C30 prefix is {LONGEST_PREFIX_TOKENS} tokens and the "
                f"action cap is {MAX_ACTION_TOKENS}, so this server cannot "
                f"serve this corpus. Relaunch with --max-model-len "
                f"{REQUIRED_CONTEXT}."
            )
        log(f"endpoint ok: {model}, context unreported but a {need}-token "
            "prompt was accepted")
        return {"served_model": model, "max_model_len": None,
                "probe_tokens_accepted": need, "models": served}

    if int(ctx) < need:
        raise ReOPDSampleError(
            f"endpoint context {ctx} < {need} required "
            f"({LONGEST_PREFIX_TOKENS} longest prefix + {MAX_ACTION_TOKENS} cap). "
            f"Relaunch with --max-model-len {REQUIRED_CONTEXT}."
        )
    log(f"endpoint ok: {model}, context {ctx}")
    return {"served_model": model, "max_model_len": int(ctx), "models": served}


def refresh_serving_policy(args, idx: int, checkpoint_path, state) -> dict:
    """Serve the checkpoint update `idx-1` produced, before sampling update idx.

    ReOPD is on-policy in the action: `log pi_old` must come from the policy
    that sampled it. An endpoint still serving CK35 at update 7 makes every
    importance ratio compare two different distributions, with a finite loss and
    nothing in any log to show for it.

    Update 0 needs no refresh -- CK35 *is* the current policy there, which is
    why a one-update canary is valid against a static endpoint.
    """
    from vektori_trace.tau2.reopd_refresh import refresh_policy

    name = f"{args.model}-u{idx - 1:03d}"
    rep = refresh_policy(
        args.api_base,
        new_name=name,
        new_path=str(checkpoint_path),
        probe_prompt_ids=args.probe_prompt_ids,
        previous_logprobs=getattr(args, "probe_logprobs", None),
        previous_name=getattr(args, "served_name", None),
    )
    args.model = name
    args.served_name = name
    args.probe_logprobs = rep["probe_logprobs"]
    log(f"  endpoint now serves {name} "
        f"(probe delta {rep.get('max_logprob_delta', 'n/a')})")
    telemetry(args.run_dir_resolved,
              {"event": "refresh", "update": idx, **rep})
    return rep


# ---------------------------------------------------------------------------
# one update
# ---------------------------------------------------------------------------


def run_update(
    idx: int,
    prefixes_by_id: dict[str, Any],
    schedule: dict,
    run: RunState,
    args,
    *,
    student_tok,
    teacher_tok,
    pool,
    trainer,
    policy_version: str,
) -> dict:
    """PLANNED -> SAMPLED -> SCORED -> TRAINED for one update."""
    from vektori_trace.replay_opd import run_replay_chunk_opd
    from vektori_trace.replay_score import score_replay_batch
    from vektori_trace.tau2.reopd_sample import (
        capture_to_sampled_action,
        sample_batch,
    )

    u = run.update(idx)
    u.validate()

    # Before anything is sampled or paid for: the endpoint must serve the
    # policy this update is meant to sample from.
    if idx > 0 and not u.reached("SAMPLED"):
        prev = run.update(idx - 1)
        prev.validate_checkpoint()
        refresh_serving_policy(args, idx, prev.checkpoint_path, None)

    batch_ids = batch_for(schedule, idx)
    prefixes = [prefixes_by_id[pid] for pid in batch_ids]

    if not u.reached("PLANNED"):
        u.mark("PLANNED", {"prefix_ids": batch_ids,
                           "policy_version": policy_version})

    # --- sample -----------------------------------------------------------
    if not u.reached("SAMPLED"):
        prior = {r["key"]: r for r in _read(u.actions_path)}
        if prior:
            log(f"  reusing {len(prior)} captures already on disk")
        t_sample = time.time()
        n_done = [0]

        def _landed(c: dict) -> None:
            append_jsonl(u.actions_path, c)
            # Behaviour logprobs cannot be recreated after sampling, so each
            # capture is published the moment it lands.
            commit("capture")
            n_done[0] += 1
            telemetry(args.run_dir_resolved, {
                "event": "sample", "update": idx, "key": c["key"],
                "n_prompt_tokens": len(c["prompt_token_ids"]),
                "n_action_tokens": len(c["action_token_ids"]),
                "finish_reason": c.get("finish_reason"),
                "elapsed_s": round(time.time() - t_sample, 2),
                "i": n_done[0], "of": len(prefixes),
            })
            log(f"    sample {n_done[0]}/{len(prefixes)} {c['key']} "
                f"prompt={len(c['prompt_token_ids'])} "
                f"action={len(c['action_token_ids'])} tok")

        captures = sample_batch(
            prefixes,
            api_base=args.api_base, model=args.model,
            tokenizer=student_tok, policy_version=policy_version,
            max_tokens=MAX_ACTION_TOKENS, temperature=args.temperature,
            n_samples=1, already=prior,
            on_capture=_landed,
        )
        sample_s = round(time.time() - t_sample, 1)
        telemetry(args.run_dir_resolved, {
            "event": "stage", "stage": "SAMPLED", "update": idx,
            "seconds": sample_s, "n_actions": len(captures),
            "n_action_tokens": sum(len(c["action_token_ids"]) for c in captures),
        })
        u.mark("SAMPLED", {"n_actions": len(captures), "seconds": sample_s})
        commit("sampled marker")
        log(f"  sampling took {sample_s}s")
    captures = _read(u.actions_path)
    log(f"  sampled {len(captures)} actions")

    actions = [capture_to_sampled_action(c) for c in captures]
    rendered = {p.prefix_id: p.canonical_messages for p in prefixes}

    # --- score ------------------------------------------------------------
    if not u.reached("SCORED"):
        paid = run.paid_scores(idx)
        if paid:
            log(f"  reusing {len(paid)} teacher scores already paid for")

        import base64
        already = {
            k: ([base64.b64decode(b) for b in r["teacher_token_bytes_b64"]],
                [float(x) for x in r["teacher_logprobs"]])
            for k, r in paid.items() if r.get("teacher_token_bytes_b64")
        }

        def _persist(sc) -> None:
            # Every one of these was billed. Committing per score is the
            # difference between a crash costing seconds and costing the batch.
            append_jsonl(u.scores_path, {
                "key": sc.key,
                "teacher_token_bytes_b64": [
                    base64.b64encode(b).decode() for b in sc.teacher_token_bytes],
                "teacher_logprobs": list(sc.teacher_logprobs),
                "n_prefix_tokens": sc.n_prefix_tokens,
                "n_trailing_dropped": sc.n_trailing_dropped,
            })
            commit("score")

        t_score = time.time()
        scored, ledger = score_replay_batch(
            actions, rendered, teacher_tok, pool,
            on_scored=_persist, already_scored=already,
        )
        score_s = round(time.time() - t_score, 1)
        tin = ledger.get("teacher_input_tokens", 0)
        telemetry(args.run_dir_resolved, {
            "event": "stage", "stage": "SCORED", "update": idx,
            "seconds": score_s, "n_scores": len(scored),
            "teacher_input_tokens": tin,
            "repeated_prefix_tokens": ledger.get("repeated_prefix_tokens"),
            # $0.22/M uncached, $0.007/M cached (Fireworks deepseek-v4-flash).
            # Upper bound: assumes nothing cached.
            "est_cost_usd_uncached": round(tin * 0.22 / 1e6, 4),
        })
        u.mark("SCORED", {"n_scores": len(scored), "seconds": score_s,
                          "teacher_input_tokens": tin})
        commit("scored marker")
        log(f"  scored {len(scored)} in {score_s}s; {tin:,} teacher input "
            f"tokens (<= ${tin * 0.22 / 1e6:.4f})")
    else:
        import base64
        scored = {
            k: ([base64.b64decode(b) for b in r["teacher_token_bytes_b64"]],
                [float(x) for x in r["teacher_logprobs"]])
            for k, r in run.paid_scores(idx).items()
        }

    # --- train ------------------------------------------------------------
    if u.reached("TRAINED"):
        log("  already trained; skipping")
        return json.loads((u.checkpoint_path / "state.json").read_text())

    t_train = time.time()
    telemetry(args.run_dir_resolved,
              {"event": "gpu", "when": "pre_train", "update": idx,
               **gpu_snapshot()})
    report = run_replay_chunk_opd(
        prefixes, actions, scored, trainer.step,
        max_new_tokens=MAX_ACTION_TOKENS,
        n_samples_per_prefix=1,
        # Task share, not trace share: in C30 one task is one trace, and the
        # concentration that matters is a single task dominating an update.
        max_trace_share=args.max_task_share,
        selection_policy="tau2-c30-task-first-frozen",
    )
    atomic_write_json(u.report_path, report)

    train_s = round(time.time() - t_train, 1)
    gpu = gpu_snapshot()
    # run_replay_chunk_opd nests the optimizer's own report under "optimizer";
    # reading `loss` off the top level silently logs null for the one number
    # the canary exists to produce.
    opt = report.get("optimizer") or {}
    # Advantage spread is the collapse signal: if ck35 samples the same action
    # every time the teacher scores it identically and every advantage is ~0.
    advs = [a for st in (report.get("per_action_stats") or [])
            for a in ([st.get("mean_advantage")] if isinstance(st, dict) else [])
            if a is not None]
    telemetry(args.run_dir_resolved, {
        "event": "stage", "stage": "TRAINED", "update": idx,
        "seconds": train_s,
        "loss": opt.get("loss"),
        "clip_fraction": opt.get("clip_fraction"),
        "n_examples": opt.get("n_examples"),
        "grad_norm": opt.get("grad_norm"),
        "supervised_tokens": report.get("global_supervised_tokens"),
        "advantage_spread": (
            {"min": min(advs), "max": max(advs),
             "n_distinct": len({round(a, 6) for a in advs})} if advs else None),
        "spread": report.get("spread"),
        "realized_step_histogram": report.get("realized_step_histogram"),
        **gpu,
    })

    state = trainer.checkpoint(
        u.checkpoint_path, update_index=idx, policy_version=policy_version,
    )
    # The adapter is the run's entire output. Commit it before anything else
    # can fail, and again after the marker, so a container killed between the
    # two still leaves reloadable weights on the volume.
    commit("checkpoint")
    u.mark("TRAINED", {"loss": opt.get("loss"), "seconds": train_s,
                       "adapter_hash": state.get("adapter_hash")})
    commit("trained marker")
    log(f"  trained in {train_s}s; loss={opt.get('loss')} "
        f"adapter={state.get('adapter_hash')} "
        f"peak={gpu.get('torch_peak_gib')}GiB "
        f"util={gpu.get('gpu0', {}).get('util_pct')}%")
    return state


def _read(path: Path) -> list[dict]:
    from vektori_trace.tau2.reopd_state import read_jsonl
    return read_jsonl(path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--simulations-dir", default="/data/tau2/data/simulations")
    ap.add_argument("--run-dir", default=None,
                    help="default: ./runs/reopd-<timestamp>")
    ap.add_argument("--parent", required=False,
                    default="/adapters/tau2/runs/a_warm_20260825_003343/checkpoint-35",
                    help="CK35: the frozen A_warm both branches start from")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE", ""))
    ap.add_argument("--model", default="ck35")
    ap.add_argument("--teacher-model",
                    default="accounts/fireworks/models/deepseek-v4-flash-0731")
    # Must match the served teacher. A different DeepSeek tokenizer produces
    # different token boundaries for the same bytes, so every logprob would be
    # indexed against a span the teacher never scored -- finite, plausible, and
    # wrong. This is the pin OPD-MULTITURN-PLAN records and run_replay_opd uses.
    ap.add_argument("--teacher-tokenizer",
                    default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-task-share", type=float, default=0.5)
    ap.add_argument("--n-updates", type=int, default=N_UPDATES)
    ap.add_argument("--n-per-update", type=int, default=N_PER_UPDATE)
    ap.add_argument("--expect-manifest-hash", default="8e78c7b96161d024")
    ap.add_argument("--tools-file", default=None,
                    help="retail tool schemas as JSON. Required wherever the "
                         "tau2 package is not installed -- a Modal training "
                         "image has no reason to carry a benchmark harness just "
                         "to read a static schema list. The hash is verified "
                         "against the corpus either way.")
    ap.add_argument("--policy-file", default=None,
                    help="read the system policy from this file instead of "
                         "recovering it from the simulation files. Required "
                         "wherever the simulations are not reachable (a Modal "
                         "container cannot see the box's /data/tau2). The "
                         "policy is hashed into the run manifest either way, "
                         "and render parity proves it is the right one.")

    ap.add_argument("--dry-run", action="store_true",
                    help="plan and validate only: no endpoint, teacher or GPU")
    ap.add_argument("--canary", type=int, default=0,
                    help="run ONE update with this many prefixes, then stop")
    ap.add_argument("--yes", action="store_true",
                    help="required for the full paid run")
    a = ap.parse_args()

    run_dir = a.run_dir or f"runs/reopd-{time.strftime('%Y%m%d-%H%M%S')}"
    a.run_dir_resolved = run_dir

    # --- load and prove the context --------------------------------------
    if a.policy_file:
        import hashlib
        policy = Path(a.policy_file).read_text()
        prep = {"policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                "policy_chars": len(policy), "source": a.policy_file}
    else:
        policy, prep = recover_system_policy(a.artifacts,
                                             simulations_dir=a.simulations_dir)
    log(f"policy {prep['policy_sha256'][:16]} ({prep['policy_chars']:,} chars)")

    tools = None
    if a.tools_file:
        tools = json.load(open(a.tools_file))
        log(f"tools {len(tools)} schemas from {a.tools_file}")

    prefixes, corpus = load_c30_prefixes(
        a.artifacts, system_policy=policy, tools=tools,
        expect_manifest_hash=a.expect_manifest_hash,
    )
    log(f"corpus {corpus['n_prefixes']} prefixes / {corpus['n_tasks']} tasks, "
        f"manifest {corpus['prefix_manifest_hash']}")

    schedule = build_schedule(
        [p.prefix_id for p in prefixes],
        n_updates=1 if a.canary else a.n_updates,
        n_per_update=a.canary or a.n_per_update,
    )
    log(f"schedule {describe(schedule)}")

    run = RunState(run_dir, n_updates=schedule["n_updates"])
    manifest = {
        "artifacts": a.artifacts,
        "parent": a.parent,
        "base_model": a.base_model,
        "teacher_model": a.teacher_model,
        "teacher_tokenizer": a.teacher_tokenizer,
        "policy_sha256": prep["policy_sha256"],
        "prefix_manifest_hash": corpus["prefix_manifest_hash"],
        "split_manifest_hash": corpus["split_manifest_hash"],
        "schedule_hash": schedule["schedule_hash"],
        "n_updates": schedule["n_updates"],
        "n_per_update": schedule["n_per_update"],
        "max_action_tokens": MAX_ACTION_TOKENS,
        "required_context": REQUIRED_CONTEXT,
        "learning_rate": a.learning_rate,
        "temperature": a.temperature,
        "max_task_share": a.max_task_share,
        "sampling": "task-first frozen order (NOT kappa^t)",
        "mode": "canary" if a.canary else "full",
    }

    if a.dry_run:
        os.makedirs(run_dir, exist_ok=True)
        freeze_schedule(os.path.join(run_dir, "schedule.json"), schedule)
        run.freeze_manifest(manifest)
        atomic_write_json(Path(run_dir) / "dry_run.json", {
            "manifest": manifest, "corpus": corpus,
            "first_update": batch_for(schedule, 0),
            "prompt_tokens": corpus["prompt_tokens"],
        })
        log(f"DRY RUN ok -> {run_dir}")
        log(f"  update 0 would sample: {batch_for(schedule, 0)[:4]} ...")
        log("  nothing sampled, scored, or trained")
        return 0

    if not a.canary and not a.yes:
        raise SystemExit(
            "refusing the full run without --yes. Use --dry-run to validate, "
            "or --canary N for one small paid update first."
        )
    if not a.api_base:
        raise SystemExit("--api-base is required (or set STUDENT_API_BASE)")

    # --- verify before spending ------------------------------------------
    endpoint = verify_endpoint(a.api_base, a.model)
    manifest["endpoint"] = endpoint

    from transformers import AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(a.base_model)
    log("checking render parity before any paid call ...")
    parity = assert_render_parity(prefixes, student_tok,
                                  max_length=REQUIRED_CONTEXT)
    manifest["render_parity"] = parity
    log(f"parity ok: {parity['n_checked']} prefixes")

    os.makedirs(run_dir, exist_ok=True)
    freeze_schedule(os.path.join(run_dir, "schedule.json"), schedule)
    run.freeze_manifest(manifest)

    # --- resume ------------------------------------------------------------
    start = run.resume_point()
    if start >= schedule["n_updates"]:
        log(f"run already complete ({start}/{schedule['n_updates']})")
        return 0
    if start > 0:
        log(f"resuming at update {start}")

    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.tau2.reopd_trainer import ReOPDTrainer
    from vektori_trace.vocab_bridge import load_tokenizer

    teacher_tok = load_tokenizer(a.teacher_tokenizer)
    pool = FireworksTeacherPool(model=a.teacher_model)
    trainer = ReOPDTrainer(
        base_model=a.base_model, parent_adapter=a.parent,
        learning_rate=a.learning_rate, run_dir=run_dir,
    )
    trainer.load(resume_from=run.update(start - 1).checkpoint_path
                 if start > 0 else None)

    by_id = {p.prefix_id: p for p in prefixes}
    # A fixed prompt whose greedy logprobs fingerprint the served policy. Taken
    # from the first prefix's real ids so the probe exercises the same shape the
    # run does; truncated because a fingerprint needs a few tokens, not 12k.
    a.probe_prompt_ids = prefixes[0].prompt_token_ids[:256]
    a.served_name = a.model

    for idx in range(start, schedule["n_updates"]):
        log(f"update {idx}/{schedule['n_updates'] - 1}")
        run_update(idx, by_id, schedule, run, a,
                   student_tok=student_tok, teacher_tok=teacher_tok,
                   pool=pool, trainer=trainer,
                   policy_version=f"reopd-u{idx:03d}")

    log(f"done: {run.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
