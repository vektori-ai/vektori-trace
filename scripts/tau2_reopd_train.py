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

    from vektori_trace.tau2.reopd_sample import ReOPDSampleError

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
    ctx = entry.get("max_model_len") or entry.get("context_length")
    if ctx is None:
        raise ReOPDSampleError(
            f"endpoint did not report a context length for {model!r}; it cannot "
            "be verified against the 14,928 tokens this corpus needs"
        )
    need = LONGEST_PREFIX_TOKENS + MAX_ACTION_TOKENS
    if int(ctx) < need:
        raise ReOPDSampleError(
            f"endpoint context {ctx} < {need} required "
            f"({LONGEST_PREFIX_TOKENS} longest prefix + {MAX_ACTION_TOKENS} cap). "
            f"Relaunch with --max-model-len {REQUIRED_CONTEXT}."
        )
    log(f"endpoint ok: {model}, context {ctx}")
    return {"served_model": model, "max_model_len": int(ctx), "models": served}


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
        captures = sample_batch(
            prefixes,
            api_base=args.api_base, model=args.model,
            tokenizer=student_tok, policy_version=policy_version,
            max_tokens=MAX_ACTION_TOKENS, temperature=args.temperature,
            n_samples=1, already=prior,
            on_capture=lambda c: append_jsonl(u.actions_path, c),
        )
        u.mark("SAMPLED", {"n_actions": len(captures)})
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
            append_jsonl(u.scores_path, {
                "key": sc.key,
                "teacher_token_bytes_b64": [
                    base64.b64encode(b).decode() for b in sc.teacher_token_bytes],
                "teacher_logprobs": list(sc.teacher_logprobs),
                "n_prefix_tokens": sc.n_prefix_tokens,
                "n_trailing_dropped": sc.n_trailing_dropped,
            })

        scored, ledger = score_replay_batch(
            actions, rendered, teacher_tok, pool,
            on_scored=_persist, already_scored=already,
        )
        u.mark("SCORED", {"n_scores": len(scored),
                          "teacher_input_tokens": ledger.get("teacher_input_tokens")})
        log(f"  scored {len(scored)}; "
            f"{ledger.get('teacher_input_tokens', 0):,} teacher input tokens")
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

    state = trainer.checkpoint(
        u.checkpoint_path, update_index=idx, policy_version=policy_version,
    )
    u.mark("TRAINED", {"loss": report.get("loss"),
                       "adapter_hash": state.get("adapter_hash")})
    log(f"  trained; adapter {state.get('adapter_hash')}")
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
    ap.add_argument("--teacher-tokenizer", default="deepseek-ai/DeepSeek-V3")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-task-share", type=float, default=0.5)
    ap.add_argument("--n-updates", type=int, default=N_UPDATES)
    ap.add_argument("--n-per-update", type=int, default=N_PER_UPDATE)
    ap.add_argument("--expect-manifest-hash", default="8e78c7b96161d024")

    ap.add_argument("--dry-run", action="store_true",
                    help="plan and validate only: no endpoint, teacher or GPU")
    ap.add_argument("--canary", type=int, default=0,
                    help="run ONE update with this many prefixes, then stop")
    ap.add_argument("--yes", action="store_true",
                    help="required for the full paid run")
    a = ap.parse_args()

    run_dir = a.run_dir or f"runs/reopd-{time.strftime('%Y%m%d-%H%M%S')}"

    # --- load and prove the context --------------------------------------
    policy, prep = recover_system_policy(a.artifacts,
                                         simulations_dir=a.simulations_dir)
    log(f"policy {prep['policy_sha256'][:16]} ({prep['policy_chars']:,} chars)")

    prefixes, corpus = load_c30_prefixes(
        a.artifacts, system_policy=policy,
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
