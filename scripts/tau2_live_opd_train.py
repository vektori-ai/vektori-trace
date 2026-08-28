#!/usr/bin/env python3
"""Run the Tau2 live multi-turn OPD arm, resumably.

The loop `TAU2-OPD-DEEP-DIVE.md` §"Still to build" item 5 specifies::

    for update in 0..N-1:
        serve the checkpoint update-1 produced        (skipped at update 0)
        roll out E complete Tau2 episodes on-policy   -> SAMPLED
        DeepSeek scores every sampled action          -> SCORED
        chunk-OPD over all supervised turns           -> TRAINED
        save adapter + optimizer + scheduler + RNG
        reload and prove the adapter changed the logits
        the next update's episodes are sampled from THAT checkpoint

The last line is the whole point. Replay samples one action from a frozen
prefix; live samples a whole trajectory, and after an update changes the policy
the student visits *different* downstream conversation and tool states. That is
the property static SFT and stored-prefix training cannot reproduce, and it is
why this driver exists rather than another capture script.

Staged, and each stage separately authorized:

    --dry-run            no endpoint, no teacher, no GPU: plan and validate only
    --canary             one update, then stop (small paid spend)
    --two-update-proof   exactly two updates: the smallest run that proves
                         checkpoint -> reload -> rollout from the NEW adapter
    --yes                the full multi-update run

`--canary` deliberately cannot prove the loop. One update ends at a saved
checkpoint, so it exercises everything *except* the transition this driver
exists for. `--two-update-proof` is its own authorization rather than a flavour
of `--yes` for that reason: it is the cheapest run that can fail in the way
that matters.

Nothing here is authorized to spend by default. A GPU or paid teacher call
requires explicit per-run approval; see CLAUDE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.tau2.live_rollout import (  # noqa: E402
    EpisodePlan,
    RolloutSettings,
    Tau2EpisodeRunner,
    capture_live_update,
)
from vektori_trace.tau2.live_train import (  # noqa: E402
    live_max_trace_share,
    load_live_update_inputs,
    refresh_live_policy,
    train_live_update,
)
from vektori_trace.tau2.opd_stages import (  # noqa: E402
    commit_scores,
    log,
    telemetry,
)
from vektori_trace.tau2.reopd_state import RunState, atomic_write_json  # noqa: E402

#: Live actions carry reasoning, so the replay arm's 2048 is not a safe cap
#: here. A cap hit is archived as a FailedTurn and fails the episode rather
#: than being trained: a fragment is not a completed action, which is exactly
#: what the 256-token cap did to the 0/13 run.
MAX_ACTION_TOKENS = 4096


def episode_id_for(update: int, task_id: str, seed: int) -> str:
    """A stable, collision-free episode identity.

    The update index is part of it because the same (task, seed) is rolled out
    again under a *different* policy at every update, and two trajectories that
    share a key would collide in one archive.
    """
    return f"u{update:03d}-task{task_id}-seed{seed}"


def plan_update(
    update: int, task_ids: list[str], seeds: list[int]
) -> list[EpisodePlan]:
    """The episode set for one update: every task crossed with every seed.

    Task-balanced by construction, which is what the per-task share limit is
    there to enforce rather than discover.
    """
    return [
        EpisodePlan(
            episode_id=episode_id_for(update, task_id, seed),
            task_id=task_id,
            seed=seed,
        )
        for task_id in task_ids
        for seed in seeds
    ]


def gen_config_hash(args: argparse.Namespace) -> str:
    """Fingerprint the generation contract an episode was sampled under.

    `batch_report` refuses a batch spanning two of these, so a mid-run
    temperature change becomes an error instead of a silent confound.
    """
    payload = {
        "max_tokens": MAX_ACTION_TOKENS,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "require_reasoning": not args.allow_missing_reasoning,
        "user_model": args.user_model,
        "user_model_args": json.loads(args.user_model_args),
        "domain": args.domain,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]


def build_render_ids(tokenizer: Any, tools: list[dict[str, Any]] | None) -> Any:
    """A `messages -> prompt ids` callable for the prompt-parity proof.

    Returns None when tool schemas are unavailable: parity cannot be proven
    without the exact tools the prompt was rendered with, and a proof run
    against the wrong tools is worse than no proof, because it would pass.
    """
    if tools is None:
        return None
    from vektori_trace.tau2.live_agent import render_prompt_ids

    def _render(messages: list[dict[str, Any]]) -> list[int]:
        return render_prompt_ids(tokenizer, messages, tools)

    return _render


def run_update(
    idx: int,
    args: argparse.Namespace,
    run: RunState,
    *,
    settings: RolloutSettings,
    tokenizer: Any,
    teacher_tok: Any,
    pool: Any,
    trainer: Any,
    render_ids: Any,
    policy_version: str,
) -> dict[str, Any]:
    """PLANNED -> SAMPLED -> SCORED -> TRAINED for one live update."""
    u = run.update(idx)
    u.validate()

    # Before anything is sampled or paid for: the endpoint must serve the
    # policy this update is meant to sample from. Update 0 is the SFT
    # checkpoint and needs no refresh.
    if idx > 0 and not u.reached("SAMPLED"):
        prev = run.update(idx - 1)
        prev.validate_checkpoint()
        refresh_live_policy(args, idx, prev.checkpoint_path, run_dir=args.run_dir)
        # `refresh_live_policy` advanced both the served name and the adapter
        # hash. Carrying the old hash here would archive this update's episodes
        # under the parent SFT adapter while sampling from the new one.
        settings = RolloutSettings(
            **{**settings.__dict__, "student_model": args.student_model,
               "adapter_hash": args.adapter_hash,
               "policy_version": policy_version}
        )

    plans = plan_update(idx, args.task_ids, args.seeds)

    # --- sample (live rollouts) -------------------------------------------
    if not u.reached("SAMPLED"):
        t0 = time.time()
        log(f"  rolling out {len(plans)} episode(s) on {settings.student_model}")
        report = capture_live_update(
            u,
            plans,
            settings=settings,
            teacher_context={
                "model": args.teacher_model,
                "tokenizer": args.teacher_tokenizer,
                "renderer": args.teacher_renderer,
            },
            runner=Tau2EpisodeRunner(settings, tokenizer),
        )
        rollout_s = round(time.time() - t0, 1)
        commit_scores("live rollout")
        telemetry(
            args.run_dir,
            {
                "event": "stage",
                "stage": "SAMPLED",
                "update": idx,
                "seconds": rollout_s,
                "n_episodes": report.get("trainable"),
                "n_turns": report.get("trainable_turns"),
                "failed": report.get("failed"),
                "discarded": report.get("discarded"),
                "failed_turns": report.get("failed_turns"),
            },
        )
        log(
            f"  rollout took {rollout_s}s: {report.get('trainable')} episodes, "
            f"{report.get('trainable_turns')} turns, "
            f"{report.get('failed')} failed, {report.get('discarded')} discarded"
        )

    inputs = load_live_update_inputs(
        u, policy_version=policy_version, render_ids=render_ids,
        headroom=args.share_headroom,
    )
    log(
        f"  {len(inputs.actions)} supervised turns from "
        f"{inputs.n_episodes} episode(s)"
    )

    # Concentration is REPORTED, never enforced, on the live path. An episode's
    # length is an outcome of the policy, so refusing a batch for producing a
    # long trajectory selects against hard tasks -- exactly the states OPD
    # exists to learn from. Balance is a property of the preregistered roster
    # (equal episode counts per task), decided before the rollout; pathology is
    # bounded by `max_turns` and MAX_ACTION_TOKENS, which cap a runaway
    # generation without discarding a legitimate long one.
    from vektori_trace.tau2.live_batch import estimate_batch_shares

    n_tasks = len({p.task for p in inputs.prefixes})
    est = estimate_batch_shares(
        inputs.prefixes, inputs.actions,
        max_task_share=min(1.0, (1.0 / n_tasks) * 1.5) if n_tasks else 1.0,
        max_trace_share=inputs.max_trace_share,
    )
    log(f"  concentration (raw-token estimate): "
        f"task={est.get('worst_task_share')} "
        f"trace={est.get('worst_trace_share')} -- telemetry only")
    telemetry(
        args.run_dir,
        {
            "event": "batch_concentration",
            "update": idx,
            "estimated_task_share": est.get("estimated_task_share"),
            "estimated_trace_share": est.get("estimated_trace_share"),
            "worst_task_share": est.get("worst_task_share"),
            "worst_trace_share": est.get("worst_trace_share"),
            "enforced": False,
            "basis": est.get("basis"),
        },
    )

    return train_live_update(
        u,
        inputs,
        run,
        teacher_tok=teacher_tok,
        pool=pool,
        trainer=trainer,
        run_dir=args.run_dir,
        policy_version=policy_version,
        max_new_tokens=MAX_ACTION_TOKENS,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=None,
                   help="default: ./runs/live-opd-<timestamp>")
    p.add_argument("--n-updates", type=int, default=5)
    p.add_argument("--task-ids", required=True,
                   help="comma-separated C30 task ids to roll out each update")
    p.add_argument("--seeds", default="0,1",
                   help="comma-separated seeds; episodes = tasks x seeds")

    p.add_argument("--domain", default="retail")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    p.add_argument("--parent", required=False,
                   help="the frozen A_sft_new adapter this run is parented on")
    p.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE", ""))
    p.add_argument("--reload-url", default=os.environ.get("STUDENT_RELOAD_URL", ""))
    p.add_argument("--student-model", required=True)
    p.add_argument("--adapter-hash", default="",
                   help="DEPRECATED override. The parent hash is derived from "
                        "--parent's weights; pass this only to pin a value when "
                        "the parent is not readable from this process.")

    p.add_argument("--teacher-model",
                   default="accounts/fireworks/models/deepseek-v4-flash-0731")
    p.add_argument("--teacher-tokenizer",
                   default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--teacher-renderer", default="deepseek-v4-native")

    p.add_argument("--max-input-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-errors", type=int, default=10)
    p.add_argument("--user", default="user_simulator")
    # Fireworks, not OpenAI: this environment has a FIREWORKS_API_KEY and no
    # OpenAI Modal secret, and a `gpt-4o-mini` default fails mid-episode after
    # turn 0's student generation has already been paid for.
    p.add_argument(
        "--user-model",
        default="fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731",
    )
    p.add_argument("--user-model-args", default="{}")
    p.add_argument("--allow-missing-reasoning", action="store_true",
                   help="diagnostic only: admit actions with no <think> span")

    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--share-headroom", type=float, default=1.5,
                   help="an episode may exceed its even share of the batch by "
                        "this factor before it is judged to dominate")
    p.add_argument("--tools-file", default=None,
                   help="retail tool schemas as JSON; required for the "
                        "prompt-parity proof")

    p.add_argument("--dry-run", action="store_true",
                   help="plan and validate only: no endpoint, teacher or GPU")
    p.add_argument("--canary", action="store_true",
                   help="one update only, then stop. Proves SAMPLED -> SCORED "
                        "-> TRAINED -> checkpoint, but NOT the reload and "
                        "next-policy rollout; use --two-update-proof for that")
    p.add_argument("--two-update-proof", action="store_true",
                   help="exactly two updates: the smallest run that proves "
                        "checkpoint -> serving reload -> update-1 rollout from "
                        "the NEW adapter. Separately authorized from --yes.")
    p.add_argument("--yes", action="store_true",
                   help="required for the full multi-update paid run")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    a.task_ids = [t.strip() for t in a.task_ids.split(",") if t.strip()]
    a.seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if not a.task_ids:
        raise SystemExit("--task-ids must name at least one task")
    if not a.seeds:
        raise SystemExit("--seeds must name at least one seed")

    if a.canary and a.two_update_proof:
        raise SystemExit(
            "--canary and --two-update-proof are different authorizations; "
            "pass exactly one"
        )
    if a.two_update_proof:
        # Exactly two. The whole point is the transition between them, and a
        # third update buys nothing this proof is for while costing another
        # full batch of teacher calls.
        n_updates = 2
    elif a.canary:
        n_updates = 1
    else:
        n_updates = a.n_updates
    a.run_dir = a.run_dir or f"runs/live-opd-{time.strftime('%Y%m%d-%H%M%S')}"

    # Provenance is DERIVED, never an optional CLI string. `adapter_hash` is
    # stamped onto every archived episode and is what `batch_report` checks a
    # batch against; the 2026-08-28 two-update proof recorded "" for update 0
    # because the wrapper never passed the flag, leaving that rollout's
    # provenance blank while the checkpoint's own parent_policy_hash was
    # correct. A hash that can be forgotten is not provenance.
    if a.parent:
        try:
            from vektori_trace.tau2.reopd_checkpoint import adapter_hash
            derived = adapter_hash(a.parent)
        except Exception as exc:
            raise SystemExit(
                f"cannot derive the parent adapter hash from {a.parent}: {exc}. "
                "Rollout provenance would be blank, so refusing to sample."
            ) from exc
        if a.adapter_hash and a.adapter_hash != derived:
            raise SystemExit(
                f"--adapter-hash {a.adapter_hash!r} disagrees with the weights "
                f"at --parent ({derived!r}). One of them names a different "
                "adapter than the run will actually sample from."
            )
        a.adapter_hash = derived
        log(f"parent adapter {a.parent} -> {derived}")
    elif not a.adapter_hash:
        raise SystemExit(
            "no --parent to derive the adapter hash from, and no explicit "
            "--adapter-hash. Every episode would archive blank provenance."
        )

    n_episodes = len(a.task_ids) * len(a.seeds)
    share = live_max_trace_share(n_episodes, headroom=a.share_headroom)
    log(
        f"plan: {n_updates} update(s) x {n_episodes} episode(s) "
        f"({len(a.task_ids)} tasks x {len(a.seeds)} seeds), "
        f"max_trace_share={share:.3f}"
    )
    if n_episodes < 4:
        log(
            f"  WARNING {n_episodes} episodes/update is below the Phase 1 "
            "four-episode proof; the trace-share limit cannot measure "
            "concentration this small"
        )

    tools = json.load(open(a.tools_file)) if a.tools_file else None
    gch = gen_config_hash(a)

    run = RunState(a.run_dir, n_updates=n_updates)
    manifest = {
        "kind": "tau2_live_opd",
        "domain": a.domain,
        "n_updates": n_updates,
        "task_ids": a.task_ids,
        "seeds": a.seeds,
        "episodes_per_update": n_episodes,
        "base_model": a.base_model,
        "parent": a.parent,
        "student_model": a.student_model,
        "student_tokenizer": a.tokenizer,
        "adapter_hash": a.adapter_hash,
        "teacher_model": a.teacher_model,
        "teacher_tokenizer": a.teacher_tokenizer,
        "teacher_renderer": a.teacher_renderer,
        "max_action_tokens": MAX_ACTION_TOKENS,
        "max_input_tokens": a.max_input_tokens,
        "temperature": a.temperature,
        "learning_rate": a.learning_rate,
        "require_reasoning": not a.allow_missing_reasoning,
        "gen_config_hash": gch,
        "max_trace_share": share,
        "share_headroom": a.share_headroom,
        # The live boundary deliberately differs from the frozen action-only
        # corpus by the empty-think wrapper. Recording it means a later reader
        # does not have to infer which convention a run used.
        "prompt_boundary": "generation-prompt (no empty-think wrapper)",
        "mode": ("two-update-proof" if a.two_update_proof
                 else "canary" if a.canary else "full"),
    }

    if a.dry_run:
        os.makedirs(a.run_dir, exist_ok=True)
        run.freeze_manifest(manifest)
        plans = plan_update(0, a.task_ids, a.seeds)
        atomic_write_json(
            Path(a.run_dir) / "dry_run.json",
            {
                "manifest": manifest,
                "first_update": [p.__dict__ for p in plans],
            },
        )
        log(f"DRY RUN ok -> {a.run_dir}")
        log(f"  update 0 would roll out: {[p.episode_id for p in plans][:4]} ...")
        log("  nothing sampled, scored, or trained")
        return 0

    if not (a.canary or a.two_update_proof or a.yes):
        raise SystemExit(
            "refusing the full run without --yes. Use --dry-run to validate, "
            "--canary for one small paid update, or --two-update-proof for the "
            "smallest run that proves the reload and next-policy rollout."
        )
    if a.two_update_proof and not a.reload_url:
        raise SystemExit(
            "--two-update-proof requires --reload-url (or STUDENT_RELOAD_URL): "
            "the transition it exists to prove IS the serving reload, and "
            "without it update 1 would resample the same policy as update 0 "
            "while reporting success."
        )
    if not a.api_base:
        raise SystemExit("--api-base is required (or set STUDENT_API_BASE)")
    if n_updates > 1 and not a.reload_url:
        raise SystemExit(
            "--reload-url (or STUDENT_RELOAD_URL) is required for more than one "
            "update; without it the endpoint keeps serving update 0's policy "
            "while later updates claim to be on-policy, and every importance "
            "ratio compares two different distributions with a finite loss."
        )
    if tools is None:
        raise SystemExit(
            "--tools-file is required: the prompt-parity proof cannot run "
            "without the exact tool schemas the prompt was rendered with, and "
            "an unproven semantic history is one the teacher may score under a "
            "conversation that never happened."
        )

    from vektori_trace.tau2.reopd_refresh import served_models

    advertised = served_models(a.api_base, timeout=min(a.timeout, 30.0))
    if a.student_model not in advertised:
        raise SystemExit(
            f"endpoint does not advertise {a.student_model!r}; it serves "
            f"{advertised}. vLLM resolves an unknown name against the base "
            "model, so the adapter would silently do nothing."
        )

    os.makedirs(a.run_dir, exist_ok=True)
    run.freeze_manifest(manifest)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    render_ids = build_render_ids(tokenizer, tools)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device in the training container: the live update cannot "
            "run here. The student endpoint being reachable is not evidence of "
            "a local GPU -- it is a separate host."
        )

    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.tau2.reopd_trainer import ReOPDTrainer
    from vektori_trace.vocab_bridge import load_tokenizer

    teacher_tok = load_tokenizer(a.teacher_tokenizer)
    pool = FireworksTeacherPool(model=a.teacher_model)

    start = run.resume_point()
    if start >= n_updates:
        log(f"run already complete ({start}/{n_updates})")
        return 0
    if start > 0:
        log(f"resuming at update {start}")

    trainer = ReOPDTrainer(
        base_model=a.base_model,
        parent_adapter=a.parent,
        learning_rate=a.learning_rate,
        run_dir=a.run_dir,
        device="cuda",
    )
    trainer.load(
        resume_from=run.update(start - 1).checkpoint_path if start > 0 else None
    )

    a.initial_served_name = a.student_model
    a.served_name = a.student_model

    settings = RolloutSettings(
        domain=a.domain,
        student_model=a.student_model,
        api_base=a.api_base,
        policy_version=f"live-u{start:03d}",
        adapter_hash=a.adapter_hash,
        gen_config_hash=gch,
        max_tokens=MAX_ACTION_TOKENS,
        max_input_tokens=a.max_input_tokens,
        temperature=a.temperature,
        timeout=a.timeout,
        max_steps=a.max_steps,
        max_errors=a.max_errors,
        user=a.user,
        user_model=a.user_model,
        user_model_args=json.loads(a.user_model_args),
        require_reasoning=not a.allow_missing_reasoning,
    )

    # A fixed prompt whose greedy logprobs fingerprint the served policy, so the
    # first refresh is verified against the policy it replaces rather than
    # merely accepted because its name appeared in /models.
    from vektori_trace.tau2.reopd_refresh import probe_logprobs

    a.probe_prompt_ids = tokenizer(
        "You are a retail agent.", add_special_tokens=False
    )["input_ids"][:256]
    a.probe_logprobs = probe_logprobs(
        a.api_base, a.student_model, a.probe_prompt_ids, timeout=120.0
    )
    log(
        f"baseline fingerprint captured for {a.student_model} "
        f"({len(a.probe_logprobs)} tokens)"
    )

    for idx in range(start, n_updates):
        log(f"update {idx}/{n_updates - 1}")
        policy_version = f"live-u{idx:03d}"
        # `a.student_model` and `a.adapter_hash` both advance inside
        # `refresh_live_policy`; read them together so the served policy and
        # its recorded identity can never disagree.
        settings = RolloutSettings(
            **{**settings.__dict__, "policy_version": policy_version,
               "student_model": a.student_model,
               "adapter_hash": a.adapter_hash}
        )
        run_update(
            idx,
            a,
            run,
            settings=settings,
            tokenizer=tokenizer,
            teacher_tok=teacher_tok,
            pool=pool,
            trainer=trainer,
            render_ids=render_ids,
            policy_version=policy_version,
        )

    log(f"done: {run.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
