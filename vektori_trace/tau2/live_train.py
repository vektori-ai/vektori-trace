"""The live SAMPLED -> SCORED -> TRAINED -> reload -> next-rollout bridge.

`capture_live_update` stops at SAMPLED. This module carries one live update the
rest of the way and hands the next one a refreshed endpoint, which is what makes
the loop on-policy rather than a sequence of independent capture runs.

Everything after sampling is the existing ReOPD backend, called through
`opd_stages`. (That module is a shared destination, not yet shared execution:
the replay driver still has its own copy of those stages. See its docstring.)
What lives here is only what live genuinely adds:

- a **share limit derived from the episode count**, because an episode's turns
  all carry one `trace_id` and the replay default of 0.35 rejects every live
  batch outright (`live_turns.LivePrefix` documents this as *the* live-vs-replay
  semantic difference the driver must handle);
- **stale-score filtering**, because a live turn key can recur inside one update
  after a discard-and-resample, where identical action bytes may follow a
  different history;
- **score provenance**, so the staleness check has something to compare on the
  next resume and `RunState.validate()` can enforce the binding without the
  driver's cooperation;
- a **prompt-parity proof** over the persisted semantic histories, before any
  teacher call is paid for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vektori_trace.tau2.live_turns import (
    LiveTurnError,
    live_prefixes,
    score_row_provenance,
    stale_score_keys,
    verify_prompt_parity,
)
from vektori_trace.tau2.opd_stages import (
    commit_scores,
    gpu_snapshot,
    log,
    telemetry,
)
from vektori_trace.tau2.reopd_state import RunState, UpdateDir, read_jsonl

__all__ = [
    "LiveTrainError",
    "LiveUpdateInputs",
    "live_max_trace_share",
    "load_live_update_inputs",
    "refresh_live_policy",
    "train_live_update",
]

#: `run_replay_chunk_opd` records this so a report says which experiment it ran.
SELECTION_POLICY = "tau2-live-episode-balanced"


class LiveTrainError(RuntimeError):
    """A live update that must not reach the optimizer."""


def live_max_trace_share(n_episodes: int, *, headroom: float = 1.5) -> float:
    """The trace-share limit a live batch of `n_episodes` can actually satisfy.

    On the replay path one trace is one task and 0.35 keeps a single task from
    dominating an update. Live is different in a way that is easy to miss: a
    whole *episode* is one trace, so a 4-episode batch has a natural share of
    0.25 per episode and a 1-episode batch has a share of 1.0. Inheriting 0.35
    would reject every batch of three or fewer episodes -- including the Phase 1
    four-episode proof, whose episodes will not be equal length.

    So the limit is derived from the batch, not inherited: the even share
    `1/n` times `headroom`, clamped to 1.0. With the default that is 0.375 for
    four episodes -- an episode may run half again as long as its share of the
    batch before it is judged to dominate. A single-episode batch necessarily
    yields 1.0, which is honest: there is no concentration to measure with one
    trajectory, and the *episode count* is the thing to raise, not the limit.
    """
    if n_episodes <= 0:
        raise LiveTrainError("a live update must contain at least one episode")
    return min(1.0, (1.0 / n_episodes) * headroom)


@dataclass
class LiveUpdateInputs:
    """The `(actions, rendered, prefixes)` triple assembled from live turns."""

    capture_rows: list[dict[str, Any]]
    actions: list[Any]
    rendered: dict[str, list[dict[str, Any]]]
    prefixes: list[Any]
    episode_ids: list[str]
    max_trace_share: float

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ids)


def load_live_update_inputs(
    update: UpdateDir,
    *,
    policy_version: str,
    render_ids: Callable[[list[dict[str, Any]]], list[int]] | None = None,
    headroom: float = 1.5,
) -> LiveUpdateInputs:
    """Read a SAMPLED live update off disk and prepare it for scoring.

    `capture_live_update` already wrote `actions.jsonl` and `rendered.json` in
    the exact shapes the replay stages consume, so this reads rather than
    re-derives -- re-deriving would make a resumed run depend on the archive
    and the tokenizer still agreeing, which is precisely what the persisted
    rows exist to avoid.

    `render_ids`, when supplied, re-renders every persisted semantic history
    through the pinned Qwen path and requires it to reproduce the captured
    prompt ids. That closes the seam between "the history we stored" and "the
    state the student actually sampled in": a history that does not reproduce
    the prompt is one the teacher would score under a conversation that never
    happened, and the resulting loss would be finite and wrong.
    """
    if not update.reached("SAMPLED"):
        raise LiveTrainError(
            f"update {update.index} has not reached SAMPLED; there is nothing "
            "to score. Run the live rollout first."
        )
    update.validate()

    capture_rows = read_jsonl(update.actions_path)
    if not capture_rows:
        raise LiveTrainError(
            f"update {update.index} is marked SAMPLED but actions.jsonl is "
            "empty; an optimizer step over zero supervised tokens is not a step"
        )

    rendered_path = update.path / "rendered.json"
    if not rendered_path.exists():
        raise LiveTrainError(
            f"update {update.index} has no rendered.json; Qwen prompt ids are "
            "not a teacher context and the semantic histories cannot be rebuilt"
        )
    rendered = json.loads(rendered_path.read_text())

    stale_versions = sorted(
        {
            str(r.get("policy_version"))
            for r in capture_rows
            if r.get("policy_version") != policy_version
        }
    )
    if stale_versions:
        raise LiveTrainError(
            f"update {update.index} holds actions sampled under "
            f"{stale_versions}, but this update is {policy_version!r}. The "
            "student policy is fixed for an entire episode and an entire batch."
        )

    missing = [r["key"] for r in capture_rows if r["prefix_id"] not in rendered]
    if missing:
        raise LiveTrainError(
            f"update {update.index}: {len(missing)} action(s) have no rendered "
            f"teacher context: {missing[:4]}"
        )

    if render_ids is not None:
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in capture_rows:
            by_episode.setdefault(str(row["episode_id"]), []).append(
                {
                    "capture": {
                        "episode_id": row["episode_id"],
                        "turn_index": row["turn_index"],
                        "prompt_token_ids": row["prompt_token_ids"],
                    },
                    "semantic_history": rendered[row["prefix_id"]],
                }
            )
        parity = verify_prompt_parity(by_episode, render_ids)
        log(f"  prompt parity ok: {parity['checked']} live turns re-render")

    from vektori_trace.tau2.reopd_sample import capture_to_sampled_action

    actions = [capture_to_sampled_action(r) for r in capture_rows]
    prefixes = live_prefixes(capture_rows)
    episode_ids = sorted({str(r["episode_id"]) for r in capture_rows})

    return LiveUpdateInputs(
        capture_rows=capture_rows,
        actions=actions,
        rendered=rendered,
        prefixes=prefixes,
        episode_ids=episode_ids,
        max_trace_share=live_max_trace_share(len(episode_ids), headroom=headroom),
    )


def train_live_update(
    update: UpdateDir,
    inputs: LiveUpdateInputs,
    run: RunState,
    *,
    teacher_tok: Any,
    pool: Any,
    trainer: Any,
    run_dir: str | Path,
    policy_version: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """SCORED -> TRAINED for one already-sampled live update.

    The stages themselves are `opd_stages`'; what this adds is the live score
    cache discipline. `stale_score_keys` drops a paid score whose recorded
    history is not the history now attached to its key -- the discard-and-
    resample case, where reusing it would train on a number bought for a
    trajectory that no longer exists, with every downstream assertion passing.
    """
    paid = run.paid_scores(update.index)
    stale = set(stale_score_keys(inputs.capture_rows, paid))

    by_key = {r["key"]: r for r in inputs.capture_rows}

    # --- SCORED, through the semantic projection ---------------------------
    # NOT `score_replay_batch`. That scorer asks DeepSeek for the likelihood of
    # the raw Qwen action bytes, which for a reasoning-inclusive live action
    # includes control markup the teacher never emits -- measured at -55 per
    # `<tool_call>` on 2026-08-28, and forbidden by the pinned reference
    # (`response content only, no chat template special tokens`).
    result = run_projected_score_stage(
        update,
        inputs,
        teacher_tok=teacher_tok,
        pool=pool,
        run_dir=run_dir,
        paid=paid,
        drop_keys=stale,
        by_key=by_key,
    )

    log(
        f"  training over {len(inputs.actions)} live turns from "
        f"{inputs.n_episodes} episode(s), max_trace_share="
        f"{inputs.max_trace_share:.3f}"
    )
    telemetry(
        run_dir,
        {
            "event": "live_batch",
            "update": update.index,
            "n_episodes": inputs.n_episodes,
            "n_turns": len(inputs.actions),
            "episode_ids": inputs.episode_ids,
            "max_trace_share": inputs.max_trace_share,
            "n_stale_scores_dropped": len(stale),
            "n_supervised_tokens": sum(
                sc.n_supervised for sc in result.projected.values()
            ),
            "scoring": "semantic projection",
        },
    )

    return run_projected_train_stage(
        update,
        inputs,
        result.projected,
        trainer,
        run_dir=run_dir,
        policy_version=policy_version,
        max_trace_share=inputs.max_trace_share,
    )


@dataclass
class ProjectedScoreResult:
    """What the projected SCORED stage produced."""

    projected: dict[str, Any]
    n_newly_scored: int
    n_reused: int
    teacher_input_tokens: int
    seconds: float


def run_projected_score_stage(
    update: UpdateDir,
    inputs: LiveUpdateInputs,
    *,
    teacher_tok: Any,
    pool: Any,
    run_dir: str | Path,
    paid: dict[str, Any],
    drop_keys: set[str] | None = None,
    by_key: dict[str, Any] | None = None,
) -> ProjectedScoreResult:
    """Score every live turn through its DeepSeek-native semantic equivalent.

    Persists one row per action carrying the per-student-token teacher credit
    and the exclusion mask, so a resume reloads the *projected* representation
    rather than re-deriving it from raw bytes -- which is what re-introduced
    the contamination in the first place.
    """
    import base64
    import json as _json
    import time as _time

    from vektori_trace.tau2.live_score import score_live_action
    from vektori_trace.tau2.reopd_state import append_jsonl, read_jsonl

    drop = set(drop_keys or ())
    by_key = by_key or {}
    rows_on_disk = {
        r["key"]: r for r in read_jsonl(update.scores_path)
        if r.get("key") and r.get("projection") == "semantic"
    }

    projected: dict[str, Any] = {}
    n_new = n_reused = 0
    t0 = _time.time()

    from vektori_trace.tau2.live_score import ProjectedScore

    for row, action in zip(inputs.capture_rows, inputs.actions):
        key = row["key"]
        cached = rows_on_disk.get(key)
        if cached is not None and key not in drop:
            projected[key] = ProjectedScore(
                key=key,
                teacher_logprob_by_index={
                    int(k): float(v)
                    for k, v in cached["teacher_logprob_by_index"].items()
                },
                excluded={int(k): v for k, v in cached["excluded"].items()},
                n_prefix_tokens=int(cached.get("n_prefix_tokens", 0)),
                n_teacher_tokens=int(cached.get("n_teacher_tokens", 0)),
                payload_report=cached.get("payloads", {}),
            )
            n_reused += 1
            continue

        raw = base64.b64decode(row["action_bytes_b64"]).decode("utf-8", "replace")
        for special in ("<|im_end|>", "<|endoftext|>"):
            if raw.endswith(special):
                raw = raw[: -len(special)]
        history = inputs.rendered[row["prefix_id"]]
        sc = score_live_action(
            key=key,
            raw_text=raw,
            student_token_bytes=[
                base64.b64decode(b) for b in row["action_token_bytes_b64"]
            ],
            semantic_history=history,
            teacher_tokenizer=teacher_tok,
            pool=pool,
        )
        projected[key] = sc
        n_new += 1
        # A paid stage that prints nothing until it finishes is one you cannot
        # tell apart from a hung one. 34 actions took 410 s on 2026-08-28.
        if n_new % 5 == 0 or n_new == 1:
            done = n_new + n_reused
            spent = sum(s.n_prefix_tokens + s.n_teacher_tokens
                        for s in projected.values())
            log(f"    scoring {done}/{len(inputs.capture_rows)} "
                f"({_time.time() - t0:.0f}s, ~{spent:,} teacher tokens, "
                f"~${spent * 0.22 / 1e6:.4f})")
        # Persist immediately: every teacher call is billed.
        out = {
            "key": key,
            "projection": "semantic",
            "teacher_logprob_by_index": {
                str(k): v for k, v in sc.teacher_logprob_by_index.items()
            },
            "excluded": {str(k): v for k, v in sc.excluded.items()},
            "n_prefix_tokens": sc.n_prefix_tokens,
            "n_teacher_tokens": sc.n_teacher_tokens,
            "payloads": sc.payload_report,
        }
        src = by_key.get(key)
        if src is not None:
            out["fingerprint"] = src.get("score_fingerprint")
            out["semantic_history_hash"] = src.get("semantic_history_hash")
            out["teacher_context_hash"] = src.get("teacher_context_hash")
            out["policy_version"] = src.get("policy_version")
        append_jsonl(update.scores_path, out)
        commit_scores("projected score")

    seconds = round(_time.time() - t0, 1)
    tin = sum(sc.n_prefix_tokens + sc.n_teacher_tokens
              for sc in projected.values())
    if not update.reached("SCORED"):
        update.mark("SCORED", {
            "n_scores": len(projected), "seconds": seconds,
            "teacher_input_tokens": tin, "projection": "semantic",
        })
        commit_scores("scored marker")
    log(f"  scored {n_new} new / {n_reused} reused in {seconds}s "
        f"(semantic projection; ~{tin:,} teacher input tokens)")
    telemetry(run_dir, {
        "event": "stage", "stage": "SCORED", "update": update.index,
        "seconds": seconds, "n_scores": len(projected),
        "teacher_input_tokens": tin, "projection": "semantic",
        "est_cost_usd_uncached": round(tin * 0.22 / 1e6, 4),
    })
    return ProjectedScoreResult(projected, n_new, n_reused, tin, seconds)


def run_projected_train_stage(
    update: UpdateDir,
    inputs: LiveUpdateInputs,
    projected: dict[str, Any],
    trainer: Any,
    *,
    run_dir: str | Path,
    policy_version: str,
    max_trace_share: float,
    clamp: float | None = None,
) -> dict[str, Any]:
    """One optimizer step over the projected batch.

    Does NOT call `run_replay_chunk_opd`: that re-aligns raw student bytes
    against teacher bytes, which for a projected score would undo the
    projection. `build_projected_batch` consumes the already-aligned per-token
    credit instead.
    """
    import json as _json
    import time as _time

    from vektori_trace.tau2.live_batch import build_projected_batch
    from vektori_trace.tau2.reopd_state import atomic_write_json

    if update.reached("TRAINED"):
        log("  already trained; skipping")
        return _json.loads((update.checkpoint_path / "state.json").read_text())

    t0 = _time.time()
    # Task share is derived from the batch, exactly as the trace share is. A
    # live update rolls out a fixed task set, so a 4-episode single-task batch
    # legitimately has share 1.0; inheriting the replay default of 0.5 would
    # reject it for a concentration the schedule chose on purpose.
    n_tasks = len({p.task for p in inputs.prefixes})
    max_task_share = min(1.0, (1.0 / n_tasks) * 1.5) if n_tasks else 1.0
    batch = build_projected_batch(
        inputs.prefixes, inputs.actions, projected,
        policy_version=policy_version,
        max_task_share=max_task_share,
        max_trace_share=max_trace_share,
        # Telemetry, not refusal -- see `build_projected_batch`. A live
        # episode's length is an outcome, so rejecting on realized length
        # selects against hard tasks, which is the opposite of what on-policy
        # distillation is for. Balance is enforced before the rollout by equal
        # episode counts per task; the shares are reported below.
        enforce_shares=False,
        clamp=clamp,
    )
    opt = trainer.step(batch)
    train_s = round(_time.time() - t0, 1)
    gpu = gpu_snapshot()

    report = {
        "policy_version": policy_version,
        "selection_policy": SELECTION_POLICY,
        "projection": "semantic",
        "global_supervised_tokens": batch.global_supervised_tokens,
        "n_actions": len(batch.keys),
        "spread": batch.spread_report,
        "optimizer": opt,
    }
    atomic_write_json(update.report_path, report)
    telemetry(run_dir, {
        "event": "stage", "stage": "TRAINED", "update": update.index,
        "seconds": train_s, "loss": (opt or {}).get("loss"),
        "grad_norm": (opt or {}).get("grad_norm"),
        "supervised_tokens": batch.global_supervised_tokens,
        "projection": "semantic", **gpu,
    })

    state = trainer.checkpoint(
        update.checkpoint_path, update_index=update.index,
        policy_version=policy_version,
    )
    commit_scores("checkpoint")
    update.mark("TRAINED", {
        "loss": (opt or {}).get("loss"), "seconds": train_s,
        "adapter_hash": state.get("adapter_hash"),
    })
    commit_scores("trained marker")
    log(f"  trained in {train_s}s; loss={(opt or {}).get('loss')} "
        f"adapter={state.get('adapter_hash')} "
        f"supervised={batch.global_supervised_tokens}")
    return state


def refresh_live_policy(
    args: Any,
    idx: int,
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Serve the checkpoint update `idx-1` produced, before sampling update idx.

    This is the step that makes the loop on-policy and is the one the live arm
    could not previously take. OPD is on-policy in the action: `log pi_old` must
    come from the policy that sampled it, so an endpoint still serving the SFT
    checkpoint at update 3 makes every importance ratio compare two different
    distributions -- with a finite loss and nothing in any log to show for it.

    Update 0 needs no refresh: the SFT checkpoint *is* the current policy there,
    which is why a one-update canary is valid against a static endpoint.

    **The adapter hash moves with the policy.** `RolloutSettings.adapter_hash`
    is stamped onto every archived episode and is what `batch_report` checks a
    batch against. Leaving it at the parent SFT hash after the endpoint has
    been repointed produces a batch that samples from update *k-1*'s adapter
    while claiming provenance of the SFT one -- a contradiction no assertion
    downstream would catch, because every episode in the batch agrees with
    every other. So the new hash is read from the checkpoint that was just
    served, and a checkpoint that cannot state its own hash is refused rather
    than defaulted.
    """
    from vektori_trace.tau2.reopd_refresh import refresh_policy

    state_path = Path(checkpoint_path) / "state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError) as exc:
        raise LiveTrainError(
            f"cannot read {state_path} to learn the adapter hash update "
            f"{idx - 1} produced: {exc}. Refusing to sample update {idx} while "
            "archiving the previous adapter's provenance."
        ) from exc
    new_hash = state.get("adapter_hash")
    if not new_hash:
        raise LiveTrainError(
            f"{state_path} records no adapter_hash; update {idx} would archive "
            "episodes under the wrong adapter identity"
        )

    name = f"{args.initial_served_name}-u{idx - 1:03d}"
    rep = refresh_policy(
        args.api_base,
        new_name=name,
        new_path=str(checkpoint_path),
        probe_prompt_ids=args.probe_prompt_ids,
        previous_logprobs=getattr(args, "probe_logprobs", None),
        previous_name=getattr(args, "served_name", None),
        reload_url=args.reload_url,
    )
    args.student_model = name
    args.served_name = name
    args.adapter_hash = new_hash
    args.probe_logprobs = rep["probe_logprobs"]
    log(
        f"  endpoint now serves {name} "
        f"(adapter {str(new_hash)[:16]}, "
        f"probe delta {rep.get('max_logprob_delta', 'n/a')})"
    )
    telemetry(
        run_dir,
        {"event": "refresh", "update": idx, "adapter_hash": new_hash, **rep},
    )
    commit_scores("refresh", required=False)
    return rep
