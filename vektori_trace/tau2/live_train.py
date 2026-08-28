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
    log,
    run_score_stage,
    run_train_stage,
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

    def _provenance(scored_action: Any) -> dict[str, Any]:
        row = by_key.get(getattr(scored_action, "key", None))
        if row is None:
            raise LiveTurnError(
                f"scored action {getattr(scored_action, 'key', None)!r} is not "
                "in this batch; refusing to write provenance for a foreign score"
            )
        return score_row_provenance(scored_action, row)

    result = run_score_stage(
        update,
        inputs.actions,
        inputs.rendered,
        teacher_tok=teacher_tok,
        pool=pool,
        run_dir=run_dir,
        paid=paid,
        drop_keys=stale,
        provenance=_provenance,
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
        },
    )

    return run_train_stage(
        update,
        inputs.prefixes,
        inputs.actions,
        result.scored,
        trainer,
        run_dir=run_dir,
        policy_version=policy_version,
        max_new_tokens=max_new_tokens,
        max_trace_share=inputs.max_trace_share,
        selection_policy=SELECTION_POLICY,
        n_samples_per_prefix=1,
    )


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
