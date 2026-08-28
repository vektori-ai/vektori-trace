"""Live Tau2 turns, in the shapes the existing ReOPD backend already consumes.

This is deliberately not a second OPD stack. `scripts/tau2_reopd_train.py` runs
PLANNED -> SAMPLED -> SCORED -> TRAINED over `RunState`, and every stage after
sampling is indifferent to where the actions came from:

    score_replay_batch(actions, rendered, ...)   # rendered: prefix_id -> messages
    run_replay_chunk_opd(...)
    ReOPDTrainer.step / .checkpoint
    refresh_policy(...)

A replay update supplies those two arguments from frozen C30 prefixes. A live
update supplies them from complete Tau2 episodes. That substitution is the
entire delta, so this module only builds the pair:

    live turn rows  ->  (list[SampledAction], {prefix_id: canonical_messages})

using the existing `capture_to_sampled_action` and the existing base64 row
format, so a live `actions.jsonl` is readable by the same code as a replay one.

What live rollouts add that replay does not have is an *episode*: turns are no
longer independent, the policy must be fixed across a whole trajectory, and an
episode that dies mid-flight cannot be resumed -- Tau2's environment and user
simulator are stateful and cannot be rewound to turn k. `EpisodeStatus` and
`batch_report` in `live_episode.py` carry that; this module enforces it at the
point where turns become a batch.
"""

from __future__ import annotations

import base64
from typing import Any

import hashlib
import json
from dataclasses import dataclass, field

from vektori_trace.tau2.live_episode import semantic_hash

__all__ = [
    "LiveTurnError",
    "LivePrefix",
    "live_score_fingerprint",
    "teacher_context_hash",
    "verify_prompt_parity",
    "live_turn_key",
    "capture_row_from_turn",
    "flatten_live_turns",
    "live_prefixes",
    "stale_score_keys",
    "score_row_provenance",
]


class LiveTurnError(RuntimeError):
    """A live batch that must not reach the optimizer."""


def teacher_context_hash(context: dict[str, Any]) -> str:
    """Hash the teacher model, tokenizer and rendering contract."""
    required = {"model", "tokenizer", "renderer"}
    missing = sorted(required - set(context))
    if missing:
        raise LiveTurnError(f"teacher context lacks required fields {missing}")
    return hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class LivePrefix:
    """One live turn, in the shape `run_replay_chunk_opd` reads off a prefix.

    That function and `build_replay_batch` touch four things: `prefix_id`,
    `task`, `trace_id` and `step_index` — used for the per-task and per-trace
    share limits and for `supervised_tokens_by_step`. A live turn has natural
    values for all four, and `ReplayPrefix.prefix_id` is already
    `f"{trace_id}@{step_index}"`, which is exactly the live key convention.

    So this is a four-field adapter, not a parallel batch implementation.
    `prefix_turns` is not carried: nothing in the training path reads it, and
    the live equivalent — the semantic history — is already passed separately
    as the teacher-render map.

    One live-vs-replay semantic difference the driver must handle: an episode's
    turns all share a `trace_id`, so `max_trace_share` (0.35 by default) will
    reject any live batch of few episodes. A live update must pass a share
    limit derived from its episode count, not inherit the replay default.
    """

    task: str
    trace_id: str
    step_index: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def prefix_id(self) -> str:
        return f"{self.trace_id}@{self.step_index}"


def live_turn_key(episode_id: str, turn_index: int) -> str:
    """`prefix_id` for a live turn.

    The existing action key is `f"{prefix_id}#{sample_index}"`, so this keeps
    live rows in the same namespace as replay rows while remaining unambiguous:
    one episode, one turn, one sample.
    """
    return f"{episode_id}@{turn_index}"


def capture_row_from_turn(
    turn_row: dict[str, Any], *, teacher_context: str
) -> dict[str, Any]:
    """Archived `LiveTurn` -> the capture-row shape `actions.jsonl` stores.

    `live_agent.TurnCapture.to_json` writes hex and names turns by episode;
    `reopd_sample.build_capture` writes base64 and names them by prefix. Rather
    than fork `capture_to_sampled_action`, translate here -- one small function,
    against a format the scoring, loss and resume paths already understand.

    `semantic_history_hash` rides along because the teacher-score cache keys on
    the action key and validates action *bytes*. For a frozen replay prefix that
    is sufficient. For a live turn it is not: an episode discarded and resampled
    inside one update can produce the same action bytes at the same
    `episode@turn` key under a different preceding history, and reusing that
    paid score would train on a number bought for a trajectory that no longer
    exists -- with every downstream assertion still passing.
    """
    cap = turn_row["capture"]
    token_bytes = [bytes.fromhex(h) for h in cap["action_token_bytes_hex"]]
    row = {
        "prefix_id": live_turn_key(cap["episode_id"], cap["turn_index"]),
        "sample_index": 0,
        "key": f"{live_turn_key(cap['episode_id'], cap['turn_index'])}#0",
        "action_bytes_b64": base64.b64encode(b"".join(token_bytes)).decode(),
        "action_token_bytes_b64": [
            base64.b64encode(b).decode() for b in token_bytes
        ],
        "action_token_ids": [int(t) for t in cap["sampled_token_ids"]],
        "behavior_logprobs": [float(x) for x in cap["behavior_logprobs"]],
        "prompt_token_ids": [int(t) for t in cap["prompt_token_ids"]],
        "policy_version": cap["policy_version"],
        "finish_reason": cap.get("finish_reason"),
        # Live-only provenance. Ignored by the replay path, checked below.
        "episode_id": cap["episode_id"],
        "task_id": cap["task_id"],
        "turn_index": cap["turn_index"],
        "semantic_history_hash": semantic_hash(turn_row["semantic_history"]),
        "teacher_context_hash": teacher_context,
    }
    # The half of the binding `RunState.validate()` reads off the action.
    row["score_fingerprint"] = live_score_fingerprint(row)
    return row


def flatten_live_turns(
    turn_rows_by_episode: dict[str, list[dict[str, Any]]],
    *,
    policy_version: str,
    teacher_context: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Complete episodes -> `(capture_rows, rendered)` for the existing stages.

    `rendered` is the `prefix_id -> canonical_messages` map `score_replay_batch`
    takes. Its values are each turn's semantic history *before* the action --
    the conversation the student acted in -- so DeepSeek renders that state
    natively instead of being handed Qwen prompt ids, which are not a teacher
    context.

    Refuses, rather than silently producing a smaller or mixed batch:

    - a turn tagged with an episode it is not filed under;
    - a turn whose policy version is not this update's;
    - a turn with no semantic history;
    - an episode whose turn indices are not contiguous;
    - a duplicate turn key.

    Episode completeness and discards are `EpisodeArchive.batch_report`'s job
    and must be checked before this is called; the caller passes only the
    episodes that report cleared.
    """
    if not turn_rows_by_episode:
        raise LiveTurnError(
            "no episodes to flatten; an empty batch would take an optimizer "
            "step over zero supervised tokens"
        )

    rows: list[dict[str, Any]] = []
    rendered: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()

    for episode_id in sorted(turn_rows_by_episode):
        turns = sorted(
            turn_rows_by_episode[episode_id],
            key=lambda r: r["capture"]["turn_index"],
        )
        if not turns:
            raise LiveTurnError(f"{episode_id}: no turns to flatten")

        indices = [r["capture"]["turn_index"] for r in turns]
        if indices != list(range(len(indices))):
            raise LiveTurnError(
                f"{episode_id}: turn indices {indices} are not contiguous from "
                "0; a generation is missing and its behaviour logprobs cannot "
                "be recreated"
            )

        for turn in turns:
            cap = turn["capture"]
            # The row is built from the capture's own episode_id, so a capture
            # tagged with a different episode than the one it is filed under
            # would key the row one way and `rendered` another -- two turns
            # colliding on one key, one silently overwriting the other's
            # teacher context.
            if cap["episode_id"] != episode_id:
                raise LiveTurnError(
                    f"{episode_id}: turn {cap['turn_index']} is tagged "
                    f"{cap['episode_id']!r}; a capture filed under the wrong "
                    "episode collides with a real turn key"
                )
            key = live_turn_key(episode_id, cap["turn_index"])
            if cap["policy_version"] != policy_version:
                raise LiveTurnError(
                    f"{key}: sampled under {cap['policy_version']!r} but this "
                    f"update is {policy_version!r}; the student policy is fixed "
                    "for an entire episode and an entire batch"
                )
            if not turn.get("semantic_history"):
                raise LiveTurnError(
                    f"{key}: no semantic history; Qwen prompt ids are not a "
                    "teacher context"
                )
            if key in seen:
                raise LiveTurnError(f"{key}: duplicate turn")
            seen.add(key)

            rows.append(
                capture_row_from_turn(turn, teacher_context=teacher_context)
            )
            rendered[key] = turn["semantic_history"]

    return rows, rendered


def stale_score_keys(
    capture_rows: list[dict[str, Any]],
    paid_scores: dict[str, dict[str, Any]],
) -> list[str]:
    """Paid scores that must not be reused, despite matching action bytes.

    `score_replay_batch` reuses a cached score when its bytes reconstruct the
    action. That is exactly right for a frozen prefix. A live turn key can be
    re-used within one update after a discard-and-resample, so a score whose
    recorded history differs from the history now attached to the key is stale
    and must be dropped from `already_scored` before it is passed along.

    A live score row with **no** recorded history is also dropped. Replay rows
    legitimately lack the field -- a frozen prefix id names a fixed history, so
    the byte check governs them -- but a key in the live namespace without
    provenance cannot be shown to belong to this trajectory, and the safe
    reading of an unprovable score is that it is not reusable.

    Returns the keys to drop. Paying twice is the cheap failure.
    """
    by_key = {r["key"]: r for r in capture_rows}
    stale: list[str] = []
    for key, score in paid_scores.items():
        row = by_key.get(key)
        if row is None:
            continue  # not in this batch; not ours to judge
        recorded = score.get("semantic_history_hash")
        if recorded is None or recorded != row["semantic_history_hash"]:
            stale.append(key)
    return sorted(stale)


def live_prefixes(capture_rows: list[dict[str, Any]]) -> list[LivePrefix]:
    """The `prefixes` argument for `run_replay_chunk_opd`, one per live turn.

    Order matches `capture_rows`, and `prefix_id` matches each row's, so the
    two arguments line up exactly as they do on the replay path.
    """
    return [
        LivePrefix(
            task=str(r["task_id"]),
            trace_id=str(r["episode_id"]),
            step_index=int(r["turn_index"]),
            meta={"policy_version": r["policy_version"]},
        )
        for r in capture_rows
    ]


def live_score_fingerprint(capture_row: dict[str, Any]) -> str:
    """What a paid teacher score for this live turn is valid for.

    `RunState.validate()` already enforces a binding between an action row's
    `score_fingerprint` and its score row's `fingerprint`, on **every resume**.
    Putting the live binding in that same field means the check runs without
    the driver having to remember `stale_score_keys` -- which stays as the
    cheaper in-batch filter that drops a stale score before it is passed to
    `already_scored`.

    `reopd_sample.capture_fingerprint` covers prefix, model, policy version,
    temperature, cap and prompt ids. That is sufficient for a frozen replay
    prefix, whose id names a fixed history. It is not sufficient here: two live
    turns can share a key, an action and a policy version and still follow
    different conversations, so the semantic history is part of the identity.
    """
    h = hashlib.sha256()
    for part in (
        capture_row["key"],
        capture_row["policy_version"],
        capture_row["semantic_history_hash"],
        capture_row["teacher_context_hash"],
        capture_row["action_bytes_b64"],
    ):
        h.update(str(part).encode())
        h.update(b"\x00")
    h.update(json.dumps(capture_row["prompt_token_ids"]).encode())
    return h.hexdigest()[:32]


def score_row_provenance(
    scored: Any, capture_row: dict[str, Any]
) -> dict[str, Any]:
    """The extra fields a live score row must carry when it is persisted.

    `ScoredAction` has no history field, and the ReOPD score row format does
    not add one. Without this, a live paid score on disk cannot be checked
    against the history it was bought for, and `stale_score_keys` has nothing
    to compare. The driver merges this into the row it appends.

    Refuses to describe a score with a row it does not belong to: attaching
    episode A's history to episode B's score would forge exactly the
    provenance `stale_score_keys` exists to check, and the forged row would
    then pass every later validation.
    """
    key = getattr(scored, "key", None)
    if key != capture_row["key"]:
        raise LiveTurnError(
            f"score key {key!r} does not belong to capture row "
            f"{capture_row['key']!r}; provenance must describe its own score"
        )
    return {
        # `RunState.validate()` compares this against the action row's
        # `score_fingerprint` on every resume.
        "fingerprint": live_score_fingerprint(capture_row),
        "semantic_history_hash": capture_row["semantic_history_hash"],
        "teacher_context_hash": capture_row["teacher_context_hash"],
        "policy_version": capture_row["policy_version"],
        "episode_id": capture_row["episode_id"],
        "turn_index": capture_row["turn_index"],
    }


def verify_prompt_parity(
    turn_rows_by_episode: dict[str, list[dict[str, Any]]], render_ids: Any
) -> dict[str, Any]:
    """Require semantic histories to reproduce captured Qwen prompt ids.

    The driver supplies `render_ids(messages)` using the pinned Qwen tokenizer,
    tool schemas and renderer. This closes the persisted-history seam without
    making this data adapter depend on Tau2 or a concrete tokenizer.
    """
    checked = 0
    for episode_id, turns in turn_rows_by_episode.items():
        for turn in turns:
            cap = turn["capture"]
            expected = [int(x) for x in cap["prompt_token_ids"]]
            got = [int(x) for x in render_ids(turn["semantic_history"])]
            if got != expected:
                mismatch = next(
                    (i for i, pair in enumerate(zip(got, expected))
                     if pair[0] != pair[1]),
                    min(len(got), len(expected)),
                )
                raise LiveTurnError(
                    f"{episode_id}@{cap['turn_index']}: semantic history does "
                    f"not reproduce captured prompt ids (first mismatch "
                    f"{mismatch}, rendered={len(got)}, "
                    f"captured={len(expected)})"
                )
            checked += 1
    if checked == 0:
        raise LiveTurnError("no live turns supplied for prompt-parity proof")
    return {"checked": checked, "ok": True}
