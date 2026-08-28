"""Durable episode archive for live Tau2 OPD.

`TurnCapture` (`live_agent.py`) holds what one generation produced. It is not
enough to train on later, for two reasons this module exists to fix:

1. **The teacher cannot read Qwen ids.** `prompt_token_ids` is a Qwen encoding
   of the history. DeepSeek scores its *own* render of the same conversation,
   so the archive must store the semantic history -- the canonical message
   dicts -- alongside the ids, not instead of them.

2. **A generation that fails to parse is still paid for.** `build_capture`
   raises on malformed Hermes before any `TurnCapture` exists, discarding the
   behaviour logprobs, which "are captured here or they do not exist". A
   `FailedTurn` is written *before* that raise propagates.

The archive is append-only. Each paid generation and each paid teacher score
lands on disk the moment it exists, because a crash on turn 6 must not cost the
five turns before it. Episode metadata is a separate record from the turns, so
a torn write to one cannot corrupt the other.

Nothing here calls a model, a GPU or the teacher. It is the schema and its
integrity checks only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from vektori_trace.tau2.live_agent import TurnCapture
from vektori_trace.tau2.reopd_state import append_jsonl

__all__ = [
    "EpisodeStatus",
    "build_failed_turn",
    "classify_failure",
    "LiveTurn",
    "FailedTurn",
    "LiveEpisode",
    "EpisodeArchive",
    "semantic_hash",
    "canonical_json",
    "validate_episode_id",
]


class EpisodeStatus:
    """Lifecycle of one **episode**. Deliberately short.

    Scoring, training, checkpointing and serving-refresh are properties of an
    *update*, not of an episode, and `RunState`'s
    `PLANNED -> SAMPLED -> SCORED -> TRAINED` already owns them. Duplicating
    them here produced a contradiction: an episode had to reach `COMPLETE` to
    be trainable, but `COMPLETE` was terminal, so it could never advance into
    the scoring state it was also supposed to pass through.

    An episode is therefore only ever being sampled, or finished one of three
    ways:

    - `SAMPLED`  -- ran to a Tau2 termination; eligible for a batch.
    - `FAILED`   -- ended on an error the episode itself could not survive.
    - `DISCARDED`-- abandoned; a crash mid-sampling cannot be resumed, because
      Tau2's environment and user simulator are stateful and cannot be rewound
      to turn *k*. The only honest recovery is discard-and-resample.

    `FAILED` and `DISCARDED` are distinct on purpose: one is a measurement of
    the policy, the other is an infrastructure event, and a run that conflates
    them cannot report either.
    """

    SAMPLING = "sampling"
    SAMPLED = "sampled"
    FAILED = "failed"
    DISCARDED = "discarded"

    #: Terminal states. An episode in one of these is never advanced again.
    TERMINAL = frozenset({SAMPLED, FAILED, DISCARDED})

    #: Terminal states that are not eligible for a training batch. Both must be
    #: counted, never silently skipped -- that is what shrinks a batch below
    #: the size the run reports.
    UNUSABLE = frozenset({FAILED, DISCARDED})

    @classmethod
    def can_advance(cls, current: str, nxt: str) -> bool:
        """Only `SAMPLING` -> one terminal state. Nothing else is legal."""
        return current == cls.SAMPLING and nxt in cls.TERMINAL


#: Episode ids become filenames. They are internally generated, so this is
#: defence in depth rather than untrusted input -- but an id containing a path
#: separator would write outside the archive, and refusing by construction is
#: cheaper than auditing every call site that might one day build one.
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_episode_id(episode_id: str) -> str:
    """Refuse anything that is not a safe filename component."""
    if not _SAFE_EPISODE_ID.fullmatch(episode_id) or ".." in episode_id:
        raise ValueError(
            f"unsafe episode id {episode_id!r}: expected 1-128 chars of "
            "[A-Za-z0-9_.-] starting alphanumeric, with no '..'"
        )
    return episode_id


def canonical_json(payload: Any) -> str:
    """Stable serialisation for hashing. Sorted keys and no incidental
    whitespace, so the same history hashes identically across processes and
    Python versions."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def semantic_hash(messages: list[dict[str, Any]]) -> str:
    """Fingerprint of the semantic history a teacher score was bought against.

    A resumed run that reuses a cached score computed under a *different*
    history would be training on a number that never described this
    trajectory, and every downstream assertion would still pass. This hash is
    what makes that detectable.
    """
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


@dataclass
class LiveTurn:
    """One assistant turn that parsed, with everything needed to score and
    train it.

    `semantic_history` is the conversation *before* this action, in the shape
    `replay_score.score_action` takes as `prefix_messages` -- i.e. what
    `tau2_messages_to_canonical` emits. Storing the rendered Qwen prompt ids
    without it would leave the turn unscoreable.
    """

    capture: TurnCapture
    semantic_history: list[dict[str, Any]]
    parsed_message: dict[str, Any]

    #: Observation the environment/user returned in response to this action.
    #: `None` until the next step runs -- the archive writes the turn as soon
    #: as it is generated, so the observation is filled in afterwards.
    observation: dict[str, Any] | None = None

    #: `environment.get_db_hash()` after the action executed. `None` when the
    #: environment exposes no database (tau2 returns `None` there itself).
    env_state_hash: str | None = None

    def semantic_history_hash(self) -> str:
        return semantic_hash(self.semantic_history)

    def observation_hash(self) -> str | None:
        if self.observation is None:
            return None
        return hashlib.sha256(
            canonical_json(self.observation).encode("utf-8")
        ).hexdigest()

    def parsed_message_hash(self) -> str:
        return hashlib.sha256(
            canonical_json(self.parsed_message).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> dict[str, Any]:
        """The `turn_sampled` event: everything that exists at generation time.

        `observation` and `env_state_hash` are normally still `None` here --
        the environment has not run yet -- and are filled in later by a
        separate `turn_observed` event. They are included when already set only
        so a caller that has both can write one record.
        """
        return {
            "kind": "turn",
            "capture": self.capture.to_json(),
            "semantic_history": self.semantic_history,
            "semantic_history_hash": self.semantic_history_hash(),
            "parsed_message": self.parsed_message,
            "parsed_message_hash": self.parsed_message_hash(),
            "observation": self.observation,
            "observation_hash": self.observation_hash(),
            "env_state_hash": self.env_state_hash,
        }


@dataclass
class FailedTurn:
    """A paid generation that did not become a trainable action.

    Malformed Hermes, empty output, or `finish_reason == "length"`. These are
    written before the parse error propagates, so the raw ids and behaviour
    logprobs survive for diagnosis.

    Scope: a 200 response that did not yield a trainable capture. A failure
    with no generation to salvage -- a non-200, a transport exception, a
    prompt over budget -- produces no `FailedTurn`, because there are no
    tokens to archive; the driver records those against the episode.

    They count in validity metrics and are **excluded from OPD training**. A
    cap termination in particular is a truncated fragment, not a completed
    action: training it as one is the mistake the 256-token cap made in the
    0/13 run, where the teacher scored fragments as finished actions and
    nothing in the logs showed it.
    """

    episode_id: str
    task_id: str
    turn_index: int
    policy_version: str

    raw_text: str
    finish_reason: str
    failure_kind: str
    failure_detail: str

    prompt_token_ids: list[int] = field(default_factory=list)
    sampled_token_ids: list[int] = field(default_factory=list)
    behavior_logprobs: list[float] = field(default_factory=list)
    semantic_history: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    #: `True` only for failures that end the episode rather than one turn.
    fatal: bool = True

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "failed_turn"
        return d


@dataclass
class LiveEpisode:
    """Metadata for one complete (or discarded) Tau2 episode.

    Turns live in their own file; this record carries what identifies the
    trajectory and what makes it eligible for a training batch. `adapter_hash`
    and `gen_config_hash` are stored because a batch that mixes two adapters or
    two sampling configurations is not on-policy, and after the fact the alias
    alone cannot prove which weights served.
    """

    episode_id: str
    task_id: str
    seed: int
    policy_version: str
    adapter_hash: str
    gen_config_hash: str

    status: str = EpisodeStatus.SAMPLING
    initial_env_state_hash: str | None = None
    final_env_state_hash: str | None = None
    termination_reason: str | None = None
    reward: float | None = None
    reward_info: dict[str, Any] | None = None
    num_turns: int = 0
    num_failed_turns: int = 0
    discard_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def __post_init__(self) -> None:
        validate_episode_id(self.episode_id)

    def finish(self, nxt: str, *, reason: str | None = None) -> None:
        """End the episode in `nxt`. An episode is finished exactly once.

        Refusing a second call is the point: re-finishing would let a driver
        re-admit an already-consumed episode to a later batch.
        """
        if not EpisodeStatus.can_advance(self.status, nxt):
            raise ValueError(
                f"{self.episode_id}: illegal transition {self.status!r} -> "
                f"{nxt!r}; an episode goes from sampling to exactly one "
                "terminal state"
            )
        if nxt in EpisodeStatus.UNUSABLE:
            if not reason:
                raise ValueError(
                    f"{self.episode_id}: {nxt} must carry a reason -- an "
                    "unexplained loss silently shrinks the batch"
                )
            self.discard_reason = reason
        self.status = nxt
        self.ended_at = time.time()

    @property
    def trainable(self) -> bool:
        """Only a `SAMPLED` episode contributes turns to a batch.

        Not "anything not discarded": an episode still in `SAMPLING` has no
        guarantee its last turn landed, and a truncated trajectory is
        indistinguishable from a complete one once its turns are in a batch.
        """
        return self.status == EpisodeStatus.SAMPLED

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = "episode"
        return d


class EpisodeArchive:
    """Append-only on-disk archive, one directory per run.

    ```text
    <root>/episodes.jsonl        one record per episode state change
    <root>/turns/<episode>.jsonl one record per generation, parsed or failed
    ```

    Every write is `append_jsonl`, which fsyncs. Episode records are appended
    rather than rewritten, so the file is a state *history*: the current status
    of an episode is its last record. That costs a little space and buys the
    ability to see, after a crash, exactly which boundary was crossed.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.turns_dir = self.root / "turns"
        self.root.mkdir(parents=True, exist_ok=True)
        self.turns_dir.mkdir(parents=True, exist_ok=True)

    @property
    def episodes_path(self) -> Path:
        return self.root / "episodes.jsonl"

    def turns_path(self, episode_id: str) -> Path:
        return self.turns_dir / f"{validate_episode_id(episode_id)}.jsonl"

    # -- writes ------------------------------------------------------------

    def record_episode(self, episode: LiveEpisode) -> None:
        """Persist the episode's current state. Called on every transition."""
        append_jsonl(self.episodes_path, episode.to_json())

    def record_turn(self, turn: LiveTurn) -> None:
        """Persist one parsed generation, the moment it lands.

        Called *before* the environment runs, so `observation` is normally
        still absent. Waiting for it would risk the whole point of this write:
        behaviour logprobs cannot be recreated after sampling.
        """
        append_jsonl(self.turns_path(turn.capture.episode_id), turn.to_json())

    def record_observation(
        self,
        episode_id: str,
        turn_index: int,
        observation: dict[str, Any] | None,
        env_state_hash: str | None,
    ) -> None:
        """The `turn_observed` event, appended after the environment responds.

        A separate record rather than a rewrite: the turns file is append-only,
        and `load_turns` merges the latest observation onto its turn. Rewriting
        the earlier line would put a paid capture at risk to record an
        observation that can simply be re-derived.
        """
        append_jsonl(
            self.turns_path(episode_id),
            {
                "kind": "turn_observed",
                "episode_id": episode_id,
                "turn_index": turn_index,
                "observation": observation,
                "observation_hash": (
                    None
                    if observation is None
                    else hashlib.sha256(
                        canonical_json(observation).encode("utf-8")
                    ).hexdigest()
                ),
                "env_state_hash": env_state_hash,
            },
        )

    def record_failure(self, failure: FailedTurn) -> None:
        """Persist one unparseable generation, before the error propagates."""
        append_jsonl(self.turns_path(failure.episode_id), failure.to_json())

    # -- reads -------------------------------------------------------------

    def load_episodes(self) -> dict[str, dict[str, Any]]:
        """Latest record per episode id after validating its full history."""
        out: dict[str, dict[str, Any]] = {}
        if not self.episodes_path.exists():
            return out
        for row in _iter_jsonl(self.episodes_path):
            episode_id = row.get("episode_id")
            if not episode_id:
                raise ValueError(
                    f"{self.episodes_path}: an episode record has no "
                    "'episode_id'; the archive index cannot be built"
                )
            validate_episode_id(str(episode_id))
            status = row.get("status")
            if status not in ({EpisodeStatus.SAMPLING} | EpisodeStatus.TERMINAL):
                raise ValueError(f"{episode_id}: unknown episode status {status!r}")
            previous = out.get(episode_id)
            if previous is None:
                if status != EpisodeStatus.SAMPLING:
                    raise ValueError(
                        f"{episode_id}: first state is {status!r}, not "
                        f"{EpisodeStatus.SAMPLING!r}"
                    )
            else:
                immutable = (
                    "task_id", "seed", "policy_version", "adapter_hash",
                    "gen_config_hash",
                )
                drift = {
                    k: (previous.get(k), row.get(k))
                    for k in immutable if previous.get(k) != row.get(k)
                }
                if drift:
                    raise ValueError(
                        f"{episode_id}: immutable episode identity changed: {drift}"
                    )
                if previous.get("status") in EpisodeStatus.TERMINAL:
                    raise ValueError(
                        f"{episode_id}: record follows terminal state "
                        f"{previous.get('status')!r}"
                    )
                if status == EpisodeStatus.SAMPLING:
                    raise ValueError(
                        f"{episode_id}: duplicate sampling-state record"
                    )
                if row.get("started_at") != previous.get("started_at"):
                    raise ValueError(
                        f"{episode_id}: started_at changed across records"
                    )
                if (
                    row.get("ended_at") is not None
                    and row.get("started_at") is not None
                    and float(row["ended_at"]) < float(row["started_at"])
                ):
                    raise ValueError(f"{episode_id}: ended_at precedes started_at")
            out[episode_id] = row
        return out

    def load_turns(self, episode_id: str) -> list[dict[str, Any]]:
        """Parsed turns for one episode, in turn order, observations merged.

        A later `turn_observed` event wins over whatever the `turn` record
        carried, so a re-observed turn reads as its latest state without any
        line ever being rewritten.
        """
        path = self.turns_path(episode_id)
        if not path.exists():
            return []
        turns: dict[int, dict[str, Any]] = {}
        observed: dict[int, dict[str, Any]] = {}
        for row in _iter_jsonl(path):
            kind = row.get("kind")
            if kind == "turn":
                idx = row["capture"]["turn_index"]
                if idx in turns:
                    raise ValueError(
                        f"{episode_id}: duplicate archived capture for turn "
                        f"{idx}; sampled generations are not last-write-wins"
                    )
                turns[idx] = row
            elif kind == "turn_observed":
                observed[row["turn_index"]] = row
        for idx, obs in observed.items():
            turn = turns.get(idx)
            if turn is None:
                continue  # an observation with no capture; verify_episode reports it
            turn["observation"] = obs["observation"]
            turn["observation_hash"] = obs["observation_hash"]
            turn["env_state_hash"] = obs["env_state_hash"]
        return [turns[i] for i in sorted(turns)]

    def load_failures(self, episode_id: str) -> list[dict[str, Any]]:
        path = self.turns_path(episode_id)
        if not path.exists():
            return []
        rows = [r for r in _iter_jsonl(path) if r.get("kind") == "failed_turn"]
        rows.sort(key=lambda r: r["turn_index"])
        return rows

    # -- integrity ---------------------------------------------------------

    def verify_episode(self, episode_id: str) -> list[str]:
        """Structural problems that must block this episode from a batch.

        Returns a list of complaints; empty means the episode is structurally
        fit to train on. This checks shape, not learning: byte alignment and
        teacher agreement are the scoring layer's job.
        """
        problems: list[str] = []
        episodes = self.load_episodes()
        meta = episodes.get(episode_id)
        if meta is None:
            return [f"{episode_id}: no episode record"]

        turns = self.load_turns(episode_id)
        failures = self.load_failures(episode_id)

        if meta["status"] in EpisodeStatus.UNUSABLE:
            problems.append(
                f"{episode_id}: {meta['status']} ({meta.get('discard_reason')}) "
                "-- not trainable, and must be counted, not skipped"
            )
            return problems

        if meta["status"] != EpisodeStatus.SAMPLED:
            problems.append(
                f"{episode_id}: status {meta['status']!r}, not "
                f"{EpisodeStatus.SAMPLED!r}; sampling did not finish"
            )

        if not turns:
            problems.append(f"{episode_id}: no parsed turns")

        # A SAMPLED episode ran to a Tau2 termination; the reason is what
        # distinguishes a finished conversation from a loop that hit max_steps,
        # and a batch cannot report validity without it.
        if not meta.get("termination_reason"):
            problems.append(
                f"{episode_id}: status {EpisodeStatus.SAMPLED!r} with no "
                "termination_reason; an episode that ran to completion records "
                "how Tau2 ended it"
            )

        # A capture failure ends the episode. `CapturingLLMAgent` re-raises
        # every one of them, so a turn after a failed turn cannot normally
        # exist -- and an episode with a hole in it is a partial trajectory
        # whose remaining turns condition on a state the archive cannot
        # describe. Such an episode must be recorded FAILED, not SAMPLED.
        #
        # This also keeps the archive gate and `flatten_live_turns` in
        # agreement: the adapter rejects non-contiguous parsed indices, so an
        # episode accepted here would otherwise fail at batch assembly.
        if failures:
            kinds = sorted({f["failure_kind"] for f in failures})
            problems.append(
                f"{episode_id}: {len(failures)} archived capture failure(s) "
                f"{kinds} but status is {meta['status']!r}; a capture failure "
                f"ends the episode, which must be recorded "
                f"{EpisodeStatus.FAILED!r} and excluded from training"
            )

        indices = [r["capture"]["turn_index"] for r in turns]
        expected = list(range(len(indices)))
        if indices != expected:
            problems.append(
                f"{episode_id}: turn indices {indices} != {expected}; a "
                "generation is missing and its behaviour logprobs cannot be "
                "recreated"
            )

        # A driver that claims five turns while four are on disk would train a
        # batch one turn smaller than the run reports, changing the global
        # denominator invisibly.
        declared = meta.get("num_turns")
        if declared is not None and int(declared) != len(turns):
            problems.append(
                f"{episode_id}: episode record declares {declared} turns but "
                f"{len(turns)} are archived"
            )
        declared_failed = meta.get("num_failed_turns")
        if declared_failed is not None and int(declared_failed) != len(failures):
            problems.append(
                f"{episode_id}: episode record declares {declared_failed} failed "
                f"turns but {len(failures)} are archived"
            )

        # An observation whose capture never landed means the turns file is
        # describing a generation that is not there.
        turns_file = self.turns_path(episode_id)
        observed = (
            {
                r["turn_index"]
                for r in _iter_jsonl(turns_file)
                if r.get("kind") == "turn_observed"
            }
            if turns_file.exists()
            else set()
        )
        orphans = sorted(observed - set(indices))
        if orphans:
            problems.append(
                f"{episode_id}: observations for turns {orphans} with no "
                "archived capture"
            )

        # `LivePrefix.task` is read off the turn, and `max_task_share` limits
        # how much of a batch one task may supply. A turn mislabelled with
        # another task would skew exactly the balance that check enforces.
        tasks = {r["capture"]["task_id"] for r in turns}
        if tasks - {meta["task_id"]}:
            problems.append(
                f"{episode_id}: turns claim tasks {sorted(tasks)} but the "
                f"episode is task {meta['task_id']!r}; per-task batch shares "
                "are computed from the turn"
            )

        versions = {r["capture"]["policy_version"] for r in turns}
        if len(versions) > 1:
            problems.append(
                f"{episode_id}: turns span policy versions {sorted(versions)}; "
                "the student policy must be fixed for an entire episode"
            )
        elif versions and meta["policy_version"] not in versions:
            problems.append(
                f"{episode_id}: turns are {sorted(versions)} but the episode "
                f"records {meta['policy_version']!r}"
            )

        for r in turns:
            cap = r["capture"]
            idx = cap["turn_index"]
            n_ids = len(cap["sampled_token_ids"])
            n_lps = len(cap["behavior_logprobs"])
            if n_ids != n_lps:
                problems.append(
                    f"{episode_id} turn {idx}: {n_ids} sampled ids against "
                    f"{n_lps} behaviour logprobs"
                )
            if n_ids == 0:
                problems.append(f"{episode_id} turn {idx}: empty generation")
            if not r.get("semantic_history"):
                problems.append(
                    f"{episode_id} turn {idx}: no semantic history; the "
                    "teacher cannot be given Qwen prompt ids as its context"
                )
            if cap.get("finish_reason") == "length":
                problems.append(
                    f"{episode_id} turn {idx}: finish_reason 'length' -- a cap "
                    "termination is a fragment, not a completed action"
                )

        return problems

    def batch_report(
        self,
        episode_ids: list[str],
        *,
        expected_episode_ids: list[str] | None = None,
        policy_version: str | None = None,
        adapter_hash: str | None = None,
        gen_config_hash: str | None = None,
    ) -> dict[str, Any]:
        """Whether these episodes may be trained on together.

        Refuses a short, empty, duplicated or mixed-policy batch, and reports
        failures and discards explicitly so a shrunken batch cannot pass as a
        full one.

        `policy_version`, when given, is the version this update *asked* to
        sample under. Without it the report can only confirm the episodes agree
        with each other, not that they agree with the update -- a batch
        uniformly sampled under the previous adapter would pass.
        """
        problems: list[str] = []

        # An empty batch is not vacuously fine: it would take an optimizer step
        # over zero supervised tokens.
        if not episode_ids:
            problems.append("empty batch: no episodes requested")

        # The same episode twice is not two episodes. It would double every one
        # of its turns in the denominator while halving the batch's actual
        # diversity, and every other check would still pass.
        duplicates = sorted({e for e in episode_ids if episode_ids.count(e) > 1})
        if duplicates:
            problems.append(f"batch lists {duplicates} more than once")

        if expected_episode_ids is not None:
            expected_duplicates = sorted({
                e for e in expected_episode_ids
                if expected_episode_ids.count(e) > 1
            })
            if expected_duplicates:
                problems.append(
                    f"planned episode list contains duplicates "
                    f"{expected_duplicates}"
                )
            got_set, expected_set = set(episode_ids), set(expected_episode_ids)
            absent = sorted(expected_set - got_set)
            foreign = sorted(got_set - expected_set)
            if absent:
                problems.append(
                    f"batch is short; missing planned episodes {absent}"
                )
            if foreign:
                problems.append(f"batch contains unplanned episodes {foreign}")

        episodes = self.load_episodes()
        episode_ids = list(dict.fromkeys(episode_ids))
        missing = [e for e in episode_ids if e not in episodes]
        unusable = [
            e
            for e in episode_ids
            if episodes.get(e, {}).get("status") in EpisodeStatus.UNUSABLE
        ]
        discarded = [
            e for e in unusable
            if episodes[e]["status"] == EpisodeStatus.DISCARDED
        ]
        failed = [
            e for e in unusable
            if episodes[e]["status"] == EpisodeStatus.FAILED
        ]
        for e in missing:
            problems.append(f"{e}: no episode record")
        for e in failed:
            problems.append(
                f"{e}: failed ({episodes[e].get('discard_reason')}) -- the "
                "planned batch contains a policy/capture failure"
            )
        for e in discarded:
            problems.append(
                f"{e}: discarded ({episodes[e].get('discard_reason')}) -- the "
                "planned batch contains an infrastructure loss"
            )

        versions = {
            episodes[e]["policy_version"] for e in episode_ids if e in episodes
        }
        adapters = {episodes[e]["adapter_hash"] for e in episode_ids if e in episodes}
        gen_cfgs = {
            episodes[e]["gen_config_hash"] for e in episode_ids if e in episodes
        }
        if len(versions) > 1:
            problems.append(f"batch spans policy versions {sorted(versions)}")
        if policy_version is not None and versions - {policy_version}:
            problems.append(
                f"batch was sampled under {sorted(versions)} but this update "
                f"is {policy_version!r}; the next batch must be sampled from "
                "the newly reloaded checkpoint"
            )
        if len(adapters) > 1:
            problems.append(f"batch spans adapter hashes {sorted(adapters)}")
        if adapter_hash is not None and adapters - {adapter_hash}:
            problems.append(
                f"batch was sampled under adapter hashes {sorted(adapters)} "
                f"but the planned adapter is {adapter_hash!r}"
            )
        if len(gen_cfgs) > 1:
            problems.append(
                f"batch spans generation configs {sorted(gen_cfgs)}; the "
                "episodes were not sampled under one policy"
            )
        if gen_config_hash is not None and gen_cfgs - {gen_config_hash}:
            problems.append(
                f"batch was sampled under generation configs "
                f"{sorted(gen_cfgs)} but the planned config is "
                f"{gen_config_hash!r}"
            )

        trainable: list[str] = []
        for e in episode_ids:
            if e in missing or e in unusable:
                continue
            episode_problems = self.verify_episode(e)
            if episode_problems:
                problems.extend(episode_problems)
            else:
                trainable.append(e)

        n_turns = sum(len(self.load_turns(e)) for e in trainable)
        n_failed = sum(len(self.load_failures(e)) for e in episode_ids)
        return {
            "requested": len(episode_ids),
            "duplicates": duplicates,
            "trainable": len(trainable),
            "trainable_episode_ids": trainable,
            "discarded": len(discarded),
            "discarded_episode_ids": discarded,
            "failed": len(failed),
            "failed_episode_ids": failed,
            "trainable_turns": n_turns,
            "failed_turns": n_failed,
            "policy_versions": sorted(versions),
            "problems": problems,
            "ok": not problems and len(trainable) == len(episode_ids),
        }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file, tolerating a torn final line.

    A crash mid-append leaves a partial last line. Every line before it is
    intact and paid for; refusing to read the file would throw that away.
    """
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                return  # torn tail from a crash mid-write
            raise


# ---------------------------------------------------------------------------
# Failure archival
# ---------------------------------------------------------------------------

#: Failure kinds, in the order `classify_failure` tests them.
FAILURE_CAP = "cap_termination"
FAILURE_EMPTY = "empty_generation"
FAILURE_MALFORMED_TOOL = "malformed_tool_call"
FAILURE_NO_TOKEN_IDS = "no_token_ids"
FAILURE_LOGPROB = "logprob_mismatch"
FAILURE_BYTE_MISMATCH = "byte_mismatch"
FAILURE_TRANSPORT = "transport"
FAILURE_OTHER = "other"


def classify_failure(message: str) -> str:
    """Bucket a `LiveCaptureError` for the validity metrics.

    Kept as message matching deliberately: the alternative is an exception
    hierarchy in `live_agent`, and these strings are asserted by its tests, so
    a change to one breaks the other loudly rather than silently reclassifying
    a failure mode.
    """
    m = message.lower()
    if "finish_reason=length" in m or "token cap" in m:
        return FAILURE_CAP
    if "no token ids" in m or "return_token_ids" in m:
        return FAILURE_NO_TOKEN_IDS
    if "logprob" in m:
        return FAILURE_LOGPROB
    if "do not reconstruct" in m:
        return FAILURE_BYTE_MISMATCH
    if "tool_call" in m or "json" in m or "hermes" in m:
        return FAILURE_MALFORMED_TOOL
    if "no choices" in m or "no 'text'" in m or "http " in m:
        return FAILURE_TRANSPORT
    if "empty" in m:
        return FAILURE_EMPTY
    return FAILURE_OTHER


def build_failed_turn(
    *,
    body: dict[str, Any],
    episode_id: str,
    task_id: str,
    turn_index: int,
    policy_version: str,
    prompt_ids: list[int],
    semantic_history: list[dict[str, Any]],
    error: BaseException,
) -> FailedTurn:
    """Salvage everything the response body still holds.

    Every field is read defensively: this runs precisely when the body did not
    have the shape the capture path expected, so a `KeyError` here would
    destroy the record meant to explain that.
    """
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    logprob_block = choice.get("logprobs") or {}
    raw_lps = logprob_block.get("token_logprobs") or []
    return FailedTurn(
        episode_id=episode_id,
        task_id=task_id,
        turn_index=turn_index,
        policy_version=policy_version,
        raw_text=str(choice.get("text") or ""),
        finish_reason=str(choice.get("finish_reason") or ""),
        failure_kind=classify_failure(str(error)),
        failure_detail=str(error),
        prompt_token_ids=list(prompt_ids),
        sampled_token_ids=list(choice.get("token_ids") or []),
        behavior_logprobs=[float(x) for x in raw_lps if x is not None],
        semantic_history=list(semantic_history),
        usage=dict(body.get("usage") or {}),
    )
