"""Terminus-2 subclass that seeds one format-calibration exchange before turn 1.

Phase 7 established that repaired-v1 copies its output protocol from the visible
assistant history rather than from the instruction block: the dose-response probe
flips at 0 -> 1 prior assistant turns, and the ablation flips it back by rewriting
that single turn into v1's envelope. The instruction block is byte-identical in
every prefix and is demonstrably ignored at turn 1, so more instruction text
cannot help. An example in the *assistant* role is the only lever that moves it.

This subclass is the smallest thing that tests whether that lever is deployable.
It seeds one task-neutral user/assistant pair into the chat history and then hands
control back to harbor unchanged. It does NOT proxy requests, rewrite completions,
translate tool calls, or synthesize observations.

Read `native_json` in the probe log as the primary signal, and `echoed_seed` as
its guard: a completion that merely regurgitates the demonstration is bare native
JSON that harbor accepts, and would otherwise score as a pass while meaning
nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.llms.chat import Chat

# Harbor renders instructions, task and the live terminal state into ONE user
# message. We split that rendered prompt exactly once, at the marker below, so the
# demonstration lands between the instructions and the real terminal state:
#
#   user       instructions + task + "wait for the terminal state"
#   assistant  calibration JSON, commands: []          <- synthetic few-shot
#   user       the genuine live terminal state + "begin"
#
# This reproduces the geometry Phase 7 measured as working (action -> short
# observation -> generate) WITHOUT fabricating an observation: the terminal state
# in the final message is the real one harbor captured, merely relocated.
#
# The earlier version placed the pair *before* the whole prompt, which left the
# 5,277-char instruction block as the final message -- byte-identical to the
# unseeded turn-1 condition Phase 7 measured at 0/8 native. Same number of prior
# assistant turns, opposite result: the operative variable is adjacency to the
# generation point, not the count.
SPLIT_MARKER = "Current terminal state:"

WAIT_INSTRUCTION = (
    "\n\nBefore you begin: do not act yet. The live terminal state follows in the "
    "next message. Acknowledge with your standard response format and an empty "
    "commands list."
)

BEGIN_INSTRUCTION = "\n\nBegin the task."

# `commands: []` is the *correct* reply to the instruction above, not an
# unmotivated no-op -- and "Begin the task." then supersedes it, so copying this
# turn is visibly wrong. The non-empty-keystroke gate is the tripwire if the model
# copies it anyway.
CALIBRATION_ASSISTANT = json.dumps(
    {
        "analysis": "Acknowledged. I have the instructions and the task, but the "
        "live terminal state has not been provided yet, so there is nothing to "
        "inspect and no basis for choosing a command.",
        "plan": "Take no action this turn. Wait for the terminal state in the next "
        "message, then begin by orienting in the working directory.",
        "commands": [],
        "task_complete": False,
    },
    indent=2,
)


class FirstTurnProtocolFailure(RuntimeError):
    """Turn 1 failed the protocol gate.

    Raised to abort the rollout immediately rather than let harbor feed the parse
    error back and start the loop that produced 8 identical `<tool_call>` turns in
    the previous run. That loop costs GPU and teaches nothing: once turn 1 emits
    the legacy envelope, every later turn has the model's own legacy output as the
    nearest assistant turn to copy.
    """


def is_native_json(text: str) -> bool:
    """True when the completion *is* the object: no fence, no prose, no think block.

    This is the strict tier from Phase 7. Harbor itself is more lenient -- it
    salvages a fenced object with only a warning -- so a checkpoint that is nearly
    repaired but still wrapping its output would pass the parser and fail here,
    which is the distinction the probe exists to record.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        return isinstance(json.loads(stripped), dict)
    except (json.JSONDecodeError, ValueError):
        return False


def has_legacy_envelope(text: str) -> bool:
    """True when the completion carries v1's `<tool_call>` envelope or a think block."""
    return "<tool_call>" in text or "<think>" in text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


class Terminus2PromptSeed(Terminus2):
    """Terminus-2 with one calibration exchange seeded ahead of the real prompt.

    Message geometry at turn 1, and reproduced after every compaction:

        user      calibration request
        assistant bare native JSON, no executable command
        user      harbor's unchanged initial prompt (instructions, task, terminal)

    The pair lives only in `chat.messages`. It is never appended to
    `_trajectory_steps`, so nothing in it can reach tmux -- only parsed model
    completions do.
    """

    @staticmethod
    def name() -> str:
        return "terminus-2-prompt-seed"

    def __init__(self, *args: Any, probe_log: str | Path | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # `vektori passk` exposes no --agent-kwargs, and harbor constructs the agent
        # itself, so the env var is the only channel that reaches this __init__
        # without new CLI plumbing. The explicit kwarg still wins when harbor is
        # driven directly via `--ak probe_log=...`.
        # Deterministic sampling, so a rerun of this exact geometry reproduces.
        # NOTE: this means the run tests "new geometry + temperature 0", not
        # position in isolation -- both moved relative to the previous run.
        kwargs["temperature"] = float(os.environ.get("PROMPT_SEED_TEMPERATURE", "0"))
        resolved = probe_log or os.environ.get("PROMPT_SEED_PROBE_LOG")
        self._probe_log = Path(resolved) if resolved else None
        self._probe_turn = 0
        self._probe_episode = 0
        self._reseed_turn = -1
        self._seed_events: list[dict[str, Any]] = []
        self._first_completion: str | None = None

    def _reset_per_run_state(self) -> None:
        """Harbor reuses one agent instance across `run()` calls, and the log is
        opened in append mode. Without an episode counter a retried rollout writes
        a second turn-1 record into the same file and the summarizer grades the
        abandoned attempt, because it reads the first record it finds.
        """
        super()._reset_per_run_state()
        self._probe_episode += 1
        self._probe_turn = 0
        self._reseed_turn = -1
        self._first_completion = None

    # ---- seeding -------------------------------------------------------------

    @staticmethod
    def split_initial_prompt(prompt: str) -> tuple[str, str]:
        """Split harbor's rendered prompt into (instructions+task, terminal state).

        Fails closed. A marker that is absent, or present more than once, means the
        template changed or the task text happens to contain the phrase -- either
        way the geometry we are testing is not the geometry we would build, and a
        silently mis-split prompt would produce a result nobody could interpret.
        """
        count = prompt.count(SPLIT_MARKER)
        if count != 1:
            raise FirstTurnProtocolFailure(
                f"expected exactly 1 {SPLIT_MARKER!r} in the rendered prompt, found {count}"
            )
        head, tail = prompt.split(SPLIT_MARKER)
        return head.rstrip(), SPLIT_MARKER + tail

    def seed_messages(self, head: str) -> list[dict[str, str]]:
        """The synthetic few-shot context: harbor's real instructions + task, a
        wait instruction, and a calibration reply. Only the assistant turn and the
        two added sentences are ours; everything else is harbor's own text."""
        return [
            {"role": "user", "content": head + WAIT_INSTRUCTION},
            {"role": "assistant", "content": CALIBRATION_ASSISTANT},
        ]

    def _seed_present(self, chat: Chat) -> bool:
        return any(
            m.get("role") == "assistant"
            and _normalize(m.get("content") or "") == _normalize(CALIBRATION_ASSISTANT)
            for m in chat.messages
        )

    def _strip_seed(self, chat: Chat) -> int:
        """Remove the synthetic assistant turn and the user turn carrying the wait
        instruction, so harbor sees a history shaped like an unseeded one before we
        delegate to code that makes structural assumptions about it."""
        cal = _normalize(CALIBRATION_ASSISTANT)
        wait = _normalize(WAIT_INSTRUCTION)
        before = len(chat.messages)
        chat._messages = [
            m
            for m in chat.messages
            if _normalize(m.get("content") or "") != cal
            and not _normalize(m.get("content") or "").endswith(wait)
        ]
        removed = before - len(chat.messages)
        if removed:
            chat.reset_response_chain()
        return removed

    def _record_seed_event(self, chat: Chat, reason: str, geometry: str, **extra: Any) -> None:
        if reason == "post_compaction":
            self._reseed_turn = self._probe_turn
        event = {
            "reason": reason,
            "geometry": geometry,
            "episode": self._probe_episode,
            "turn": self._probe_turn,
            "n_messages": len(chat.messages),
            **extra,
        }
        self._seed_events.append(event)
        self.logger.info("PROMPT-SEED: %s", json.dumps(event))

    # ---- hooks ---------------------------------------------------------------

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        original_instruction: str = "",
    ) -> None:
        """Install the split geometry, then hand control back to harbor unchanged.

        `run()` gives us an empty Chat and the rendered initial prompt. We seed the
        head plus the calibration reply, and pass the *tail* on as the prompt harbor
        will send -- so harbor's own `chat.chat(prompt)` appends the genuine terminal
        state as the final user message, immediately after the demonstration.
        """
        head, tail = self.split_initial_prompt(initial_prompt)
        chat._messages = self.seed_messages(head)
        chat.reset_response_chain()
        self._record_seed_event(
            chat,
            reason="initial",
            geometry="split",
            head_chars=len(head),
            tail_chars=len(tail),
            head_sha256=hashlib.sha256(head.encode()).hexdigest()[:16],
            tail_sha256=hashlib.sha256(tail.encode()).hexdigest()[:16],
        )
        return await super()._run_agent_loop(
            initial_prompt=tail + BEGIN_INSTRUCTION,
            chat=chat,
            original_instruction=original_instruction,
        )

    async def _summarize(self, chat: Chat, original_instruction: str, session: TmuxSession):
        """Reinstate the seed after compaction, without corrupting compaction itself.

        Harbor rebuilds history as `[chat.messages[0], question_prompt, questions]`
        (terminus_2.py:939) and treats `messages[0]` as the anchor carrying the
        Terminus contract and the task. With the seed installed, `messages[0]` is
        our modified head, so we strip first and reinstate after.

        The handoff prompt carries no terminal-state marker, so the split geometry
        cannot be reproduced there. We fall back to appending the pair and record
        `geometry: fallback` -- a post-compaction turn is therefore NOT evidence
        about the split hypothesis, and the log says so rather than leaving it to
        be inferred.
        """
        removed = self._strip_seed(chat)
        self.logger.info("PROMPT-SEED: stripped %d seed messages before summarization", removed)
        try:
            return await super()._summarize(chat, original_instruction, session)
        finally:
            chat._messages = [
                *chat.messages,
                {"role": "assistant", "content": CALIBRATION_ASSISTANT},
            ]
            chat.reset_response_chain()
            self._record_seed_event(chat, reason="post_compaction", geometry="fallback")

    async def _handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        original_instruction: str = "",
        session: TmuxSession | None = None,
    ) -> tuple[list[Command], bool, str, str, str, Any]:
        seed_in_context = self._seed_present(chat)
        result = await super()._handle_llm_interaction(
            chat=chat,
            prompt=prompt,
            original_instruction=original_instruction,
            session=session,
        )
        commands, is_complete, _feedback, _analysis, _plan, llm_response = result

        self._probe_turn += 1
        raw = llm_response.content or ""
        if self._first_completion is None:
            self._first_completion = raw

        parsed = self._parser.parse_response(raw)
        keystrokes = [c.keystrokes for c in commands]
        # Turn 1 is the whole experiment; the turn after a compaction is where the
        # reseed either held or did not.
        verbatim_context = self._probe_turn == 1 or self._probe_turn == self._reseed_turn + 1
        record = {
            "episode": self._probe_episode,
            "turn": self._probe_turn,
            "seed_in_context": seed_in_context,
            # Verbatim context on the turns where the finding lives -- turn 1 and
            # the first turn after a compaction -- and a digest elsewhere. Prefixes
            # run to ~40k tokens, so logging every turn in full would produce a file
            # large enough that nobody opens it, which is its own kind of not
            # logging. The digest still pins roles, sizes and content hashes, so any
            # later claim about what the model saw stays checkable.
            "messages": (
                [{"role": m.get("role"), "content": m.get("content")} for m in chat.messages]
                if verbatim_context
                else None
            ),
            "messages_digest": [
                {
                    "role": m.get("role"),
                    "chars": len(m.get("content") or ""),
                    "sha256": hashlib.sha256((m.get("content") or "").encode()).hexdigest()[:16],
                }
                for m in chat.messages
            ],
            "prompt": prompt if verbatim_context else None,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "native_json": is_native_json(raw),
            "legacy_envelope": has_legacy_envelope(raw),
            # A completion that just replays the demonstration is native JSON that
            # harbor accepts while meaning nothing. Without this the protocol gate
            # can pass vacuously.
            "echoed_seed": _normalize(raw) == _normalize(CALIBRATION_ASSISTANT),
            "parser_error": parsed.error or None,
            "parser_warning": parsed.warning or None,
            "harbor_accepts": not parsed.error,
            "n_commands": len(commands),
            # Named for what this actually is. We sit between parsing and
            # `_execute_commands`, so these are the keystrokes harbor *intends* to
            # send; nothing here is evidence that any of them reached tmux. Real
            # execution has to be read off the trajectory and terminal output.
            "parsed_keystrokes": keystrokes,
            # Guards the copied-emptiness failure mode. A first action that parses
            # cleanly but sends nothing is a stall, not a working protocol.
            "has_keystrokes": any(k.strip() for k in keystrokes),
            "task_complete": is_complete,
            "raw_completion": raw,
            "prompt_chars": len(prompt),
            "n_messages": len(chat.messages),
        }
        self._write_record(record)
        # Abort on a failed first turn instead of letting harbor feed the parse
        # error back: once turn 1 emits the legacy envelope, every later turn has
        # the model's own legacy output as the nearest assistant turn to copy, and
        # the loop burns GPU producing eight identical failures (previous run).
        if self._probe_turn == 1:
            gate = {
                "harbor_accepts": record["harbor_accepts"],
                "no_legacy_envelope": not record["legacy_envelope"],
                "has_keystrokes": record["has_keystrokes"],
                "not_a_seed_echo": not record["echoed_seed"],
            }
            record["first_turn_gate"] = gate
            self.logger.info("PROMPT-SEED first-turn gate: %s", json.dumps(gate))
            if not all(gate.values()):
                failed = [k for k, v in gate.items() if not v]
                raise FirstTurnProtocolFailure(
                    f"turn 1 failed the protocol gate: {failed}; parser_error="
                    f"{record['parser_error']!r}"
                )
        self.logger.info(
            "PROMPT-SEED turn %d: seeded=%s native_json=%s legacy=%s cmds=%d err=%s",
            record["turn"],
            record["seed_in_context"],
            record["native_json"],
            record["legacy_envelope"],
            record["n_commands"],
            record["parser_error"],
        )
        return result

    def _write_record(self, record: dict[str, Any]) -> None:
        if self._probe_log is None:
            return
        self._probe_log.parent.mkdir(parents=True, exist_ok=True)
        with self._probe_log.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


# ---- preflight ---------------------------------------------------------------


def preflight(out_dir: Path) -> int:
    """Prove the split geometry without an environment, an endpoint, or a GPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import harbor.agents.terminus_2.terminus_2 as t2mod

    template_path = Path(t2mod.__file__).parent / "templates" / "terminus-json-plain.txt"
    initial_prompt = template_path.read_text().format(
        instruction="<TASK INSTRUCTION>",
        terminal_state="root@0000000000:/workspace#",
    )

    head, tail = Terminus2PromptSeed.split_initial_prompt(initial_prompt)
    messages = [
        *Terminus2PromptSeed.seed_messages(Terminus2PromptSeed, head),
        {"role": "user", "content": tail + BEGIN_INSTRUCTION},
    ]

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "geometry_user_assistant_user",
            [m["role"] for m in messages] == ["user", "assistant", "user"],
            "demonstration sits between the instructions and the terminal state",
        )
    )
    checks.append(
        (
            "final_message_is_the_terminal_state",
            messages[-1]["content"].startswith(SPLIT_MARKER)
            and "root@0000000000:/workspace#" in messages[-1]["content"],
            "generation happens right after the REAL terminal state, as in Phase 7",
        )
    )
    checks.append(
        (
            "final_message_is_short",
            len(messages[-1]["content"]) < 500,
            "the last message is a short observation, not the 5k instruction block",
        )
    )
    checks.append(
        (
            "instructions_and_task_precede_the_demo",
            "Format your response as JSON" in messages[0]["content"]
            and "<TASK INSTRUCTION>" in messages[0]["content"],
            "harbor's contract and the task survive, above the demonstration",
        )
    )
    checks.append(
        (
            "split_round_trips",
            (head + "\n\n" + tail).replace("\n", "").replace(" ", "")
            == initial_prompt.replace("\n", "").replace(" ", ""),
            "head + tail reconstruct harbor's rendered prompt exactly",
        )
    )
    checks.append(
        (
            "demonstration_is_native_json",
            is_native_json(CALIBRATION_ASSISTANT)
            and not has_legacy_envelope(CALIBRATION_ASSISTANT),
            "the demonstration is a bare JSON object with no legacy envelope",
        )
    )
    demo = json.loads(CALIBRATION_ASSISTANT)
    checks.append(
        (
            "demonstration_carries_no_commands",
            demo["commands"] == [] and demo["task_complete"] is False,
            "commands: [] is the correct reply to the wait instruction above it",
        )
    )
    checks.append(
        (
            "copied_demonstration_would_fail_the_gate",
            not any(c.get("keystrokes", "").strip() for c in demo["commands"]),
            "a verbatim copy fails the non-empty-keystroke gate",
        )
    )
    checks.append(
        (
            "split_fails_closed",
            _split_fails_closed(),
            "a missing or duplicated marker raises instead of mis-splitting",
        )
    )

    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "agent_sha256": sha,
        "agent_source": str(Path(__file__).resolve()),
        "template_path": str(template_path),
        "split_marker": SPLIT_MARKER,
        "head_sha256": hashlib.sha256(head.encode()).hexdigest(),
        "tail_sha256": hashlib.sha256(tail.encode()).hexdigest(),
        "head": head,
        "tail": tail,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "all_passed": all(p for _, p, _ in checks),
        "first_request_messages": messages,
        "synthetic": [
            "messages[1] (assistant calibration)",
            "WAIT_INSTRUCTION",
            "BEGIN_INSTRUCTION",
        ],
    }
    (out_dir / "preflight.json").write_text(json.dumps(report, indent=2))

    print(f"agent sha256: {sha}")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} -- {detail}")
    print("\nsynthetic content (everything else is harbor's own text):")
    for item in report["synthetic"]:
        print(f"  - {item}")
    print("\n--- first request, message sequence ---")
    for i, m in enumerate(messages):
        body = m["content"]
        shown = (
            body
            if len(body) <= 700
            else body[:350] + f"\n    ... [{len(body) - 700} chars elided] ...\n" + body[-350:]
        )
        print(f"\n[{i}] role={m['role']}  ({len(body)} chars)")
        print("    " + shown.replace("\n", "\n    "))
    print(f"\nwrote {out_dir / 'preflight.json'}")
    return 0 if report["all_passed"] else 1


def _split_fails_closed() -> bool:
    for bad in ("no marker here", f"{SPLIT_MARKER} a {SPLIT_MARKER} b"):
        try:
            Terminus2PromptSeed.split_initial_prompt(bad)
            return False
        except FirstTurnProtocolFailure:
            continue
    return True


def summarize(log_path: Path) -> int:
    """Grade a probe log against the primary protocol gate.

    Deliberately does not touch the secondary capability evidence (orientation,
    edit/test behaviour, verifier outcome) -- those are read from the trajectory
    and the verifier, and folding them in here would blur a protocol result into
    a capability result.
    """
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        print(f"no records in {log_path}")
        return 1

    # The log is append-mode and harbor may retry a rollout on the same agent
    # instance. Grading records[0] blindly would score an abandoned attempt, so
    # episodes are separated explicitly and the last one -- the attempt that
    # actually stood -- is the one graded.
    episodes = sorted({r.get("episode", 1) for r in records})
    if len(episodes) > 1:
        print(f"WARNING: {len(episodes)} episodes in this log ({episodes}); grading the last")
        for ep in episodes:
            n = sum(1 for r in records if r.get("episode", 1) == ep)
            print(f"  episode {ep}: {n} turns")
    graded_episode = episodes[-1]
    records = [r for r in records if r.get("episode", 1) == graded_episode]
    print(f"grading episode {graded_episode}")

    first = records[0]
    if first.get("turn") != 1:
        print(f"WARNING: first graded record is turn {first.get('turn')}, not 1")
    gates = {
        "first_response_is_native_json": first["native_json"],
        "first_response_not_legacy_envelope": not first["legacy_envelope"],
        "harbor_accepts_first_response": first["harbor_accepts"],
        "first_response_is_not_a_seed_echo": not first["echoed_seed"],
        "first_action_has_keystrokes": first["has_keystrokes"],
        "no_parser_loop": not _has_parser_loop(records),
    }

    print(f"turns: {len(records)}   log: {log_path}")
    print(
        f"seed events: {sum(1 for r in records if r['seed_in_context'])} turns with seed in context"
    )
    print("\nprimary protocol gate (turn 1):")
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    native = sum(1 for r in records if r["native_json"])
    legacy = sum(1 for r in records if r["legacy_envelope"])
    print(
        f"\nacross all turns: native_json {native}/{len(records)}, legacy_envelope {legacy}/{len(records)}"
    )
    print("note: command *execution* is not graded here -- confirm from trajectory/terminal output")

    print(f"\nPROTOCOL GATE: {'PASS' if all(gates.values()) else 'FAIL'}")
    return 0 if all(gates.values()) else 1


def _has_parser_loop(records: list[dict[str, Any]], threshold: int = 2) -> bool:
    """Two consecutive parser failures is the stop condition, per the run plan."""
    run = 0
    for r in records:
        run = run + 1 if r["parser_error"] else 0
        if run >= threshold:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="dump the exact first request and assert its geometry; no GPU, no endpoint",
    )
    ap.add_argument(
        "--out-dir",
        default="/data/prompt-seed-probe",
        help="where the preflight report is written",
    )
    ap.add_argument(
        "--summarize",
        metavar="PROBE_LOG",
        help="grade a probe JSONL against the primary protocol gate",
    )
    args = ap.parse_args()
    if args.preflight:
        return preflight(Path(args.out_dir))
    if args.summarize:
        return summarize(Path(args.summarize))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
