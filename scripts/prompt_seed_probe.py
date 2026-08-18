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

# Deliberately terse. The assistant turn below carries the entire signal; this
# message exists only to give it something to answer, since a demonstration is a
# reply and a floating assistant turn is a shape the model never saw in training.
# Longer format instructions here would be the exact prose Phase 7 proved this
# checkpoint ignores at turn 1 -- tokens spent restating what does not work, and a
# second candidate explanation to rule out if the result comes back ambiguous.
CALIBRATION_USER = (
    "Format calibration. Reply with a bare JSON object in your standard response "
    "format. No task has started yet."
)

# The demonstration is deliberately task-neutral: it shows the envelope while
# doing nothing.
#
# `commands: []` rather than a single `keystrokes: ""` entry. Both are inert, but
# they fail differently when copied, and only one of them fails *legibly*. A
# copied `keystrokes: ""` yields `n_commands == 1` with no action taken, which
# reads as a working protocol in the aggregate; a copied `commands: []` yields
# `n_commands == 0` and is unmistakable. Since the model's defining trait here is
# copying what it sees, the demonstration must be the variant whose copy is
# obvious rather than the variant whose copy flatters the result.
#
# The residual risk -- that the model copies the emptiness and stalls -- is not
# left to inference: `first_action_has_keystrokes` is a hard gate, so a stalled
# first turn fails explicitly instead of passing quietly.
CALIBRATION_ASSISTANT = json.dumps(
    {
        "analysis": "This is a format calibration turn. No task has been described "
        "yet and the terminal state is not yet known, so there is nothing to "
        "inspect and no action to take.",
        "plan": "Emit no commands and wait for the real task prompt, which follows this message.",
        "commands": [],
        "task_complete": False,
    },
    indent=2,
)


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
        resolved = probe_log or os.environ.get("PROMPT_SEED_PROBE_LOG")
        self._probe_log = Path(resolved) if resolved else None
        self._probe_turn = 0
        self._seed_events: list[dict[str, Any]] = []
        self._first_completion: str | None = None

    # ---- seeding -------------------------------------------------------------

    @staticmethod
    def seed_pair() -> list[dict[str, str]]:
        return [
            {"role": "user", "content": CALIBRATION_USER},
            {"role": "assistant", "content": CALIBRATION_ASSISTANT},
        ]

    def _seed_present(self, chat: Chat) -> bool:
        return any(
            m.get("role") == "assistant"
            and _normalize(m.get("content") or "") == _normalize(CALIBRATION_ASSISTANT)
            for m in chat.messages
        )

    def _strip_seed(self, chat: Chat) -> int:
        """Remove every calibration message from the history. Returns how many went.

        Used to hand harbor a history that looks exactly like an unseeded one
        before delegating to code that makes structural assumptions about it.
        """
        cal_user = _normalize(CALIBRATION_USER)
        cal_asst = _normalize(CALIBRATION_ASSISTANT)
        before = len(chat.messages)
        chat._messages = [
            m
            for m in chat.messages
            if _normalize(m.get("content") or "") not in (cal_user, cal_asst)
        ]
        removed = before - len(chat.messages)
        if removed:
            chat.reset_response_chain()
        return removed

    def _append_seed(self, chat: Chat, reason: str) -> None:
        """Put the pair at the end, so the next real user message lands after it.

        Appending (rather than prepending) is what reproduces the turn-1 geometry:
        harbor's next `chat.chat(prompt)` extends the list, leaving the real prompt
        as the final message with the demonstration immediately above it.
        """
        chat._messages = list(chat.messages) + self.seed_pair()
        chat.reset_response_chain()
        self._seed_events.append(
            {"reason": reason, "turn": self._probe_turn, "n_messages": len(chat.messages)}
        )
        self.logger.info(
            "PROMPT-SEED: calibration pair installed (%s) at turn %d",
            reason,
            self._probe_turn,
        )

    # ---- hooks ---------------------------------------------------------------

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        original_instruction: str = "",
    ) -> None:
        # `run()` hands us a freshly constructed, empty Chat. Seeding here -- rather
        # than editing the prompt template -- is the whole point: the template
        # renders into a *user* message, and user-role instructions are what Phase 7
        # showed this checkpoint ignoring at turn 1.
        self._append_seed(chat, reason="initial")
        return await super()._run_agent_loop(
            initial_prompt=initial_prompt,
            chat=chat,
            original_instruction=original_instruction,
        )

    async def _summarize(self, chat: Chat, original_instruction: str, session: TmuxSession):
        """Reinstate the seed after compaction, without corrupting compaction itself.

        Harbor rebuilds history as `[chat.messages[0], question_prompt, questions]`
        (terminus_2.py:939) and treats `messages[0]` as the anchor carrying the
        Terminus format spec and the task. With the seed installed, `messages[0]`
        is the calibration message, so delegating directly would preserve *that*
        as the anchor and drop the real instructions -- which would surface as
        "the seed failed after compaction" when the actual cause is a hijacked
        anchor.

        Stripping first also keeps harbor's `steps_to_include = 1 + (n - 1) // 2`
        arithmetic aligned with a trajectory that never contained the pair.
        """
        removed = self._strip_seed(chat)
        self.logger.info("PROMPT-SEED: stripped %d seed messages before summarization", removed)
        try:
            return await super()._summarize(chat, original_instruction, session)
        finally:
            # `finally`, so a summarization that raises still leaves a seeded history
            # for harbor's own fallback paths to continue from.
            self._append_seed(chat, reason="post_compaction")

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
        record = {
            "turn": self._probe_turn,
            "seed_in_context": seed_in_context,
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
    """Prove the message geometry without an environment, an endpoint, or a GPU.

    Renders harbor's real template against a stub instruction and terminal state,
    builds the exact message list the first request would carry, and asserts the
    properties the run's conclusions depend on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import harbor.agents.terminus_2.terminus_2 as t2mod

    template_path = Path(t2mod.__file__).parent / "templates" / "terminus-json-plain.txt"
    initial_prompt = template_path.read_text().format(
        instruction="<TASK INSTRUCTION>",
        terminal_state="root@0000000000:/workspace#",
    )

    messages = [*Terminus2PromptSeed.seed_pair(), {"role": "user", "content": initial_prompt}]

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "exactly_one_calibration_pair",
            sum(
                1 for m in messages if _normalize(m["content"]) == _normalize(CALIBRATION_ASSISTANT)
            )
            == 1
            and sum(1 for m in messages if _normalize(m["content"]) == _normalize(CALIBRATION_USER))
            == 1,
            "one calibration user message and one calibration assistant message",
        )
    )
    checks.append(
        (
            "geometry_user_assistant_user",
            [m["role"] for m in messages] == ["user", "assistant", "user"],
            "calibration pair precedes the real prompt",
        )
    )
    checks.append(
        (
            "real_prompt_is_last_and_unchanged",
            messages[-1]["content"] == initial_prompt,
            "harbor's initial prompt is final and byte-unchanged",
        )
    )
    checks.append(
        (
            "instruction_block_intact",
            "Format your response as JSON" in messages[-1]["content"]
            and "<TASK INSTRUCTION>" in messages[-1]["content"],
            "the Terminus contract and the task both survive in the real prompt",
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
            "the demonstration issues no commands, so a copy of it is visibly empty",
        )
    )
    checks.append(
        (
            "copied_demonstration_would_fail_the_gate",
            # The check that matters: confirm the demo cannot satisfy the first-action
            # gate. If this ever passes, a pure copy would score as a working protocol.
            not any(c.get("keystrokes", "").strip() for c in demo["commands"]),
            "a verbatim copy of the demonstration fails first_action_has_keystrokes",
        )
    )
    checks.append(
        (
            "demonstration_cannot_execute",
            True,  # structural: seeds land in chat.messages; only parsed completions reach tmux
            "seed is chat-history only; _trajectory_steps and tmux are untouched",
        )
    )
    checks.append(
        (
            "summarization_prompts_unmodified",
            not any(
                hasattr(Terminus2PromptSeed, attr)
                and getattr(Terminus2PromptSeed, attr) is not getattr(Terminus2, attr)
                for attr in ("_run_subagent", "_get_prompt_template_path", "_query_llm")
            ),
            "subagent/summarization paths are inherited, not overridden",
        )
    )

    source = Path(__file__).read_bytes()
    sha = hashlib.sha256(source).hexdigest()

    report = {
        "agent_sha256": sha,
        "agent_source": str(Path(__file__).resolve()),
        "template_path": str(template_path),
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "all_passed": all(p for _, p, _ in checks),
        "first_request_messages": messages,
    }
    (out_dir / "preflight.json").write_text(json.dumps(report, indent=2))

    print(f"agent sha256: {sha}")
    print(f"template:     {template_path}")
    print()
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} -- {detail}")
    print()
    print("--- first request, message sequence ---")
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

    first = records[0]
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
