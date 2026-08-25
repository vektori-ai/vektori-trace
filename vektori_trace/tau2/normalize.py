"""Semantic normalization of stored Tau2 simulations.

The input is a raw `flash_*.json` simulation record — the complete recording of
one episode, not Tau2's evaluation summary. Every assistant message carries the
provider's original response under `raw_data`, including `reasoning_content`.

Two rules govern this module, both settled 2026-08-24:

**The scripted greeting is not a decision.** Every retail trace opens with an
identical `"Hi! How can I help you today?"` injected by the simulator at message
index 0. Measured over all 78 passing traces: exactly one per trace, always at
index 0, always the first assistant message, always missing `raw_data`, and no
genuine action anywhere shares its text. It is excluded as a *target* and kept
in every downstream *prompt*, because the simulator supplies it at serving time
too. Training on it would teach the model to emit something it never needs to
emit, on 1 row in 6.

**Private reasoning never reaches the model.** `reasoning_content` is DeepSeek's
scratchpad. Putting it in the prompt would train
`p(action | DeepSeek's hidden reasoning)` while serving has no such thing —
privileged-information leakage and a guaranteed train/serve mismatch.
Supervising it is a different experiment (reasoning distillation, not action
distillation). It is preserved verbatim in the audit record and excluded from
both prompt and target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCRIPTED_GREETING = "Hi! How can I help you today?"


class GreetingProvenanceError(ValueError):
    """The scripted-greeting invariants do not hold for this trace."""


class MalformedTraceError(ValueError):
    """A trace could not be normalized without reconstructing meaning."""


@dataclass
class Decision:
    """One genuine teacher assistant message, with its exact prompt state."""

    task_id: str
    position: int          # index among genuine decisions, 0-based
    message_index: int     # index in the original message list
    prompt: list[dict[str, Any]]   # externally visible history, greeting included
    target: dict[str, Any]         # the supervised assistant message
    action_type: str               # message | toolcall | toolcall+text
    tool_names: list[str]
    reasoning_content: str | None  # audit only; never rendered
    raw_data: dict[str, Any] | None

    def semantic_hash(self) -> str:
        payload = json.dumps(
            {"prompt": self.prompt, "target": self.target}, sort_keys=True, default=str
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class NormalizedTrace:
    task_id: str
    trial: int | None
    seed: int | None
    trace_hash: str
    source_file: str
    decisions: list[Decision] = field(default_factory=list)
    n_messages: int = 0


def _visible(msg: dict[str, Any]) -> dict[str, Any]:
    """The externally visible form of a message.

    Drops provider bookkeeping (`raw_data`, `cost`, `usage`, timestamps) and,
    critically, never surfaces `reasoning_content`. Tool calls keep their id,
    name and typed arguments — no re-serialization of meaning.
    """
    role = msg.get("role")
    out: dict[str, Any] = {"role": role, "content": msg.get("content") or ""}

    if role == "assistant" and msg.get("tool_calls"):
        calls = []
        for tc in msg["tool_calls"]:
            name = tc.get("name")
            args = tc.get("arguments")
            if name is None or args is None:
                raise MalformedTraceError(
                    f"tool call missing name/arguments: {tc!r}"
                )
            if not isinstance(args, dict):
                raise MalformedTraceError(
                    f"tool call arguments are {type(args).__name__}, not a dict; "
                    "refusing to parse a string back into structure"
                )
            calls.append({"id": tc.get("id"), "type": "function",
                          "function": {"name": name, "arguments": args}})
        out["tool_calls"] = calls

    if role == "tool":
        out["tool_call_id"] = msg.get("id") or msg.get("tool_call_id")

    return out


def _action_type(msg: dict[str, Any]) -> str:
    has_tc = bool(msg.get("tool_calls"))
    has_text = bool((msg.get("content") or "").strip())
    if has_tc and has_text:
        return "toolcall+text"
    if has_tc:
        return "toolcall"
    return "message"


def assert_greeting_provenance(msgs: list[dict[str, Any]], task_id: str) -> int:
    """Prove the scripted greeting, return its message index.

    Every clause is a measured invariant over the 78 passing retail traces. An
    assistant message missing `raw_data` that is *not* this greeting is a data
    defect and fails the trace rather than being silently dropped.
    """
    assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    no_raw = [i for i in assistant_idx if msgs[i].get("raw_data") is None]

    if len(no_raw) != 1:
        raise GreetingProvenanceError(
            f"task {task_id}: expected exactly one assistant message without "
            f"raw_data, found {len(no_raw)} at {no_raw}. Every other assistant "
            "message must carry provider provenance."
        )
    gi = no_raw[0]
    if gi != 0:
        raise GreetingProvenanceError(
            f"task {task_id}: scripted greeting at message index {gi}, not 0"
        )
    if not assistant_idx or gi != assistant_idx[0]:
        raise GreetingProvenanceError(
            f"task {task_id}: greeting is not the first assistant message"
        )
    text = (msgs[gi].get("content") or "").strip()
    if text != SCRIPTED_GREETING:
        raise GreetingProvenanceError(
            f"task {task_id}: greeting text {text!r} != {SCRIPTED_GREETING!r}"
        )
    if msgs[gi].get("tool_calls"):
        raise GreetingProvenanceError(f"task {task_id}: greeting carries tool calls")

    for i in assistant_idx:
        if i == gi:
            continue
        if (msgs[i].get("content") or "").strip() == SCRIPTED_GREETING:
            raise GreetingProvenanceError(
                f"task {task_id}: genuine action at {i} collides with greeting text"
            )
    return gi


def normalize_trace(sim: dict[str, Any], source_file: str) -> NormalizedTrace:
    """Raw simulation -> every genuine decision with its exact prompt state.

    The prompt for decision i is all externally visible messages before it,
    greeting included. No truncation, no reconstruction, no private reasoning.
    """
    task_id = str(sim.get("task_id"))
    msgs = sim.get("messages") or []
    if not msgs:
        raise MalformedTraceError(f"task {task_id}: no messages")

    greeting_idx = assert_greeting_provenance(msgs, task_id)

    payload = json.dumps(msgs, sort_keys=True, default=str).encode()
    trace = NormalizedTrace(
        task_id=task_id,
        trial=sim.get("trial"),
        seed=sim.get("seed"),
        trace_hash=hashlib.sha256(payload).hexdigest()[:16],
        source_file=source_file,
        n_messages=len(msgs),
    )

    history: list[dict[str, Any]] = []
    position = 0
    for i, m in enumerate(msgs):
        vis = _visible(m)
        is_genuine_action = (
            m.get("role") == "assistant" and i != greeting_idx
        )
        if is_genuine_action:
            raw = m.get("raw_data") or {}
            reasoning = ((raw.get("message") or {}).get("reasoning_content")
                         if isinstance(raw, dict) else None)
            trace.decisions.append(Decision(
                task_id=task_id,
                position=position,
                message_index=i,
                prompt=[dict(h) for h in history],
                target=vis,
                action_type=_action_type(m),
                tool_names=[c["function"]["name"] for c in vis.get("tool_calls", [])],
                reasoning_content=reasoning,
                raw_data=raw or None,
            ))
            position += 1
        history.append(vis)

    if not trace.decisions:
        raise MalformedTraceError(f"task {task_id}: no genuine decisions")
    return trace


def select_trace(traces: list[NormalizedTrace]) -> NormalizedTrace:
    """One trace per task, deterministically: lowest trial, seed, then hash."""
    return sorted(traces, key=lambda t: (
        t.trial if t.trial is not None else 1 << 30,
        t.seed if t.seed is not None else 1 << 30,
        t.trace_hash,
    ))[0]
