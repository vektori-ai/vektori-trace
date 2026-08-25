"""Render Tau2 decisions into supervised rows, and prove they will serve.

Every row is `(prompt state, exactly one teacher decision)`. Loss falls on the
target only; the greeting, user turns, tool results and every earlier assistant
action stay visible and unsupervised.

The tokenization itself is `vektori_trace.dataset.tokenize_messages`, which does
the two-encode construction Qwen3 requires: `render(messages[:-1],
add_generation_prompt=True)` must be an exact token prefix of
`render(messages)`. Per-message masking is not prefix-stable on this template
and overshoots into the next turn.

`max_length` is **not** defaulted here. The pinned context comes from the
measured W30/C30 corpus and must be identical across SFT, continued SFT, ReOPD
and serving. An over-length row is rejected and reported, never truncated and
never swapped for a shorter neighbour by inspecting content.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ..dataset import (
    PrefixInstabilityError,
    NonLastSupervisionError,
    tokenize_messages,
)
from .normalize import Decision


class ToolRoundTripError(ValueError):
    """A rendered tool call does not parse back to the same structured action."""


@dataclass
class Row:
    task_id: str
    position: int
    action_type: str
    tool_names: list[str]
    n_prefix: int
    n_target: int
    n_total: int
    semantic_hash: str
    input_ids: list[int]
    labels: list[int]

    def census(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("input_ids"), d.pop("labels")
        return d


def _canonical(call: dict[str, Any]) -> tuple[str, str]:
    """Structured identity of a tool call: name plus key-sorted arguments.

    Byte comparison would fail on `{"a":1,"b":2}` vs `{"b":2,"a":1}`, which are
    the same action. Compare the structure.
    """
    fn = call.get("function") or call
    return fn.get("name"), json.dumps(fn.get("arguments") or {}, sort_keys=True,
                                      default=str)


def _serving_parser(tokenizer: Any):
    """The exact tool parser the serving path uses, or None if unavailable.

    vLLM is served with `--tool-call-parser hermes` for Qwen3 (see
    `vektori_trace/runtime/serve.py`), and that parser -- not a hand-rolled
    marker scan -- is what decides whether a generated action becomes a tool
    call at inference. Importing it here makes the round trip a serving-parity
    check rather than a structural one.
    """
    try:
        from vllm.entrypoints.openai.tool_parsers import Hermes2ProToolParser
    except Exception:
        return None
    try:
        return Hermes2ProToolParser(tokenizer)
    except Exception:
        return None


def _parse_with_serving_parser(parser: Any, text: str) -> list[tuple[str, str]] | None:
    """Run the serving parser over the decoded target. None if it cannot run."""
    try:
        info = parser.extract_tool_calls(text, request=None)
    except Exception:
        return None
    if not getattr(info, "tools_called", False):
        return []
    out = []
    for tc in (getattr(info, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        raw = getattr(fn, "arguments", None)
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            raise ToolRoundTripError(
                f"serving parser produced non-JSON arguments: {raw!r}"
            )
        out.append((name, json.dumps(args, sort_keys=True, default=str)))
    return out


def assert_tool_round_trip(target: dict[str, Any], tokenizer: Any,
                           rendered_ids: list[int], start: int,
                           parser: Any = None, *, require_parser: bool = True) -> str:
    """Decode the supervised span and prove it parses back to the same action.

    "It tokenizes" does not prove "it will execute". This decodes exactly what
    the model is trained to emit and re-parses it, preferring the serving-time
    parser so the check establishes parity rather than mere well-formedness.

    Called for *every* target, conversational ones included: a plain message
    that the parser reads as a tool call is a serving bug, and only running the
    parser on it can catch that.

    Returns the parser used ("serving" or "structural").
    """
    intended = [_canonical(c) for c in (target.get("tool_calls") or [])]
    text = tokenizer.decode(rendered_ids[start:], skip_special_tokens=False)

    found = _parse_with_serving_parser(parser, text) if parser is not None else None
    used = "serving"

    if found is None:
        if require_parser:
            raise ToolRoundTripError(
                "the serving tool parser is unavailable, so serving parity "
                "cannot be established. Install vLLM in the export environment "
                "or pass require_parser=False to accept the structural check "
                "and record that the corpus was not parity-verified."
            )
        used = "structural"
        found = []
        marker = "<tool_call>"
        idx = text.find(marker)
        while idx != -1:
            end = text.find("</tool_call>", idx)
            if end == -1:
                raise ToolRoundTripError("unterminated <tool_call> in target span")
            blob = text[idx + len(marker):end].strip()
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError as e:
                raise ToolRoundTripError(
                    f"tool call is not valid JSON: {blob!r}") from e
            found.append((parsed.get("name"),
                          json.dumps(parsed.get("arguments") or {}, sort_keys=True,
                                     default=str)))
            idx = text.find(marker, end)

    if found != intended:
        raise ToolRoundTripError(
            f"round trip changed the action ({used} parser).\n"
            f"  intended: {intended}\n  parsed:   {found}"
        )
    if not intended and found:
        raise ToolRoundTripError(
            f"conversational target parsed as a tool call ({used}): {found!r}"
        )
    return used


def build_row(decision: Decision, tokenizer: Any, *, max_length: int,
              system: str, tools: list[dict[str, Any]] | None,
              check_tools: bool = True, parser: Any = None,
              require_parser: bool = True) -> Row | None:
    """One decision -> one tokenized row, or None if it does not fit.

    Returns None only for over-length; every other failure raises, because a
    silent drop is how a corpus loses the rows it most needed.
    """
    messages = [{"role": "system", "content": system}]
    messages.extend(decision.prompt)
    messages.append(decision.target)

    supervise = [False] * (len(messages) - 1) + [True]

    # Every serving-time template setting is named here. Relying on tokenizer
    # defaults is how a corpus ends up rendered differently from the way it is
    # served -- and Qwen3's template has already cost this repo one silent
    # label bug (`docs/SFT-SCRATCH-PLAN.md`). `enable_thinking=True` is the
    # pinned serving mode; it gates the generation prompt, and the wrapper it
    # produces is masked out of the loss by `tokenize_messages`.
    kwargs = {"tools": tools, "enable_thinking": True}

    try:
        ex = tokenize_messages(
            messages, tokenizer, supervise,
            max_length=max_length, truncate=False,
            template_kwargs=kwargs, mask_think_wrapper=True,
        )
    except (PrefixInstabilityError, NonLastSupervisionError):
        raise
    if ex is None:
        return None

    n_target = sum(1 for lab in ex.labels if lab != -100)
    if n_target == 0:
        raise ValueError(
            f"task {decision.task_id} position {decision.position}: zero "
            "supervised tokens after masking"
        )
    start = len(ex.labels) - n_target

    if check_tools:
        # Every target, including conversational ones: the guard against a
        # plain message parsing as a tool call is only reachable if it runs on
        # messages that have no tool calls.
        assert_tool_round_trip(decision.target, tokenizer, ex.input_ids, start,
                               parser=parser, require_parser=require_parser)

    return Row(
        task_id=decision.task_id,
        position=decision.position,
        action_type=decision.action_type,
        tool_names=list(decision.tool_names),
        n_prefix=start,
        n_target=n_target,
        n_total=len(ex.input_ids),
        semantic_hash=decision.semantic_hash(),
        input_ids=ex.input_ids,
        labels=ex.labels,
    )


def census(rows: list[Row], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    """What was actually built. This, not a predicted count, sets the budget."""
    import collections

    def quantiles(vals: list[int]) -> dict[str, int]:
        if not vals:
            return {}
        v = sorted(vals)
        def p(x): return v[min(len(v) - 1, int(len(v) * x))]
        return {"min": v[0], "p50": p(.5), "p90": p(.9), "p95": p(.95),
                "p99": p(.99), "max": v[-1]}

    per_task = collections.Counter(r.task_id for r in rows)
    mass = collections.Counter()
    for r in rows:
        mass[r.task_id] += r.n_target

    return {
        "n_rows": len(rows),
        "n_tasks": len(per_task),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "rows_per_task": quantiles(list(per_task.values())),
        "total_tokens": quantiles([r.n_total for r in rows]),
        "target_tokens": quantiles([r.n_target for r in rows]),
        "prefix_tokens": quantiles([r.n_prefix for r in rows]),
        "supervised_mass_per_task": quantiles(list(mass.values())),
        "action_types": dict(collections.Counter(r.action_type for r in rows)),
        "tool_frequency": dict(collections.Counter(
            n for r in rows for n in r.tool_names).most_common()),
        "over_4096": sum(1 for r in rows if r.n_total > 4096),
        "over_8192": sum(1 for r in rows if r.n_total > 8192),
    }
