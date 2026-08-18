"""Phase 7 gates: does a checkpoint emit Terminus's native protocol unaided?

Every number from the corrective SFT run is teacher-forced. Teacher forcing
hands the model a correct prefix at every position, which is exactly the crutch
that is absent at generation time — and generation time is where v1 failed, 305
consecutive `Missing required fields`. These gates are the first measurement
that removes the crutch.

The authority on "is this a valid action" is **harbor's own parser**, not a
schema written here. A gate that disagrees with the thing that will actually
run in a rollout is measuring the wrong object, so `harbor_accepts` delegates
to `TerminusJSONPlainParser` and the rest of the gates add the strictness that
the parser deliberately does not enforce.

That leniency is the reason there are two format tiers rather than one:

    harbor_accepts  — the parser returns no error. The rollout would proceed.
    native_json     — additionally, the completion *is* the JSON object: no
                      ```json fence, no <think> block, no prose preamble.

The parser accepts a fenced object with only a warning ("Extra text detected
before JSON object"). Reporting that as a pass would hide a checkpoint that is
*nearly* repaired but still wrapping its output, and `enable_thinking=False`
would be unverified. Reporting it as a total failure would be equally wrong —
it runs. So both are recorded and a checkpoint is selected on the strict one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Operation classes, kept identical to the dataset audit's
# (`scripts/sft_repair_dataset.py`) so "emitted an edit" means the same thing
# in the gate report as it does in the corpus the model trained on.
READ_RE = re.compile(
    r"^\s*(ls|cat|sed|grep|head|tail|find|rg|less|wc|git status|git diff|git log)\b"
)
EDIT_RE = re.compile(
    r"^\s*(patch|git apply|python -c|cat\s*>|tee|sed -i|>\s*\S|>>)|<<\s*['\"]?EOF"
)
TEST_RE = re.compile(r"^\s*(pytest|python -m pytest|tox|nox|make test|python -m unittest)\b")

CLONE_RE = re.compile(r"\bgit\s+clone\b")

# The v1 envelope. A *substring* check on the raw completion rather than a check
# on the parsed object, because the whole point is to catch text the parser threw
# away.
#
# `<think>` used to live in this tuple, from when serving pinned
# `enable_thinking=False` and a reasoning block meant the template had been
# bypassed. Thinking is on now (`docs/SFT-SCRATCH-PLAN.md` step 2), so a correct
# completion *opens with* the wrapper and ORing the two would report "still
# emitting the tool_call envelope" for output that had dropped the tags — the
# measurement bug prompt-seed run 3 hit. One flag per failure mode.
LEGACY_MARKERS = ("<tool_call>", "</tool_call>")

#: Emitted before the action when the model uses its reasoning channel. Not a
#: format failure — harbor extracts commands from `<think>…</think>{json}` with
#: only an "Extra text detected before JSON object" warning.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# Top-level keys terminus defines. Anything else is the model inventing
# protocol, which is how a silently-ignored field becomes a silently-dropped
# action.
ALLOWED_TOP_LEVEL = frozenset({"analysis", "plan", "commands", "task_complete"})
REQUIRED_TOP_LEVEL = ("analysis", "plan", "commands")

# Per-command keys terminus's ParsedCommand actually consumes. `is_blocking`
# and `timeout_sec` are the two most common inventions — harbor downgrades them
# to a warning and drops them, so a model that relies on them is issuing
# commands whose semantics it does not get.
ALLOWED_COMMAND_KEYS = frozenset({"keystrokes", "duration"})

# A batch that repeats one keystroke this many times is looping, not acting.
REPETITION_LIMIT = 3


@dataclass
class GateResult:
    """One completion, graded. `gates` holds only the gates that applied."""

    prefix_id: str
    checkpoint: str
    category: str
    suite: str
    completion: str
    finish_reason: str | None = None
    parser_error: str | None = None
    parser_warning: str | None = None
    n_commands: int = 0
    first_command: str | None = None
    #: Text between `<think>` and `</think>`, "" when the model emitted none.
    #: Reported, never gated — see `grade`.
    think_body: str = ""
    gates: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    @property
    def failed_gates(self) -> list[str]:
        return sorted(k for k, v in self.gates.items() if not v)


def _parse_with_harbor(text: str):
    """harbor's parser, imported lazily so the gates import without harbor."""
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    return TerminusJSONPlainParser().parse_response(text)


def _raw_object(text: str) -> dict[str, Any] | None:
    """The completion parsed as JSON *without* harbor's salvage.

    Harbor will dig an object out of surrounding prose. The strict gates need
    to know whether it had to.
    """
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _keystrokes(obj: dict[str, Any] | None) -> list[str]:
    if not obj:
        return []
    return [
        c.get("keystrokes", "")
        for c in (obj.get("commands") or [])
        if isinstance(c, dict)
    ]


def _strip_think(completion: str) -> str:
    """The completion with one leading `<think>…</think>` block removed.

    Only a *leading* block, and only the first: a `</think>` appearing later is
    the model emitting the literal string inside its action, which is content,
    not a channel boundary.
    """
    text = completion.lstrip()
    if not text.startswith(THINK_OPEN):
        return completion
    end = text.find(THINK_CLOSE)
    if end == -1:
        return completion
    return text[end + len(THINK_CLOSE):].lstrip()


def _think_body(completion: str) -> str:
    """What the model put between `<think>` and `</think>`; "" if nothing."""
    text = completion.lstrip()
    if not text.startswith(THINK_OPEN):
        return ""
    end = text.find(THINK_CLOSE)
    if end == -1:
        return text[len(THINK_OPEN):].strip()
    return text[len(THINK_OPEN):end].strip()


def _first_line(keystrokes: str) -> str:
    stripped = keystrokes.strip()
    return stripped.splitlines()[0][:80] if stripped else ""


def grade(
    completion: str,
    *,
    prefix_id: str,
    checkpoint: str,
    category: str,
    suite: str,
    finish_reason: str | None = None,
    git_present: bool = True,
) -> GateResult:
    """Grade one completion against the format gates plus this category's.

    `git_present` is a property of the prefix, not the response: the "don't
    clone" gate only means something in a conversation whose terminal output
    has already shown the repository is there.
    """
    res = GateResult(
        prefix_id=prefix_id,
        checkpoint=checkpoint,
        category=category,
        suite=suite,
        completion=completion,
        finish_reason=finish_reason,
    )

    parsed = _parse_with_harbor(completion)
    res.parser_error = parsed.error or None
    res.parser_warning = parsed.warning or None
    harbor_ok = not parsed.error

    salvaged = _raw_object(completion)
    if salvaged is None and harbor_ok:
        # harbor found an object the bare loader could not: fenced, prefixed, or
        # trailed. Grade the object harbor recovered, but the strict gate below
        # has already been decided by this branch.
        salvaged = {
            "analysis": parsed.analysis,
            "plan": parsed.plan,
            "commands": [
                {"keystrokes": c.keystrokes, "duration": c.duration}
                for c in (parsed.commands or [])
            ],
            "task_complete": bool(parsed.is_task_complete),
        }
        res.notes.append("object recovered by harbor's salvage, not a bare parse")

    ks = _keystrokes(salvaged)
    res.n_commands = len(ks)
    res.first_command = _first_line(ks[0]) if ks else None

    # ---- format tier: applies to every prefix ---------------------------
    res.gates["harbor_accepts"] = harbor_ok

    # `native_json` asks "is the completion *only* the JSON object" — no fence,
    # no prose preamble. A reasoning block is not prose the model invented, it is
    # the channel the template gave it, so it is removed before the question is
    # asked. What remains must still be a bare object.
    body = _strip_think(completion)
    res.gates["native_json"] = (
        harbor_ok
        and _raw_object(body) is not None
        and not any(m in completion for m in LEGACY_MARKERS)
    )

    res.gates["no_legacy_envelope"] = not any(m in completion for m in LEGACY_MARKERS)

    # Recorded, never gating. Stage A supervises the action, not the reasoning;
    # think *length* belongs to the instruction-tuned prior. A run where this is
    # 0 everywhere means the wrapper mask failed and the model was trained to
    # close its reasoning block immediately — a dataset bug to go fix, not a
    # reason to reject a checkpoint whose format is correct.
    res.think_body = _think_body(completion)

    obj = _raw_object(completion) or (salvaged if harbor_ok else None)
    res.gates["required_fields"] = bool(obj) and all(
        k in obj for k in REQUIRED_TOP_LEVEL
    )

    res.gates["no_invented_fields"] = bool(obj) and not (
        set(obj) - ALLOWED_TOP_LEVEL
    )

    cmds = (obj or {}).get("commands")
    res.gates["command_structure"] = (
        isinstance(cmds, list)
        and all(
            isinstance(c, dict)
            and isinstance(c.get("keystrokes"), str)
            and not (set(c) - ALLOWED_COMMAND_KEYS)
            for c in cmds
        )
    )

    # `length` means the model never chose to stop. A completion truncated at
    # the token cap may be valid JSON only by accident of where it was cut.
    res.gates["eos_before_limit"] = finish_reason != "length"

    counts: dict[str, int] = {}
    for k in ks:
        counts[k] = counts.get(k, 0) + 1
    res.gates["non_repetition"] = all(v < REPETITION_LIMIT for v in counts.values())

    # ---- behavior tier: only the gates this prefix's state can test ------
    joined = "\n".join(ks)

    if category in ("orientation", "repo_present", "post_compaction"):
        # v1 is not empty — it opens with `ls -la` 114x and `git status` 23x.
        # Orientation is therefore a behavior the repair must *preserve*, and a
        # checkpoint that lost it while gaining the envelope is a regression.
        res.gates["orientation"] = bool(ks) and any(
            READ_RE.search(_first_line(k)) for k in ks[:2]
        )

    if git_present:
        res.gates["no_clone_when_git_exists"] = not CLONE_RE.search(joined)

    if category == "first_edit":
        res.gates["edit_emission"] = bool(EDIT_RE.search(joined))

    if category == "test_exec":
        res.gates["test_emission"] = bool(TEST_RE.search(joined))

    if category == "parse_error_recovery":
        # The whole content of "recovery" is producing an action the parser
        # accepts, at the exact turn where the teacher did not.
        res.gates["recovery"] = harbor_ok and bool(ks)

    return res


def summarize(results: Iterable[GateResult]) -> dict[str, Any]:
    """Aggregate by checkpoint × suite, and by gate, for selection.

    Selection needs per-cell numbers, not one rate: the acquisition suite and
    the held-out suite answer different questions, and the held-out suite is
    only interpretable next to the trained-task-failed-rollout control that
    shares its rollout-outcome confound.
    """
    rows = list(results)
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        cell = cells.setdefault(
            (r.checkpoint, r.suite),
            {"n": 0, "passed": 0, "gates": {}, "categories": {}},
        )
        cell["n"] += 1
        cell["passed"] += int(r.passed)
        for gate, ok in r.gates.items():
            g = cell["gates"].setdefault(gate, {"n": 0, "passed": 0})
            g["n"] += 1
            g["passed"] += int(ok)
        c = cell["categories"].setdefault(r.category, {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += int(r.passed)

    return {
        "cells": {f"{ck}|{suite}": v for (ck, suite), v in sorted(cells.items())},
        "n_results": len(rows),
    }


# Selection reads these and only these. The behavior gates are reported at 1-3
# samples per category, which is enough to notice a regression and nowhere near
# enough to select on; treating an anecdote as a gate would reject a working
# checkpoint on one prefix. The plan originally listed edit/orientation/no-clone
# among the selection gates — this narrows that deliberately, because the repair
# being measured is the envelope.
#: `harbor_accepts` and not `native_json`: the parser that will actually run the
#: rollout is the authority on whether an action is valid, and with thinking on a
#: correct completion legitimately carries text before the object. `native_json`
#: stays in the report as the stricter reading. On the 168 graded completions of
#: the previous sweep the two never disagreed, so this loosening is honest about
#: what gates rather than a claim that it buys headroom.
#:
#: Nothing else gates. `think_body` in particular is reported, not required:
#: Stage A was never asked to teach reasoning *content*.
SELECTION_GATES = ("harbor_accepts", "required_fields", "no_legacy_envelope")

# The suite selection is read from. Acquisition prefixes are training inputs: a
# checkpoint can reproduce a memorised continuation there, so a pass proves
# acquisition, not protocol.
SELECTION_SUITE = "generalization"


def clears(
    results: Iterable[GateResult],
    *,
    checkpoint: str,
    suite: str,
    require: Iterable[str] = SELECTION_GATES,
    expected_prefix_ids: Iterable[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Did `checkpoint` clear `require` on every prefix of `suite`?

    `expected_prefix_ids` is not optional in practice. A request the endpoint
    dropped produces no `GateResult` at all, so a checkpoint that answered 7 of
    8 prefixes perfectly and lost the eighth to an HTTP error would otherwise
    read as a clean sweep — silence scoring as success. Passing the manifest's
    prefix ids makes an ungraded prefix block the checkpoint, which is the same
    correction `no_gradeable_rollouts` needed in the pass@k reports.
    """
    require = tuple(require)
    group = [r for r in results if r.suite == suite and r.checkpoint == checkpoint]
    graded = {r.prefix_id for r in group}

    detail: dict[str, Any] = {
        "n_graded": len(group),
        "failures": {g: sum(1 for r in group if r.gates.get(g) is False) for g in require},
    }
    if expected_prefix_ids is not None:
        missing = sorted(set(expected_prefix_ids) - graded)
        detail["n_expected"] = len(set(expected_prefix_ids))
        detail["ungraded"] = missing
        if missing:
            detail["passed"] = False
            return False, detail
    if not group:
        detail["passed"] = False
        detail["status"] = "no results"
        return False, detail

    ok = not any(detail["failures"].values())
    detail["passed"] = ok
    return ok, detail


def select_checkpoint(
    results: Iterable[GateResult],
    *,
    order: list[str],
    suite: str = SELECTION_SUITE,
    require: Iterable[str] = SELECTION_GATES,
    expected_prefix_ids: Iterable[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """The earliest checkpoint that clears `require` on every prefix of `suite`.

    Earliest, not best and not lowest-loss. Loss fell monotonically across all
    63 steps, so a loss rule always returns the final checkpoint — the one most
    overfit to 34 tasks — and the checkpoints exist precisely so that choice can
    be made on behavior instead.
    """
    rows = list(results)
    expected = list(expected_prefix_ids) if expected_prefix_ids is not None else None
    trace: dict[str, Any] = {}
    chosen: str | None = None
    for ck in order:
        ok, detail = clears(
            rows,
            checkpoint=ck,
            suite=suite,
            require=require,
            expected_prefix_ids=expected,
        )
        if detail["n_graded"] == 0 and expected is None:
            trace[ck] = {"status": "no results"}
            continue
        trace[ck] = detail
        if ok and chosen is None:
            chosen = ck
    return chosen, trace
