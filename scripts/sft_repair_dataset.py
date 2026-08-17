"""Rebuild the v1 SFT set in Terminus's *native* protocol.

v1 learned the right objective in the wrong envelope. It was trained on
OpenAI-style `tool_calls` — which Qwen3's template renders as
`<tool_call>{...}</tool_call>` — with `role="tool"` observations. terminus-2
wants literal flat JSON in assistant *text* and sends observations as
`role="user"` (`terminus_2.py:1537`). Hence 305 consecutive
`Missing required fields`. `docs/SFT-REPAIR-PLAN.md` Phase 2.

Two representations of every teacher action exist, and neither alone is enough:

  * The **raw capture** (`<run>/captures/token_captures.jsonl`) holds the exact
    JSON DeepSeek emitted. It is one global file per run, written by
    `CaptureProxy` across concurrent rollouts, so it carries no rollout identity
    and has to be joined on content.
  * The **trajectory** holds rollout structure and the parsed commands, but not
    the raw string: `save_raw_content_in_trajectory` was off, so terminus wrote
    `message = f"Analysis: {analysis}\nPlan: {plan}"` (`terminus_2.py:1345`)
    plus structured tool calls.

That render is a *semantically* lossless decomposition, so this script is
hybrid: rebuild every action from the trajectory, and swap in the raw bytes
wherever a capture joins and verifies. Nothing is truncated for a failed join —
the rebuild covers it, and the join doubles as the proof that rebuilding is
sound.

    python scripts/sft_repair_dataset.py \
        --run /data/vektori-out/dsv4-corpus60 \
        --run /data/vektori-out/dsv4-corpus60-b \
        --out-dir /data/sft-repaired
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sft_export_traces import BOUNDARY_MESSAGE, passing_rollouts

# terminus clamps every command's duration before recording it
# (`Command(duration_sec=min(parsed_cmd.duration, 60))`, terminus_2.py:1180).
# A raw response asking for 120 is recorded as 60, so the join compares clamped
# against clamped or every long-duration command reads as a mismatch.
DURATION_CLAMP = 60.0

# The parser's default when a command omits `duration` (ParsedCommand default).
DEFAULT_DURATION = 1.0

# Assistant turn classes. Only `action` is supervised; `unknown` fails the audit.
ACTION = "action"
HANDOFF_QUESTION = "handoff_question"
SUMMARIZATION = "summarization"
PARSE_ERROR = "parse_error"
UNKNOWN = "unknown"

# The two prompts that mark a non-action exchange. Both are terminus literals
# (terminus_2.py:807, :845, :913) — matched on their stable openings because the
# task text is interpolated into the middle of each.
HANDOFF_QUESTION_PROMPT = "You are picking up work from a previous AI agent on this task:"
SUMMARY_REQUEST_PROMPT = "You are about to hand off your work to another AI agent."
ANSWER_REQUEST_PROMPT = "The next agent has a few questions for you"


# --------------------------------------------------------------------------
# The action record
# --------------------------------------------------------------------------


@dataclass
class Action:
    """One parent-agent action, in Terminus's own vocabulary."""

    analysis: str
    plan: str
    commands: list[dict[str, Any]]  # {"keystrokes": str, "duration": float}
    task_complete: bool

    def signature(self) -> tuple:
        """The part of an action two representations must agree on.

        Durations are clamped on both sides (see DURATION_CLAMP) so the
        comparison is between what terminus *recorded* and what it *would have*
        recorded, not between a request and its clamped execution.
        """
        return (
            tuple(
                (c["keystrokes"], min(float(c.get("duration", DEFAULT_DURATION)), DURATION_CLAMP))
                for c in self.commands
            ),
            self.task_complete,
        )

    def rendered_message(self) -> str:
        """Invert terminus_2.py:1338-1348 — the `message` this action produced.

        Inverting is used rather than splitting ATIF's `message` because
        splitting is ambiguous for any analysis that itself contains
        `"\\nPlan: "`, and inverting never is. This string is the join key's
        first component.
        """
        parts = []
        if self.analysis:
            parts.append(f"Analysis: {self.analysis}")
        if self.plan:
            parts.append(f"Plan: {self.plan}")
        return "\n".join(parts)

    def join_key(self) -> str:
        """Full semantic tuple: rendered analysis/plan + commands + completion.

        A full analysis/plan string alone removes the *known* 300-char collision
        but does not make collisions impossible — two responses can share both
        and differ in their commands.
        """
        payload = json.dumps(
            [self.rendered_message(), self.signature()], default=list, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Canonical Terminus JSON, for actions with no verified raw capture.

        `task_complete` is emitted only when true, matching how the teacher
        writes it and how the parser reads it (it defaults to false when absent,
        `terminus_json_plain_parser.py`). Two-space indent is the shape the
        system prompt itself demonstrates.
        """
        obj: dict[str, Any] = {
            "analysis": self.analysis,
            "plan": self.plan,
            "commands": [
                {"keystrokes": c["keystrokes"], "duration": c["duration"]}
                for c in self.commands
            ],
        }
        if self.task_complete:
            obj["task_complete"] = True
        return json.dumps(obj, indent=2, ensure_ascii=False)


def action_from_step(step: dict) -> tuple[Action | None, str | None]:
    """Rebuild an Action from a trajectory step. Returns (action, problem)."""
    message = step.get("message") or ""
    analysis, plan, split_problem = split_rendered_message(message)

    commands: list[dict[str, Any]] = []
    task_complete = False
    for tc in step.get("tool_calls") or []:
        name = tc.get("function_name")
        args = tc.get("arguments") or {}
        if name == "bash_command":
            keystrokes = args.get("keystrokes")
            if not isinstance(keystrokes, str):
                return None, f"bash_command with no keystrokes: {args!r}"
            commands.append(
                {
                    "keystrokes": keystrokes,
                    "duration": float(args.get("duration", DEFAULT_DURATION)),
                }
            )
        elif name == "mark_task_complete":
            # Never a `commands[]` entry — it is a top-level flag in the schema.
            task_complete = True
        else:
            return None, f"unknown tool verb {name!r}"

    return Action(analysis, plan, commands, task_complete), split_problem


def split_rendered_message(message: str) -> tuple[str, str, str | None]:
    """Undo `"Analysis: {a}\\nPlan: {p}"`. Third element flags ambiguity.

    terminus joins the two parts with a newline and omits either if it is empty.
    Splitting on the first `"\\nPlan: "` is correct unless the analysis itself
    contains that sequence, which is recorded rather than guessed at: the join
    against the raw capture is what settles it for real.
    """
    if not message:
        return "", "", None
    ambiguous = None
    n_markers = message.count("\nPlan: ")
    if n_markers > 1:
        ambiguous = f"{n_markers} occurrences of the '\\nPlan: ' marker"

    if message.startswith("Analysis: "):
        body = message[len("Analysis: ") :]
        head, sep, tail = body.partition("\nPlan: ")
        return (head, tail, ambiguous) if sep else (head, "", ambiguous)
    if message.startswith("Plan: "):
        return "", message[len("Plan: ") :], ambiguous
    return message, "", "message has neither 'Analysis: ' nor 'Plan: ' prefix"


# --------------------------------------------------------------------------
# Raw captures
# --------------------------------------------------------------------------


def iter_capture_texts(path: Path):
    """Yield each capture's `text` without parsing the whole line.

    A capture line is ~620 KB (logprob_detail carries top-5 per token) and the
    two files total ~6.2 GB, so `json.loads` per line is wasteful when one
    string is wanted. `"text"` is a plain JSON string value: find it and decode
    only that span.
    """
    key = '"text": "'
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find(key)
            if i < 0:
                continue
            j = i + len(key) - 1  # the opening quote
            k = j + 1
            close = -1
            while True:
                close = line.find('"', k)
                if close < 0:
                    break
                backslashes = 0
                p = close - 1
                while p >= 0 and line[p] == "\\":
                    backslashes += 1
                    p -= 1
                if backslashes % 2 == 0:
                    break
                k = close + 1
            if close < 0:
                continue
            try:
                yield json.loads(line[j : close + 1])
            except json.JSONDecodeError:
                continue


def is_bare_json(text: str) -> bool:
    """True when the whole string is one JSON object and nothing else.

    This is stricter than Harbor's parser on purpose. The parser's job at
    runtime was to salvage whatever the teacher sent; ours is to decide what the
    student should learn to send.
    """
    try:
        return isinstance(json.loads(text), dict)
    except (json.JSONDecodeError, TypeError):
        return False


def parse_raw(text: str) -> Action | None:
    """Parse a raw teacher completion with Harbor's own Terminus parser.

    Using the real parser rather than `json.loads` matters: it is what decided
    at runtime whether a response was an action at all, including its
    auto-corrections. A response this rejects was never an action.
    """
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    result = TerminusJSONPlainParser().parse_response(text)
    if result.error:
        return None
    return Action(
        analysis=result.analysis or "",
        plan=result.plan or "",
        commands=[
            {"keystrokes": c.keystrokes, "duration": float(c.duration)}
            for c in result.commands
        ],
        task_complete=bool(result.is_task_complete),
    )


def build_capture_index(paths: list[Path]) -> tuple[dict[str, str], dict]:
    """Map join key -> raw text, over every capture in every run.

    Collisions are only a problem when they disagree. Two captures with the same
    key and byte-identical text are the same action sampled twice; two with the
    same key and different text mean the key is not identifying an action, which
    fails loudly rather than picking one.
    """
    index: dict[str, str] = {}
    stats = {"scanned": 0, "parsed": 0, "benign_duplicates": 0, "conflicts": []}
    for path in paths:
        n = 0
        for text in iter_capture_texts(path):
            n += 1
            action = parse_raw(text)
            if action is None:
                continue
            stats["parsed"] += 1
            key = action.join_key()
            prior = index.get(key)
            if prior is None:
                index[key] = text
            elif prior == text:
                stats["benign_duplicates"] += 1
            else:
                stats["conflicts"].append(
                    {"key": key, "a": prior[:300], "b": text[:300]}
                )
        stats["scanned"] += n
        print(f"  {path}: {n} captures scanned", flush=True)
    stats["index_size"] = len(index)
    return index, stats


# --------------------------------------------------------------------------
# Segment assembly
# --------------------------------------------------------------------------


@dataclass
class Segment:
    task: str
    jobs_dir: str
    rollout_index: int | None
    segment_index: int
    messages: list[dict[str, str]] = field(default_factory=list)
    supervise: list[bool] = field(default_factory=list)
    turn_meta: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, role: str, content: str, *, supervise: bool, meta: dict) -> None:
        self.messages.append({"role": role, "content": content})
        self.supervise.append(supervise)
        self.turn_meta.append(meta)


def classify(step: dict, prev_user_text: str) -> str:
    """What kind of assistant turn is this?

    The prompt that elicited it is the discriminator, not the response: a
    handoff question and an action are both free text from the model's side, but
    only one of them was asked for under the Terminus action instructions.
    """
    if prev_user_text.startswith(HANDOFF_QUESTION_PROMPT):
        return HANDOFF_QUESTION
    if prev_user_text.startswith(SUMMARY_REQUEST_PROMPT) or prev_user_text.startswith(
        ANSWER_REQUEST_PROMPT
    ):
        return SUMMARIZATION
    if step.get("tool_calls"):
        return ACTION
    # No tool calls under an action prompt means the parser rejected the
    # response and terminus recorded the raw text instead (terminus_2.py:1355-
    # 1370). The teacher's own protocol failures: prose with a ```bash fence, or
    # JSON cut off at the output limit. v1 trained on these as targets, which is
    # 18 demonstrations of the exact malformed output this run repairs.
    if parse_raw(step.get("message") or "") is None:
        return PARSE_ERROR
    return UNKNOWN


def parse_error_prompt(message: str) -> str | None:
    """Re-derive the reply terminus sent after a rejected response.

    The error path never writes a user step, so this prompt is absent from the
    trajectory — but it is a deterministic function of the raw response
    (terminus_2.py:1355-1360, with `_get_error_response_type()` returning
    "JSON response" for the json parser). Recomputing it from the same input
    with the same formatting is reconstruction, not fabrication; inventing a
    plausible-sounding prompt would be the latter.
    """
    from harbor.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )

    result = TerminusJSONPlainParser().parse_response(message)
    if not result.error:
        return None
    feedback = f"ERROR: {result.error}"
    if result.warning:
        feedback += f"\nWARNINGS: {result.warning}"
    return (
        f"Previous response had parsing errors:\n{feedback}\n\n"
        "Please fix these issues and provide a proper JSON response."
    )


def observation_text(step: dict) -> str | None:
    """The next `role="user"` message terminus sends: `prompt = observation`.

    In the ordinary path terminus appends exactly one ObservationResult holding
    the whole observation string (terminus_2.py:1455-1487, :1540), so more than
    one result means this step is not shaped the way the loop's own reply is.
    """
    observation = step.get("observation") or {}
    results = observation.get("results") or []
    if len(results) != 1:
        return None
    return results[0].get("content")


def load_steps(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("steps") or []


def handoff_head_steps(agent_dir: Path, boundary_ordinal: int) -> list[dict]:
    """The opening of the conversation *after* compaction number N.

    The main trajectory holds only the tail of a handoff — the step carrying the
    previous agent's answers. The head lives in
    `trajectory.summarization-N-questions.json`, which is why reading only
    `agent/trajectory.json` yields a conversation with no beginning.
    """
    path = agent_dir / f"trajectory.summarization-{boundary_ordinal}-questions.json"
    if not path.exists():
        return []
    return load_steps(path)


def split_on_compaction(steps: list[dict], agent_dir: Path) -> list[list[dict]]:
    """One list of steps per linear conversation the model actually saw."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    boundary_ordinal = 0
    for step in steps:
        message = step.get("message") or ""
        if step.get("source") == "system" and message.strip() == BOUNDARY_MESSAGE:
            segments.append(current)
            boundary_ordinal += 1
            current = list(handoff_head_steps(agent_dir, boundary_ordinal))
            continue
        if step.get("source") == "system":
            continue
        current.append(step)
    segments.append(current)
    return [s for s in segments if s]


def build_segment(
    steps: list[dict],
    index: dict[str, str],
    *,
    task: str,
    jobs_dir: Path,
    rollout_index: int | None,
    segment_index: int,
) -> Segment:
    seg = Segment(task, str(jobs_dir), rollout_index, segment_index)
    prev_user_text = ""

    for step in steps:
        source = step.get("source")
        message = step.get("message") or ""

        if source == "user":
            seg.add("user", message, supervise=False, meta={"kind": "prompt"})
            prev_user_text = message
            continue
        if source != "agent":
            seg.problems.append(f"unexpected step source {source!r}")
            continue

        kind = classify(step, prev_user_text)
        if kind == UNKNOWN:
            # Fails the audit rather than being guessed at, per the plan.
            seg.problems.append(f"unclassified assistant turn at step {step.get('step_id')}")
            seg.add("assistant", message, supervise=False, meta={"kind": UNKNOWN})
            prev_user_text = ""
            continue

        if kind == PARSE_ERROR:
            # Kept as context, never a target. Dropping it would leave the next
            # action's analysis ("the previous command batch was not executed
            # due to invalid JSON in my response") referring to an event that is
            # no longer there — and the recovery that follows is one of the
            # behaviours Phase 7 tests for.
            seg.add("assistant", message, supervise=False, meta={"kind": PARSE_ERROR})
            reply = parse_error_prompt(message)
            if reply is not None:
                seg.add("user", reply, supervise=False, meta={"kind": "parse_error_reply"})
                prev_user_text = reply
            else:
                prev_user_text = ""
            continue

        if kind != ACTION:
            # Context, never a target: the model already writes prose, and this
            # run's whole objective is action serialization.
            seg.add("assistant", message, supervise=False, meta={"kind": kind})
            prev_user_text = ""
            continue

        action, problem = action_from_step(step)
        if action is None:
            seg.problems.append(f"step {step.get('step_id')}: {problem}")
            break  # truncate: an action we cannot represent at all
        if problem:
            seg.problems.append(f"step {step.get('step_id')}: {problem}")

        raw = index.get(action.join_key())
        source_kind = "atif_rebuilt"
        content = action.to_json()
        equivalent = None
        if raw is not None:
            parsed = parse_raw(raw)
            # Harbor's parser tolerates text around the object — it warns
            # "Extra text detected before JSON object" and proceeds — so
            # parsing is not the same as being a clean target. A raw response
            # wrapped in prose is very nearly the failure this run repairs, and
            # it must not become a target just because the runtime accepted it.
            # The canonical rebuild is the right target for those.
            if parsed is not None and not is_bare_json(raw):
                seg.problems.append(
                    f"step {step.get('step_id')}: raw capture is not bare JSON, rebuilding"
                )
                parsed = None
            if parsed is not None and parsed.signature() == action.signature():
                content = raw
                source_kind = "raw_capture"
                # The rebuild-justification test: on every joined turn the two
                # representations must agree on every field, not just the ones
                # the key covered.
                equivalent = (
                    parsed.analysis == action.analysis and parsed.plan == action.plan
                )
                if not equivalent:
                    seg.problems.append(
                        f"step {step.get('step_id')}: raw/rebuild analysis-plan mismatch"
                    )

        seg.add(
            "assistant",
            content,
            supervise=True,
            meta={
                "kind": ACTION,
                "target_source": source_kind,
                "equivalent": equivalent,
                "n_commands": len(action.commands),
                "task_complete": action.task_complete,
                "step_id": step.get("step_id"),
            },
        )

        obs = observation_text(step)
        if obs is None:
            # No observation means nothing followed this action in the real
            # conversation; ending here is correct, and leaves no user message
            # answering a command that was dropped.
            break
        seg.add("user", obs, supervise=False, meta={"kind": "observation"})
        prev_user_text = obs

    # An observation is only meaningful as the reply to the action before it.
    while seg.messages and seg.messages[-1]["role"] == "user":
        seg.messages.pop()
        seg.supervise.pop()
        seg.turn_meta.pop()

    return seg


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

READ_RE = re.compile(r"^\s*(ls|cat|sed|grep|head|tail|find|rg|less|wc|git status|git diff|git log)\b")
EDIT_RE = re.compile(r"^\s*(patch|git apply|python -c|cat\s*>|tee|sed -i|>\s*\S|>>)|<<\s*['\"]?EOF")
TEST_RE = re.compile(r"^\s*(pytest|python -m pytest|tox|nox|make test|python -m unittest)\b")


def audit(segments: list[Segment], capture_stats: dict) -> dict:
    turns = [m for s in segments for m in s.turn_meta if m.get("kind") == ACTION]
    commands = [
        c
        for s in segments
        for msg, meta in zip(s.messages, s.turn_meta, strict=True)
        if meta.get("kind") == ACTION
        for c in _keystrokes(msg["content"])
    ]
    first_commands = Counter()
    for s in segments:
        for msg, meta in zip(s.messages, s.turn_meta, strict=True):
            if meta.get("kind") == ACTION:
                ks = _keystrokes(msg["content"])
                if ks:
                    first_commands[_command_label(ks[0])] += 1
                break

    by_source = Counter(t.get("target_source") for t in turns)
    joined = [t for t in turns if t.get("target_source") == "raw_capture"]
    kinds = Counter(m.get("kind") for s in segments for m in s.turn_meta)

    return {
        "source_segments": len(segments),
        "segments_with_problems": sum(1 for s in segments if s.problems),
        "turn_kinds": dict(kinds),
        "supervised_action_turns": len(turns),
        "target_source": dict(by_source),
        "rebuild_justification": {
            "joined_turns": len(joined),
            "field_equivalent": sum(1 for t in joined if t.get("equivalent") is True),
            "field_mismatch": sum(1 for t in joined if t.get("equivalent") is False),
        },
        "commands_retained": len(commands),
        "operations": {
            "read": sum(1 for c in commands if READ_RE.search(c)),
            "edit": sum(1 for c in commands if EDIT_RE.search(c)),
            "test": sum(1 for c in commands if TEST_RE.search(c)),
        },
        "task_complete_mappings": sum(1 for t in turns if t.get("task_complete")),
        "teacher_parse_failures": kinds.get(PARSE_ERROR, 0),
        "segments_containing_edit": sum(
            1
            for s in segments
            if any(
                EDIT_RE.search(c)
                for msg, meta in zip(s.messages, s.turn_meta, strict=True)
                if meta.get("kind") == ACTION
                for c in _keystrokes(msg["content"])
            )
        ),
        "first_command_distribution": dict(first_commands.most_common(15)),
        "captures": {
            k: v for k, v in capture_stats.items() if k != "conflicts"
        }
        | {"conflicts": len(capture_stats.get("conflicts", []))},
        "problems": [
            {"task": s.task, "segment_index": s.segment_index, "problems": s.problems}
            for s in segments
            if s.problems
        ],
    }


def _command_label(keystrokes: str) -> str:
    """A short label for the first-command histogram.

    Keystrokes are sent to a terminal verbatim, so a perfectly ordinary command
    is a bare `"\\n"` (press Enter) or `"C-c"` — both of which strip to nothing
    and have no first line. Those are real actions and belong in the
    distribution under their own name, not as a crash.
    """
    stripped = keystrokes.strip()
    if not stripped:
        return "<enter>" if keystrokes else "<empty>"
    return stripped.splitlines()[0][:40]


def _keystrokes(content: str) -> list[str]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return []
    return [
        c.get("keystrokes", "")
        for c in obj.get("commands") or []
        if isinstance(c, dict)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="a pass@k output dir (repeatable)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-captures", action="store_true",
                    help="rebuild from ATIF only; skips the 6.2 GB scan")
    args = ap.parse_args()

    rollouts: list[dict] = []
    for run in args.run:
        got = passing_rollouts(run)
        print(f"{run}: {len(got)} passing rollouts")
        rollouts.extend(got)

    index: dict[str, str] = {}
    capture_stats: dict = {"skipped": True}
    if not args.no_captures:
        paths = [r / "captures" / "token_captures.jsonl" for r in args.run]
        paths = [p for p in paths if p.exists()]
        print(f"indexing raw captures from {len(paths)} file(s)...")
        index, capture_stats = build_capture_index(paths)
        print(f"  index size {capture_stats['index_size']}, "
              f"{capture_stats['benign_duplicates']} benign duplicates, "
              f"{len(capture_stats['conflicts'])} conflicts")
        if capture_stats["conflicts"]:
            print("CONFLICT: identical join keys with different text — the key "
                  "is not identifying an action", file=sys.stderr)
            for c in capture_stats["conflicts"][:3]:
                print(f"  {c['key'][:16]}\n    a={c['a'][:160]}\n    b={c['b'][:160]}",
                      file=sys.stderr)
            return 1

    segments: list[Segment] = []
    skipped: list[str] = []
    for r in rollouts:
        task = r["task"]
        jobs_dir = Path(r["jobs_dir"])
        traj = next(jobs_dir.rglob("agent/trajectory.json"), None)
        if traj is None:
            skipped.append(f"{task}: no agent/trajectory.json under {jobs_dir}")
            continue
        steps = load_steps(traj)
        for i, seg_steps in enumerate(split_on_compaction(steps, traj.parent)):
            seg = build_segment(
                seg_steps, index,
                task=task, jobs_dir=jobs_dir,
                rollout_index=r.get("rollout_index"), segment_index=i,
            )
            if not any(seg.supervise):
                skipped.append(f"{task} seg{i}: no supervised action turns")
                continue
            segments.append(seg)

    for s in skipped:
        print(f"  skip {s}", file=sys.stderr)

    report = audit(segments, capture_stats)
    unknown = report["turn_kinds"].get(UNKNOWN, 0)

    print(f"\n{len(rollouts)} rollouts -> {len(segments)} segments over "
          f"{len({s.task for s in segments})} tasks")
    print(f"turn kinds: {report['turn_kinds']}")
    print(f"supervised action turns: {report['supervised_action_turns']}")
    print(f"target source: {report['target_source']}")
    rj = report["rebuild_justification"]
    print(f"rebuild justification: {rj['field_equivalent']}/{rj['joined_turns']} "
          f"joined turns field-equivalent, {rj['field_mismatch']} mismatched")
    print(f"commands: {report['commands_retained']}  ops {report['operations']}")
    print(f"segments containing an edit: {report['segments_containing_edit']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = args.out_dir / "sft_repaired.jsonl"
    with train.open("w") as f:
        for seg in segments:
            f.write(json.dumps({
                "task": seg.task,
                "messages": seg.messages,
                "supervise": seg.supervise,
            }) + "\n")
    manifest = args.out_dir / "repair_manifest.jsonl"
    with manifest.open("w") as f:
        for seg in segments:
            f.write(json.dumps({
                "task": seg.task,
                "jobs_dir": seg.jobs_dir,
                "rollout_index": seg.rollout_index,
                "segment_index": seg.segment_index,
                "n_messages": len(seg.messages),
                "n_supervised": sum(seg.supervise),
                "turn_meta": seg.turn_meta,
                "problems": seg.problems,
            }) + "\n")
    (args.out_dir / "repair_audit.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / "skipped.json").write_text(json.dumps(skipped, indent=2))
    digest = hashlib.sha256(train.read_bytes()).hexdigest()
    (args.out_dir / "dataset_sha256.txt").write_text(digest + "\n")
    print(f"\nwrote {train}\ndataset sha256 {digest}")

    if unknown:
        print(f"\nAUDIT FAILED: {unknown} unclassified assistant turns", file=sys.stderr)
        return 1
    if rj["field_mismatch"]:
        print(f"\nAUDIT FAILED: {rj['field_mismatch']} raw/rebuild field mismatches — "
              "rebuilding the unjoined turns is not justified", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
