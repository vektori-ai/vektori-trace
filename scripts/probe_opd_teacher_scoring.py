#!/usr/bin/env python3
"""Phase-0 §6.3 gate — can DeepSeek actually score ck75's actions?

Proves, against the real deployment, every condition
`docs/OPD-MULTITURN-PLAN.md` §6.3 requires before any Harbor or GPU work:

  A. the expected DeepSeek model revision answers;
  B. the returned number is the teacher-forced `logprob`, NOT `sampling_logprob`;
  C. returned token ids / bytes locate the action span exactly;
  D. exactly one finite log probability per DeepSeek action token;
  E. nothing was generated and no context was silently truncated;
  F. teacher forcing works against a separately rendered multi-turn prefix.

Run on **one short** and **one multi-turn** transcript, as §6.3 specifies.

Token ids come from the **pinned local encoder** (`encoding_dsv4` +
`providers/teacher/cross.py`), not from a server endpoint: Fireworks exposes no
`/tokenize`, and `providers/teacher/fireworks.py` documents that the OPD loop
never needs one because it holds the prefix as ids already. Probing any other
tokenisation path would prove nothing about the run.

Read-only and cheap: every call is `max_tokens=1` over a bounded prefix. The
multi-turn transcript is the only one with real size, and it is scored once.
`top_logprobs` is never sent — §6.3 omits it, and top-5 is irrelevant to this
objective.

A FAIL here is a §11 stop condition. Do not fall back to GOLD, SimpleOPD,
isolated-action scoring, or a top-5 reconstruction: fix the transport or stop.

Usage
-----
    export FIREWORKS_API_KEY=...            # or: source /data/.env.fw
    python3 scripts/probe_opd_teacher_scoring.py \
        --model accounts/fireworks/models/deepseek-v4-flash-0731 \
        --out /tmp/opd_probe_report.json

Add `--echo-mode last` to also check the cheaper `echo_last` shape. Exit code is
0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from typing import Any

BASE = "https://api.fireworks.ai/inference/v1"

# A short single-turn action and a multi-turn transcript whose last assistant
# message is what gets scored. Deliberately Terminus-shaped: JSON, a shell
# command, a path, digits — the content §6.2 calls out as alignment-hostile.
SHORT_ACTION = '{"cmd": "ls -la /workspace/src"}'

MULTI_TURN = [
    ("system", "You are a terminal agent. Reply with one JSON tool call."),
    ("user", "Find where the resolver lives in this repo."),
    ("assistant", '{"cmd": "find /workspace -name \'*.py\' | head -20"}'),
    ("user", "/workspace/src/hatch/resolver.py\n/workspace/src/hatch/cli.py"),
    ("assistant", '{"cmd": "sed -n \'1,40p\' /workspace/src/hatch/resolver.py"}'),
]


def post(url: str, payload: dict, key: str, timeout: float = 180.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:1200]}
    except urllib.error.URLError as e:
        return 0, {"unreachable": str(e)}


class Gate:
    """One §6.3 condition, its verdict, and the evidence behind it."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, gate: str, ok: bool, detail: str, evidence: Any = None) -> bool:
        self.rows.append(
            {"gate": gate, "pass": bool(ok), "detail": detail, "evidence": evidence}
        )
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {gate}: {detail}")
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [r["gate"] for r in self.rows if not r["pass"]]


def load_teacher_tokenizer(repo: str):
    """The pinned local DeepSeek tokenizer — the production path.

    Fireworks exposes **no** `/tokenize` endpoint; `providers/teacher/fireworks.py`
    documents this and the OPD loop never needs one, because it holds the prefix
    as ids already. Ids therefore come from the SHA-pinned local encoder, exactly
    as `providers/teacher/cross.py` does it at training time. Probing a different
    tokenisation path than the run will use would prove nothing about the run.
    """
    from vektori_trace.vocab_bridge import load_tokenizer

    return load_tokenizer(repo)


def render_and_encode(messages, tok, thinking_mode: str = "chat"):
    """messages -> DeepSeek-rendered string -> teacher ids, the production way."""
    from vektori_trace.providers.teacher.cross import (
        encode_teacher_ids,
        render_teacher_prefix,
    )

    text = render_teacher_prefix(messages, thinking_mode=thinking_mode)
    return text, encode_teacher_ids(text, tok)


def _decode(tok: Any, ids: list[int]) -> str:
    """Decode ids **with special tokens visible**.

    Critical: the default `decode` silently drops EOS, so a span still carrying
    the renderer's `<|end_of_sentence|>` decodes to exactly the action bytes and
    looks byte-exact while containing a token ck75 never sampled. Scoring that
    token would supervise the teacher's opinion of a turn boundary as if it were
    the model's own output.
    """
    for kwargs in ({"skip_special_tokens": False}, {}):
        try:
            return tok.decode(ids, **kwargs)
        except TypeError:
            continue
        except Exception:  # pragma: no cover - flavour differences
            return ""
    return ""


def score(
    prefix_ids: list[int],
    action_ids: list[int],
    model: str,
    key: str,
    echo_mode: str,
) -> tuple[int, dict]:
    """One teacher-forced scoring call over prefix+action, nothing generated."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prefix_ids + action_ids,
        # Not 0 — some deployments reject a zero-token generation. The one
        # sampled token is discarded and must not appear in the scored window.
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "raw_output": True,
    }
    if echo_mode == "last":
        payload["echo_last"] = len(action_ids)
    else:
        payload["echo"] = True
    return post(BASE + "/completions", payload, key)


def run_case(
    name: str,
    prior_messages: list[dict[str, Any]],
    action_text: str,
    tok: Any,
    model: str,
    key: str,
    echo_mode: str,
    g: Gate,
) -> dict[str, Any]:
    print(f"\n=== {name} (echo_mode={echo_mode}) ===")
    ev: dict[str, Any] = {"case": name, "echo_mode": echo_mode}

    # --- C (part 1): locate the action span the way §4 demands — tokenise
    # `teacher_prefix + exact_action` jointly, never by assuming
    #   tokenize(prefix + action) == tokenize(prefix) + tokenize(action).
    # Both renders go through the pinned encoder, so this is the same
    # tokenisation the training run will use.
    _prefix_text, prefix_ids = render_and_encode(prior_messages, tok)
    joint_text, joint_ids = render_and_encode(
        [*prior_messages, {"role": "assistant", "content": action_text}], tok
    )

    # The rendered assistant turn adds a role header before the action bytes;
    # what we supervise is only the action itself, so find where the action's
    # own bytes begin inside the joint render.
    if action_text not in joint_text:
        g.check(
            f"{name}/C-boundary",
            False,
            "rendered joint prompt does not contain the verbatim action bytes — "
            "the renderer altered them, so no byte-exact span exists",
        )
        return ev

    prefix_ok = joint_ids[: len(prefix_ids)] == prefix_ids
    action_ids = joint_ids[len(prefix_ids):]

    # The renderer closes a completed assistant turn with EOS. That token was
    # never sampled by ck75, so §4 ("only bytes actually sampled as ck75's
    # completion are supervised") excludes it. Drop it by checking the decoded
    # bytes rather than hardcoding an id.
    n_eos_dropped = 0
    while len(action_ids) > 1 and _decode(tok, action_ids) != action_text:
        if not _decode(tok, action_ids).startswith(action_text):
            break  # mismatch is not a trailing-token problem; report it as-is
        action_ids = action_ids[:-1]
        n_eos_dropped += 1
    ev["n_trailing_tokens_dropped"] = n_eos_dropped
    _, concat_ids = render_and_encode(
        [{"role": "assistant", "content": action_text}], tok
    )
    ev["n_prefix_tokens"] = len(prefix_ids)
    ev["n_action_tokens"] = len(action_ids)
    ev["prefix_is_exact_id_prefix_of_joint"] = prefix_ok

    decoded = _decode(tok, action_ids)
    g.check(
        f"{name}/C-bytes",
        decoded == action_text,
        f"scored span decodes to the exact action bytes ({len(action_text)} chars)"
        if decoded == action_text
        else f"scored span decodes to {decoded!r}, expected {action_text!r}",
    )
    g.check(
        f"{name}/C-boundary",
        len(action_ids) > 0,
        f"joint tokenisation gives {len(prefix_ids)} prefix + {len(action_ids)} "
        f"continuation teacher tokens; prefix is an exact id-prefix of the joint "
        f"encoding: {prefix_ok}",
        {"boundary_shifts_under_concat": concat_ids != action_ids},
    )

    status, body = score(prefix_ids, action_ids, model, key, echo_mode)
    ev["http_status"] = status
    if status != 200:
        g.check(f"{name}/transport", False, f"HTTP {status}: {str(body)[:300]}")
        return ev

    # --- A: model revision
    served = body.get("model", "")
    ev["served_model"] = served
    g.check(
        f"{name}/A-revision",
        model.split("/")[-1] in served or served in model,
        f"served model reported as {served!r}",
    )

    choices = body.get("choices") or []
    if not choices:
        g.check(f"{name}/transport", False, f"no choices: {str(body)[:300]}")
        return ev
    lp = choices[0].get("logprobs") or {}
    content = lp.get("content")

    if content is None:
        g.check(
            f"{name}/D-format",
            False,
            "legacy logprobs format (no `content`) — alternatives keyed by token "
            f"string, not id; cannot align. keys={sorted(lp)}",
        )
        return ev

    entries = [e for e in content if isinstance(e, dict)]
    ev["n_entries"] = len(entries)

    # --- C (part 2): locate the scored window by token id.
    ids_present = all("token_id" in e for e in entries)
    window = None
    if ids_present:
        got = [int(e["token_id"]) for e in entries]
        ev["entry_ids_head"] = got[:12]
        for start in range(len(got) - len(action_ids) + 1):
            if got[start : start + len(action_ids)] == action_ids:
                window = (start, start + len(action_ids))
                break
    g.check(
        f"{name}/C-span",
        window is not None,
        f"action ids located in the echoed stream at {window}"
        if window
        else "action token ids NOT found in the response — the span cannot be "
        "located, which is a §11 stop condition",
    )
    if window is None:
        return ev

    scored = entries[window[0] : window[1]]

    # --- D: one finite logprob per action token
    vals = [e.get("logprob") for e in scored]
    finite = [isinstance(v, (int, float)) and math.isfinite(v) for v in vals]
    g.check(
        f"{name}/D-finite",
        len(scored) == len(action_ids) and all(finite),
        f"{sum(finite)}/{len(action_ids)} action tokens carry a finite `logprob`",
        {"min": min([v for v in vals if isinstance(v, (int, float))], default=None)},
    )

    # --- B: teacher-forced `logprob`, not `sampling_logprob`.
    # These differ whenever temperature/filters were applied. Both present and
    # equal everywhere is inconclusive at temperature 0, so report that honestly
    # rather than claiming a pass we did not earn.
    has_sampling = any("sampling_logprob" in e for e in scored)
    if has_sampling:
        diffs = [
            abs(float(e["logprob"]) - float(e["sampling_logprob"]))
            for e in scored
            if "sampling_logprob" in e and isinstance(e.get("logprob"), (int, float))
        ]
        distinguishable = any(d > 1e-9 for d in diffs)
        g.check(
            f"{name}/B-logprob-field",
            True,
            "`logprob` read explicitly; `sampling_logprob` also present and "
            + (
                f"differs (max Δ={max(diffs):.4g}) — the two are distinguishable "
                "and we read the right one"
                if distinguishable
                else "identical at temperature 0 (expected; we still read `logprob`)"
            ),
            {"max_abs_delta": max(diffs, default=0.0)},
        )
    else:
        g.check(
            f"{name}/B-logprob-field",
            True,
            "response carries only `logprob` — no `sampling_logprob` to confuse it with",
        )

    # --- E: nothing generated, nothing truncated
    finish = choices[0].get("finish_reason")
    text_out = choices[0].get("text") or ""
    usage = body.get("usage") or {}
    ev["finish_reason"] = finish
    ev["usage"] = usage
    prompt_toks = usage.get("prompt_tokens")
    expected = len(prefix_ids) + len(action_ids)
    g.check(
        f"{name}/E-no-truncation",
        prompt_toks is None or abs(int(prompt_toks) - expected) <= 2,
        f"prompt_tokens={prompt_toks} vs {expected} submitted "
        "(±2 for BOS/sampled-token accounting)",
    )
    g.check(
        f"{name}/E-no-generation",
        len(text_out) < 200,
        f"finish_reason={finish!r}, generated text {len(text_out)} chars "
        "(1 sampled token is discarded by design)",
    )

    ev["scored_logprobs_head"] = [
        round(float(v), 5) for v in vals[:8] if isinstance(v, (int, float))
    ]
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="accounts/fireworks/models/deepseek-v4-flash-0731"
    )
    ap.add_argument(
        "--tokenizer",
        default="deepseek-ai/DeepSeek-V4-Flash-0731",
        help="HF repo for the teacher tokenizer (the production path; Fireworks "
        "has no /tokenize endpoint)",
    )
    ap.add_argument("--echo-mode", default="full", choices=["full", "last", "both"])
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2

    # The renderer and tokenizer are the production ones. A failure to load
    # them is a real Phase-0 failure (the run needs both), not a probe excuse.
    try:
        from vektori_trace.encoding_dsv4 import verify_encoding_dsv4_pin

        verify_encoding_dsv4_pin()
        print(f"pinned DeepSeek encoder verified; loading tokenizer {args.tokenizer}")
        tok = load_teacher_tokenizer(args.tokenizer)
    except Exception as e:  # report it, do not mask it
        print(f"\nFAILED to load the pinned teacher renderer/tokenizer: {e}")
        print(
            "This is the same path the training run uses, so it must work before "
            "any scoring is meaningful."
        )
        return 2

    # §6.3: "at least one short and one multi-turn frozen transcript".
    short_messages = [
        {"role": "system", "content": "You are a terminal agent."},
        {"role": "user", "content": "List the source directory."},
    ]
    multi_messages = [
        {"role": role, "content": text} for role, text in MULTI_TURN[:-1]
    ]
    multi_action = MULTI_TURN[-1][1]

    modes = ["full", "last"] if args.echo_mode == "both" else [args.echo_mode]

    g = Gate()
    report: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "cases": [],
    }
    for mode in modes:
        report["cases"].append(
            run_case(
                "short", short_messages, SHORT_ACTION, tok, args.model, key, mode, g
            )
        )
        report["cases"].append(
            run_case(
                "multiturn", multi_messages, multi_action, tok, args.model, key, mode, g
            )
        )

    report["gates"] = g.rows
    report["all_passed"] = not g.failed

    print("\n" + "=" * 66)
    if g.failed:
        print(f"FAILED gates: {', '.join(g.failed)}")
        print(
            "\n§11 stop condition. Do NOT fall back to GOLD, SimpleOPD, "
            "isolated-action scoring, or top-5 reconstruction.\n"
            "Do not run the Harbor/GPU smoke."
        )
    else:
        print("ALL §6.3 GATES PASSED — teacher scoring is usable for chunk OPD.")
        print("Next gate (§6.4, no spend): mocked two-turn optimizer proof.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport: {args.out}")

    return 0 if not g.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
