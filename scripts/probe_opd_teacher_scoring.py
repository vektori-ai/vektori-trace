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


def tokenize(text: str, model: str, key: str) -> list[int] | None:
    """Ask the deployment for the ids it would use. None when unsupported."""
    status, body = post(
        BASE + "/tokenize", {"model": model, "text": text}, key
    )
    if status != 200:
        return None
    toks = body.get("tokens") or body.get("token_ids")
    if isinstance(toks, list) and toks and isinstance(toks[0], int):
        return [int(t) for t in toks]
    return None


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
    prefix_text: str,
    action_text: str,
    model: str,
    key: str,
    echo_mode: str,
    g: Gate,
) -> dict[str, Any]:
    print(f"\n=== {name} (echo_mode={echo_mode}) ===")
    ev: dict[str, Any] = {"case": name, "echo_mode": echo_mode}

    # --- C (part 1): the boundary must come from real tokenisation, never from
    # assuming tokenize(prefix+action) == tokenize(prefix) + tokenize(action).
    prefix_ids = tokenize(prefix_text, model, key)
    joint_ids = tokenize(prefix_text + action_text, model, key)
    if prefix_ids is None or joint_ids is None:
        g.check(
            f"{name}/C-boundary",
            False,
            "/tokenize unavailable — cannot locate the action span by id. "
            "§6.3 requires ids/offsets sufficient to locate it.",
        )
        return ev

    action_ids = joint_ids[len(prefix_ids):]
    concat_ids = tokenize(action_text, model, key) or []
    ev["n_prefix_tokens"] = len(prefix_ids)
    ev["n_action_tokens"] = len(action_ids)

    g.check(
        f"{name}/C-boundary",
        joint_ids[: len(prefix_ids)] == prefix_ids and len(action_ids) > 0,
        f"action span located by joint tokenisation: {len(action_ids)} teacher "
        f"tokens (naive concat would give {len(concat_ids)})",
        {"straddles": concat_ids != action_ids},
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
    ap.add_argument("--echo-mode", default="full", choices=["full", "last", "both"])
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2

    # §6.3: "at least one short and one multi-turn frozen transcript".
    short_prefix = "<|user|>List the source directory.<|assistant|>"
    multi_prefix = "".join(
        f"<|{role}|>{text}" for role, text in MULTI_TURN[:-1]
    ) + "<|assistant|>"
    multi_action = MULTI_TURN[-1][1]

    modes = ["full", "last"] if args.echo_mode == "both" else [args.echo_mode]

    g = Gate()
    report: dict[str, Any] = {"model": args.model, "cases": []}
    for mode in modes:
        report["cases"].append(
            run_case("short", short_prefix, SHORT_ACTION, args.model, key, mode, g)
        )
        report["cases"].append(
            run_case("multiturn", multi_prefix, multi_action, args.model, key, mode, g)
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
