"""Is the repaired adapter *copying* the protocol from context, or does it know it?

Phase 7 found that `native_json` tracks how many prior assistant turns a prefix
contains almost perfectly: it fails at turn 1-4 and passes at turn 6+, with the
instruction block byte-identical in every prefix. That is a correlation. This
script tries to break it into a cause.

Two experiments, both on one checkpoint, both cheap:

  A. DOSE-RESPONSE — walk one real rollout and ask the same checkpoint for an
     action at turn 1, 2, 3, ... Nothing is synthesised; these are real prefixes
     from the same conversation, so the only thing that grows is the amount of
     native JSON visible above. Locates the threshold rather than assuming one.

  B. ABLATION — take a prefix the checkpoint answers correctly and rewrite its
     visible assistant turns from native JSON into v1's `<tool_call>` envelope,
     changing nothing else. If the answer flips to the v1 envelope, the visible
     format is doing the causal work. If it stays native JSON, the model knows
     the protocol and the turn-ordinal correlation has some other explanation.

B is the decisive one: A can be explained by "later turns are easier", B cannot.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.evaluate.phase7 import grade


def to_v1_envelope(native: str) -> str | None:
    """Render a native-JSON action the way v1 emitted it.

    Not invented: this is the shape the checkpoint itself produces at turn 1 —
    terminus's rendered `Analysis:/Plan:` prose (terminus_2.py:1345) followed by
    Qwen's `<tool_call>` serialisation of a `bash_command` call. Using the
    model's own failure mode as the ablation keeps the comparison honest.
    """
    try:
        obj = json.loads(native)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "commands" not in obj:
        return None
    out = [f"Analysis: {obj.get('analysis', '')}", f"Plan: {obj.get('plan', '')}"]
    for c in obj.get("commands") or []:
        if not isinstance(c, dict):
            continue
        call = {
            "name": "bash_command",
            "arguments": {
                "keystrokes": c.get("keystrokes", ""),
                "duration": c.get("duration", 1.0),
            },
        }
        out.append("<tool_call>\n" + json.dumps(call) + "\n</tool_call>")
    if obj.get("task_complete"):
        out.append(
            "<tool_call>\n"
            + json.dumps({"name": "mark_task_complete", "arguments": {}})
            + "\n</tool_call>"
        )
    return "\n".join(out)


def ask(api: str, model: str, messages: list[dict], timeout: float = 180.0):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        api.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    c = d["choices"][0]
    return c["message"]["content"] or "", c.get("finish_reason")


def verdict(text: str, finish: str | None, category: str) -> dict[str, Any]:
    g = grade(
        text,
        prefix_id="probe",
        checkpoint="probe",
        category=category,
        suite="generalization",
        finish_reason=finish,
        git_present=True,
    )
    return {
        "native_json": g.gates["native_json"],
        "harbor_accepts": g.gates["harbor_accepts"],
        "legacy_envelope": "<tool_call>" in text,
        "parser_error": g.parser_error,
        "head": text.lstrip()[:70],
        "finish_reason": finish,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", default="Qwen3-14B-ck63")
    ap.add_argument("--corpus", default="/data/phase7-corpora/heldout-failing")
    ap.add_argument("--out", type=Path, default=Path("/data/phase7/context_probe.json"))
    args = ap.parse_args()

    segs = [
        json.loads(ln)
        for ln in (Path(args.corpus) / "sft_repaired.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    # The longest segment gives the widest turn range from a single conversation,
    # so dose-response is measured within one task rather than across tasks of
    # differing difficulty.
    seg = max(segs, key=lambda s: sum(s["supervise"]))
    idx = [i for i, sup in enumerate(seg["supervise"]) if sup]
    print(f"task {seg['task']}  {len(idx)} supervised actions\n")

    report: dict[str, Any] = {
        "model": args.model,
        "task": seg["task"],
        "n_actions": len(idx),
        "dose_response": [],
        "ablation": [],
    }

    # ---- A: dose-response -------------------------------------------------
    print("=== A. DOSE-RESPONSE (real prefixes, increasing native-JSON history)")
    picks = [n for n in (1, 2, 3, 4, 5, 6, 8, 12, 20) if n <= len(idx)]
    for n in picks:
        mi = idx[n - 1]
        msgs = seg["messages"][:mi]
        text, finish = ask(args.api_base, args.model, msgs)
        v = verdict(text, finish, "orientation" if n == 1 else "first_inspection")
        v |= {"turn": n, "n_messages": len(msgs), "prior_assistant_turns": n - 1}
        report["dose_response"].append(v)
        print(f"  turn {n:2}  prior_assistant={n - 1:2}  native_json={str(v['native_json']):5}"
              f"  legacy={str(v['legacy_envelope']):5}  {v['head']!r}")

    # ---- B: ablation ------------------------------------------------------
    print("\n=== B. ABLATION (same prefix; visible assistant turns rewritten to v1)")
    passing = None
    for v in report["dose_response"]:
        if v["native_json"]:
            passing = v["turn"]
            break
    if passing is None:
        print("  no dose-response prefix passed; nothing to ablate")
        args.out.write_text(json.dumps(report, indent=2))
        return 0

    mi = idx[passing - 1]
    base_msgs = seg["messages"][:mi]

    # control: unmodified, to prove the flip is the rewrite and not sampling
    t0, f0 = ask(args.api_base, args.model, base_msgs)
    v0 = verdict(t0, f0, "first_inspection") | {"arm": "control (native history)"}

    rewritten, n_rw = [], 0
    for m in base_msgs:
        if m["role"] == "assistant":
            alt = to_v1_envelope(m["content"])
            if alt:
                rewritten.append({"role": "assistant", "content": alt})
                n_rw += 1
                continue
        rewritten.append(m)
    t1, f1 = ask(args.api_base, args.model, rewritten)
    v1 = verdict(t1, f1, "first_inspection") | {
        "arm": "ablated (v1-envelope history)",
        "assistant_turns_rewritten": n_rw,
    }
    report["ablation"] = [v0, v1]
    report["ablation_turn"] = passing

    for v in (v0, v1):
        print(f"  {v['arm']:32} native_json={str(v['native_json']):5}"
              f"  legacy={str(v['legacy_envelope']):5}  {v['head']!r}")

    flipped = v0["native_json"] and not v1["native_json"]
    report["conclusion"] = (
        "CONFIRMED: rewriting the visible history to v1 flipped the output to v1 — "
        "the model is copying the format it can see."
        if flipped
        else "NOT CONFIRMED: output did not follow the visible format; the "
             "turn-ordinal effect needs another explanation."
    )
    print(f"\n{report['conclusion']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
