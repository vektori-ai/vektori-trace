#!/usr/bin/env python3
"""Serving probe: can ck75 sample at a stored replay prefix with logprobs?

The one thing the replay update needs from the serving path that nothing has
verified end to end:

  A. the endpoint answers as ck75 (base + the pinned adapter, not the base alone);
  B. a stored replay prefix renders and is accepted;
  C. the response carries per-token ids for the sampled action;
  D. it carries one behaviour log probability per sampled token;
  E. those survive `replay_sample.sampled_action_from_capture` into a
     `SampledAction` whose bytes reconstruct exactly;
  F. four independent draws at one prefix differ (temperature is live, so the
     four samples §8.3 asks for are not four copies);
  G. no sample hit the token cap — 9216 clears the corpus max (8,842) outright,
     but those are the *teacher's* lengths, so the student's own cap-hit rate is
     the only evidence the cap is right for ck75.

D is the load-bearing one. `log pi_old` is the denominator of §5's importance
ratio and cannot be recovered after sampling — if it is missing the whole
rollout has to be repeated, so it is worth one cheap probe to find out.

This does **not** train, score with the teacher, or touch the optimizer. It
samples a handful of short actions and exits.

Usage
-----
Start the endpoint first (see the module docstring of `scripts/serve_student.py`;
it must be launched with the adapter and `--max-lora-rank 32`), then:

    export STUDENT_API_BASE=<url printed by serve_student.py>
    .venv/bin/python scripts/probe_ck75_replay_sampling.py \\
        --corpus /data/vektori-out/dsv4-corpus60 \\
        --out /data/ck75_sampling_probe.json

Exit code is 0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CORPUS = "/data/vektori-out/dsv4-corpus60"
DEFAULT_MODEL = "qwen3-14b"


class Gate:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str, evidence: Any = None) -> bool:
        self.rows.append(
            {"gate": name, "pass": bool(ok), "detail": detail, "evidence": evidence}
        )
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [r["gate"] for r in self.rows if not r["pass"]]


def post(url: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
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


def get(url: str, timeout: float = 60.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:
        return 0, {"error": str(e)}


def pick_prefix(corpus: Path, *, step: int | None):
    """One passing trace's replay state, via the production loader."""
    from vektori_trace.replay_corpus import candidates_from_traces, load_corpus

    traces = load_corpus(corpus, passing_only=True, limit=1)
    if not traces:
        raise SystemExit(f"no passing traces under {corpus}")
    trace = traces[0]
    cands = candidates_from_traces([trace])
    if not cands:
        raise SystemExit(f"{trace.trace_id}: no usable replay states")
    if step is not None:
        for c in cands:
            if c.step_index == step:
                return trace, c
        raise SystemExit(f"step {step} not among {len(cands)} candidates")
    # Middle of the trace: a real long-horizon state rather than a cold start.
    return trace, cands[len(cands) // 2]


def render_prefix(prefix, tokenizer) -> str:
    """The stored prefix as ck75's serving renderer would see it.

    Uses the student chat template, not the DeepSeek one: this is what ck75
    conditions on. The teacher-side render is a separate concern handled by
    `providers/teacher/cross.py` when the action is scored.
    """
    from vektori_trace.dataset import turns_to_messages

    messages = turns_to_messages(prefix.prefix_turns)
    if not messages:
        raise SystemExit(f"{prefix.prefix_id}: prefix rendered to zero messages")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE"))
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="served model name; the LoRA name if one is attached")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--step", type=int, default=None,
                    help="replay step to probe; default is mid-trace")
    ap.add_argument("--n-samples", type=int, default=4,
                    help="§8.3 draws four independent actions per prefix")
    ap.add_argument("--max-tokens", type=int, default=9216,
                    help="measured cap (docs/action-length-measurement.md); "
                         "must exceed the old 256 (§7.1)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--allow-truncated",
        action="store_true",
        help="record a cap-hit instead of failing. Off by default: the replay "
             "run must fail closed on any truncation, so a probe that tolerates "
             "it would not be probing the real path.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.api_base:
        print("set --api-base or STUDENT_API_BASE", file=sys.stderr)
        return 2
    base = args.api_base.rstrip("/")

    from vektori_trace.chunk_opd import assert_token_cap_is_task_derived
    from vektori_trace.replay_sample import (
        CaptureAdaptError,
        sampled_action_from_capture,
        summarize_cap_hits,
    )

    # §7.1 refuses the old cap before anything is sampled under it.
    assert_token_cap_is_task_derived(args.max_tokens)

    g = Gate()
    report: dict[str, Any] = {"api_base": base, "model": args.model}

    # --- A: the endpoint is serving what we think it is.
    status, models = get(base + "/models")
    served = [m.get("id") for m in (models.get("data") or [])] if status == 200 else []
    report["served_models"] = served
    g.check(
        "A-endpoint",
        status == 200 and args.model in served,
        f"/models -> {served}" if status == 200 else f"HTTP {status}: {models}",
    )
    if status != 200:
        return _finish(g, report, args)

    # --- B: a real stored replay state renders.
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    trace, prefix = pick_prefix(Path(args.corpus), step=args.step)
    prompt = render_prefix(prefix, tokenizer)
    n_prompt = len(tokenizer(prompt)["input_ids"])
    report["trace"] = {
        "task": trace.task,
        "trace_id": trace.trace_id,
        "n_steps": trace.n_steps,
        "passed": trace.passed,
    }
    report["prefix"] = {
        "prefix_id": prefix.prefix_id,
        "step_index": prefix.step_index,
        "n_prefix_turns": prefix.n_prefix_turns,
        "n_prompt_tokens": n_prompt,
    }
    g.check(
        "B-prefix",
        n_prompt > 0,
        f"{trace.task} {prefix.prefix_id} (step {prefix.step_index} of "
        f"{trace.n_steps}) renders to {n_prompt} prompt tokens",
    )

    # --- C/D: sample with capture on.
    actions, raw = [], []
    for i in range(args.n_samples):
        status, body = post(
            base + "/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "logprobs": 0,          # chosen-token logprob only; no alternatives
                "return_token_ids": True,
            },
        )
        if status != 200:
            g.check("C-sampling", False, f"sample {i}: HTTP {status}: {str(body)[:300]}")
            return _finish(g, report, args)
        raw.append(body)

        choice = (body.get("choices") or [{}])[0]
        lp = choice.get("logprobs") or {}
        cap = _Capture(
            token_ids=choice.get("token_ids") or body.get("token_ids") or [],
            logprobs=lp.get("token_logprobs"),
            finish_reason=choice.get("finish_reason"),
            prompt_token_ids=body.get("prompt_token_ids") or [],
            text=choice.get("text"),
        )
        try:
            actions.append(
                sampled_action_from_capture(
                    cap,
                    tokenizer,
                    prefix_id=prefix.prefix_id,
                    sample_index=i,
                    policy_version="ck75-v0",
                    allow_truncated=args.allow_truncated,
                )
            )
        except CaptureAdaptError as e:
            gate = "G-cap" if "mid-sequence" in str(e) else "D-logprobs"
            g.check(gate, False, f"sample {i}: {e}")
            return _finish(g, report, args)

    g.check(
        "C-token-ids",
        all(a.action_token_ids for a in actions),
        f"{len(actions)} samples carry token ids "
        f"({[len(a.action_token_ids) for a in actions]} tokens)",
    )
    g.check(
        "D-logprobs",
        all(len(a.behavior_logprobs) == len(a.action_token_ids) for a in actions),
        "one finite behaviour logprob per sampled token in every sample — "
        "log pi_old is available",
    )

    # --- E: bytes reconstruct.
    ok_bytes = all(
        b"".join(a.action_token_bytes) == a.action_bytes for a in actions
    )
    g.check(
        "E-bytes",
        ok_bytes,
        "token bytes reconstruct each sampled action exactly",
    )

    # --- F: the four draws are independent.
    distinct = len({a.action_bytes for a in actions})
    g.check(
        "F-distinct",
        distinct > 1 or args.temperature == 0.0,
        f"{distinct} distinct actions out of {len(actions)} at "
        f"temperature {args.temperature}",
    )

    # --- G: no sample hit the cap. 9216 clears the corpus's observed max
    # (8,842) with room, so no stored action would have been truncated by it.
    # But these are DeepSeek's lengths, not ck75's: a non-zero rate here means
    # the cap is wrong for the student and must be re-derived before the run.
    cap_report = summarize_cap_hits(actions)
    g.check(
        "G-cap",
        cap_report["n_cap_hits"] == 0,
        f"no sample hit the {args.max_tokens}-token cap "
        f"(max sampled {cap_report['max_action_tokens']}, "
        f"mean {cap_report['mean_action_tokens']:.0f})"
        if cap_report["n_cap_hits"] == 0
        else f"{cap_report['n_cap_hits']}/{cap_report['n_actions']} samples hit "
        f"the cap — re-derive it from ck75's own lengths before the run",
    )
    report["cap"] = cap_report
    report["samples"] = [
        {
            "key": a.key,
            "n_tokens": len(a.action_token_ids),
            "finish_reason": a.termination_reason,
            "mean_logprob": sum(a.behavior_logprobs) / len(a.behavior_logprobs),
            "action_preview": a.action_bytes[:160].decode("utf-8", "replace"),
        }
        for a in actions
    ]
    report["raw_first_response_keys"] = sorted(raw[0].keys()) if raw else []
    return _finish(g, report, args)


class _Capture:
    """Duck-typed stand-in for `CapturedCompletion` (see replay_sample)."""

    def __init__(self, token_ids, logprobs, finish_reason, prompt_token_ids, text):
        self.token_ids = [int(t) for t in (token_ids or [])]
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.prompt_token_ids = prompt_token_ids
        self.text = text
        self.request_id = None
        self.model = None


def _finish(g: Gate, report: dict[str, Any], args) -> int:
    report["gates"] = g.rows
    report["all_passed"] = not g.failed

    print("\n" + "=" * 66)
    if g.failed:
        print(f"FAILED: {', '.join(g.failed)}")
        print(
            "\nDo not build the scoring bridge or optimizer callback on an "
            "unproven sampling path. Tear the endpoint down before diagnosing."
        )
    else:
        print("ALL GATES PASSED — ck75 sampling supplies ids and log pi_old.")
        print("Next: Fireworks scoring bridge, then the optimizer callback.")
    print("\nTear down the endpoint now if this was its only use.")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"report: {args.out}")
    return 0 if not g.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
