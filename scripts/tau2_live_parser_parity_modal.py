"""Prove the live agent's client-side splitter agrees with vLLM's. CPU only.

Why this exists
---------------
The live capture path calls `/completions`, not `/chat/completions`, because it
needs token ids, behaviour logprobs and raw bytes -- none of which survive the
chat endpoint. But `/completions` also means vLLM's `--reasoning-parser qwen3`
and `--tool-call-parser hermes` never run: the generation arrives as one blob
with `<think>` and `<tool_call>` inline.

`live_agent.split_generation` does that splitting client-side. If it disagrees
with the server-side parser by even one character, the model is *evaluated*
through one splitter and *trained* through another -- and every number on both
sides still looks fine. That is precisely the class of defect this repo has
already paid for six times.

`tau2_parser_parity_modal.py` proved the corpus round-trips through vLLM's real
`Hermes2ProToolParser` (25/25). This runs the **same spans** through both
parsers and requires all three to agree:

    intended calls  ==  vLLM hermes parser  ==  live_agent.split_generation

No GPU: both parsers are text transforms.

    python scripts/tau2_live_parser_parity_modal.py --build   # on the box
    modal run scripts/tau2_live_parser_parity_modal.py

Comparison is on structured values, not bytes: `{"a":1,"b":2}` and
`{"b":2,"a":1}` are the same call. A conversational span that either parser
reads *as* a call is a failure, so every span is checked, not only the ones
that carry tool calls.
"""
from __future__ import annotations

import modal

app = modal.App("tau2-live-parser-parity")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same pin and same reasoning as `tau2_parser_parity_modal.py`: this
    # container never re-tokenizes. Spans arrive as text; only the parsers are
    # under test.
    .pip_install("vllm==0.11.2")
    .add_local_file("/tmp/parity_rows.jsonl", remote_path="/root/rows.jsonl")
    # The splitter under test, shipped as source so the container runs the
    # exact code the live agent will run.
    .add_local_file(
        "vektori_trace/tau2/live_agent.py", remote_path="/root/live_agent_src.py"
    )
)


def _load_splitter():
    """Import `split_generation` without dragging in the package.

    `live_agent` imports `..replay_sample` at module scope for
    `token_bytes_from_ids`, which the splitter does not need and which would
    require the whole package inside this container. Stub the parent package so
    the one function under test is importable on its own.
    """
    import sys
    import types

    pkg = types.ModuleType("vektori_trace")
    pkg.__path__ = []
    sub = types.ModuleType("vektori_trace.replay_sample")

    def _unused(*a, **k):  # pragma: no cover - never called by the splitter
        raise AssertionError("token_bytes_from_ids is not part of this check")

    sub.token_bytes_from_ids = _unused
    sys.modules.setdefault("vektori_trace", pkg)
    sys.modules["vektori_trace.replay_sample"] = sub

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vektori_trace.tau2.live_agent", "/root/live_agent_src.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "vektori_trace.tau2"
    tau2_pkg = types.ModuleType("vektori_trace.tau2")
    tau2_pkg.__path__ = []
    sys.modules["vektori_trace.tau2"] = tau2_pkg
    spec.loader.exec_module(mod)
    return mod.split_generation, mod.LiveCaptureError


@app.function(image=image, timeout=20 * 60, cpu=4.0, memory=16384)
def check() -> dict:
    import json

    rows = [json.loads(line) for line in open("/root/rows.jsonl")]
    print(f"loaded {len(rows)} target spans", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    from vllm.entrypoints.openai.tool_parsers import ToolParserManager

    available = sorted(ToolParserManager.tool_parsers.keys())
    name = "hermes" if "hermes" in available else available[0]
    vllm_parser = ToolParserManager.get_tool_parser(name)(tok)
    print(f"vLLM parser: {name}", flush=True)

    split_generation, LiveCaptureError = _load_splitter()
    print("loaded live_agent.split_generation", flush=True)

    def canon(n, a):
        return (n, json.dumps(a or {}, sort_keys=True, default=str))

    results, failures = [], []
    for r in rows:
        text = r["decoded_target"]
        intended = [canon(c["name"], c["arguments"]) for c in r["intended_calls"]]
        rec = {
            "task_id": r["task_id"],
            "position": r["position"],
            "action_type": r["action_type"],
            "intended": intended,
        }

        # -- server-side: what evaluation will see -------------------------
        try:
            info = vllm_parser.extract_tool_calls(text, request=None)
            vllm_calls = []
            for tc in getattr(info, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                raw = getattr(fn, "arguments", None)
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                vllm_calls.append(canon(getattr(fn, "name", None), args))
            rec["vllm"] = vllm_calls
        except Exception as e:
            rec["vllm_error"] = f"{type(e).__name__}: {e}"
            rec["verdict"] = "VLLM_PARSER_ERROR"
            failures.append(rec)
            results.append(rec)
            continue

        # -- client-side: what training will see ---------------------------
        try:
            reasoning, content, live = split_generation(text)
            live_calls = [canon(c["name"], c["arguments"]) for c in live]
            rec["live"] = live_calls
            rec["live_has_reasoning"] = reasoning is not None
            rec["live_content_empty"] = not content
        except LiveCaptureError as e:
            rec["live_error"] = str(e)
            rec["verdict"] = "LIVE_SPLITTER_ERROR"
            failures.append(rec)
            results.append(rec)
            continue

        # -- all three must agree ------------------------------------------
        if rec["live"] != rec["vllm"]:
            rec["verdict"] = "SPLITTER_DISAGREES_WITH_VLLM"
            failures.append(rec)
        elif rec["vllm"] != intended:
            # Already covered by tau2_parser_parity, but a regression here
            # would otherwise be invisible in this report.
            rec["verdict"] = "VLLM_DISAGREES_WITH_CORPUS"
            failures.append(rec)
        elif not intended and rec["live"]:
            rec["verdict"] = "CONVERSATIONAL_PARSED_AS_CALL"
            failures.append(rec)
        else:
            rec["verdict"] = "ok"
        results.append(rec)

    ok = sum(1 for r in results if r["verdict"] == "ok")
    print(f"\n{ok}/{len(results)} spans agree across corpus/vLLM/live", flush=True)

    by_kind: dict[tuple[str, str], int] = {}
    for r in results:
        k = (r["action_type"], r["verdict"])
        by_kind[k] = by_kind.get(k, 0) + 1
    for k, v in sorted(by_kind.items()):
        print(f"  {k[0]:16s} {k[1]:34s} {v}", flush=True)

    for f in failures[:10]:
        print(f"\nFAIL {f['task_id']}#{f['position']} {f['verdict']}", flush=True)
        print(f"  intended: {f.get('intended')}", flush=True)
        print(f"  vllm:     {f.get('vllm', f.get('vllm_error'))}", flush=True)
        print(f"  live:     {f.get('live', f.get('live_error'))}", flush=True)

    return {
        "parser": name,
        "n": len(results),
        "ok": ok,
        "failures": failures[:40],
        "by_kind": {f"{k[0]}|{k[1]}": v for k, v in by_kind.items()},
    }


@app.local_entrypoint()
def main():
    import json

    out = check.remote()
    print("\n" + "=" * 64)
    print(f"LIVE PARSER PARITY: {out['ok']}/{out['n']} agree (parser={out['parser']})")
    print("=" * 64)
    if out["ok"] != out["n"]:
        print(
            f"{out['n'] - out['ok']} span(s) failed. The live splitter and the "
            "serving parser do not agree, so the model would be trained "
            "through one and evaluated through the other."
        )
    json.dump(out, open("/tmp/live_parity_result.json", "w"), indent=1)
    print("wrote /tmp/live_parity_result.json")
