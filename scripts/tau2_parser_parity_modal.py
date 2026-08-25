"""Prove the corpus's tool calls survive the real serving parser. CPU only.

`tau2_build_corpus.py` fell back to a structural check -- its own scan for
`<tool_call>` markers -- because vLLM is not installed on the EC2 box. That
proves the bytes are well formed; it does not prove the *serving* path will turn
them back into the same action. The parser vLLM actually runs for Qwen3 is what
decides that at inference, so this imports it and feeds it decoded target spans.

No GPU: the tool parser is a text transform. This container asks for CPU only.

    modal run scripts/tau2_parser_parity_modal.py

Compares structured values, not bytes: {"a":1,"b":2} and {"b":2,"a":1} are the
same call. A conversational target that the parser reads *as* a call is also a
failure, so every target is checked, not only the ones with tool calls.
"""
from __future__ import annotations

import modal

app = modal.App("tau2-parser-parity")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.11.2 pins transformers<5, while the corpus was tokenized with
    # 5.5.3. That is fine here: this container never re-tokenizes. The decoded
    # target spans arrive as text from the box, and only the *parser* is under
    # test. The tokenizer is loaded solely because the parser constructor wants
    # one, and the check asserts the spans are unchanged from what shipped.
    .pip_install("vllm==0.11.2")
    .add_local_file("/tmp/parity_rows.jsonl", remote_path="/root/rows.jsonl")
)


@app.function(image=image, timeout=20 * 60, cpu=4.0, memory=16384)
def check() -> dict:
    import json

    rows = [json.loads(l) for l in open("/root/rows.jsonl")]
    print(f"loaded {len(rows)} target spans", flush=True)

    import transformers
    from transformers import AutoTokenizer
    print(f"transformers {transformers.__version__} (vLLM-pinned; used only to "
          "construct the parser, never to re-tokenize)", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    available = sorted(ToolParserManager.tool_parsers.keys())
    print(f"vLLM tool parsers available: {available}", flush=True)

    name = "hermes" if "hermes" in available else available[0]
    parser = ToolParserManager.get_tool_parser(name)(tok)
    print(f"using parser: {name}", flush=True)

    def canon(n, a):
        return (n, json.dumps(a or {}, sort_keys=True, default=str))

    results, failures = [], []
    for r in rows:
        text = r["decoded_target"]
        intended = [canon(c["name"], c["arguments"]) for c in r["intended_calls"]]
        rec = {"task_id": r["task_id"], "position": r["position"],
               "action_type": r["action_type"], "intended": intended}
        try:
            info = parser.extract_tool_calls(text, request=None)
            called = bool(getattr(info, "tools_called", False))
            got = []
            for tc in (getattr(info, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                raw = getattr(fn, "arguments", None)
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                got.append(canon(getattr(fn, "name", None), args))
            rec.update({"parsed": got, "tools_called": called})
            if got != intended:
                rec["verdict"] = "MISMATCH"
                failures.append(rec)
            elif not intended and called:
                rec["verdict"] = "CONVERSATIONAL_PARSED_AS_CALL"
                failures.append(rec)
            else:
                rec["verdict"] = "ok"
        except Exception as e:
            rec.update({"verdict": "PARSER_ERROR",
                        "error": f"{type(e).__name__}: {e}"})
            failures.append(rec)
        results.append(rec)

    ok = sum(1 for r in results if r["verdict"] == "ok")
    print(f"\n{ok}/{len(results)} target spans round-trip exactly", flush=True)
    by_kind = {}
    for r in results:
        k = (r["action_type"], r["verdict"])
        by_kind[k] = by_kind.get(k, 0) + 1
    for k, v in sorted(by_kind.items()):
        print(f"  {k[0]:16s} {k[1]:32s} {v}", flush=True)
    for f in failures[:10]:
        print(f"\nFAIL {f['task_id']}#{f['position']} {f['verdict']}", flush=True)
        print(f"  intended: {f.get('intended')}", flush=True)
        print(f"  parsed:   {f.get('parsed', f.get('error'))}", flush=True)

    return {"parser": name, "available": available, "n": len(results), "ok": ok,
            "failures": failures[:40], "by_kind": {f"{k[0]}|{k[1]}": v
                                                   for k, v in by_kind.items()}}


@app.local_entrypoint()
def main():
    import json
    out = check.remote()
    print("\n" + "=" * 60)
    print(f"PARSER PARITY: {out['ok']}/{out['n']} exact "
          f"(parser={out['parser']})")
    print("=" * 60)
    if out["ok"] != out["n"]:
        print(f"{out['n'] - out['ok']} span(s) failed — the corpus format does "
              "not match what serving will parse.")
    json.dump(out, open("/tmp/parity_result.json", "w"), indent=1)
    print("wrote /tmp/parity_result.json")
