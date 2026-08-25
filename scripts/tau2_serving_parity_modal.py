"""Prove the corpus matches the SERVING stack, not just that it is well formed.

vLLM 0.11.2 pins `transformers>=4.56,<5` (vllm#30466); this repository pins
`transformers==5.5.3` everywhere it trains. So training and serving cannot share
one environment, and the chat template -- which lives in transformers and is
what rendered every row -- may not be the same code on both sides.

That makes two independent questions, and passing the second says nothing about
the first:

  PROMPT PARITY   re-render the frozen semantic messages inside the serving
                  image and compare text and token ids against the frozen
                  `rows.tokenized.jsonl`. A template that drifted between
                  transformers versions silently invalidates every row.

  PARSER PARITY   feed the frozen decoded target spans to the tool parser vLLM
                  actually runs and compare structured values.

A corpus can pass parser parity while failing prompt parity: the parser only
ever sees the completion. Both are reported separately.

CPU only -- no `gpu=` on the function, so no GPU is allocated. Modal still bills
CPU-seconds (~$0.047/core-hour), which for this is cents, not zero.

    modal run scripts/tau2_serving_parity_modal.py
"""
from __future__ import annotations

import modal

app = modal.App("tau2-serving-parity")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # No transformers pin: pip resolves whatever vLLM requires, which is the
    # point. We are measuring what the serving stack does, not imposing the
    # training stack on it.
    .pip_install("vllm==0.11.2")
    .add_local_file("/tmp/parity_input.json", remote_path="/root/input.json")
)


@app.function(image=image, timeout=30 * 60, cpu=4.0, memory=16384)
def check() -> dict:
    import json

    import transformers
    from transformers import AutoTokenizer

    data = json.load(open("/root/input.json"))
    print(f"serving-side transformers {transformers.__version__}", flush=True)
    print(f"corpus-side  transformers {data['built_with_transformers']}", flush=True)

    tok = AutoTokenizer.from_pretrained(data["model"], trust_remote_code=True)
    out: dict = {
        "serving_transformers": transformers.__version__,
        "corpus_transformers": data["built_with_transformers"],
        "tokenizer_class": type(tok).__name__,
    }

    # ---------- 1. PROMPT PARITY ----------
    print("\n=== PROMPT PARITY: re-render frozen messages ===", flush=True)
    prompt_fail, prompt_ok = [], 0
    for row in data["prompt_rows"]:
        messages = row["messages"]
        try:
            ids = tok.apply_chat_template(
                messages, tools=data["tools"], tokenize=True,
                add_generation_prompt=False, enable_thinking=True,
            )
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = list(ids)
        except Exception as e:
            prompt_fail.append({"task_id": row["task_id"],
                                "position": row["position"],
                                "reason": "render_error",
                                "detail": f"{type(e).__name__}: {e}"})
            continue

        want = row["input_ids"]
        if ids == want:
            prompt_ok += 1
            continue
        diverge = next((i for i in range(min(len(ids), len(want)))
                        if ids[i] != want[i]), min(len(ids), len(want)))
        prompt_fail.append({
            "task_id": row["task_id"], "position": row["position"],
            "reason": "token_mismatch",
            "serving_len": len(ids), "corpus_len": len(want),
            "first_divergence": diverge,
            "serving_context": tok.decode(ids[max(0, diverge - 20):diverge + 20]),
            "corpus_context": tok.decode(want[max(0, diverge - 20):diverge + 20]),
        })
    out["prompt_parity"] = {"n": len(data["prompt_rows"]), "ok": prompt_ok,
                            "failures": prompt_fail[:12]}
    print(f"  {prompt_ok}/{len(data['prompt_rows'])} rows render identically",
          flush=True)
    for f in prompt_fail[:5]:
        print(f"  MISMATCH {f['task_id']}#{f['position']}: {f['reason']} "
              f"len {f.get('serving_len')} vs {f.get('corpus_len')} "
              f"@tok {f.get('first_divergence')}", flush=True)
        if f.get("serving_context"):
            print(f"    serving: {f['serving_context']!r}", flush=True)
            print(f"    corpus : {f['corpus_context']!r}", flush=True)

    # ---------- 2. PARSER PARITY ----------
    print("\n=== PARSER PARITY: frozen targets through the serving parser ===",
          flush=True)
    # The registry populates as each parser module is imported; importing the
    # package alone leaves `tool_parsers` empty.
    # Import the concrete parser by name. The registry-walk approach kept
    # reporting an empty registry with no error, which is exactly the kind of
    # silent failure worth replacing with a real traceback.
    from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import (
        Hermes2ProToolParser,
    )
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager

    direct = Hermes2ProToolParser
    available = sorted(ToolParserManager.tool_parsers.keys())
    print(f"  imported Hermes2ProToolParser directly", flush=True)
    print(f"  registered parsers ({len(available)}): {available}", flush=True)
    if available:
        name = ("hermes" if "hermes" in available else
                next((a for a in available if "qwen" in a.lower()), available[0]))
        parser = ToolParserManager.get_tool_parser(name)(tok)
    elif direct is not None:
        name = direct.__name__
        parser = direct(tok)
    else:
        out["parser_parity"] = {"error": "no tool parser could be constructed",
                                "n": len(data["target_spans"]), "ok": 0,
                                "parser": None, "failures": []}
        print("  NO PARSER AVAILABLE — parser parity not established", flush=True)
        return out
    print(f"  using parser: {name}", flush=True)

    def canon(n, a):
        return (n, json.dumps(a or {}, sort_keys=True, default=str))

    parse_fail, parse_ok = [], 0
    for r in data["target_spans"]:
        intended = [canon(c["name"], c["arguments"]) for c in r["intended_calls"]]
        rec = {"task_id": r["task_id"], "position": r["position"],
               "action_type": r["action_type"], "intended": intended}
        try:
            info = parser.extract_tool_calls(r["decoded_target"], request=None)
            got = []
            for tc in (getattr(info, "tool_calls", None) or []):
                fn = getattr(tc, "function", None)
                raw = getattr(fn, "arguments", None)
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                got.append(canon(getattr(fn, "name", None), args))
            rec["parsed"] = got
            rec["tools_called"] = bool(getattr(info, "tools_called", False))
            if got != intended:
                rec["verdict"] = "MISMATCH"
                parse_fail.append(rec)
            elif not intended and rec["tools_called"]:
                rec["verdict"] = "CONVERSATIONAL_PARSED_AS_CALL"
                parse_fail.append(rec)
            else:
                parse_ok += 1
        except Exception as e:
            rec["verdict"] = "PARSER_ERROR"
            rec["error"] = f"{type(e).__name__}: {e}"
            parse_fail.append(rec)
    out["parser_parity"] = {"parser": name, "n": len(data["target_spans"]),
                            "ok": parse_ok, "failures": parse_fail[:12]}
    print(f"  {parse_ok}/{len(data['target_spans'])} targets round-trip exactly",
          flush=True)
    for f in parse_fail[:5]:
        print(f"  FAIL {f['task_id']}#{f['position']} {f['verdict']}", flush=True)
        print(f"    intended: {f.get('intended')}", flush=True)
        print(f"    parsed:   {f.get('parsed', f.get('error'))}", flush=True)

    return out


@app.local_entrypoint()
def main():
    import json

    out = check.remote()
    pp = out["prompt_parity"]
    qq = out.get("parser_parity") or {}
    print("\n" + "=" * 66)
    print(f"transformers: serving {out['serving_transformers']} vs "
          f"corpus {out['corpus_transformers']}")
    print(f"PROMPT PARITY: {pp['ok']}/{pp['n']} rows render identically")
    if qq.get("parser"):
        print(f"PARSER PARITY: {qq['ok']}/{qq['n']} targets parse identically "
              f"(parser={qq['parser']})")
    else:
        print(f"PARSER PARITY: NOT ESTABLISHED — {qq.get('error', 'no result')}")
    print("=" * 66)
    if not qq.get("parser"):
        print("\nParser parity could not run in this image. Prompt parity is "
              "still meaningful on its own, but the serving parser check "
              "remains open.")
    elif pp["ok"] != pp["n"]:
        print("\nBLOCKER: the serving stack does not reproduce the corpus's "
              "prompts. The model would be trained on renders it will never "
              "see at inference. Do not run the GPU probe.")
    elif qq["ok"] != qq["n"]:
        print("\nBLOCKER: targets do not survive the serving parser.")
    else:
        print("\nBoth parities clean.")
    json.dump(out, open("/tmp/parity_result.json", "w"), indent=1)
    print("wrote /tmp/parity_result.json")
