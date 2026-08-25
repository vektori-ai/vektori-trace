#!/usr/bin/env python3
"""Prove the recovered teacher context reproduces the frozen student prompts.

CPU only. No GPU, no Modal, no paid call. This is the cheapest and strongest
prerequisite to any ReOPD run: if all 289 prefixes re-render to their frozen
`prompt_token_ids` byte for byte, then the recovered policy, the tool schemas,
the template settings, the tokenizer and the action boundary *jointly* reproduce
the exact context the student was trained and sampled under.

Why it has to run before the controller
---------------------------------------
`rows.semantic.jsonl` stores only `decision.prompt`. The student's ids were
rendered from `[system + tools] + prompt + [target]`. Both halves of that head
have to be reconstructed at scoring time, and both were wrong in this module's
first two drafts -- once by omitting the policy, once by attaching the tools
where the DeepSeek renderer never looks. Neither failure changes a single
downstream metric: alignment succeeds, the loss is finite, the logs look clean.

A hash check cannot catch this. Only re-rendering can.

    python scripts/tau2_reopd_parity.py --artifacts /data/tau2/artifacts_16384 \
        --simulations-dir /data/tau2/data/simulations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.tau2.c30_loader import (  # noqa: E402
    C30LoadError,
    assert_render_parity,
    load_c30_prefixes,
    recover_system_policy,
    selected_trace_hashes,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--simulations-dir", default="/data/tau2/data/simulations",
                    help="where the corpus's selected simulation files live")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B",
                    help="must be the pinned serving tokenizer")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None,
                    help="check only the first N prefixes (smoke use only; a "
                         "partial pass does not discharge the gate)")
    ap.add_argument("--policy-file", default=None,
                    help="use this policy text instead of recovering it")
    ap.add_argument("--out", default=None,
                    help="default: ./reopd_parity_report.json. Deliberately NOT "
                         "inside --artifacts: a frozen corpus directory should "
                         "be read-only, and writing a report into it would "
                         "change bytes that artifact_hashes.json pins.")
    a = ap.parse_args()

    art = a.artifacts
    print(f"artifacts {art}")

    # --- 1. the policy ----------------------------------------------------
    if a.policy_file:
        policy = Path(a.policy_file).read_text()
        import hashlib
        prep = {"policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                "policy_chars": len(policy), "source": a.policy_file}
    else:
        policy, prep = recover_system_policy(
            art, simulations_dir=a.simulations_dir
        )
    print(f"policy    {prep['policy_sha256'][:16]} "
          f"({prep['policy_chars']:,} chars, "
          f"{prep.get('n_tasks_agreeing', '?')} tasks agree)")

    # --- 2. the join ------------------------------------------------------
    prefixes, report = load_c30_prefixes(art, system_policy=policy)
    traces = selected_trace_hashes(art)
    print(f"prefixes  {report['n_prefixes']} over {report['n_tasks']} tasks "
          f"(manifest {report['prefix_manifest_hash']})")
    print(f"traces    {len(traces)} task->trace_hash records")
    missing_traces = [p.prefix_id for p in prefixes
                      if p.trace_id == p.task_id and p.task_id not in traces]
    if missing_traces:
        print(f"  WARNING {len(missing_traces)} prefixes have no recorded trace "
              f"hash; archive provenance will name the task id instead")

    tok_max = max(p.n_prompt_tokens for p in prefixes)
    print(f"prompts   max {tok_max:,} tokens")

    # --- 3. render parity -------------------------------------------------
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"tokenizer {a.tokenizer}")
    print(f"rendering {len(prefixes) if not a.limit else a.limit} prefixes ...")

    parity = assert_render_parity(
        prefixes, tokenizer, max_length=a.max_length, limit=a.limit
    )
    print(f"PARITY OK {parity['n_checked']} prefixes re-render exactly")

    out = a.out or "reopd_parity_report.json"
    payload = {
        "artifacts": art,
        "tokenizer": a.tokenizer,
        "max_length": a.max_length,
        "policy": prep,
        "corpus": report,
        "parity": parity,
        "n_trace_hashes": len(traces),
        "max_prompt_tokens": tok_max,
        "partial": bool(a.limit),
    }
    try:
        json.dump(payload, open(out, "w"), indent=1)
        print(f"wrote     {out}")
    except OSError as e:
        # The gate already passed; failing to file the receipt must not read as
        # a parity failure.
        print(f"PARITY PASSED but the report could not be written: {e}",
              file=sys.stderr)
        print(json.dumps(payload, indent=1))
        return 0

    if a.limit:
        print("\nNOTE partial run: the gate requires all prefixes, not a sample.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except C30LoadError as e:
        print(f"\nPARITY FAILED\n{e}", file=sys.stderr)
        raise SystemExit(2)
