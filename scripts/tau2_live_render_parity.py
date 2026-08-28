"""Prove the live reasoning prompt differs only by the empty-think wrapper.

Why this is a gate and not a nicety
-----------------------------------
The live agent rebuilds the prompt from Tau2's conversation state. The corpus
built its prompts from stored semantic rows. Those two paths must produce
byte-identical token ids, or the student is trained on a state it was never
sampled in.

Two defects in the *corpus* build were caught by exactly this comparison and by
nothing else:

1. `canonical_messages` omitted the system policy entirely.
2. Tools were attached to the prefix object but never reached the renderer,
   because `encoding_dsv4` reads `msg["tools"]` off the system turn.

**Neither is caught by any hash.** Both produce a finite loss, a successful
alignment, and clean logs. The live path can reintroduce both -- it constructs
the same head from a different source -- so it gets the same gate.

`TAU2-REOPD-PLAN.md` §5 records the corpus-side result: 289/289 prefixes
re-render exactly. This asserts the same property for
`live_agent.render_prompt_ids`, by replaying each frozen prefix's semantic
history through the live rendering path. The frozen action-only prompt must be
exactly ``live_ids + empty_think_wrapper_ids``. Equality without that explicit
suffix would suppress reasoning; any other difference is semantic skew.

    python3 scripts/tau2_live_render_parity.py \
        --artifacts /data/tau2/artifacts_16384 \
        --simulations /data/tau2/data/simulations

Exit code is non-zero on any mismatch. Run it before any episode is paid for.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--simulations", default="/data/tau2/data/simulations")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N prefixes (0 = all)")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    from vektori_trace.dataset import _think_wrapper_ids
    from vektori_trace.tau2.c30_loader import (
        load_c30_prefixes,
        recover_system_policy,
    )
    from vektori_trace.tau2.live_agent import render_prompt_ids

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    policy, policy_report = recover_system_policy(
        a.artifacts, simulations_dir=a.simulations
    )
    sha = policy_report.get("policy_sha256") or "?"
    print(f"policy sha256 {sha[:16]} ({len(policy)} chars)")

    prefixes, report = load_c30_prefixes(
        a.artifacts, system_policy=policy, domain=a.domain
    )
    print(f"loaded {len(prefixes)} prefixes "
          f"(manifest {report.get('prefix_manifest_hash')})")

    rows = prefixes[: a.limit] if a.limit else prefixes

    n_ok = 0
    failures: list[dict] = []
    for p in rows:
        # `canonical_messages` is `[system+tools] + history`; the frozen ids
        # were rendered from that plus a generation prompt. The live agent
        # renders the identical structure from Tau2 state, so feeding the
        # corpus's own messages through the *live* function is what isolates
        # the renderer from the message-construction path.
        got = render_prompt_ids(tok, p.canonical_messages, p.tools)
        want = list(p.prompt_token_ids)
        wrapper = list(_think_wrapper_ids(tok))
        if got + wrapper == want and got[-len(wrapper):] != wrapper:
            n_ok += 1
            continue
        diverge = next(
            (i for i in range(min(len(got), len(want))) if got[i] != want[i]),
            min(len(got), len(want)),
        )
        failures.append({
            "prefix_id": p.prefix_id,
            "task_id": p.task_id,
            "position": p.position,
            "n_got": len(got),
            "n_want": len(want),
            "diverge_at": diverge,
            "got_tail": tok.decode(got[diverge:diverge + 24]),
            "want_tail": tok.decode(want[diverge:diverge + 24]),
        })

    print(
        f"\n{n_ok}/{len(rows)} live prefixes match the frozen semantic head "
        "with only the intentional empty-think suffix removed"
    )
    for f in failures[:10]:
        print(f"\nFAIL {f['prefix_id']} (task {f['task_id']} pos {f['position']})")
        print(f"  lengths  live={f['n_got']} frozen={f['n_want']}")
        print(f"  diverge at token {f['diverge_at']}")
        print(f"  live  -> {f['got_tail']!r}")
        print(f"  frozen-> {f['want_tail']!r}")

    out = {
        "n": len(rows),
        "ok": n_ok,
        "prefix_manifest_hash": report.get("prefix_manifest_hash"),
        "policy_sha256": policy_report.get("policy_sha256"),
        "failures": failures[:40],
    }
    with open("/tmp/live_render_parity.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote /tmp/live_render_parity.json")

    if n_ok != len(rows):
        print(
            f"\nREFUSED: {len(rows) - n_ok} prefix(es) do not re-render. The "
            "live reasoning boundary differs from the frozen action-only "
            "boundary by more than the required empty-think suffix."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
