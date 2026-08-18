"""Run the frozen Phase 7 suites against every checkpoint on one endpoint.

No shim. The point of the exercise is what the adapter emits on its own, so
nothing here repairs, reformats or retries a completion — a malformed response
is the measurement.

One endpoint hosts all seven checkpoints as seven LoRA modules over a single
base load, because the base weights are ~27.6 GiB and paying that seven times
would cost more than the whole rest of the evaluation. Requests select a
checkpoint by `model`.

The base model is served in **BF16** while the corrective run trained against an
**NF4** base (`docs/SFT-REPAIR-PLAN.md` Phase 5 — the BF16 arm was never run
under chunked loss and does not fit in 80 GiB). Evaluating through anything but
the real serving stack would certify a configuration that will never be
deployed, so this driver talks to vLLM over the OpenAI API exactly as harbor
will.

    python scripts/phase7_eval.py \
        --manifest /data/phase7/manifest.json \
        --api-base https://...modal.run/v1 \
        --checkpoints ck10=...-ck10 ck20=...-ck20 \
        --out /data/phase7/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.evaluate.phase7 import (
    SELECTION_GATES,
    SELECTION_SUITE,
    clears,
    grade,
    select_checkpoint,
    summarize,
)

# The plan's greedy suite. `enable_thinking` is pinned off because the dataset
# pins it off; a reasoning block would be off-protocol at generation time and
# the gates would (correctly) fail it, but the mismatch would be ours.
GREEDY = {"temperature": 0.0, "max_tokens": 512}
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def load_corpus(root: str, cache: dict[str, list]) -> list[dict]:
    if root not in cache:
        cache[root] = [
            json.loads(ln)
            for ln in (Path(root) / "sft_repaired.jsonl").read_text().splitlines()
            if ln.strip()
        ]
    return cache[root]


def load_prefix_messages(entry: dict[str, Any], cache: dict[str, list]) -> list[dict]:
    """The conversation before the reference action, byte-identical to training.

    Read straight out of the corpus the model trained on rather than rebuilt
    here: a second renderer is a second chance to differ from what the adapter
    actually saw.
    """
    seg = load_corpus(entry["corpus_root"], cache)[entry["line_no"]]
    return seg["messages"][: entry["message_index"]]


def complete(
    api_base: str,
    model: str,
    messages: list[dict],
    *,
    timeout: float,
    seed: int | None = None,
    temperature: float | None = None,
    max_tokens: int = 512,
    retries: int = 2,
) -> tuple[str, str | None, str | None]:
    """(text, finish_reason, error). A transport failure is never a model failure.

    Retried, because an ungraded prefix now blocks its checkpoint from being
    selected — which is correct (silence must not score as success) but makes a
    single dropped request expensive. Retrying transport is not retrying the
    model: the completion itself is never re-rolled to get a better answer.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": GREEDY["temperature"] if temperature is None else temperature,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = "no attempt"
    payload = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            body_text = e.read()[:400].decode(errors="replace")
            last = f"HTTP {e.code}: {body_text}"
            # A 4xx is the request being wrong, not the network being flaky.
            # Retrying it just repeats the same mistake more expensively.
            if 400 <= e.code < 500:
                return "", None, last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    if payload is None:
        return "", None, last
    choice = (payload.get("choices") or [{}])[0]
    return (
        (choice.get("message") or {}).get("content") or "",
        choice.get("finish_reason"),
        None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label=served_model_name pairs, in training order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sampled-seeds", type=int, default=0,
                    help="temperature-0.7 seeds per prefix, catching a format "
                         "that holds greedily and collapses under sampling. 0 "
                         "skips the sampled suite; run it only after a "
                         "checkpoint passes greedily.")
    ap.add_argument("--sampled-temperature", type=float, default=0.7)
    ap.add_argument("--strategy", choices=("staged", "all"), default="staged",
                    help="staged: smoke the last checkpoint, then walk the "
                         "checkpoints in training order and stop at the first "
                         "that passes. Identical answer to `all` — selection is "
                         "`earliest passing`, so every checkpoint after the "
                         "winner is a generation whose result is discarded.")
    ap.add_argument("--retries", type=int, default=2,
                    help="transport retries per request (default 2)")
    ap.add_argument("--abort-on-smoke", action="store_true",
                    help="let a failed smoke on the last checkpoint stop the "
                         "run. Off by default: the smoke is a heuristic, and "
                         "loss fell for all 63 steps on 34 tasks, so the last "
                         "checkpoint is also the most overfit — a middle one "
                         "can emit the protocol where it does not. Being wrong "
                         "here abandons a working repair to save ~20 minutes.")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the endpoint preflight. Do not: it is the only "
                         "thing that verifies enable_thinking is actually "
                         "pinned at eval time.")
    ap.add_argument("--suites", nargs="*", default=None,
                    help="restrict to these suites. The control suite only "
                         "interprets a gap between the other two; if there is "
                         "no gap it was never needed.")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    prefixes = manifest["prefixes"]
    if args.suites:
        prefixes = [p for p in prefixes if p["suite"] in set(args.suites)]
        if not prefixes:
            print(f"no prefixes in suites {args.suites}", file=sys.stderr)
            return 1
    order = [c.split("=", 1)[0] for c in args.checkpoints]
    models = dict(c.split("=", 1) for c in args.checkpoints)
    available_suites = {p["suite"] for p in prefixes}
    cache: dict[str, list] = {}

    def jobs_for(label: str) -> list[dict[str, Any]]:
        out = []
        for entry in prefixes:
            out.append({"label": label, "entry": entry, "seed": None})
            for s in range(args.sampled_seeds):
                out.append({"label": label, "entry": entry, "seed": s})
        return out

    per_ck = len(jobs_for(order[0]))
    print(
        f"{len(prefixes)} prefixes"
        f"{f' x (1 greedy + {args.sampled_seeds} sampled)' if args.sampled_seeds else ''}"
        f" = {per_ck} generations per checkpoint; "
        f"{'staged' if args.strategy == 'staged' else f'all {len(order)} checkpoints'}"
        f" (worst case {per_ck * len(order)})"
    )

    # Fail before the first generation, not during. A corpus that cannot be
    # read is a broken manifest, and discovering it mid-sweep throws away every
    # completion bought so far — on an endpoint billed by the minute.
    for root in sorted({p["corpus_root"] for p in prefixes}):
        path = Path(root) / "sft_repaired.jsonl"
        if not path.is_file():
            print(f"manifest points at a corpus that does not exist: {path}",
                  file=sys.stderr)
            return 1
    for entry in prefixes:
        seg_count = len(load_corpus(entry["corpus_root"], cache))
        if entry["line_no"] >= seg_count:
            print(f"prefix {entry['prefix_id']} wants line {entry['line_no']} of "
                  f"{entry['corpus_root']} which has {seg_count} segments",
                  file=sys.stderr)
            return 1

    results = []
    infra_failures = []
    started = time.time()

    if not args.no_preflight:
        # `chat_template_kwargs` is a vLLM extension, not core OpenAI. If the
        # server ignores it or rejects it, `enable_thinking=False` is not pinned
        # and every completion could carry a reasoning block the gates would
        # then fail — a mismatch that would be ours, not the checkpoint's.
        # Cheapest possible check: one 16-token request per registered model.
        print("preflight:")
        bad = False
        for label in order:
            text, _finish, err = complete(
                args.api_base,
                models[label],
                [{"role": "user", "content": "Reply with the single word: ok"}],
                timeout=args.timeout,
                max_tokens=16,
                retries=args.retries,
            )
            thinking = "<think>" in text
            ok = err is None and not thinking
            bad = bad or not ok
            why = err or ("emitted <think> despite enable_thinking=False"
                          if thinking else "ok")
            print(f"  {label:8} {models[label]:44} {'OK' if ok else 'FAIL'}  {why}")
        if bad:
            print(
                "\npreflight failed — every checkpoint must be reachable and "
                "must honour enable_thinking=False before the sweep is worth "
                "running. Check that the endpoint registered each adapter "
                "(--adapter NAME=PATH) and started with --max-lora-rank 32.",
                file=sys.stderr,
            )
            return 4
        print()

    def run(job: dict[str, Any]):
        entry = job["entry"]
        try:
            return _run(job)
        except Exception as exc:
            # One malformed prefix must not discard every completion already
            # paid for. It is recorded as ungraded, which blocks its checkpoint
            # from being selected rather than quietly excusing it.
            return None, {
                "prefix_id": entry["prefix_id"],
                "checkpoint": job["label"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _run(job: dict[str, Any]):
        entry = job["entry"]
        messages = load_prefix_messages(entry, cache)
        sampled = job["seed"] is not None
        text, finish, err = complete(
            args.api_base,
            models[job["label"]],
            messages,
            timeout=args.timeout,
            seed=job["seed"],
            temperature=args.sampled_temperature if sampled else None,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        if err is not None:
            return None, {
                "prefix_id": entry["prefix_id"],
                "checkpoint": job["label"],
                "error": err,
            }
        res = grade(
            text,
            prefix_id=entry["prefix_id"],
            checkpoint=job["label"],
            category=entry["category"],
            suite=entry["suite"] + ("-sampled" if sampled else ""),
            finish_reason=finish,
            git_present=entry.get("git_present", True),
        )
        row = asdict(res)
        row["seed"] = job["seed"]
        row["task"] = entry["task"]
        row["repo"] = entry["repo"]
        row["reference"] = entry["reference"]
        return res, row

    def run_checkpoint(label: str, pool: ThreadPoolExecutor) -> list:
        batch = jobs_for(label)
        got = []
        for i, (res, row) in enumerate(pool.map(run, batch), 1):
            if res is None:
                infra_failures.append(row)
                print(f"  [{label} {i}/{len(batch)}] INFRA {row['prefix_id']}: "
                      f"{row['error'][:120]}")
                continue
            results.append(res)
            got.append(res)
            print(
                f"  [{label} {i}/{len(batch)}] {res.suite:22} {res.category:22} "
                f"{'PASS' if res.passed else 'FAIL ' + ','.join(res.failed_gates)}"
            )
        return got

    # The staged stop must ask the *same* question selection asks, on the same
    # suite and the same prefix set. Reading "every result we happen to hold"
    # instead would keep generating past the real winner whenever a non-selection
    # suite fails, and — worse — would stop early on a partial set when the
    # endpoint dropped a request.
    sel_suite = SELECTION_SUITE if SELECTION_SUITE in available_suites else (
        "acquisition" if "acquisition" in available_suites else sorted(available_suites)[0]
    )
    if sel_suite != SELECTION_SUITE:
        print(
            f"\nNOTE: selecting on the {sel_suite!r} suite — {SELECTION_SUITE!r} "
            "is not in this manifest. Acquisition prefixes are training inputs, "
            "so a pass there proves the protocol was acquired, NOT that it "
            "transfers. Do not read a held-out claim off this run.",
            file=sys.stderr,
        )
    sel_ids = [p["prefix_id"] for p in prefixes if p["suite"] == sel_suite]
    print(f"selecting on suite {sel_suite!r} ({len(sel_ids)} prefixes), "
          f"gates {SELECTION_GATES}\n")

    def checkpoint_clears(label: str) -> tuple[bool, dict]:
        return clears(
            results,
            checkpoint=label,
            suite=sel_suite,
            require=SELECTION_GATES,
            expected_prefix_ids=sel_ids,
        )

    evaluated: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.strategy == "all":
            for label in order:
                run_checkpoint(label, pool)
                evaluated.append(label)
        else:
            # Smoke the last checkpoint. It has seen the most training, so if it
            # cannot emit the protocol nothing earlier is going to. This is a
            # heuristic, not a proof — the run could have overfit past a working
            # middle checkpoint — but it aborts a dead repair for the price of
            # one checkpoint instead of seven, and `--strategy all` tests them
            # all when the difference matters. The bar here is deliberately
            # lower than selection's: *any* prefix clearing the format gates is
            # enough to keep going.
            last = order[-1]
            smoke = run_checkpoint(last, pool)
            evaluated.append(last)
            alive = any(
                all(r.gates.get(g, False) for g in SELECTION_GATES)
                for r in smoke
                if r.suite == sel_suite
            )
            if not alive and args.abort_on_smoke:
                print(
                    f"\nSMOKE FAILED: {last} produced no completion on the "
                    f"{sel_suite!r} suite passing {SELECTION_GATES}. Stopping "
                    f"before the remaining {len(order) - 1} checkpoints because "
                    "--abort-on-smoke was passed. This is a heuristic, not "
                    "proof they cannot work.",
                    file=sys.stderr,
                )
            else:
                if not alive:
                    print(
                        f"\nSMOKE: {last} cleared {SELECTION_GATES} on no "
                        f"{sel_suite} prefix. Continuing anyway — it is the "
                        "most-trained checkpoint and therefore the most "
                        "overfit, so an earlier one can still emit the "
                        "protocol. ~20 minutes to rule that out is cheaper "
                        "than abandoning a working repair.",
                        file=sys.stderr,
                    )
                for label in order[:-1]:
                    run_checkpoint(label, pool)
                    evaluated.append(label)
                    ok, detail = checkpoint_clears(label)
                    if ok:
                        print(f"\n{label} clears {SELECTION_GATES} on all "
                              f"{detail['n_graded']} {sel_suite} prefixes — "
                              "stopping, it is the earliest.")
                        break
                    if detail.get("ungraded"):
                        print(f"  {label}: {len(detail['ungraded'])} prefix(es) "
                              "ungraded, cannot clear", file=sys.stderr)

    rows = [
        {**asdict(r), "failed_gates": r.failed_gates, "passed": r.passed}
        for r in results
    ]
    chosen, trace = select_checkpoint(
        results, order=order, suite=sel_suite, expected_prefix_ids=sel_ids
    )
    report = {
        "manifest": str(args.manifest),
        "manifest_sha256": (args.manifest.parent / "manifest_sha256.txt").read_text().strip()
        if (args.manifest.parent / "manifest_sha256.txt").exists()
        else None,
        "api_base": args.api_base,
        "checkpoint_order": order,
        "strategy": args.strategy,
        # Which checkpoints were actually generated for. A checkpoint absent
        # from this list was never tested, which is not the same as tested and
        # failed — `selection_trace` marks it "no results" for that reason.
        "checkpoints_evaluated": evaluated,
        "suites": sorted({p["suite"] for p in prefixes}),
        "models": models,
        "elapsed_sec": round(time.time() - started, 1),
        "n_generations": len(results) + len(infra_failures),
        # An endpoint that refused a request is not a checkpoint that failed a
        # gate. Recording these separately is the same correction that
        # `fallback_exitcode` needed in the pass@k reports.
        "infra_failures": infra_failures,
        "selection_suite": sel_suite,
        "selection_gates": list(SELECTION_GATES),
        "selection_prefix_ids": sel_ids,
        "selected_checkpoint": chosen,
        "selection_trace": trace,
        "summary": summarize(results),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\nelapsed {report['elapsed_sec']}s, "
          f"{len(infra_failures)} infra failures")
    for cell, v in report["summary"]["cells"].items():
        print(f"  {cell:34} {v['passed']}/{v['n']}")
    print(f"\nselected checkpoint: {chosen or 'NONE PASSED'}")
    if infra_failures:
        print("WARNING: infra failures present — rates are over what was "
              "gradeable, not over what was requested", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
