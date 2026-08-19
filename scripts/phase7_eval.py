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
    SELECTION_CATEGORIES,
    SELECTION_GATES,
    SELECTION_SUITES,
    STAGE_A_BASELINE_ARTIFACT,
    STAGE_A_BASELINE_SHA256,
    clears,
    grade,
    select_checkpoint,
    select_stage_b_checkpoint,
    selection_prefix_ids,
    stage_b_clears,
    summarize,
    tripwire_prefix_ids,
)

# The plan's greedy suite. `enable_thinking` is pinned ON because the dataset is
# tokenized that way (`docs/SFT-SCRATCH-PLAN.md` step 2). Qwen3's template
# default is already thinking-on, so this pins a decision rather than changing
# behaviour — but an unpinned default is how a rollout and a sweep of the same
# checkpoint end up measuring different prompts.
GREEDY = {"temperature": 0.0, "max_tokens": 512}
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}


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


#: `<think>`, `\n\n`, `</think>`, `\n\n` — what `enable_thinking=False` appends to
#: the generation prompt. The preflight asserts exactly this difference.
THINK_WRAPPER_TOKENS = 4


def _prompt_tokens(
    api_base: str,
    model: str,
    messages: list[dict],
    *,
    timeout: float,
    retries: int,
    chat_template_kwargs: dict[str, Any],
) -> tuple[int, str | None]:
    """`usage.prompt_tokens` for one 1-token request. (count, error).

    Reads what the server rendered rather than what it replied, so the check
    does not depend on the model choosing to open a reasoning block.
    """
    _text, _finish, err, payload = complete(
        api_base, model, messages, timeout=timeout, max_tokens=1, retries=retries,
        chat_template_kwargs=chat_template_kwargs, return_payload=True,
    )
    if err is not None:
        return -1, err
    try:
        return int(payload["usage"]["prompt_tokens"]), None
    except (KeyError, TypeError, ValueError):
        return -1, "response carried no usage.prompt_tokens"


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
    chat_template_kwargs: dict[str, Any] | None = None,
    return_payload: bool = False,
) -> tuple[str, str | None, str | None] | tuple[str, str | None, str | None, Any]:
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
        "chat_template_kwargs": (
            CHAT_TEMPLATE_KWARGS if chat_template_kwargs is None else chat_template_kwargs
        ),
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
                return ("", None, last, None) if return_payload else ("", None, last)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    if payload is None:
        return ("", None, last, None) if return_payload else ("", None, last)
    choice = (payload.get("choices") or [{}])[0]
    out = (
        (choice.get("message") or {}).get("content") or "",
        choice.get("finish_reason"),
        None,
    )
    return (*out, payload) if return_payload else out


def main() -> int:
    # Line-buffer stdout. `print()` block-buffers in 4-8 KiB chunks the moment
    # stdout is a file rather than a terminal, so a sweep redirected to a log
    # writes nothing for its entire run and then everything at exit. A sweep you
    # cannot watch is a sweep you cannot tell from a hung one — this cost a live
    # eval its observability on 2026-08-18.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label=served_model_name pairs, in training order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="generation budget per prefix. 512 was v1's "
                         "figure, when the think block was empty and the "
                         "JSON started immediately. Thinking is on now "
                         "and step 1 masks the wrapper, so the model "
                         "reasons for real and 512 becomes a think "
                         "budget, not a JSON budget — at 512 every one "
                         "of ck84's 30 harbor_accepts failures had hit "
                         "the cap, and none finished inside it.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sampled-seeds", type=int, default=0,
                    help="temperature-0.7 seeds per prefix, catching a format "
                         "that holds greedily and collapses under sampling. 0 "
                         "skips the sampled suite; run it only after a "
                         "checkpoint passes greedily.")
    ap.add_argument("--sampled-temperature", type=float, default=0.7)
    ap.add_argument("--strategy", choices=("staged", "all"), default="staged",
                    help="staged: with stage-a, smoke the last checkpoint then "
                         "walk earlier checkpoints; with stage-b, walk in true "
                         "training order and stop at the first combined pass. "
                         "`all` evaluates every checkpoint.")
    ap.add_argument(
        "--selection-policy", choices=("stage-a", "stage-b"), default="stage-a",
        help="stage-b additionally requires first_edit > 0/3 and test_exec "
             "> 1/3. It evaluates checkpoints in true earliest-first order.",
    )
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
                         "thing that verifies chat_template_kwargs is actually "
                         "honoured at eval time.")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.out.with_suffix(".partial.jsonl")
    partial_fh = partial_path.open("w")
    print(f"streaming results to {partial_path} as they are graded")
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
        # `chat_template_kwargs` is a vLLM extension, not core OpenAI. A server
        # that rejects or ignores it renders with the template's own default,
        # and the sweep then measures a prompt nobody chose. The check has to be
        # differential: send the same message with thinking on and off and
        # require the outputs to *differ* in whether a reasoning block appears.
        # Asserting only "thinking on -> <think> present" would pass on a server
        # that drops the kwarg entirely, since the default is already on.
        print("preflight:")
        bad = False
        probe = [{"role": "user", "content": "Reply with the single word: ok"}]
        for label in order:
            # The distinguishing evidence is in the *prompt*, not the
            # completion. With thinking off the template appends
            # `<think>\n\n</think>\n\n` to the generation prompt; with it on the
            # prompt stops at `<|im_start|>assistant\n`. Those differ by exactly
            # four tokens, and vLLM reports the count it actually rendered. A
            # server that drops the kwarg renders one prompt twice and the
            # counts match — which no assertion about completion text can see,
            # because Qwen3's default is already thinking-on and a short reply
            # may open no block either way.
            on_ids, on_err = _prompt_tokens(
                args.api_base, models[label], probe, timeout=args.timeout,
                retries=args.retries, chat_template_kwargs={"enable_thinking": True},
            )
            off_ids, off_err = _prompt_tokens(
                args.api_base, models[label], probe, timeout=args.timeout,
                retries=args.retries, chat_template_kwargs={"enable_thinking": False},
            )
            err = on_err or off_err
            delta = None if err else off_ids - on_ids
            honoured = delta == THINK_WRAPPER_TOKENS
            ok = err is None and honoured
            bad = bad or not ok
            if err:
                why = err
            elif delta == 0:
                why = ("prompt length identical with thinking on and off — the "
                       "server is ignoring chat_template_kwargs")
            elif not honoured:
                why = (f"prompt grew by {delta} tokens with thinking off, "
                       f"expected {THINK_WRAPPER_TOKENS}")
            else:
                why = "ok"
            print(f"  {label:8} {models[label]:44} {'OK' if ok else 'FAIL'}  {why}")
        if bad:
            print(
                "\npreflight failed — every checkpoint must be reachable and the "
                "endpoint must honour chat_template_kwargs before the sweep is "
                "worth running, or the sweep measures a template nobody chose. "
                "Check that the endpoint registered each adapter "
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
            # Persist as we go. `results.json` is written once at the end, so a
            # sweep killed at prefix 40 of 540 used to leave nothing at all —
            # every completion, finish_reason and think_body it had already
            # graded died with the process. Twice, on 2026-08-18. One line per
            # result means a partial sweep is still evidence.
            if partial_fh is not None:
                partial_fh.write(json.dumps(
                    {**asdict(res), "failed_gates": res.failed_gates,
                     "passed": res.passed}
                ) + "\n")
                partial_fh.flush()
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
    sel_ids = selection_prefix_ids(prefixes)
    edit_ids = tripwire_prefix_ids(prefixes, "first_edit")
    test_ids = tripwire_prefix_ids(prefixes, "test_exec")
    missing_suites = sorted(set(SELECTION_SUITES) - available_suites)
    if missing_suites:
        print(
            f"\nNOTE: selection suites {missing_suites} are not in this "
            "manifest, so the selection set is smaller than the 45 the plan "
            "specifies. Acquisition prefixes are training inputs: a pass on "
            "them proves the protocol was acquired, NOT that it transfers. Do "
            "not read a held-out claim off a run missing 'generalization'.",
            file=sys.stderr,
        )
    if not sel_ids:
        print("no prefix in a selection category — nothing to select on",
              file=sys.stderr)
        return 1
    if args.selection_policy == "stage-b":
        if len(edit_ids) != 3 or len(test_ids) != 3:
            print(
                "Stage B selection requires exactly three first_edit and three "
                f"test_exec prefixes; found {len(edit_ids)} and {len(test_ids)}",
                file=sys.stderr,
            )
            return 1
        prefix_by_id = {p["prefix_id"]: p for p in prefixes}
        for category, ids in (("first_edit", edit_ids), ("test_exec", test_ids)):
            suites = {prefix_by_id[p]["suite"] for p in ids}
            if suites != set(SELECTION_SUITES):
                print(
                    f"Stage B {category} tripwires must contain one prefix per "
                    f"suite; found {sorted(suites)}",
                    file=sys.stderr,
                )
                return 1
    by_suite = {
        s: sum(1 for p in prefixes
               if p["prefix_id"] in set(sel_ids) and p["suite"] == s)
        for s in sorted(available_suites)
    }
    print(f"selecting with {args.selection_policy} policy on {len(sel_ids)} prefixes in categories "
          f"{SELECTION_CATEGORIES} across suites {by_suite}, "
          f"gates {SELECTION_GATES}\n")

    def checkpoint_clears(label: str) -> tuple[bool, dict]:
        if args.selection_policy == "stage-b":
            return stage_b_clears(
                results,
                checkpoint=label,
                format_prefix_ids=sel_ids,
                edit_prefix_ids=edit_ids,
                test_prefix_ids=test_ids,
            )
        return clears(
            results,
            checkpoint=label,
            require=SELECTION_GATES,
            expected_prefix_ids=sel_ids,
        )

    evaluated: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.strategy == "all":
            for label in order:
                run_checkpoint(label, pool)
                evaluated.append(label)
        elif args.selection_policy == "stage-b":
            # Stage B selection is genuinely earliest-first.  Do not pay for a
            # full ck93 smoke before testing ck25: ck93 cannot affect the answer
            # once an earlier checkpoint clears the combined rule.
            for label in order:
                run_checkpoint(label, pool)
                evaluated.append(label)
                ok, detail = checkpoint_clears(label)
                if ok:
                    print(
                        f"\n{label} clears Stage B format + behavior gates — "
                        "stopping, it is the earliest."
                    )
                    break
                ungraded = (
                    detail["format"].get("ungraded", [])
                    + detail["first_edit"].get("ungraded", [])
                    + detail["test_exec"].get("ungraded", [])
                )
                if ungraded:
                    print(f"  {label}: {len(ungraded)} required result(s) "
                          "ungraded, cannot clear", file=sys.stderr)
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
                if r.prefix_id in set(sel_ids)
            )
            if not alive and args.abort_on_smoke:
                print(
                    f"\nSMOKE FAILED: {last} produced no completion on the "
                    f"selection set passing {SELECTION_GATES}. Stopping "
                    f"before the remaining {len(order) - 1} checkpoints because "
                    "--abort-on-smoke was passed. This is a heuristic, not "
                    "proof they cannot work.",
                    file=sys.stderr,
                )
            else:
                if not alive:
                    print(
                        f"\nSMOKE: {last} cleared {SELECTION_GATES} on no "
                        f"selection prefix. Continuing anyway — it is the "
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
                              f"{detail['n_graded']} selection prefixes — "
                              "stopping, it is the earliest.")
                        break
                    if detail.get("ungraded"):
                        print(f"  {label}: {len(detail['ungraded'])} prefix(es) "
                              "ungraded, cannot clear", file=sys.stderr)

    rows = [
        {**asdict(r), "failed_gates": r.failed_gates, "passed": r.passed}
        for r in results
    ]
    if args.selection_policy == "stage-b":
        chosen, trace = select_stage_b_checkpoint(
            results,
            order=order,
            format_prefix_ids=sel_ids,
            edit_prefix_ids=edit_ids,
            test_prefix_ids=test_ids,
        )
    else:
        chosen, trace = select_checkpoint(
            results, order=order, expected_prefix_ids=sel_ids
        )
    report = {
        "manifest": str(args.manifest),
        "manifest_sha256": (args.manifest.parent / "manifest_sha256.txt").read_text().strip()
        if (args.manifest.parent / "manifest_sha256.txt").exists()
        else None,
        "api_base": args.api_base,
        "checkpoint_order": order,
        "strategy": args.strategy,
        "selection_policy": args.selection_policy,
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
        "selection_categories": list(SELECTION_CATEGORIES),
        "selection_suites": sorted(by_suite),
        "selection_gates": list(SELECTION_GATES),
        "selection_prefix_ids": sel_ids,
        "stage_b_behavior_prefix_ids": {
            "first_edit": edit_ids,
            "test_exec": test_ids,
        } if args.selection_policy == "stage-b" else None,
        "stage_b_baseline": {
            "first_edit": "0/3",
            "test_exec": "1/3",
            "comparison": "strictly_greater",
            "artifact": STAGE_A_BASELINE_ARTIFACT,
            "artifact_sha256": STAGE_A_BASELINE_SHA256,
        } if args.selection_policy == "stage-b" else None,
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
    if partial_fh is not None:
        partial_fh.close()
    print(f"\nselected checkpoint: {chosen or 'NONE PASSED'}")
    if infra_failures:
        print("WARNING: infra failures present — rates are over what was "
              "gradeable, not over what was requested", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
