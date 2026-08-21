#!/usr/bin/env python3
"""One replay-prefix OPD update, end to end (plan §8).

    corpus -> 8 prefixes -> 4 ck75 actions each -> DeepSeek scores each
           -> chunk alignment + semantic-prior advantages
           -> one optimizer step -> v_replay

Every stage is an existing, tested module; this is the assembly. It is written
as one script rather than a library entry point because the run is a *single
event* with an approval attached to it, and the things worth getting right are
ordering and refusal, not reuse.

Three orderings that are not arbitrary
--------------------------------------
1. **Select and sample before scoring.** Teacher scoring is the paid step; a
   batch that would fail spread checks or arrive short should fail before it
   costs anything.
2. **Score everything before training.** §8.3 takes exactly one optimizer step,
   so a partially-scored batch cannot be trained on — the denominator would be
   silently wrong.
3. **Load `v0` last.** It is the largest allocation, and there is no reason to
   hold 14B of weights while waiting on HTTP.

Sampling needs a served ck75 (`scripts/serve_student.py`), and scoring needs
`FIREWORKS_API_KEY`. `--dry-run` does everything except spend: it selects,
renders, and reports what the run *would* cost, which is the number an approval
should be granted against.

    .venv/bin/python scripts/run_replay_opd.py --dry-run
    .venv/bin/python scripts/run_replay_opd.py --api-base "$STUDENT_API_BASE" \\
        --model "$STUDENT_MODEL" --adapter-path /adapters/... --out /data/replay-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CORPUS = ["/data/vektori-out/dsv4-corpus60", "/data/vektori-out/dsv4-corpus60-b"]
DEFAULT_TEACHER = "accounts/fireworks/models/deepseek-v4-flash-0731"
POLICY_VERSION = "ck75-v0"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post(url: str, payload: dict, timeout: float = 900.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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
            return e.code, {"raw": raw[:800]}
    except urllib.error.URLError as e:
        return 0, {"unreachable": str(e)}


class _Capture:
    """Duck-typed `CapturedCompletion` over a raw vLLM completion choice."""

    def __init__(self, body: dict, choice: dict):
        lp = choice.get("logprobs") or {}
        self.token_ids = choice.get("token_ids") or body.get("token_ids") or []
        self.prompt_token_ids = body.get("prompt_token_ids") or []
        self.logprobs = lp.get("token_logprobs")
        self.finish_reason = choice.get("finish_reason")
        self.text = choice.get("text")
        self.request_id = body.get("id")
        self.model = body.get("model")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def select_prefixes(args, student_tok=None) -> tuple[list, dict[str, Any]]:
    from vektori_trace.replay_corpus import (
        candidates_from_traces,
        corpus_report,
        load_corpus,
    )
    from vektori_trace.replay_select import select_replay_prefixes

    traces = []
    for root in args.corpus:
        traces.extend(load_corpus(Path(root), passing_only=True))
    if not traces:
        raise SystemExit(f"no passing traces under {args.corpus}")
    log(f"corpus: {len(traces)} passing traces")

    candidates = candidates_from_traces(traces, max_step=args.max_step)
    log(f"candidate replay states: {len(candidates)}")

    # Context-budget filter, *before* selection. 38% of the measured pool
    # cannot be sampled at all, and overflow is concentrated in late-stage
    # prefixes — so filtering after selection would re-draw from an unknown
    # subset, and filtering inside the sampling loop would kill the batch after
    # some actions were already paid for. Only a pre-filter lets the spread
    # constraints operate on states that are actually reachable.
    budget_report: dict[str, Any] = {"skipped": "no tokenizer supplied"}
    if student_tok is not None:
        from vektori_trace.dataset import turns_to_messages
        from vektori_trace.replay_context import filter_candidates_by_budget

        def _render(c):
            return student_tok.apply_chat_template(
                turns_to_messages(c.prefix_turns),
                tokenize=False,
                add_generation_prompt=True,
            )

        candidates, budget_report = filter_candidates_by_budget(
            candidates,
            _render,
            student_tok,
            max_new_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            progress=lambda i, n: log(f"  budget {i}/{n}") if i and i % 2000 == 0 else None,
        )
        log(
            f"context filter: {budget_report['n_fitting']}/{budget_report['n_candidates']} "
            f"fit (overflow {budget_report['overflow_rate']}); "
            f"tasks {budget_report['eligible_tasks_before']}->"
            f"{budget_report['eligible_tasks_after']}, "
            f"post-compaction {budget_report['post_compaction_before']}->"
            f"{budget_report['post_compaction_after']}"
        )
        if not candidates:
            raise SystemExit(
                "every candidate overflows the context window; nothing to sample"
            )

    chosen = select_replay_prefixes(
        candidates,
        n_prefixes=args.n_prefixes,
        require_post_compaction=args.require_post_compaction,
    )
    by_trace = {t.trace_id: t for t in traces}
    for p in chosen:
        log(f"  {p.task} {p.prefix_id} (step {p.step_index})")
    return chosen, {
        "traces": by_trace,
        "corpus": corpus_report(traces),
        "context_filter": budget_report,
    }


def sample_actions(prefixes, student_tok, args) -> tuple[list, dict[str, Any]]:
    """Four ck75 actions per prefix, with ids and behaviour logprobs."""
    from vektori_trace.dataset import turns_to_messages
    from vektori_trace.replay_sample import (
        sampled_action_from_capture,
        summarize_cap_hits,
    )

    from vektori_trace.replay_context import (
        assert_prefix_fits,
        measure_prefix,
        summarize_budgets,
    )

    actions, rendered, budgets = [], {}, []
    for prefix in prefixes:
        messages = turns_to_messages(prefix.prefix_turns)
        if not messages:
            raise SystemExit(f"{prefix.prefix_id}: prefix rendered to zero messages")
        rendered[prefix.prefix_id] = messages
        prompt = student_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # §4/§11: check the budget locally, before the request. A prefix that
        # overflows is dropped from the *front* by the server; sampling then
        # succeeds at a state this run cannot describe, with a finite loss and
        # every downstream assertion still passing.
        budget = measure_prefix(
            prefix.prefix_id,
            prompt,
            student_tok,
            max_new_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
        )
        budgets.append(budget)
        assert_prefix_fits(budget)
        for i in range(args.n_samples):
            status, body = post(
                args.api_base.rstrip("/") + "/completions",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "logprobs": 0,
                    "return_token_ids": True,
                },
            )
            if status != 200:
                raise SystemExit(
                    f"{prefix.prefix_id}#{i}: sampling failed HTTP {status}: "
                    f"{str(body)[:300]}"
                )
            cap = _Capture(body, (body.get("choices") or [{}])[0])
            # Refuses a missing logprob, a missing prompt id, or a cap hit —
            # all three are unrecoverable after sampling.
            actions.append(
                sampled_action_from_capture(
                    cap,
                    student_tok,
                    prefix_id=prefix.prefix_id,
                    sample_index=i,
                    policy_version=POLICY_VERSION,
                )
            )
        log(f"  sampled 4 at {prefix.prefix_id} ({budget.prefix_tokens} prefix tokens)")
    return actions, {
        "rendered": rendered,
        "cap": summarize_cap_hits(actions),
        "context": summarize_budgets(budgets),
    }


def score_actions(actions, rendered, args):
    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.replay_score import score_replay_batch
    from vektori_trace.vocab_bridge import load_tokenizer

    teacher_tok = load_tokenizer(args.teacher_tokenizer)
    pool = FireworksTeacherPool(model=args.teacher_model)
    scored, ledger = score_replay_batch(actions, rendered, teacher_tok, pool)
    log(
        f"scored {ledger['n_actions']} actions; "
        f"{ledger['teacher_input_tokens']:,} teacher input tokens "
        f"({ledger['repeated_prefix_tokens']:,} repeated prefix)"
    )
    return scored, ledger, pool.provenance()


def train(batch_inputs, args):
    from vektori_trace.replay_opd import run_replay_chunk_opd
    from vektori_trace.replay_train import (
        ReplayTrainConfig,
        build_optimizer,
        load_v0_for_training,
        make_optimizer_step,
    )

    prefixes, actions, scored, stored = batch_inputs
    cfg = ReplayTrainConfig(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=Path(args.out) / "v_replay",
        learning_rate=args.learning_rate,
        device=args.device,
    )
    log(f"loading v0 from {cfg.adapter_path}")
    model = load_v0_for_training(cfg)
    opt = build_optimizer(model, cfg)

    return run_replay_chunk_opd(
        prefixes,
        actions,
        scored,
        make_optimizer_step(model, opt, cfg),
        max_new_tokens=args.max_tokens,
        n_samples_per_prefix=args.n_samples,
        stored_teacher_actions=stored,
        selection_policy="stratified-diagnostic",
    )


def _build_manifest(args):
    """The §6.1 manifest for this invocation.

    Hashes are derived from `--adapter-path` rather than typed, so they cannot
    disagree with what is loaded; unobservable facts (Harbor revision, teacher
    precision) come from flags because guessing them would be a fake pin.
    """
    from vektori_trace.opd_manifest import build_run_manifest

    return build_run_manifest(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        task_corpus=",".join(str(c) for c in args.corpus),
        fireworks_model_id=args.teacher_model,
        student_tokenizer=args.student_tokenizer,
        tokenizer_dir=args.tokenizer_dir or args.adapter_path,
        harbor_revision=args.harbor_revision,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        teacher_serving_precision=args.teacher_precision,
        extra={"selection_policy": "stratified-diagnostic"},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", default=None)
    ap.add_argument("--n-prefixes", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-step", type=int, default=None,
                    help="cap the replay step; long prefixes eat the context window")
    ap.add_argument("--require-post-compaction", type=int, default=0,
                    help="§8.3 wants 2, but compaction boundaries are not "
                         "derived from the corpus yet (plan §15), so the pool "
                         "is empty and any value > 0 refuses. Default 0 means "
                         "this run makes no post-compaction claim.")
    # student
    ap.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE"))
    ap.add_argument("--model", default=os.environ.get("STUDENT_MODEL"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--student-tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter-path", default=None, help="the frozen v0 adapter")
    ap.add_argument("--max-tokens", type=int, default=9216)
    ap.add_argument("--max-model-len", type=int, default=40960,
                    help="serving context window; SOL-HANDOFF pins the L40S "
                         "server to 40960. Changing it here without changing "
                         "the server makes the budget check a lie.")
    ap.add_argument("--temperature", type=float, default=1.0)
    # teacher
    ap.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    ap.add_argument("--teacher-tokenizer", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--teacher-precision", default="fp8 (Fireworks serving default)",
                    help="recorded, not observed: Fireworks serves fp8 and a run "
                         "against a bf16 teacher is not comparable")
    # pins
    ap.add_argument("--tokenizer-dir", default=None,
                    help="directory whose tokenizer/chat-template files are hashed "
                         "for the pin; defaults to --adapter-path")
    ap.add_argument("--harbor-revision", default=os.environ.get("HARBOR_REVISION"),
                    help="§6.1 pin; this process cannot observe it, so it must be "
                         "supplied for a paid run")
    # training
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--device", default=os.environ.get("REPLAY_TRAIN_DEVICE"),
                    help='CUDA device for the optimizer step, e.g. "cuda" or '
                         '"cuda:0". Required for a real run and deliberately '
                         "not defaulted: the ck75 serving endpoint is a "
                         "different host, and an inferred device is how the "
                         "step silently ran on CPU.")
    ap.add_argument("--out", default="./vektori-out/opd-replay")
    ap.add_argument("--dry-run", action="store_true",
                    help="select and render only; spend nothing")
    args = ap.parse_args()
    args.corpus = args.corpus or DEFAULT_CORPUS

    from vektori_trace.chunk_opd import assert_token_cap_is_task_derived

    assert_token_cap_is_task_derived(args.max_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "n_prefixes": args.n_prefixes,
        "n_samples_per_prefix": args.n_samples,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
    }

    # The tokenizer is needed *before* selection now: the context-budget filter
    # measures rendered prompts, and it must run on the candidate pool rather
    # than on the eight already chosen.
    student_tok = None
    try:
        import transformers

        student_tok = transformers.AutoTokenizer.from_pretrained(
            args.student_tokenizer, trust_remote_code=True
        )
    except Exception as e:
        if not args.dry_run:
            print(
                f"cannot load student tokenizer {args.student_tokenizer!r}: {e}\n"
                "The context-budget filter cannot be skipped on a paid run: 38% of "
                "candidates overflow and would be sampled at silently truncated "
                "states.",
                file=sys.stderr,
            )
            return 2
        log(f"tokenizer unavailable ({type(e).__name__}); budget filter skipped")

    log("stage 1/4: selecting prefixes")
    prefixes, sel = select_prefixes(args, student_tok)
    report["corpus"] = sel["corpus"]
    report["context_filter"] = sel["context_filter"]
    report["post_compaction_coverage"] = {
        "claimed": False,
        "reason": "compaction boundaries are not derived from the corpus "
                  "(docs/OPD-MULTITURN-PLAN.md §15); every candidate is "
                  "post_compaction=False by construction",
        "required": args.require_post_compaction,
    }
    report["prefixes"] = [
        {"prefix_id": p.prefix_id, "task": p.task, "step": p.step_index,
         "post_compaction": p.post_compaction}
        # NB: post_compaction is False for every candidate until boundaries are
        # derived from the corpus (plan §15). Reported, not claimed.
        for p in prefixes
    ]

    if args.dry_run:
        from vektori_trace.dataset import turns_to_messages

        total = 0
        for p in prefixes:
            total += sum(len(m.get("content") or "") for m in turns_to_messages(p.prefix_turns))
        report["dry_run"] = True
        report["approx_prefix_chars"] = total

        # Measure the real context budget when a tokenizer is reachable. Chars
        # are not a cost estimate — an approval granted against them is granted
        # against the wrong number — and prefix overflow is the one failure the
        # dry run can catch for free that would otherwise corrupt a paid run
        # silently.
        try:
            from vektori_trace.replay_context import measure_prefix, summarize_budgets

            tok = student_tok
            if tok is None:
                raise RuntimeError("tokenizer unavailable")
            budgets = []
            for p in prefixes:
                prompt = tok.apply_chat_template(
                    turns_to_messages(p.prefix_turns),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                budgets.append(
                    measure_prefix(
                        p.prefix_id, prompt, tok,
                        max_new_tokens=args.max_tokens,
                        max_model_len=args.max_model_len,
                    )
                )
            report["context"] = summarize_budgets(budgets)
            report["context_per_prefix"] = [b.to_dict() for b in budgets]
            over = report["context"]["n_overflow"]
            log(
                f"context: max prefix {report['context']['max_prefix_tokens']} tokens, "
                f"min headroom {report['context']['min_headroom']}, "
                f"{over} would overflow"
            )
            if over:
                log(f"OVERFLOW would block sampling: {report['context']['overflow_prefix_ids']}")
        except Exception as e:  # tokenizer unreachable offline
            report["context_error"] = f"{type(e).__name__}: {e}"
            log(f"context budget not measured ({type(e).__name__}); chars only")
        # A dry run *reports* what is unpinned rather than refusing: knowing
        # which pins are missing is exactly what it is for.
        manifest = _build_manifest(args)
        report["manifest"] = manifest.to_dict()
        report["missing_pins"] = manifest.missing_pins()
        if manifest.missing_pins():
            log(f"UNPINNED (would block a paid run): {', '.join(manifest.missing_pins())}")
        report["projected_teacher_requests"] = args.n_prefixes * args.n_samples
        log(
            f"DRY RUN — would sample {args.n_prefixes * args.n_samples} actions "
            f"and issue as many teacher requests. Nothing spent."
        )
        (out / "dry_run.json").write_text(json.dumps(report, indent=2))
        log(f"report: {out / 'dry_run.json'}")
        return 0

    for need, flag in ((args.api_base, "--api-base"), (args.model, "--model"),
                       (args.adapter_path, "--adapter-path"),
                       (args.device, "--device")):
        if not need:
            print(f"{flag} is required for a real run", file=sys.stderr)
            return 2
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2

    # Validate the training device *now*, not at stage 4. `load_v0_for_training`
    # would catch a bad device anyway, but only after 32 samples and every
    # teacher call have been paid for. The check is free; the ordering is the
    # whole point.
    try:
        import torch

        if not str(args.device).startswith("cuda"):
            raise SystemExit(f"--device {args.device!r} is not a CUDA device")
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {args.device!r} but torch.cuda.is_available() is False — "
                "this host has no GPU. The ck75 serving endpoint is a separate "
                "machine and does not provide one to this process."
            )
        idx = torch.device(args.device).index or 0
        if idx >= torch.cuda.device_count():
            raise SystemExit(
                f"--device {args.device!r} but only {torch.cuda.device_count()} "
                "CUDA device(s) visible"
            )
        log(f"training device: {args.device} ({torch.cuda.get_device_name(idx)})")
    except ImportError:
        raise SystemExit("torch is not importable; this host cannot train")

    # §6.1 pin gate. Before sampling, before Fireworks, before weights: an
    # unpinned run produces numbers that cannot be attributed to a
    # configuration, which §11 makes a stop condition rather than a reporting
    # gap. `verify_reference_pins` re-hashes the vendored paper code, so an edit
    # to the reference implementation the loss claims to port fails here.
    from vektori_trace.opd_manifest import PinError, verify_reference_pins

    try:
        report["reference_sha256_observed"] = verify_reference_pins()
    except PinError as e:
        print(f"reference pin check failed: {e}", file=sys.stderr)
        return 2

    manifest = _build_manifest(args)
    report["manifest"] = manifest.to_dict()
    try:
        manifest.require_complete()
    except PinError as e:
        print(f"{e}", file=sys.stderr)
        print(
            "Fill the missing pins (adapter/tokenizer hashes are derived from "
            "--adapter-path; harbor revision via --harbor-revision) and re-run.",
            file=sys.stderr,
        )
        return 2
    log(f"pins complete; commit {manifest.vektori_trace_commit}")

    if student_tok is None:  # unreachable on a real run; guarded, not assumed
        print("student tokenizer was not loaded", file=sys.stderr)
        return 2

    log(f"stage 2/4: sampling {args.n_prefixes * args.n_samples} actions from ck75")
    actions, samp = sample_actions(prefixes, student_tok, args)
    report["cap"] = samp["cap"]
    report["context"] = samp["context"]

    log("stage 3/4: scoring with DeepSeek")
    scored, ledger, teacher_prov = score_actions(actions, samp["rendered"], args)
    report["teacher_ledger"] = ledger
    report["teacher"] = teacher_prov

    # The stored DeepSeek action at each replay step, so the driver can assert
    # ck75 did not simply reproduce it. Never a training target.
    stored = {}
    for p in prefixes:
        rec = sel["traces"].get(p.trace_id)
        if rec is not None:
            got = rec.stored_actions.get(p.step_index)
            if got:
                stored[p.prefix_id] = got

    # §10 per-example archival, *before* the optimizer step. Behaviour log
    # probabilities are `log pi_old` and cannot be recomputed once the policy
    # moves, and the paid half of the run is already complete here — a crash in
    # the training step must not cost the captures that were spent on.
    log("archiving examples (§10)")
    from vektori_trace.replay_archive import build_example_record, write_examples

    # Build the batch here and archive *from it*, so the advantages on disk are
    # the same objects the optimizer consumes. Recomputing them for the archive
    # would allow the record and the update to disagree, which is precisely the
    # kind of drift the archive exists to detect.
    from vektori_trace.replay_opd import build_replay_batch

    batch = build_replay_batch(prefixes, actions, scored, stored_teacher_actions=stored)
    by_prefix = {p.prefix_id: p for p in prefixes}
    by_action = {a.key: a for a in actions}
    records = []
    for key, adv in zip(batch.keys, batch.advantages, strict=True):
        act = by_action[key]
        pref = by_prefix[act.prefix_id]
        tb, tlp = scored.get(key, (None, None))
        records.append(
            build_example_record(
                prefix=pref,
                action=act,
                advantages=adv,
                canonical_messages=samp["rendered"].get(act.prefix_id),
                teacher_token_bytes=tb,
                teacher_logprobs=tlp,
                teacher_request=teacher_prov,
            )
        )
    report["archive"] = write_examples(out / "examples.jsonl", records)
    log(
        f"archived {report['archive']['n_examples']} examples "
        f"({report['archive']['bytes_written']:,} bytes); "
        f"max task share {report['archive']['max_task_share']}"
    )

    log("stage 4/4: one optimizer step")
    result = train((prefixes, actions, scored, stored), args)
    report["run"] = result

    (out / "replay_run.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"v_replay: {result['optimizer'].get('adapter_saved_to')}")
    log(f"report: {out / 'replay_run.json'}")
    log("TEAR DOWN the ck75 endpoint now — it is the only thing still billing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
