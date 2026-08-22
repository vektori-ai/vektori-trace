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
import base64
import hashlib
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


def endpoint_models(api_base: str, timeout: float = 30.0) -> tuple[int, dict]:
    """Cheap readiness/model check that does not advance the sampler RNG."""
    url = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"raw": e.read().decode("utf-8", "replace")[:800]}
    except urllib.error.URLError as e:
        return 0, {"unreachable": str(e)}


def require_endpoint_model(api_base: str, model: str) -> None:
    """Fail before the 4,305-prefix census if the endpoint is unusable."""
    status, body = endpoint_models(api_base)
    if status != 200:
        raise SystemExit(
            f"student endpoint readiness failed HTTP {status}: {str(body)[:300]}"
        )
    advertised = {
        str(row.get("id")) for row in (body.get("data") or []) if isinstance(row, dict)
    }
    if model not in advertised:
        raise SystemExit(
            f"student endpoint does not advertise model {model!r}; "
            f"available={sorted(advertised)}"
        )


class _Capture:
    """Duck-typed `CapturedCompletion` over a raw vLLM completion choice."""

    def __init__(self, body: dict, choice: dict):
        lp = choice.get("logprobs") or {}
        self.token_ids = choice.get("token_ids") or body.get("token_ids") or []
        # vLLM returns prompt_token_ids inside the *choice*, not at the top
        # level of the body (verified against vllm-0.21.0). Reading only the
        # body left it empty, and the parity check correctly refused rather
        # than letting the run proceed with no proof of what the server
        # actually consumed. Choice first, body as a fallback for servers that
        # place it there.
        self.prompt_token_ids = (
            choice.get("prompt_token_ids") or body.get("prompt_token_ids") or []
        )
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


def sample_actions(prefixes, student_tok, args, on_capture=None) -> tuple[list, dict[str, Any]]:
    """Four ck75 actions per prefix, with ids and behaviour logprobs."""
    from vektori_trace.dataset import turns_to_messages
    from vektori_trace.replay_sample import (
        sampled_action_from_capture,
        summarize_cap_hits,
    )

    from vektori_trace.replay_context import (
        assert_prefix_fits,
        assert_prompt_ids_match,
        measure_prefix,
        summarize_budgets,
    )

    actions, rendered, budgets, parity = [], {}, [], []
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
            # The local budget proved our rendering fits; this proves the
            # server consumed that exact rendering. Checked on every sample,
            # because drift is not per-prefix and truncation is silent.
            local_ids = student_tok(prompt, add_special_tokens=False)["input_ids"]
            parity.append(
                assert_prompt_ids_match(
                    prefix.prefix_id, list(local_ids), cap.prompt_token_ids
                )
            )
            # Refuses a missing logprob, a missing prompt id, or a cap hit —
            # all three are unrecoverable after sampling.
            act = sampled_action_from_capture(
                cap,
                student_tok,
                prefix_id=prefix.prefix_id,
                sample_index=i,
                policy_version=POLICY_VERSION,
            )
            actions.append(act)
            # Append each capture as it lands rather than after all 32. A
            # crash at sample 30 otherwise discards 29 GPU-generated actions
            # whose behaviour logprobs cannot be recreated.
            if on_capture is not None:
                on_capture(act)
        log(f"  sampled 4 at {prefix.prefix_id} ({budget.prefix_tokens} prefix tokens)")
    return actions, {
        "rendered": rendered,
        "cap": summarize_cap_hits(actions),
        "context": summarize_budgets(budgets),
        "prompt_id_parity": {
            "n_checked": len(parity),
            "all_exact": all(p.get("exact_match") for p in parity),
        },
    }


def _score_fingerprint(action, teacher_model: str, teacher_tokenizer: str) -> str:
    """What a cached teacher score is only valid for.

    Keying the cache on `prefix#sample` alone is not enough: those ids are
    positional and get reused the moment a prefix is replaced, so a stale file
    would be silently accepted for a *different* action. Bind the score to the
    bytes scored, the prompt they were conditioned on, and the teacher that
    produced them.
    """
    h = hashlib.sha256()
    h.update(action.action_bytes)
    h.update(b"\0")
    h.update(json.dumps(list(action.prompt_token_ids or [])).encode())
    h.update(b"\0")
    h.update(teacher_model.encode())
    h.update(b"\0")
    h.update(teacher_tokenizer.encode())
    return h.hexdigest()


def _load_scores(path: Path, expected: dict[str, str] | None = None) -> dict:
    """Teacher scores already on disk from an earlier attempt.

    Tolerates a truncated final line: a crash mid-write must not invalidate the
    requests before it, which is the entire point of writing them one at a time.

    Rows whose fingerprint does not match the action now in hand are dropped
    rather than reused. Paying for one request again is cheaper than training
    on a score that belongs to different bytes.
    """
    out: dict = {}
    stale = 0
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        key = row["key"]
        if expected is not None:
            want = expected.get(key)
            if want is None or row.get("fingerprint") != want:
                stale += 1
                continue
        if key in out:
            raise ValueError(f"duplicate teacher score for {key} in {path}")
        out[key] = (
            [base64.b64decode(b) for b in row["teacher_token_bytes_b64"]],
            row["teacher_logprobs"],
        )
    if stale:
        log(f"ignored {stale} cached score(s) whose fingerprint no longer matches")
    return out


def score_actions(actions, rendered, args, scores_path=None):
    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.replay_score import score_replay_batch
    from vektori_trace.vocab_bridge import load_tokenizer

    teacher_tok = load_tokenizer(args.teacher_tokenizer)
    pool = FireworksTeacherPool(model=args.teacher_model)

    fingerprints = {
        a.key: _score_fingerprint(a, args.teacher_model, args.teacher_tokenizer)
        for a in actions
    }
    prior = _load_scores(scores_path, fingerprints) if scores_path else {}
    expected = {a.key for a in actions}
    foreign = sorted(set(prior) - expected)
    if foreign:
        raise ValueError(
            f"teacher score file contains keys outside this capture batch: {foreign[:4]}"
        )
    if prior:
        log(f"reusing {len(prior)} teacher scores already on disk")

    fh = scores_path.open("a", encoding="utf-8") if scores_path else None

    def _persist(sc) -> None:
        if fh is None:
            return
        fh.write(json.dumps({
            "key": sc.key,
            "fingerprint": fingerprints.get(sc.key),
            "teacher_token_bytes_b64": [
                base64.b64encode(b).decode() for b in sc.teacher_token_bytes
            ],
            "teacher_logprobs": list(sc.teacher_logprobs),
            "n_prefix_tokens": sc.n_prefix_tokens,
            "n_trailing_dropped": sc.n_trailing_dropped,
        }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    try:
        scored, ledger = score_replay_batch(
            actions, rendered, teacher_tok, pool,
            on_scored=_persist, already_scored=prior,
        )
    finally:
        if fh is not None:
            fh.close()
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

    progress = Path(args.out) / "train_progress.jsonl"

    def _log_example(row):
        log(
            f"  ex {row['example']:>2}/{len(actions)} "
            f"{row.get('key','')} tok={row.get('total_tokens')} "
            f"sup={row.get('supervised')} "
            f"loss/tok={row.get('loss_per_token')!s:.8} "
            f"peak={row.get('peak_gib')}GiB {row.get('seconds')}s"
        )

    return run_replay_chunk_opd(
        prefixes,
        actions,
        scored,
        make_optimizer_step(
            model, opt, cfg, progress_path=progress, on_example=_log_example
        ),
        max_new_tokens=args.max_tokens,
        n_samples_per_prefix=args.n_samples,
        stored_teacher_actions=stored,
        selection_policy=_selection_policy(),
    )


def _selection_policy() -> str:
    """Name the effective policy from the reconstruction capability."""
    from vektori_trace.compaction import reconstruction_is_implemented

    return (
        "stratified-diagnostic"
        if reconstruction_is_implemented()
        else "stratified-pre-compaction-only"
    )


def _harbor_revision(explicit: str | None) -> str | None:
    """The installed harbor version, unless overridden.

    §6.1 lists this as a pin and an earlier version of this script demanded it
    be typed. That was wrong twice over: the value *is* observable here, and a
    hand-typed pin can disagree with what the corpus was actually produced
    under, which is the one thing a pin exists to prevent.
    """
    if explicit:
        return explicit
    if os.environ.get("HARBOR_REVISION"):
        return os.environ["HARBOR_REVISION"]
    try:
        import importlib.metadata as md

        return f"harbor=={md.version('harbor')}"
    except Exception:
        return None


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
        harbor_revision=_harbor_revision(args.harbor_revision),
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
    ap.add_argument("--require-post-compaction", type=int, default=2,
                    help="§8.3's two authentic post-compaction prefixes. "
                         "Satisfiable since the SFT reconstruction was ported: "
                         "candidates come from the retained segment, not a flat "
                         "slice. Selection refuses a non-zero value if "
                         "reconstruction is ever turned off again.")
    # student
    ap.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE"))
    ap.add_argument("--model", default=os.environ.get("STUDENT_MODEL"))
    ap.add_argument("--base-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--student-tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--adapter-path", default=None, help="the frozen v0 adapter")
    ap.add_argument("--max-tokens", type=int, default=9216)
    ap.add_argument("--max-train-tokens", type=int, default=35687,
                    help="measured training envelope: prefix+action that a "
                         "forward/backward/step actually survived on an "
                         "A100-80GB (67.2 GiB peak, 12.6 GiB headroom). "
                         "Distinct from --max-tokens, which is the sampling "
                         "loop guard: an action may sample fine and still be "
                         "outside what training was proven to hold.")
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
    ap.add_argument("--harbor-revision", default=None,
                    help="§6.1 pin. Auto-derived from the installed harbor "
                         "distribution when omitted; pass explicitly only to "
                         "override what is actually installed.")
    # training
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--device", default=os.environ.get("REPLAY_TRAIN_DEVICE"),
                    help='CUDA device for the optimizer step, e.g. "cuda" or '
                         '"cuda:0". Required for a real run and deliberately '
                         "not defaulted: the ck75 serving endpoint is a "
                         "different host, and an inferred device is how the "
                         "step silently ran on CPU.")
    ap.add_argument("--out", default="./vektori-out/opd-replay")
    ap.add_argument("--resume-from-captures", default=None,
                    help="path to a captures.jsonl from an earlier "
                         "--stop-after-sampling run: reload those actions "
                         "instead of paying to sample them again")
    ap.add_argument("--stop-after-sampling", action="store_true",
                    help="sample, save captures, check the training envelope "
                         "and report projected cost — then stop, before any "
                         "paid teacher call")
    ap.add_argument("--stop-after-scoring", action="store_true",
                    help="score a complete saved capture batch, persist every "
                         "teacher result, then stop before allocating a "
                         "training GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="select and render only; spend nothing")
    args = ap.parse_args()
    args.corpus = args.corpus or DEFAULT_CORPUS

    if args.stop_after_sampling and args.stop_after_scoring:
        print("choose only one of --stop-after-sampling/--stop-after-scoring",
              file=sys.stderr)
        return 2
    if args.stop_after_scoring and not args.resume_from_captures:
        print("--stop-after-scoring requires --resume-from-captures", file=sys.stderr)
        return 2
    will_sample = not bool(args.resume_from_captures)
    will_score = not args.stop_after_sampling
    will_train = will_score and not args.stop_after_scoring

    from vektori_trace.chunk_opd import assert_token_cap_is_task_derived

    assert_token_cap_is_task_derived(args.max_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def flush(stage: str) -> None:
        """Write the report after every stage, not at the end.

        The expensive states are mid-run: sampling holds a GPU, scoring is
        paid, and the optimizer step is where an OOM is most likely. A report
        written only on success loses exactly the runs worth diagnosing — and
        loses the Fireworks spend with them. Written atomically so a crash
        mid-write cannot leave a truncated file where a readable older one was.
        """
        report["last_stage"] = stage
        report["last_write"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = out / "replay_run.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, default=str))
        tmp.replace(out / "replay_run.json")
    report: dict[str, Any] = {
        "policy_version": POLICY_VERSION,
        "n_prefixes": args.n_prefixes,
        "n_samples_per_prefix": args.n_samples,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
    }

    # Cheap invocation gates belong before tokenizer loading and the 3.5-minute
    # all-candidate budget census. These used to fire only after the census,
    # leaving an already-warm L40S idle for every typo or missing credential.
    if not args.dry_run:
        needed = []
        if will_sample:
            needed.extend(((args.api_base, "--api-base"), (args.model, "--model")))
        if will_train:
            needed.append((args.adapter_path, "--adapter-path"))
            needed.append((args.device, "--device"))
        for need, flag in needed:
            if not need:
                print(f"{flag} is required for this run stage", file=sys.stderr)
                return 2
        if will_score and not os.environ.get("FIREWORKS_API_KEY"):
            print("FIREWORKS_API_KEY is not set", file=sys.stderr)
            return 2
        missing_corpus = [str(p) for p in args.corpus if not Path(p).exists()]
        if missing_corpus:
            print(f"corpus path(s) do not exist: {missing_corpus}", file=sys.stderr)
            return 2
        if will_sample:
            require_endpoint_model(args.api_base, args.model)

        # Device and pin failures are also setup failures. Check them before
        # the all-candidate census, never after minutes of idle GPU time.
        if will_train:
            try:
                import torch
            except ImportError:
                raise SystemExit("torch is not importable; this host cannot train")
            if not str(args.device).startswith("cuda"):
                raise SystemExit(f"--device {args.device!r} is not a CUDA device")
            if not torch.cuda.is_available():
                raise SystemExit(
                    f"--device {args.device!r} but torch.cuda.is_available() is False"
                )
            idx = torch.device(args.device).index or 0
            if idx >= torch.cuda.device_count():
                raise SystemExit(
                    f"--device {args.device!r} but only {torch.cuda.device_count()} "
                    "CUDA device(s) visible"
                )
            log(f"training device: {args.device} ({torch.cuda.get_device_name(idx)})")

        from vektori_trace.opd_manifest import PinError, verify_reference_pins

        try:
            report["reference_sha256_observed"] = verify_reference_pins()
        except PinError as e:
            print(f"reference pin check failed: {e}", file=sys.stderr)
            return 2
        manifest = _build_manifest(args)
        report["manifest"] = manifest.to_dict()
        report["missing_pins"] = manifest.missing_pins()
        if will_train:
            try:
                manifest.require_complete()
            except PinError as e:
                print(f"{e}", file=sys.stderr)
                return 2
        if manifest.missing_pins():
            log(
                "non-training stage: pins incomplete but deferred "
                f"[{', '.join(manifest.missing_pins())}]"
            )
        else:
            log(f"pins complete; commit {manifest.vektori_trace_commit}")

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
    report["selection_policy"] = _selection_policy()
    # Derived, never asserted: this block described the pre-port state as a
    # pair of literals and kept saying so after reconstruction landed. A report
    # that hardcodes its own provenance is the failure it exists to prevent.
    from vektori_trace.compaction import reconstruction_is_implemented

    _recon = reconstruction_is_implemented()
    _n_pc = sum(1 for p in prefixes if p.post_compaction)
    report["post_compaction_coverage"] = {
        "claimed": bool(_recon and _n_pc),
        "reconstruction_implemented": _recon,
        "boundaries_detected": True,
        "required": args.require_post_compaction,
        "selected": _n_pc,
        "eligible_pool": (
            "reconstructed — compacted traces are enumerated from "
            "compaction.current_segment (retained handoff head from the "
            "questions sidecar + subsequent turns), the definition "
            "sft_export_traces used and ck75 was trained on"
            if _recon else
            "pre-compaction only — candidates at or after a trace's first "
            "boundary are excluded, not merely un-required"
        ),
        "reason": (
            "post-compaction prefixes carry the retained state, so §8.3 "
            "coverage is real (docs/OPD-MULTITURN-PLAN.md §15)"
            if _recon else
            "prefix_turns_through_step slices from step 0, so a marked prefix "
            "carries the pre-boundary history boundary:replace discarded"
        ),
    }
    report["prefixes"] = [
        {"prefix_id": p.prefix_id, "task": p.task, "step": p.step_index,
         "post_compaction": p.post_compaction}
        # NB: post_compaction is False for every candidate until boundaries are
        # derived from the corpus (plan §15). Reported, not claimed.
        for p in prefixes
    ]
    flush("selected")

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

    # --device gates the optimizer step, so a sampling-only stage does not need
    # one. Demanding it there forced a fake value onto a run that never trains,
    # which is how a wrong device reaches a real run unnoticed.
    # Only the scoring stage spends Fireworks money. A sampling-only run must
    # not require the key: it lets the sampling half run on a host that has no
    # business holding a billing credential.

    if student_tok is None:  # unreachable on a real run; guarded, not assumed
        print("student tokenizer was not loaded", file=sys.stderr)
        return 2

    caps_path = out / "captures.jsonl"

    if args.resume_from_captures:
        # Sampling held a GPU and its behaviour logprobs cannot be recreated,
        # so a second stage must reload them rather than re-sample. Without
        # this, `--stop-after-sampling` was a dead end: the only way forward
        # was to pay for all 32 actions again.
        from vektori_trace.replay_opd import SampledAction
        from vektori_trace.replay_sample import token_bytes_from_ids

        src = Path(args.resume_from_captures)
        if not src.exists():
            print(f"--resume-from-captures: {src} does not exist", file=sys.stderr)
            return 2
        actions = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log("captures.jsonl ends in a partial line; using the complete ones")
                break
            raw = base64.b64decode(row["action_bytes_b64"])
            ids = row["action_token_ids"]
            actions.append(
                SampledAction(
                    prefix_id=row["prefix_id"],
                    sample_index=row["sample_index"],
                    action_bytes=raw,
                    action_token_ids=ids,
                    action_token_bytes=token_bytes_from_ids(student_tok, ids),
                    behavior_logprobs=row["behavior_logprobs"],
                    policy_version=row["policy_version"],
                    prompt_token_ids=row.get("prompt_token_ids") or None,
                    termination_reason=row.get("termination_reason"),
                )
            )
        if not actions:
            print(f"--resume-from-captures: no complete captures in {src}",
                  file=sys.stderr)
            return 2
        log(f"resumed {len(actions)} captures from {src}")

        # The renderings the resumed actions were sampled under. Rebuilt from
        # the same prefixes, and the prompt-id parity check below is what
        # proves the rebuild matches what the server actually consumed.
        from vektori_trace.dataset import turns_to_messages
        from vektori_trace.replay_context import assert_prompt_ids_match

        rendered = {}
        for p_ in prefixes:
            msgs = turns_to_messages(p_.prefix_turns)
            rendered[p_.prefix_id] = msgs
            prompt = student_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            local = student_tok(prompt, add_special_tokens=False)["input_ids"]
            for a in actions:
                if a.prefix_id == p_.prefix_id and a.prompt_token_ids:
                    assert_prompt_ids_match(p_.prefix_id, list(local), a.prompt_token_ids)
                    break
        samp = {"rendered": rendered, "cap": {"resumed": True}, "context": {}}
        report["resumed_from"] = str(src)
    else:
        log(f"stage 2/4: sampling {args.n_prefixes * args.n_samples} actions from ck75")
        if caps_path.exists() and caps_path.stat().st_size:
            print(
                f"refusing to overwrite non-empty capture file {caps_path}; "
                "resume it with --resume-from-captures or choose a new --out",
                file=sys.stderr,
            )
            return 2
        caps_fh = caps_path.open("w", encoding="utf-8")

        def _persist(a) -> None:
            caps_fh.write(json.dumps({
                "key": a.key,
                "prefix_id": a.prefix_id,
                "sample_index": a.sample_index,
                "policy_version": a.policy_version,
                "termination_reason": a.termination_reason,
                "action_bytes_b64": base64.b64encode(a.action_bytes).decode(),
                "action_token_ids": list(a.action_token_ids),
                "behavior_logprobs": list(a.behavior_logprobs),
                "prompt_token_ids": list(a.prompt_token_ids or []),
            }) + "\n")
            caps_fh.flush()
            os.fsync(caps_fh.fileno())

        try:
            actions, samp = sample_actions(prefixes, student_tok, args, on_capture=_persist)
        except BaseException as e:
            report["sampling_error"] = f"{type(e).__name__}: {e}"
            report["captures_path"] = str(caps_path)
            report["captures_complete"] = False
            flush("sampling_failed")
            raise
        finally:
            caps_fh.close()

    from vektori_trace.replay_opd import validate_sample_set

    validate_sample_set(
        prefixes,
        actions,
        n_samples_per_prefix=args.n_samples,
        require_prompt_ids=True,
    )

    report["cap"] = samp["cap"]
    report["context"] = samp["context"]

    # Persist the captures *before* the first paid call. `log pi_old` is
    # captured under a frozen policy and cannot be recomputed once anything
    # moves, so an unsaved batch is unrecoverable — and until now the script
    # went straight from sampling into Fireworks, where a crash would have cost
    # both the sampling and the ability to retry scoring against it.
    report["captures_path"] = str(
        Path(args.resume_from_captures) if args.resume_from_captures else caps_path
    )
    report["captures_complete"] = True
    flush("sampled")
    log(f"saved {len(actions)} captures to {caps_path}")

    # The measured training envelope (§14). An action can sample fine under the
    # 9,216 loop guard and still land outside what a forward/backward was
    # actually proven to hold, and the only place to find that out cheaply is
    # here — before Fireworks, not after the optimizer OOMs on a batch whose
    # teacher scores are already paid for.
    lengths = [
        (a.key, len(a.prompt_token_ids or []) + len(a.action_token_ids))
        for a in actions
    ]
    oversize = [(k, n) for k, n in lengths if n > args.max_train_tokens]
    report["train_envelope"] = {
        "max_train_tokens": args.max_train_tokens,
        "max_observed": max(n for _k, n in lengths),
        "median_observed": sorted(n for _k, n in lengths)[len(lengths) // 2],
        "n_oversize": len(oversize),
        "oversize": oversize[:20],
        "action_tokens": {
            "max": max(len(a.action_token_ids) for a in actions),
            "median": sorted(len(a.action_token_ids) for a in actions)[len(actions) // 2],
        },
    }
    log(
        f"train envelope: max {report['train_envelope']['max_observed']} of "
        f"{args.max_train_tokens}; actions max "
        f"{report['train_envelope']['action_tokens']['max']}"
    )
    if oversize:
        flush("envelope_refused")
        print(
            f"{len(oversize)} sampled example(s) exceed the measured training "
            f"envelope of {args.max_train_tokens} tokens: "
            f"{oversize[:5]}. Refusing to pay Fireworks for examples the "
            "optimizer was never proven to hold. Captures are saved at "
            f"{caps_path}; re-preflight at the larger size or drop them.",
            file=sys.stderr,
        )
        return 2

    # Projected teacher cost, from the real prefixes and realized actions
    # rather than a guess. DeepSeek re-reads the whole prefix for every one of
    # the four samples at a prefix, so repeated-prefix tokens dominate and are
    # reported separately — that is the number a cheaper batching decision
    # would act on.
    by_prefix_prompt = {}
    for a in actions:
        by_prefix_prompt.setdefault(a.prefix_id, len(a.prompt_token_ids or []))
    unique_prefix_tokens = sum(by_prefix_prompt.values())
    total_prompt_tokens = sum(len(a.prompt_token_ids or []) for a in actions)
    total_action_tokens = sum(len(a.action_token_ids) for a in actions)
    report["projected_teacher_cost"] = {
        "n_requests": len(actions),
        "unique_prefix_tokens": unique_prefix_tokens,
        "repeated_prefix_tokens": total_prompt_tokens - unique_prefix_tokens,
        "total_prompt_tokens_qwen": total_prompt_tokens,
        "total_action_tokens_qwen": total_action_tokens,
        "note": "Qwen-tokenizer counts; DeepSeek's tokenization differs, so "
                "these bound the request shape rather than the exact bill",
    }
    log(
        f"projected teacher input ~{total_prompt_tokens:,} prompt tokens "
        f"({total_prompt_tokens - unique_prefix_tokens:,} repeated) + "
        f"{total_action_tokens:,} action tokens over {len(actions)} requests"
    )

    if args.stop_after_sampling:
        flush("stopped_after_sampling")
        log("--stop-after-sampling: captures saved and gated, nothing spent")
        log("TEAR DOWN the ck75 endpoint now.")
        return 0

    log("stage 3/4: scoring with DeepSeek")
    scored, ledger, teacher_prov = score_actions(
        actions, samp["rendered"], args, scores_path=out / "teacher_scores.jsonl"
    )
    report["teacher_ledger"] = ledger
    report["teacher"] = teacher_prov
    # Scoring is the paid stage: flush before anything can OOM.
    flush("scored")

    if args.stop_after_scoring:
        log("--stop-after-scoring: all paid scores saved; no training GPU used")
        return 0

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
    flush("archived")
    log(
        f"archived {report['archive']['n_examples']} examples "
        f"({report['archive']['bytes_written']:,} bytes); "
        f"max task share {report['archive']['max_task_share']}"
    )

    log("stage 4/4: one optimizer step")
    try:
        result = train((prefixes, actions, scored, stored), args)
    except BaseException as e:
        # Including OOM and KeyboardInterrupt: the scores are already paid for
        # and the captures are on disk, so the report must record what killed
        # the step rather than vanishing with it.
        report["training_error"] = f"{type(e).__name__}: {e}"
        flush("training_failed")
        log(f"training failed after paid scoring: {type(e).__name__}: {e}")
        log(f"captures + scores preserved under {out}")
        raise
    report["run"] = result
    flush("trained")

    (out / "replay_run.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"v_replay: {result['optimizer'].get('adapter_saved_to')}")
    log(f"report: {out / 'replay_run.json'}")
    log("TEAR DOWN the ck75 endpoint now — it is the only thing still billing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
