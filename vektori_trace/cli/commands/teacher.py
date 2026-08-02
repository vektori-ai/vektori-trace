"""`probe-teacher`, `check-tokenizers`, `build-bridge`, `align-report`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def cmd_probe_teacher(args: argparse.Namespace) -> int:
    """Does this hosted teacher return per-token logprobs for tokens we supply?

    The whole hosted-teacher question reduces to this one request, and neither
    vendor's documentation settles it: Fireworks documents `echo_last` and the
    integer-array prompt separately without an example combining them, and AWS
    documents `prompt_logprobs` on the chat schema while claiming completion-schema
    support it never demonstrates. So: send it, print what came back.

    Exit 0 means OPD can run against this teacher. Exit 1 means it cannot, and the
    message is the reason — which is a result worth recording, not a failure.
    """
    from ...providers.teacher.base import TeacherScoringError

    # Arbitrary-but-valid ids. The check is on the shape of the response, not on
    # what the tokens mean, and there is no server-side tokenizer to ask.
    prefix_ids = [9707, 11, 1879]
    tokens = [0, 1986, 374]

    pool: Any
    if args.backend == "fireworks":
        from ...providers.teacher.fireworks import (
            DEFAULT_FIREWORKS_BASE,
            DEFAULT_FIREWORKS_TEACHER,
            FireworksTeacherPool,
        )

        try:
            pool = FireworksTeacherPool(
                model=args.model or DEFAULT_FIREWORKS_TEACHER,
                api_base=args.api_base or DEFAULT_FIREWORKS_BASE,
            )
        except TeacherScoringError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        target = f"{pool.model} @ {pool.api_base}"
    else:
        if not args.model:
            print(
                "error: --model is required for bedrock (the imported model's ARN)",
                file=sys.stderr,
            )
            return 2
        from ...providers.teacher.bedrock import BedrockTeacherPool

        try:
            pool = BedrockTeacherPool(model_id=args.model, region=args.region)
        except TeacherScoringError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        target = f"{pool.model_id} @ bedrock:{pool.region}"

    print(f"probing {args.backend}: {target}")
    result: dict[str, Any] = {
        "backend": args.backend,
        "target": target,
        "prefix_ids": prefix_ids,
        "tokens": tokens,
    }
    try:
        scored = pool.score_ids(prefix_ids, tokens)
    except TeacherScoringError as e:
        result.update({"score_ids": "failed", "error": str(e)})
        _write_probe(args, result)
        print(f"score_ids: FAILED — {e}", file=sys.stderr)
        print(
            "OPD cannot run against this teacher. This is the documented outcome "
            "to record, not something to work around.",
            file=sys.stderr,
        )
        return 1

    if len(scored) != len(tokens):
        result.update({"score_ids": "failed", "error": f"{len(scored)} logprobs for {len(tokens)} tokens"})
        _write_probe(args, result)
        print(f"score_ids: FAILED — {result['error']}", file=sys.stderr)
        return 1

    result.update({"score_ids": "ok", "logprobs": scored})
    print(f"score_ids: OK — {len(scored)} logprobs, {[round(x, 4) for x in scored]}")

    if args.top_k > 0:
        try:
            rows = pool.score_ids_topk(prefix_ids, tokens, args.top_k)
            widths = [len(r) for r in rows]
            result.update({"score_ids_topk": "ok", "row_widths": widths})
            print(f"score_ids_topk(K={args.top_k}): OK — row widths {widths}")
        except (TeacherScoringError, ValueError) as e:
            # Not fatal: the sampled-token objective is the declared one and it
            # works. Top-K is the lower-variance variant, and losing it costs
            # variance, not correctness.
            result.update({"score_ids_topk": "failed", "topk_error": str(e)})
            print(f"score_ids_topk(K={args.top_k}): FAILED — {e}")
            print("note: top_k=0 (`reverse_kl_surrogate`) is unaffected.")

    result["provenance"] = pool.provenance()
    _write_probe(args, result)
    print("this teacher can run OPD.")

    if getattr(args, "echo", False):
        # P0 echo probe. FireworksTeacherPool does not expose probe_echo_support;
        # wrap it so the same check CrossTokenizerTeacherPool uses is available.
        echo_pool = pool
        if not hasattr(echo_pool, "probe_echo_support"):
            from ...providers.teacher.cross import CrossTokenizerTeacherPool

            echo_pool = CrossTokenizerTeacherPool(
                pool=pool,
                teacher_tokenizer=None,
                thinking_mode="chat",
            )
        echo_result = echo_pool.probe_echo_support()
        result["echo"] = echo_result
        print(f"echo support: {'OK' if echo_result.get('ok') else 'FAILED'}")
        if not echo_result.get("ok"):
            print(f"  error: {echo_result.get('error')}", file=sys.stderr)
            _write_probe(args, result)
            return 1
    return 0


def _write_probe(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if not args.out:
        return
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"probe result: {path}")


def cmd_check_tokenizers(args: argparse.Namespace) -> int:
    from ...tokenizer_check import (
        DEFAULT_STUDENT,
        DEFAULT_TEACHER,
        TokenizerMismatchError,
        check_tokenizers,
    )

    teacher = args.teacher or DEFAULT_TEACHER
    student = args.student or DEFAULT_STUDENT
    try:
        t_fp, s_fp = check_tokenizers(teacher, student)
    except TokenizerMismatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "teacher": {
                    "name": t_fp.name,
                    "vocab_size": t_fp.vocab_size,
                    "merges_sha256": t_fp.merges_sha256,
                    "vocab_sha256": t_fp.vocab_sha256,
                },
                "student": {
                    "name": s_fp.name,
                    "vocab_size": s_fp.vocab_size,
                    "merges_sha256": s_fp.merges_sha256,
                    "vocab_sha256": s_fp.vocab_sha256,
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_build_bridge(args: argparse.Namespace) -> int:
    """Build a CrossTokenizerBridge JSON from a teacher/student tokenizer pair.

    The bridge maps every teacher token id to the byte-identical student token id
    (when one exists), and stores the byte tables for both tokenizers so that
    run_opd_training can align sampled student tokens with teacher re-tokenisation
    without loading either tokenizer at training time.
    """
    from ...vocab_bridge import CrossTokenizerError, check_cross_tokenizer

    try:
        bridge = check_cross_tokenizer(
            args.teacher_tokenizer,
            args.student_tokenizer,
            thinking_mode=args.thinking_mode,
        )
    except CrossTokenizerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bridge.save(out_path)
    print(f"bridge: {out_path}")
    print(f"exact-map size: {len(bridge.exact_map)} byte-identical token pairs")
    print(
        f"teacher vocab: {bridge.teacher_table.vocab_size}  "
        f"student vocab: {bridge.student_table.vocab_size}"
    )
    print(
        f"coverage: {len(bridge.exact_map) / bridge.teacher_table.vocab_size:.1%} "
        "of teacher tokens map exactly"
    )
    return 0


def cmd_align_report(args: argparse.Namespace) -> int:
    """Offline granularity report: encode text samples with both tokenizers and
    align by bytes, reporting granularity (spans / student tokens) per sample.

    Uses the bridge byte tables so no network is required; tokenizers are loaded
    locally to encode the input text.
    """
    import sys

    from ...align import AlignmentError, align_by_bytes
    from ...vocab_bridge import CrossTokenizerBridge

    bridge = CrossTokenizerBridge.load(args.bridge)

    # §10.7 — refuse a drifted encoder even for offline reporting.
    from ...encoding_dsv4 import ENCODING_DSV4_SHA256, verify_encoding_dsv4_pin

    try:
        verify_encoding_dsv4_pin()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if bridge.encoding_dsv4_hash != ENCODING_DSV4_SHA256:
        print(
            f"error: bridge encoding_dsv4 hash mismatch: "
            f"bridge={bridge.encoding_dsv4_hash!r} current={ENCODING_DSV4_SHA256!r}",
            file=sys.stderr,
        )
        return 2

    # Load both tokenizers to encode the samples. `load_tokenizer` rather than
    # AutoTokenizer: align-report is an offline stage and must not fail because
    # `transformers` cannot parse a teacher's *model* config.
    from ...vocab_bridge import encode_ids, load_tokenizer

    teacher_name = bridge.teacher_fingerprint.name
    student_name = bridge.student_fingerprint.name
    teacher_tok = load_tokenizer(teacher_name)
    student_tok = load_tokenizer(student_name)

    raw = Path(args.text).read_text() if args.text else sys.stdin.read()
    samples = [line for line in raw.splitlines() if line.strip()]
    if not samples:
        print("error: no samples to align (empty input)", file=sys.stderr)
        return 1

    rows = []
    for sample in samples:
        s_ids = encode_ids(student_tok, sample)
        t_ids = encode_ids(teacher_tok, sample)
        s_bytes = [bridge.student_table.table.get(i, b"") for i in s_ids]
        t_bytes = [bridge.teacher_table.table.get(i, b"") for i in t_ids]
        s_bytes = [b for b in s_bytes if b]
        t_bytes = [b for b in t_bytes if b]
        if not s_bytes or not t_bytes:
            rows.append({"sample": sample[:40], "error": "empty after EOS stripping"})
            continue
        try:
            al = align_by_bytes(
                s_bytes,
                t_bytes,
                max_span_student_tokens=getattr(args, "max_span_student_tokens", 8),
            )
            # §4 — log granularity with a coarse content-type tag (numeric-heavy
            # vs other); do not normalize numbers out of the data.
            from ...cross_kl import _content_type_of_bytes

            joined = b"".join(s_bytes)
            content_type = _content_type_of_bytes(joined)
            rows.append({
                "sample": sample[:40],
                "granularity": round(al.granularity, 4),
                "content_type": content_type,
                "n_student": al.n_student_tokens,
                "n_teacher": al.n_teacher_tokens,
                "n_spans": len(al.spans),
            })
        except AlignmentError as e:
            rows.append({"sample": sample[:40], "error": str(e)})

    print(json.dumps(rows, indent=2))
    gran = [r["granularity"] for r in rows if "granularity" in r]
    if gran:
        print(
            f"\nmean granularity: {sum(gran) / len(gran):.4f}  "
            f"(n={len(gran)}, min={min(gran):.4f})"
        )
        by_type: dict[str, list[float]] = {}
        for r in rows:
            if "granularity" not in r:
                continue
            by_type.setdefault(r.get("content_type", "other"), []).append(r["granularity"])
        for ctype, vals in sorted(by_type.items()):
            print(
                f"  {ctype}: mean={sum(vals)/len(vals):.4f} n={len(vals)} "
                f"min={min(vals):.4f}"
            )
    return 0
