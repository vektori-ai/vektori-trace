"""Step 0 — verify teacher and student share a tokenizer (PLAN.md).

OPD requires same-family tokenizers. Mismatch is a hard fail before any GPU work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Two pairs, both same-family so the tokenizer check can pass. Which one is the
# default matters, because `opd.py` reads these as its config defaults.
#
# PILOT is the default. The teacher is a 30B MoE with ~3B active parameters, so
# it serves on one GPU and answers fast — and OPD queries the teacher at *every
# step*, which makes teacher latency, not student size, the loop's bottleneck.
# verl's own documented OPD example is a 32B teacher with an 8B student, so this
# sits close to a configuration someone else has actually run.
PILOT_TEACHER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
PILOT_STUDENT = "Qwen/Qwen3-8B"

# SCALE is PLAN.md's stated configuration. An 80B dense teacher needs several
# GPUs and costs more per teacher query; it is the scale-up once the pilot has
# closed the loop once, not the thing to debug the loop on.
SCALE_TEACHER = "Qwen/Qwen3-Coder-Next-80B"
SCALE_STUDENT = "Qwen/Qwen3-8B"

DEFAULT_TEACHER = PILOT_TEACHER
DEFAULT_STUDENT = PILOT_STUDENT

# The cross-tokenizer teacher (FINAL-PLAN.md). Deliberately NOT DEFAULT_TEACHER:
# `check_tokenizers` still hard-fails a vocab mismatch, and that gate stays, so
# the same-vocab pair above must keep pointing at a same-family Qwen teacher.
# This id is only reachable via `cross_tokenizer=True`, which routes to
# `check_cross_tokenizer` instead.
#
# `-0731` is the released V4-Flash build and is the teacher. The unsuffixed
# `deepseek-ai/DeepSeek-V4-Flash` repo is the earlier drop; both ship a
# byte-identical tokenizer.json, so the §4 alignment measurements hold for
# either, but the encoders differ and only 0731's is vendored.
CROSS_TEACHER = "deepseek-ai/DeepSeek-V4-Flash-0731"

# The cross-tokenizer *student* is Stage-B ck75, which is Qwen3-14B — not
# `PILOT_STUDENT`. It aliased to the 8B pilot student until 2026-08-21, which
# was inert while nothing read it, but `OPDRunManifest.student_tokenizer`
# defaults to this constant: a run would have recorded "Qwen/Qwen3-8B" as the
# pinned tokenizer while `run_replay_opd.py` served 14B. A pin that disagrees
# with what ran is worse than an absent pin, because §11 treats the manifest as
# the thing a result is attributed to.
CROSS_STUDENT = "Qwen/Qwen3-14B"


class TokenizerMismatchError(RuntimeError):
    """Teacher/student tokenizers are incompatible — refuse to allocate GPU."""


@dataclass(frozen=True)
class TokenizerFingerprint:
    name: str
    vocab_size: int
    merges_sha256: str | None
    vocab_sha256: str | None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_tokenizer(name_or_path: str, tokenizer: Any | None = None) -> TokenizerFingerprint:
    """Fingerprint a tokenizer by vocab_size + hashes of merges/vocab files."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    # A bare `tokenizers.Tokenizer` has neither `.vocab_size` nor `__len__`, and
    # the cross-tokenizer path reaches one whenever `AutoTokenizer` cannot load a
    # repo (see `vocab_bridge.load_tokenizer`). The HF branch is untouched.
    vocab_size_attr = getattr(tokenizer, "vocab_size", None)
    if vocab_size_attr is not None:
        vocab_size = int(vocab_size_attr)
    else:
        get_vocab_size = getattr(tokenizer, "get_vocab_size", None)
        vocab_size = int(get_vocab_size()) if callable(get_vocab_size) else len(tokenizer)
    # Prefer on-disk files when the path is a local dir; otherwise None (remote).
    root = Path(name_or_path)
    merges = vocab = None
    if root.is_dir():
        merges = _file_sha256(root / "merges.txt")
        for candidate in ("vocab.json", "tokenizer.json"):
            vocab = _file_sha256(root / candidate)
            if vocab is not None:
                break
    else:
        # HF cache / remote. `tokenizer.merges` does not exist on fast
        # tokenizers, and `backend_tokenizer.model` is a `tokenizers.models.BPE`
        # that is not iterable — the previous attempt at both silently produced
        # None on every hub model, so PLAN.md Step 0's "hash the merges" never
        # ran. `backend_tokenizer.to_str()` is the serialized model (merges
        # included) and is the thing to hash.
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None and hasattr(tokenizer, "to_str"):
            # Already a bare `tokenizers.Tokenizer`: it *is* the backend, so the
            # merges hash stays real instead of silently degrading to None.
            backend = tokenizer
        if backend is not None and hasattr(backend, "to_str"):
            try:
                merges = hashlib.sha256(backend.to_str().encode()).hexdigest()
            except Exception:
                merges = None
        try:
            vocab_map = tokenizer.get_vocab()
            payload = "\n".join(f"{k}\t{v}" for k, v in sorted(vocab_map.items())).encode()
            vocab = hashlib.sha256(payload).hexdigest()
        except Exception:
            vocab = None

    return TokenizerFingerprint(
        name=str(name_or_path),
        vocab_size=vocab_size,
        merges_sha256=merges,
        vocab_sha256=vocab,
    )


def check_tokenizers(
    teacher: str = DEFAULT_TEACHER,
    student: str = DEFAULT_STUDENT,
    *,
    teacher_tokenizer: Any | None = None,
    student_tokenizer: Any | None = None,
) -> tuple[TokenizerFingerprint, TokenizerFingerprint]:
    """Compare vocab_size and merges/vocab hashes. Raises TokenizerMismatchError."""
    t_fp = fingerprint_tokenizer(teacher, teacher_tokenizer)
    s_fp = fingerprint_tokenizer(student, student_tokenizer)

    problems: list[str] = []
    if t_fp.vocab_size != s_fp.vocab_size:
        problems.append(
            f"vocab_size mismatch: teacher={t_fp.vocab_size} student={s_fp.vocab_size}"
        )

    # A hash present on one side and absent on the other is *unverified*, never a
    # match. The old `t and s and t != s` guard skipped the comparison entirely in
    # that case (local-dir teacher vs hub student is exactly it) and let the check
    # pass on vocab_size alone — which PLAN.md gates every GPU allocation on.
    compared = 0
    for label, t_val, s_val in (
        ("merges", t_fp.merges_sha256, s_fp.merges_sha256),
        ("vocab", t_fp.vocab_sha256, s_fp.vocab_sha256),
    ):
        if (t_val is None) != (s_val is None):
            side = "teacher" if t_val is None else "student"
            problems.append(
                f"{label} hash unverifiable: no {label} fingerprint for the {side} "
                "(one side hashed, the other not) — asymmetric, so not a match"
            )
            continue
        if t_val is None:
            continue
        compared += 1
        if t_val != s_val:
            problems.append(
                f"{label} hash mismatch: teacher={t_val[:12]}… student={s_val[:12]}…"
            )
    if compared == 0 and not problems:
        problems.append(
            "no tokenizer fingerprint could be compared — neither merges nor vocab "
            "hashed on either side; vocab_size agreement alone is not a shared "
            "tokenizer"
        )

    if problems:
        raise TokenizerMismatchError(
            "tokenizer check failed — OPD requires a shared tokenizer; "
            "hard fail before allocating GPU. " + "; ".join(problems)
        )
    return t_fp, s_fp


__all__ = [
    "CROSS_STUDENT",
    "CROSS_TEACHER",
    "DEFAULT_STUDENT",
    "DEFAULT_TEACHER",
    "PILOT_STUDENT",
    "PILOT_TEACHER",
    "SCALE_STUDENT",
    "SCALE_TEACHER",
    "TokenizerFingerprint",
    "TokenizerMismatchError",
    "check_tokenizers",
    "fingerprint_tokenizer",
]
