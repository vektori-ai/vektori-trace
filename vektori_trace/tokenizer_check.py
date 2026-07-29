"""Step 0 — verify teacher and student share a tokenizer (PLAN.md).

OPD requires same-family tokenizers. Mismatch is a hard fail before any GPU work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TEACHER = "Qwen/Qwen3-Coder-Next-80B"
DEFAULT_STUDENT = "Qwen/Qwen3-8B"


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

    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
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
    "DEFAULT_STUDENT",
    "DEFAULT_TEACHER",
    "TokenizerFingerprint",
    "TokenizerMismatchError",
    "check_tokenizers",
    "fingerprint_tokenizer",
]
