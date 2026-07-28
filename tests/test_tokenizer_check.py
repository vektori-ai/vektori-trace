"""Step 0 — tokenizer compatibility check."""

from __future__ import annotations

from pathlib import Path

import pytest

from vektori_trace.tokenizer_check import (
    TokenizerMismatchError,
    check_tokenizers,
    fingerprint_tokenizer,
)


class _FakeTok:
    def __init__(self, vocab_size: int, vocab: dict[str, int] | None = None):
        self.vocab_size = vocab_size
        self._vocab = vocab or {f"t{i}": i for i in range(vocab_size)}

    def get_vocab(self):
        return self._vocab


def test_matching_tokenizers_pass() -> None:
    tok = _FakeTok(10)
    t, s = check_tokenizers(
        "teacher",
        "student",
        teacher_tokenizer=tok,
        student_tokenizer=tok,
    )
    assert t.vocab_size == s.vocab_size == 10
    assert t.vocab_sha256 == s.vocab_sha256


def test_mismatched_vocab_size_fails_loudly() -> None:
    with pytest.raises(TokenizerMismatchError, match="vocab_size mismatch"):
        check_tokenizers(
            "teacher",
            "student",
            teacher_tokenizer=_FakeTok(10),
            student_tokenizer=_FakeTok(11),
        )


def test_mismatched_vocab_hash_fails() -> None:
    a = _FakeTok(3, {"a": 0, "b": 1, "c": 2})
    b = _FakeTok(3, {"a": 0, "b": 1, "d": 2})
    with pytest.raises(TokenizerMismatchError, match="vocab hash"):
        check_tokenizers("t", "s", teacher_tokenizer=a, student_tokenizer=b)


def test_fingerprint_uses_vocab_size() -> None:
    fp = fingerprint_tokenizer("x", _FakeTok(5))
    assert fp.vocab_size == 5
    assert fp.vocab_sha256 is not None


class _Backend:
    def __init__(self, payload: str):
        self._payload = payload

    def to_str(self) -> str:
        return self._payload


class _TokWithMerges(_FakeTok):
    def __init__(self, vocab_size: int, payload: str, vocab: dict[str, int] | None = None):
        super().__init__(vocab_size, vocab)
        self.backend_tokenizer = _Backend(payload)


class _OpaqueTok:
    """No backend and no readable vocab — nothing to fingerprint at all."""

    vocab_size = 10

    def get_vocab(self):
        raise RuntimeError("no vocab")


def test_asymmetric_merges_hash_is_not_a_match() -> None:
    """Local-dir teacher vs hub student: one side hashes merges, the other cannot.
    The old `t and s and t != s` guard skipped the comparison and passed on
    vocab_size alone — PLAN.md gates every GPU allocation on this check."""
    teacher = _TokWithMerges(10, "merges-payload")
    student = _FakeTok(10)  # same vocab, no backend → merges_sha256 is None
    assert fingerprint_tokenizer("t", teacher).merges_sha256 is not None
    assert fingerprint_tokenizer("s", student).merges_sha256 is None

    with pytest.raises(TokenizerMismatchError, match="unverifiable"):
        check_tokenizers("t", "s", teacher_tokenizer=teacher, student_tokenizer=student)


def test_symmetric_merges_hashes_still_compare_normally() -> None:
    same = check_tokenizers(
        "t",
        "s",
        teacher_tokenizer=_TokWithMerges(10, "same"),
        student_tokenizer=_TokWithMerges(10, "same"),
    )
    assert same[0].merges_sha256 == same[1].merges_sha256

    with pytest.raises(TokenizerMismatchError, match="merges hash mismatch"):
        check_tokenizers(
            "t",
            "s",
            teacher_tokenizer=_TokWithMerges(10, "a"),
            student_tokenizer=_TokWithMerges(10, "b"),
        )


def test_no_comparable_fingerprint_is_a_hard_fail_not_a_pass() -> None:
    """vocab_size agreement alone is not a shared tokenizer."""
    with pytest.raises(TokenizerMismatchError, match="no tokenizer fingerprint"):
        check_tokenizers(
            "t", "s", teacher_tokenizer=_OpaqueTok(), student_tokenizer=_OpaqueTok()
        )


def test_local_dir_merges_mismatch_is_caught(tmp_path) -> None:
    """PLAN.md's integration case: two on-disk tokenizers whose merges differ.
    vocab_size agreement alone must not let this through — it gates every GPU
    allocation in the OPD branch."""
    import json as _json

    from vektori_trace.tokenizer_check import (
        TokenizerMismatchError,
        check_tokenizers,
        fingerprint_tokenizer,
    )

    class _Tok:
        vocab_size = 1000

        def get_vocab(self):
            return {"a": 0, "b": 1}

    def _mk(name: str, merges: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "merges.txt").write_text(merges)
        (d / "vocab.json").write_text(_json.dumps({"a": 0, "b": 1}))
        return d

    teacher = _mk("teacher", "#version: 0.2\na b\nb c\n")
    student = _mk("student", "#version: 0.2\na b\nc d\n")
    same = _mk("same", "#version: 0.2\na b\nb c\n")

    t_fp = fingerprint_tokenizer(str(teacher), _Tok())
    assert t_fp.merges_sha256 is not None, "on-disk merges.txt must be hashed"

    with pytest.raises(TokenizerMismatchError, match="merges hash mismatch"):
        check_tokenizers(
            str(teacher),
            str(student),
            teacher_tokenizer=_Tok(),
            student_tokenizer=_Tok(),
        )

    # Identical merges + vocab on disk pass.
    check_tokenizers(
        str(teacher), str(same), teacher_tokenizer=_Tok(), student_tokenizer=_Tok()
    )
