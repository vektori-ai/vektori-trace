"""Cross-tokenizer OPD vocabulary bridge (§8 of FINAL-PLAN.md).

Offline-only: no network, no torch required.

Public API:
    ByteTable                  id → bytes + fingerprint
    build_byte_table(tok)      via convert_ids_to_tokens + ByteLevel unicode→byte table
    validate_byte_table        round-trip assertion over a corpus
    assert_byte_level(tok)     refuses non-ByteLevel tokenizers
    build_exact_token_map      §8's narrow teacher_id → student_id map, specials excluded
    CrossTokenizerBridge       save/load JSON artifact
    check_cross_tokenizer      sibling to tokenizer_check.check_tokenizers
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256, verify_encoding_dsv4_pin
from vektori_trace.tokenizer_check import TokenizerFingerprint, fingerprint_tokenizer

# ── ByteLevel unicode → byte table ────────────────────────────────────────────

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 ByteLevel convention: byte 0-255 → unicode char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=False))


def _unicode_to_bytes() -> dict[str, int]:
    """Invert _bytes_to_unicode(): unicode char → byte value (0-255)."""
    return {v: k for k, v in _bytes_to_unicode().items()}


_U2B: dict[str, int] | None = None


def _get_u2b() -> dict[str, int]:
    global _U2B
    if _U2B is None:
        _U2B = _unicode_to_bytes()
    return _U2B


def _token_str_to_bytes(token_str: str) -> bytes:
    """
    Convert a ByteLevel-encoded token string to raw bytes.

    Each character in a ByteLevel token maps to a single byte via the
    unicode→byte table. Falls back to UTF-8 for tokens containing characters
    outside the ByteLevel alphabet (e.g. special tokens whose literal form
    includes CJK full-width or geometric symbols like ｜ U+FF5C and ▁ U+2581).
    """
    u2b = _get_u2b()
    if all(c in u2b for c in token_str):
        return bytes([u2b[c] for c in token_str])
    return token_str.encode("utf-8")


# ── ByteTable ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ByteTable:
    """Maps every token id to its raw bytes + a SHA-256 fingerprint of the table."""

    vocab_size: int
    table: dict[int, bytes]
    fingerprint: str


def _fingerprint_table(table: dict[int, bytes]) -> str:
    payload = "\n".join(
        f"{tid}:{tbytes.hex()}" for tid, tbytes in sorted(table.items())
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# ── assert_byte_level ─────────────────────────────────────────────────────────

class ByteLevelError(RuntimeError):
    """Tokenizer does not use a ByteLevel backend decoder."""


def assert_byte_level(tok: Any) -> None:
    """
    Raise ByteLevelError unless tok has a ByteLevel backend decoder.

    Checks via tok.backend_tokenizer.to_str() JSON, falling back to
    tok.to_str() when tok itself is a bare tokenizers.Tokenizer.
    """
    # Resolve the object that has to_str()
    backend = getattr(tok, "backend_tokenizer", None)
    source = backend if (backend is not None and hasattr(backend, "to_str")) else tok

    if not hasattr(source, "to_str"):
        raise ByteLevelError(
            "Tokenizer has neither backend_tokenizer.to_str() nor to_str(); "
            "cannot verify ByteLevel decoder."
        )
    try:
        backend_json = json.loads(source.to_str())
    except Exception as exc:
        raise ByteLevelError(f"Cannot parse backend JSON: {exc}") from exc

    decoder = backend_json.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ByteLevelError(
            f"Backend decoder is not ByteLevel: {decoder!r}. "
            "OPD cross-tokenizer path requires ByteLevel BPE tokenizers."
        )


# ── build_byte_table ──────────────────────────────────────────────────────────

def build_byte_table(tok: Any) -> ByteTable:
    """
    Build a ByteTable from a tokenizer.

    Uses convert_ids_to_tokens (PreTrainedTokenizerFast style) when available,
    then falls back to get_vocab() iteration. Never calls tokenizer.decode().

    Token strings are converted to bytes via the ByteLevel unicode→byte table.
    Special tokens whose literals fall outside the ByteLevel alphabet are stored
    using UTF-8 so that the table is complete.
    """
    if hasattr(tok, "convert_ids_to_tokens"):
        # HuggingFace PreTrainedTokenizerFast path.
        # get_vocab() covers all ids including added special tokens; use its
        # key set to build the full range rather than only 0..vocab_size-1.
        if hasattr(tok, "get_vocab"):
            vocab: dict[str, int] = tok.get_vocab()
            all_ids = sorted(vocab.values())
            token_strings: list[str | None] = tok.convert_ids_to_tokens(all_ids)
        else:
            vocab_size = int(getattr(tok, "vocab_size", None) or len(tok))
            all_ids = list(range(vocab_size))
            token_strings = tok.convert_ids_to_tokens(all_ids)

        table: dict[int, bytes] = {}
        for tid, ts in zip(all_ids, token_strings, strict=False):
            if ts is None:
                continue
            table[tid] = _token_str_to_bytes(ts)

    elif hasattr(tok, "get_vocab"):
        # Bare tokenizers.Tokenizer or any object with get_vocab().
        vocab = tok.get_vocab()
        table = {tid: _token_str_to_bytes(ts) for ts, tid in vocab.items()}

    else:
        raise ValueError(
            "Cannot extract vocabulary: tokenizer has no convert_ids_to_tokens "
            "or get_vocab."
        )

    vocab_size = len(table)
    return ByteTable(
        vocab_size=vocab_size,
        table=table,
        fingerprint=_fingerprint_table(table),
    )


# ── validate_byte_table ───────────────────────────────────────────────────────

_DEFAULT_CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def foo(x): return x + 1",
    "42 + 3.14 = 45.14",
]


def validate_byte_table(
    bt: ByteTable,
    tok: Any,
    corpus: list[str] | None = None,
) -> None:
    """
    Round-trip assertion: for each text in corpus, encoding via tok then mapping
    each token id through bt must reconstruct the original bytes exactly.

    This is the non-negotiable gate confirming that bt correctly mirrors tok.

    Raises AssertionError if the round-trip fails for any text.
    """
    texts = corpus if corpus is not None else _DEFAULT_CORPUS

    # Resolve the encode function
    if hasattr(tok, "encode"):
        raw_encode = tok.encode
    else:
        raise ValueError("Tokenizer has no encode() method; cannot validate.")

    for text in texts:
        result = raw_encode(text)
        if hasattr(result, "ids"):
            ids: list[int] = result.ids
        elif isinstance(result, list):
            ids = result
        elif isinstance(result, dict):
            ids = result["input_ids"]
        else:
            ids = list(result)

        reconstructed = b""
        for tid in ids:
            if tid not in bt.table:
                raise AssertionError(
                    f"validate_byte_table: token id {tid!r} missing from ByteTable "
                    f"(corpus text={text!r})"
                )
            reconstructed += bt.table[tid]

        expected = text.encode("utf-8")
        if reconstructed != expected:
            raise AssertionError(
                f"validate_byte_table round-trip failed for {text!r}: "
                f"expected {expected!r}, got {reconstructed!r}"
            )


# ── Special-token helpers ─────────────────────────────────────────────────────

def _special_ids(tok: Any) -> frozenset[int]:
    """Best-effort extraction of special token ids from a tokenizer."""
    ids: set[int] = set()

    # transformers: all_special_ids is a list[int]
    val = getattr(tok, "all_special_ids", None)
    if isinstance(val, list):
        ids.update(int(v) for v in val)

    # tokenizers.Tokenizer.get_added_vocabulary() returns {str: int}
    backend = getattr(tok, "backend_tokenizer", tok)
    if hasattr(backend, "get_added_vocabulary"):
        try:
            ids.update(backend.get_added_vocabulary().values())
        except Exception:
            pass

    return frozenset(ids)


# ── build_exact_token_map (§8) ────────────────────────────────────────────────

def build_exact_token_map(
    bt_teacher: ByteTable,
    bt_student: ByteTable,
    special_ids_teacher: frozenset[int] | None = None,
    special_ids_student: frozenset[int] | None = None,
    tok_teacher: Any | None = None,
    tok_student: Any | None = None,
) -> dict[int, int]:
    """
    Build the §8 narrow map: teacher_id → student_id iff:
      1. Both token ids decode to the identical byte string.
      2. That byte string is a single token in both vocabs (no collision on either side).
      3. Neither token id is a special token.

    Never calls tokenizer.decode(). Correspondence is bytes(a) == bytes(b); no
    similarity metric anywhere.
    """
    t_specials: frozenset[int] = special_ids_teacher if special_ids_teacher is not None else frozenset()
    s_specials: frozenset[int] = special_ids_student if special_ids_student is not None else frozenset()

    if tok_teacher is not None and not t_specials:
        t_specials = _special_ids(tok_teacher)
    if tok_student is not None and not s_specials:
        s_specials = _special_ids(tok_student)

    # Build student: bytes → id, detecting collisions (bytes with more than one token)
    s_bytes_to_id: dict[bytes, int] = {}
    s_collision: set[bytes] = set()
    for sid, sbytes in bt_student.table.items():
        if sid in s_specials:
            continue
        if sbytes in s_bytes_to_id:
            s_collision.add(sbytes)
        else:
            s_bytes_to_id[sbytes] = sid
    for b in s_collision:
        s_bytes_to_id.pop(b, None)

    # Build teacher collision set
    t_bytes_count: dict[bytes, int] = {}
    for tid, tbytes in bt_teacher.table.items():
        if tid in t_specials:
            continue
        t_bytes_count[tbytes] = t_bytes_count.get(tbytes, 0) + 1

    # Build the map
    exact_map: dict[int, int] = {}
    for tid, tbytes in bt_teacher.table.items():
        if tid in t_specials:
            continue
        if t_bytes_count.get(tbytes, 0) > 1:
            continue  # bytes appear on multiple teacher tokens — not 1↔1
        sid = s_bytes_to_id.get(tbytes)
        if sid is not None:
            exact_map[tid] = sid

    return exact_map


# ── CrossTokenizerBridge ──────────────────────────────────────────────────────

@dataclass
class CrossTokenizerBridge:
    """
    Serialisable artifact for cross-tokenizer OPD.

    Contains both byte tables, the exact-byte map, both tokenizer fingerprints
    (mirroring tokenizer_check.fingerprint_tokenizer), the encoding_dsv4 hash
    (drift guard per §10.7), and the thinking_mode used to build the bridge.
    """

    teacher_table: ByteTable
    student_table: ByteTable
    exact_map: dict[int, int]
    teacher_fingerprint: TokenizerFingerprint
    student_fingerprint: TokenizerFingerprint
    encoding_dsv4_hash: str
    thinking_mode: str

    def save(self, path: str | Path) -> None:
        """Serialise the bridge to a JSON file."""
        data = {
            "teacher_table": {
                "vocab_size": self.teacher_table.vocab_size,
                "table": {str(k): v.hex() for k, v in self.teacher_table.table.items()},
                "fingerprint": self.teacher_table.fingerprint,
            },
            "student_table": {
                "vocab_size": self.student_table.vocab_size,
                "table": {str(k): v.hex() for k, v in self.student_table.table.items()},
                "fingerprint": self.student_table.fingerprint,
            },
            "exact_map": {str(k): v for k, v in self.exact_map.items()},
            "teacher_fingerprint": {
                "name": self.teacher_fingerprint.name,
                "vocab_size": self.teacher_fingerprint.vocab_size,
                "merges_sha256": self.teacher_fingerprint.merges_sha256,
                "vocab_sha256": self.teacher_fingerprint.vocab_sha256,
            },
            "student_fingerprint": {
                "name": self.student_fingerprint.name,
                "vocab_size": self.student_fingerprint.vocab_size,
                "merges_sha256": self.student_fingerprint.merges_sha256,
                "vocab_sha256": self.student_fingerprint.vocab_sha256,
            },
            "encoding_dsv4_hash": self.encoding_dsv4_hash,
            "thinking_mode": self.thinking_mode,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CrossTokenizerBridge:
        """Deserialise a bridge from a JSON file."""
        data = json.loads(Path(path).read_text())

        teacher_table = ByteTable(
            vocab_size=data["teacher_table"]["vocab_size"],
            table={int(k): bytes.fromhex(v) for k, v in data["teacher_table"]["table"].items()},
            fingerprint=data["teacher_table"]["fingerprint"],
        )
        student_table = ByteTable(
            vocab_size=data["student_table"]["vocab_size"],
            table={int(k): bytes.fromhex(v) for k, v in data["student_table"]["table"].items()},
            fingerprint=data["student_table"]["fingerprint"],
        )
        t_fp = data["teacher_fingerprint"]
        s_fp = data["student_fingerprint"]
        return cls(
            teacher_table=teacher_table,
            student_table=student_table,
            exact_map={int(k): v for k, v in data["exact_map"].items()},
            teacher_fingerprint=TokenizerFingerprint(
                name=t_fp["name"],
                vocab_size=t_fp["vocab_size"],
                merges_sha256=t_fp["merges_sha256"],
                vocab_sha256=t_fp["vocab_sha256"],
            ),
            student_fingerprint=TokenizerFingerprint(
                name=s_fp["name"],
                vocab_size=s_fp["vocab_size"],
                merges_sha256=s_fp["merges_sha256"],
                vocab_sha256=s_fp["vocab_sha256"],
            ),
            encoding_dsv4_hash=data["encoding_dsv4_hash"],
            thinking_mode=data["thinking_mode"],
        )


# ── Tokenizer loading ─────────────────────────────────────────────────────────

class CrossTokenizerError(RuntimeError):
    """Cross-tokenizer compatibility check failed."""


def encode_ids(tok: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    """Encode `text` to a flat `list[int]`, whatever tokenizer flavour `tok` is.

    The two flavours disagree on the return type, and the disagreement is silent
    until it isn't:

    * HF `PreTrainedTokenizerFast.encode` → `list[int]`
    * bare `tokenizers.Tokenizer.encode` → an `Encoding` object, which is **not**
      iterable — `[int(i) for i in enc]` raises `TypeError`, and `len(enc)` is
      not the token count you want either.

    Since `load_tokenizer` can hand back either, every encode site has to go
    through here. `validate_byte_table` already did this defensively; nothing
    else did.
    """
    try:
        result = tok.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        # Some tokenizers do not accept the keyword at all.
        result = tok.encode(text)

    ids = getattr(result, "ids", None)  # tokenizers.Encoding
    if ids is None:
        ids = result
    if hasattr(ids, "tolist"):  # torch / numpy
        ids = ids.tolist()
    if isinstance(ids, dict):  # {"input_ids": [...]}
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def load_tokenizer(name_or_path: str) -> Any:
    """Load a tokenizer, falling back to the raw `tokenizers` backend.

    `AutoTokenizer.from_pretrained` reads `config.json` to resolve the tokenizer
    class, so it fails on any repo whose *model* config a given `transformers`
    release cannot parse — even though the tokenizer itself is fine and is all
    this module ever needs. `deepseek-ai/DeepSeek-V4-Flash-0731` is exactly that
    case on transformers 5.5.x: its YaRN rope config raises
    ``AttributeError: 'PreTrainedConfig' object has no attribute
    'max_position_embeddings'`` before a single token is read.

    Pinning a newer `transformers` would couple the *offline* stages — which
    FINAL-PLAN.md says need no GPU and no network beyond the tokenizer files — to
    the release cadence of a library only the training stages depend on. So fall
    back to `tokenizers.Tokenizer.from_pretrained`, which downloads
    `tokenizer.json` and nothing else.

    Everything downstream already accepts a bare `tokenizers.Tokenizer`:
    `build_byte_table` has a `get_vocab()` path, `assert_byte_level` reads
    `to_str()` directly, and `_special_ids` uses `get_added_vocabulary()`.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as auto_exc:
        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_pretrained(name_or_path)
        except Exception as raw_exc:
            raise CrossTokenizerError(
                f"could not load a tokenizer for {name_or_path!r}. "
                f"AutoTokenizer failed ({type(auto_exc).__name__}: {auto_exc}); "
                f"tokenizers.Tokenizer.from_pretrained also failed "
                f"({type(raw_exc).__name__}: {raw_exc})."
            ) from raw_exc


# ── check_cross_tokenizer ─────────────────────────────────────────────────────

def check_cross_tokenizer(
    teacher: str = "teacher",
    student: str = "student",
    *,
    teacher_tokenizer: Any | None = None,
    student_tokenizer: Any | None = None,
    thinking_mode: str = "chat",
    validate_corpus: list[str] | None = None,
) -> CrossTokenizerBridge:
    """
    Sibling to tokenizer_check.check_tokenizers for the cross-tokenizer OPD path.

    Verifies:
    - Both tokenizers use a ByteLevel backend decoder (hard fail if not).
    - Both byte tables round-trip cleanly over validate_corpus (or the default
      corpus when None).
    - Bridge is buildable.

    Does NOT require equal vocab sizes or equal merges — unlike check_tokenizers,
    which keeps its stronger same-vocab guarantee and must not be weakened.

    Raises CrossTokenizerError on any failure.
    """
    # §10.7 — pin check before anything else.
    verify_encoding_dsv4_pin()

    if teacher_tokenizer is None:
        teacher_tokenizer = load_tokenizer(teacher)
    if student_tokenizer is None:
        student_tokenizer = load_tokenizer(student)

    # ── ByteLevel guard ───────────────────────────────────────────────────────
    for label, tok in (("teacher", teacher_tokenizer), ("student", student_tokenizer)):
        try:
            assert_byte_level(tok)
        except ByteLevelError as exc:
            raise CrossTokenizerError(
                f"cross-tokenizer check failed — {label} is not ByteLevel: {exc}"
            ) from exc

    # ── Build byte tables ─────────────────────────────────────────────────────
    try:
        bt_teacher = build_byte_table(teacher_tokenizer)
    except Exception as exc:
        raise CrossTokenizerError(f"Failed to build teacher byte table: {exc}") from exc

    try:
        bt_student = build_byte_table(student_tokenizer)
    except Exception as exc:
        raise CrossTokenizerError(f"Failed to build student byte table: {exc}") from exc

    # ── Round-trip validation ─────────────────────────────────────────────────
    for label, bt, tok in (
        ("teacher", bt_teacher, teacher_tokenizer),
        ("student", bt_student, student_tokenizer),
    ):
        try:
            validate_byte_table(bt, tok, validate_corpus)
        except AssertionError as exc:
            raise CrossTokenizerError(
                f"Byte table validation failed for {label}: {exc}"
            ) from exc

    # ── Fingerprints + map + bridge ───────────────────────────────────────────
    t_fp = fingerprint_tokenizer(teacher, teacher_tokenizer)
    s_fp = fingerprint_tokenizer(student, student_tokenizer)

    exact_map = build_exact_token_map(
        bt_teacher,
        bt_student,
        tok_teacher=teacher_tokenizer,
        tok_student=student_tokenizer,
    )

    return CrossTokenizerBridge(
        teacher_table=bt_teacher,
        student_table=bt_student,
        exact_map=exact_map,
        teacher_fingerprint=t_fp,
        student_fingerprint=s_fp,
        encoding_dsv4_hash=ENCODING_DSV4_SHA256,
        thinking_mode=thinking_mode,
    )


__all__ = [
    "ByteLevelError",
    "ByteTable",
    "CrossTokenizerBridge",
    "CrossTokenizerError",
    "assert_byte_level",
    "build_byte_table",
    "build_exact_token_map",
    "check_cross_tokenizer",
    "encode_ids",
    "load_tokenizer",
    "validate_byte_table",
]
