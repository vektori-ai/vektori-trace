"""Tests for vektori_trace/vocab_bridge.py — offline only, no model downloads.

Strategy:
 - Real tokenizers.Tokenizer objects (BPE + ByteLevel decoder) for round-trip
   and build_byte_table tests.
 - Lightweight mock wrappers for assert_byte_level and check_cross_tokenizer.
 - Hand-crafted ByteTable instances for exact-map logic tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256
from vektori_trace.vocab_bridge import (
    ByteLevelError,
    ByteTable,
    CrossTokenizerBridge,
    CrossTokenizerError,
    _fingerprint_table,
    _token_str_to_bytes,
    assert_byte_level,
    build_byte_table,
    build_exact_token_map,
    check_cross_tokenizer,
    validate_byte_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bytes_to_unicode() -> dict[int, str]:
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


def _make_raw_byte_level_tokenizer():
    """
    Build a bare tokenizers.Tokenizer with 256-token ByteLevel BPE vocabulary.
    Each token is a single ByteLevel unicode character mapping to one byte.
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as BLDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as BLPretok

    b2u = _bytes_to_unicode()
    vocab = {u_char: i for i, (_, u_char) in enumerate(sorted(b2u.items()))}
    raw = Tokenizer(BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = BLPretok(add_prefix_space=False)
    raw.decoder = BLDecoder(add_prefix_space=False)
    return raw


class _FakePreTrainedTok:
    """
    Minimal wrapper around a bare tokenizers.Tokenizer that mimics the
    PreTrainedTokenizerFast interface used by build_byte_table, assert_byte_level,
    and validate_byte_table.
    """

    def __init__(self, inner, special_ids: list[int] | None = None):
        self._inner = inner
        self.vocab_size = inner.get_vocab_size()
        self.backend_tokenizer = inner
        self.all_special_ids: list[int] = special_ids or []

    def get_vocab(self) -> dict[str, int]:
        return self._inner.get_vocab()

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        return [self._inner.id_to_token(i) for i in ids]

    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text).ids


def _make_fake_pretrained_tok(special_ids: list[int] | None = None) -> _FakePreTrainedTok:
    return _FakePreTrainedTok(_make_raw_byte_level_tokenizer(), special_ids)


def _make_backend_mock(decoder_type: str = "ByteLevel") -> MagicMock:
    """Return a mock backend_tokenizer whose to_str() returns the given decoder type."""
    decoder_json: dict | None
    if decoder_type:
        decoder_json = {"type": decoder_type, "add_prefix_space": False, "trim_offsets": True}
    else:
        decoder_json = None
    backend = MagicMock()
    backend.to_str.return_value = json.dumps(
        {"version": "1.0", "decoder": decoder_json, "model": {"type": "BPE"}}
    )
    return backend


def _make_mock_tok(decoder_type: str = "ByteLevel") -> MagicMock:
    """Return a mock tokenizer with a ByteLevel (or other) backend."""
    tok = MagicMock()
    tok.backend_tokenizer = _make_backend_mock(decoder_type)
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# _token_str_to_bytes
# ─────────────────────────────────────────────────────────────────────────────

def test_token_str_to_bytes_ascii() -> None:
    assert _token_str_to_bytes("H") == b"H"
    assert _token_str_to_bytes("!") == b"!"


def test_token_str_to_bytes_space_as_G() -> None:
    # Ġ (U+0120) is the ByteLevel encoding of byte 0x20 (space)
    assert _token_str_to_bytes("Ġhello") == b" hello"


def test_token_str_to_bytes_non_bytelevel_falls_back_to_utf8() -> None:
    # ｜ (U+FF5C) is outside the ByteLevel alphabet — should become UTF-8
    special = "<｜begin▁of▁sentence｜>"
    result = _token_str_to_bytes(special)
    assert result == special.encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# assert_byte_level
# ─────────────────────────────────────────────────────────────────────────────

def test_assert_byte_level_passes_for_byte_level_tok() -> None:
    tok = _make_mock_tok("ByteLevel")
    assert_byte_level(tok)  # should not raise


def test_assert_byte_level_fails_for_wordpiece() -> None:
    tok = _make_mock_tok("WordPiece")
    with pytest.raises(ByteLevelError, match="not ByteLevel"):
        assert_byte_level(tok)


def test_assert_byte_level_fails_when_no_decoder() -> None:
    tok = MagicMock()
    tok.backend_tokenizer.to_str.return_value = json.dumps(
        {"version": "1.0", "decoder": None}
    )
    with pytest.raises(ByteLevelError, match="not ByteLevel"):
        assert_byte_level(tok)


def test_assert_byte_level_no_backend_no_to_str_raises() -> None:
    tok = MagicMock(spec=[])  # no attributes
    with pytest.raises(ByteLevelError, match="to_str"):
        assert_byte_level(tok)


def test_assert_byte_level_falls_back_to_self_to_str() -> None:
    """When tok has no backend_tokenizer but has its own to_str()."""
    tok = MagicMock(spec=["to_str"])
    tok.to_str.return_value = json.dumps(
        {"version": "1.0", "decoder": {"type": "ByteLevel"}}
    )
    assert_byte_level(tok)  # should not raise


def test_assert_byte_level_with_real_tokenizer() -> None:
    raw = _make_raw_byte_level_tokenizer()
    # A bare tokenizers.Tokenizer has no backend_tokenizer; falls back to self.to_str()
    assert_byte_level(raw)


# ─────────────────────────────────────────────────────────────────────────────
# build_byte_table
# ─────────────────────────────────────────────────────────────────────────────

def test_build_byte_table_returns_256_tokens() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    assert bt.vocab_size == 256
    assert len(bt.table) == 256


def test_build_byte_table_maps_space_token_to_space_byte() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    b2u = _bytes_to_unicode()
    space_tok_str = b2u[32]  # unicode char for byte 0x20 (space)
    vocab = tok.get_vocab()
    space_id = vocab[space_tok_str]
    assert bt.table[space_id] == b" "


def test_build_byte_table_fingerprint_is_sha256() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    assert len(bt.fingerprint) == 64
    # Fingerprint is deterministic
    bt2 = build_byte_table(tok)
    assert bt.fingerprint == bt2.fingerprint


def test_build_byte_table_via_get_vocab_only() -> None:
    """When tok only has get_vocab() (no convert_ids_to_tokens)."""
    b2u = _bytes_to_unicode()
    vocab = {u_char: i for i, (_, u_char) in enumerate(sorted(b2u.items()))}

    class _MinimalTok:
        def get_vocab(self):
            return vocab

    bt = build_byte_table(_MinimalTok())
    assert bt.vocab_size == 256
    assert len(bt.table) == 256


def test_build_byte_table_fingerprint_mirrors_table_content() -> None:
    """Changing any table entry changes the fingerprint."""
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)

    # Manually corrupt one entry
    altered_table = dict(bt.table)
    first_id = next(iter(altered_table))
    altered_table[first_id] = b"\xff\xff"
    fp2 = _fingerprint_table(altered_table)
    assert fp2 != bt.fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# validate_byte_table
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_byte_table_passes_for_correct_table() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    corpus = ["Hi!", "hello world", "abc 123"]
    validate_byte_table(bt, tok, corpus)  # should not raise


def test_validate_byte_table_default_corpus() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    validate_byte_table(bt, tok)  # uses default corpus


def test_validate_byte_table_fails_on_corrupt_table() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    # Corrupt the table: map id 72 ('H') to wrong bytes
    bad_table = dict(bt.table)
    bad_table[72] = b"\xff"
    bad_bt = ByteTable(vocab_size=bt.vocab_size, table=bad_table, fingerprint=bt.fingerprint)
    with pytest.raises(AssertionError, match="round-trip failed"):
        validate_byte_table(bad_bt, tok, ["Hi!"])


def test_validate_byte_table_fails_on_missing_id() -> None:
    tok = _make_fake_pretrained_tok()
    bt = build_byte_table(tok)
    # Remove an entry that the tokenizer will produce
    bad_table = dict(bt.table)
    bad_table.pop(72, None)  # 'H' = byte 72
    bad_bt = ByteTable(vocab_size=bt.vocab_size, table=bad_table, fingerprint=bt.fingerprint)
    with pytest.raises(AssertionError, match="missing from ByteTable"):
        validate_byte_table(bad_bt, tok, ["Hi!"])


# ─────────────────────────────────────────────────────────────────────────────
# build_exact_token_map (§8)
# ─────────────────────────────────────────────────────────────────────────────

def _make_bt(mapping: dict[int, bytes]) -> ByteTable:
    return ByteTable(
        vocab_size=len(mapping),
        table=mapping,
        fingerprint=_fingerprint_table(mapping),
    )


def test_exact_map_identical_bytes_maps() -> None:
    bt_t = _make_bt({0: b"hello", 1: b"world"})
    bt_s = _make_bt({10: b"hello", 11: b"world", 12: b"foo"})
    m = build_exact_token_map(bt_t, bt_s)
    assert m == {0: 10, 1: 11}


def test_exact_map_excludes_specials_teacher() -> None:
    bt_t = _make_bt({0: b"hello", 1: b"world", 2: b"<eos>"})
    bt_s = _make_bt({10: b"hello", 11: b"world", 12: b"<eos>"})
    m = build_exact_token_map(bt_t, bt_s, special_ids_teacher=frozenset({2}))
    assert 2 not in m
    assert m == {0: 10, 1: 11}


def test_exact_map_excludes_specials_student() -> None:
    bt_t = _make_bt({0: b"hello", 1: b"world"})
    # Token 10 (b"hello") is special in student → excluded
    bt_s = _make_bt({10: b"hello", 11: b"world"})
    m = build_exact_token_map(bt_t, bt_s, special_ids_student=frozenset({10}))
    assert 0 not in m
    assert m == {1: 11}


def test_exact_map_no_match_when_bytes_differ() -> None:
    bt_t = _make_bt({0: b"foo"})
    bt_s = _make_bt({0: b"bar"})
    assert build_exact_token_map(bt_t, bt_s) == {}


def test_exact_map_excludes_collision_in_student() -> None:
    """Two student tokens with same bytes → neither maps."""
    bt_t = _make_bt({0: b"tok"})
    bt_s = _make_bt({10: b"tok", 11: b"tok"})  # collision
    assert build_exact_token_map(bt_t, bt_s) == {}


def test_exact_map_excludes_collision_in_teacher() -> None:
    """Two teacher tokens with same bytes → neither maps."""
    bt_t = _make_bt({0: b"tok", 1: b"tok"})
    bt_s = _make_bt({10: b"tok"})
    assert build_exact_token_map(bt_t, bt_s) == {}


def test_exact_map_empty_tables() -> None:
    assert build_exact_token_map(_make_bt({}), _make_bt({})) == {}


def test_exact_map_specials_from_tok_attribute() -> None:
    """build_exact_token_map reads all_special_ids from tok objects."""
    bt_t = _make_bt({0: b"hello", 99: b"<special>"})
    bt_s = _make_bt({5: b"hello", 99: b"<special>"})

    tok_t = MagicMock()
    tok_t.all_special_ids = [99]
    tok_s = MagicMock()
    tok_s.all_special_ids = [99]
    # No backend for added vocab
    del tok_t.backend_tokenizer
    del tok_s.backend_tokenizer

    m = build_exact_token_map(bt_t, bt_s, tok_teacher=tok_t, tok_student=tok_s)
    assert 99 not in m
    assert m == {0: 5}


# ─────────────────────────────────────────────────────────────────────────────
# CrossTokenizerBridge save / load
# ─────────────────────────────────────────────────────────────────────────────

def test_bridge_save_load_roundtrip(tmp_path) -> None:
    from vektori_trace.tokenizer_check import TokenizerFingerprint

    bt_t = _make_bt({0: b"hello", 1: b"world"})
    bt_s = _make_bt({10: b"hello", 11: b"world"})
    fp_t = TokenizerFingerprint(name="teacher", vocab_size=2, merges_sha256=None, vocab_sha256="aaa")
    fp_s = TokenizerFingerprint(name="student", vocab_size=2, merges_sha256="bbb", vocab_sha256="ccc")

    bridge = CrossTokenizerBridge(
        teacher_table=bt_t,
        student_table=bt_s,
        exact_map={0: 10, 1: 11},
        teacher_fingerprint=fp_t,
        student_fingerprint=fp_s,
        encoding_dsv4_hash=ENCODING_DSV4_SHA256,
        thinking_mode="chat",
    )

    p = tmp_path / "bridge.json"
    bridge.save(p)
    loaded = CrossTokenizerBridge.load(p)

    assert loaded.teacher_table.vocab_size == 2
    assert loaded.teacher_table.table == {0: b"hello", 1: b"world"}
    assert loaded.teacher_table.fingerprint == bt_t.fingerprint
    assert loaded.student_table.table == {10: b"hello", 11: b"world"}
    assert loaded.exact_map == {0: 10, 1: 11}
    assert loaded.teacher_fingerprint.name == "teacher"
    assert loaded.teacher_fingerprint.vocab_sha256 == "aaa"
    assert loaded.student_fingerprint.merges_sha256 == "bbb"
    assert loaded.encoding_dsv4_hash == ENCODING_DSV4_SHA256
    assert loaded.thinking_mode == "chat"


def test_bridge_save_is_valid_json(tmp_path) -> None:
    from vektori_trace.tokenizer_check import TokenizerFingerprint

    bridge = CrossTokenizerBridge(
        teacher_table=_make_bt({0: b"\x00\xff"}),
        student_table=_make_bt({0: b"\x00\xff"}),
        exact_map={0: 0},
        teacher_fingerprint=TokenizerFingerprint("t", 1, None, None),
        student_fingerprint=TokenizerFingerprint("s", 1, None, None),
        encoding_dsv4_hash="x" * 64,
        thinking_mode="thinking",
    )
    p = tmp_path / "b.json"
    bridge.save(p)
    parsed = json.loads(p.read_text())
    assert parsed["thinking_mode"] == "thinking"
    assert parsed["exact_map"] == {"0": 0}
    # Hex round-trip for non-printable bytes
    assert parsed["teacher_table"]["table"]["0"] == "00ff"


def test_bridge_load_restores_bytes_from_hex(tmp_path) -> None:
    from vektori_trace.tokenizer_check import TokenizerFingerprint

    bridge = CrossTokenizerBridge(
        teacher_table=_make_bt({5: b"\xde\xad\xbe\xef"}),
        student_table=_make_bt({}),
        exact_map={},
        teacher_fingerprint=TokenizerFingerprint("t", 1, None, None),
        student_fingerprint=TokenizerFingerprint("s", 0, None, None),
        encoding_dsv4_hash="0" * 64,
        thinking_mode="chat",
    )
    p = tmp_path / "b2.json"
    bridge.save(p)
    loaded = CrossTokenizerBridge.load(p)
    assert loaded.teacher_table.table[5] == b"\xde\xad\xbe\xef"


# ─────────────────────────────────────────────────────────────────────────────
# check_cross_tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def test_check_cross_tokenizer_passes_with_two_byte_level_toks() -> None:
    tok_t = _make_fake_pretrained_tok()
    tok_s = _make_fake_pretrained_tok()
    corpus = ["Hi!", "abc"]
    bridge = check_cross_tokenizer(
        "teacher",
        "student",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        validate_corpus=corpus,
    )
    assert isinstance(bridge, CrossTokenizerBridge)
    assert bridge.encoding_dsv4_hash == ENCODING_DSV4_SHA256
    assert bridge.thinking_mode == "chat"


def test_check_cross_tokenizer_thinking_mode_propagates() -> None:
    tok_t = _make_fake_pretrained_tok()
    tok_s = _make_fake_pretrained_tok()
    bridge = check_cross_tokenizer(
        "t", "s",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        thinking_mode="thinking",
        validate_corpus=["Hi!"],
    )
    assert bridge.thinking_mode == "thinking"


def test_check_cross_tokenizer_fails_if_teacher_not_byte_level() -> None:
    tok_t = _make_mock_tok("WordPiece")
    tok_s = _make_fake_pretrained_tok()
    with pytest.raises(CrossTokenizerError, match=r"teacher.*not ByteLevel"):
        check_cross_tokenizer(
            "t", "s",
            teacher_tokenizer=tok_t,
            student_tokenizer=tok_s,
            validate_corpus=["Hi!"],
        )


def test_check_cross_tokenizer_fails_if_student_not_byte_level() -> None:
    tok_t = _make_fake_pretrained_tok()
    tok_s = _make_mock_tok("WordPiece")
    with pytest.raises(CrossTokenizerError, match=r"student.*not ByteLevel"):
        check_cross_tokenizer(
            "t", "s",
            teacher_tokenizer=tok_t,
            student_tokenizer=tok_s,
            validate_corpus=["Hi!"],
        )


def test_check_cross_tokenizer_does_not_require_equal_vocab() -> None:
    """check_cross_tokenizer succeeds even when teacher != student vocab sizes."""
    tok_t = _make_fake_pretrained_tok()

    # Build a smaller student tokenizer with only ~50 tokens (ASCII letters + digits)
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as BLDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as BLPretok

    b2u = _bytes_to_unicode()
    # Pick a subset of printable ASCII bytes
    subset_bytes = list(range(ord("a"), ord("z") + 1)) + list(range(ord("0"), ord("9") + 1))
    small_vocab = {b2u[b]: i for i, b in enumerate(subset_bytes)}
    raw_small = Tokenizer(BPE(vocab=small_vocab, merges=[]))
    raw_small.pre_tokenizer = BLPretok(add_prefix_space=False)
    raw_small.decoder = BLDecoder(add_prefix_space=False)
    tok_s = _FakePreTrainedTok(raw_small)

    # Use a corpus that the small student can encode
    corpus = ["abc123", "defg"]
    bridge = check_cross_tokenizer(
        "t", "s",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        validate_corpus=corpus,
    )
    # Teacher has 256 tokens; student has ~36 — bridge should succeed
    assert bridge.teacher_table.vocab_size == 256
    assert bridge.student_table.vocab_size == len(small_vocab)


def test_check_cross_tokenizer_exact_map_populated() -> None:
    """Identical ByteLevel vocabs → exact map covers all tokens."""
    tok_t = _make_fake_pretrained_tok()
    tok_s = _make_fake_pretrained_tok()
    bridge = check_cross_tokenizer(
        "t", "s",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        validate_corpus=["Hi!"],
    )
    # Identical vocabs → all 256 tokens map 1↔1
    assert len(bridge.exact_map) == 256


def test_check_cross_tokenizer_specials_excluded_from_map() -> None:
    """Tokens in all_special_ids must not appear in the exact map."""
    # Mark ids 0 and 1 as special in both tokenizers
    tok_t = _make_fake_pretrained_tok(special_ids=[0, 1])
    tok_s = _make_fake_pretrained_tok(special_ids=[0, 1])
    bridge = check_cross_tokenizer(
        "t", "s",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        validate_corpus=["Hi!"],
    )
    assert 0 not in bridge.exact_map
    assert 1 not in bridge.exact_map
    assert len(bridge.exact_map) == 254  # 256 - 2 specials


def test_check_cross_tokenizer_bridge_roundtrip_save_load(tmp_path) -> None:
    tok_t = _make_fake_pretrained_tok()
    tok_s = _make_fake_pretrained_tok()
    bridge = check_cross_tokenizer(
        "t", "s",
        teacher_tokenizer=tok_t,
        student_tokenizer=tok_s,
        validate_corpus=["Hi!"],
    )
    p = tmp_path / "bridge.json"
    bridge.save(p)
    loaded = CrossTokenizerBridge.load(p)
    assert loaded.teacher_table.fingerprint == bridge.teacher_table.fingerprint
    assert loaded.student_table.fingerprint == bridge.student_table.fingerprint
    assert loaded.exact_map == bridge.exact_map
    assert loaded.encoding_dsv4_hash == ENCODING_DSV4_SHA256
