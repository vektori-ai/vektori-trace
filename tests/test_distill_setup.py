"""Cross-tokenizer setup, tested without torch.

`_setup_cross_tokenizer` was inline in `run_opd_training`, so its two hard
failures could only be reached by starting a training run. Coverage showed both
raise branches unexercised. They are cheap to reach directly, and they are the
ones worth failing on early — each costs a GPU allocation if it fires late.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vektori_trace.distill import OPDTrainConfig, _setup_cross_tokenizer


@dataclass
class _FakeBridge:
    encoding_dsv4_hash: str


def _cfg(**kw) -> OPDTrainConfig:
    kw.setdefault("verify_tokenizers", False)
    return OPDTrainConfig(**kw)


def test_same_vocab_path_passes_everything_through():
    """cross_tokenizer=False must not load, validate, or invent anything."""
    bridge, tok, cache = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=False), bridge=None, teacher_tokenizer=None, prefix_cache=None
    )
    assert (bridge, tok, cache) == (None, None, None)


def test_missing_bridge_is_rejected():
    with pytest.raises(ValueError, match="requires a bridge"):
        _setup_cross_tokenizer(
            _cfg(cross_tokenizer=True, bridge_path=None),
            bridge=None,
            teacher_tokenizer=object(),
            prefix_cache=None,
        )


def test_bridge_built_against_a_different_encoder_is_rejected():
    """A drifted encoding_dsv4 means the bridge's spans no longer describe what
    the encoder now produces — training would proceed on stale alignment."""
    with pytest.raises(RuntimeError, match="encoding_dsv4 hash mismatch"):
        _setup_cross_tokenizer(
            _cfg(cross_tokenizer=True),
            bridge=_FakeBridge(encoding_dsv4_hash="not-the-current-hash"),
            teacher_tokenizer=object(),
            prefix_cache=None,
        )


def test_hash_mismatch_message_names_both_hashes_and_the_fix():
    """The error has to be actionable: which hash was expected, and what to run."""
    with pytest.raises(RuntimeError) as exc:
        _setup_cross_tokenizer(
            _cfg(cross_tokenizer=True),
            bridge=_FakeBridge(encoding_dsv4_hash="stale"),
            teacher_tokenizer=object(),
            prefix_cache=None,
        )
    msg = str(exc.value)
    assert "stale" in msg
    assert "build-bridge" in msg


def test_matching_bridge_is_accepted_and_a_prefix_cache_is_created():
    from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256
    from vektori_trace.providers.teacher.cross import TeacherPrefixCache

    sentinel_tok = object()
    bridge, tok, cache = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True),
        bridge=_FakeBridge(encoding_dsv4_hash=ENCODING_DSV4_SHA256),
        teacher_tokenizer=sentinel_tok,
        prefix_cache=None,
    )
    assert bridge is not None
    assert tok is sentinel_tok
    assert isinstance(cache, TeacherPrefixCache), "cross path must get a cache"


def test_caller_supplied_prefix_cache_wins():
    """Determinism depends on reusing one cache across a run — a fresh one here
    would silently defeat the drift detector it exists to be."""
    from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256
    from vektori_trace.providers.teacher.cross import TeacherPrefixCache

    mine = TeacherPrefixCache()
    _, _, cache = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True),
        bridge=_FakeBridge(encoding_dsv4_hash=ENCODING_DSV4_SHA256),
        teacher_tokenizer=object(),
        prefix_cache=mine,
    )
    assert cache is mine


def test_kwarg_teacher_tokenizer_beats_cfg():
    from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256

    from_cfg, from_kwarg = object(), object()
    _, tok, _ = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True, teacher_tokenizer=from_cfg),
        bridge=_FakeBridge(encoding_dsv4_hash=ENCODING_DSV4_SHA256),
        teacher_tokenizer=from_kwarg,
        prefix_cache=None,
    )
    assert tok is from_kwarg


def _minimal_bridge():
    """A real CrossTokenizerBridge, small enough to build without a tokenizer."""
    from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256
    from vektori_trace.tokenizer_check import TokenizerFingerprint
    from vektori_trace.vocab_bridge import ByteTable, CrossTokenizerBridge

    table = ByteTable(vocab_size=2, table={0: b"a", 1: b"b"}, fingerprint="f" * 64)
    fp = TokenizerFingerprint(
        name="tiny", vocab_size=2, merges_sha256="0" * 64, vocab_sha256="0" * 64
    )
    return CrossTokenizerBridge(
        teacher_table=table,
        student_table=table,
        exact_map={0: 0, 1: 1},
        teacher_fingerprint=fp,
        student_fingerprint=fp,
        encoding_dsv4_hash=ENCODING_DSV4_SHA256,
        thinking_mode="chat",
    )


def test_bridge_is_loaded_from_cfg_bridge_path(tmp_path):
    """The path branch: no bridge= kwarg, so it comes off disk."""
    path = tmp_path / "bridge.json"
    _minimal_bridge().save(path)

    bridge, _, _ = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True, bridge_path=path),
        bridge=None,
        teacher_tokenizer=object(),
        prefix_cache=None,
    )
    assert bridge is not None
    assert bridge.exact_map == {0: 0, 1: 1}


def test_bridge_kwarg_beats_bridge_path(tmp_path):
    """An explicitly passed bridge must not be silently replaced by the path."""
    path = tmp_path / "bridge.json"
    _minimal_bridge().save(path)
    injected = _minimal_bridge()

    bridge, _, _ = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True, bridge_path=path),
        bridge=injected,
        teacher_tokenizer=object(),
        prefix_cache=None,
    )
    assert bridge is injected


def test_cfg_teacher_tokenizer_used_when_no_kwarg():
    from vektori_trace.encoding_dsv4 import ENCODING_DSV4_SHA256

    from_cfg = object()
    _, tok, _ = _setup_cross_tokenizer(
        _cfg(cross_tokenizer=True, teacher_tokenizer=from_cfg),
        bridge=_FakeBridge(encoding_dsv4_hash=ENCODING_DSV4_SHA256),
        teacher_tokenizer=None,
        prefix_cache=None,
    )
    assert tok is from_cfg
