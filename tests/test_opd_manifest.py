"""Phase-0 §6.1 pins — the manifest and the reference paper-code hashes.

The point of these tests is that a pin nobody filled in must be loud. A manifest
that silently defaults its way to "complete" is worse than no manifest, because
a run report would then carry pins the run never actually verified.
"""

from __future__ import annotations

import pytest

from vektori_trace.opd_manifest import (
    PAPER_CODE_REVISION,
    REFERENCE_SHA256,
    OPDRunManifest,
    PinError,
    current_commit,
    sha256_file,
    verify_reference_pins,
)
from vektori_trace.tokenizer_check import CROSS_STUDENT, CROSS_TEACHER


def _complete() -> OPDRunManifest:
    return OPDRunManifest(
        student_base_model="Qwen/Qwen3-14B",
        student_adapter_path="/data/stage-b/checkpoint-75",
        student_adapter_sha256="0" * 64,
        student_adapter_config_sha256="1" * 64,
        student_tokenizer_sha256="2" * 64,
        fireworks_model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
        harbor_revision="abc1234",
        task_corpus="cs/corpus50_v3",
        vektori_trace_commit="def5678",
    )


# ---------------------------------------------------------------------------
# Reference paper code
# ---------------------------------------------------------------------------


def test_reference_paper_code_matches_its_pin():
    """The port in chunk_opd.py is only meaningful against these exact bytes."""
    observed = verify_reference_pins()
    assert observed == REFERENCE_SHA256


def test_altered_reference_file_is_detected(tmp_path):
    for name in REFERENCE_SHA256:
        (tmp_path / name).write_text("# not the upstream file\n")

    with pytest.raises(PinError, match="does not match its pin"):
        verify_reference_pins(tmp_path)


def test_missing_reference_file_is_detected(tmp_path):
    with pytest.raises(PinError, match="missing"):
        verify_reference_pins(tmp_path)


def test_paper_revision_is_a_full_sha():
    assert len(PAPER_CODE_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in PAPER_CODE_REVISION)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "x.bin"
    p.write_bytes(b"vektori")
    assert sha256_file(p) == hashlib.sha256(b"vektori").hexdigest()


# ---------------------------------------------------------------------------
# Manifest completeness
# ---------------------------------------------------------------------------


def test_empty_manifest_reports_every_required_pin_missing():
    m = OPDRunManifest()
    missing = m.missing_pins()

    assert set(missing) == set(OPDRunManifest.REQUIRED)
    with pytest.raises(PinError, match="unpinned artifacts"):
        m.require_complete()


def test_complete_manifest_passes():
    m = _complete()
    assert m.missing_pins() == []
    m.require_complete()  # must not raise


def test_partially_filled_manifest_names_only_what_is_missing():
    m = _complete()
    m.harbor_revision = None
    m.task_corpus = ""

    missing = set(m.missing_pins())
    assert missing == {"harbor_revision", "task_corpus"}
    with pytest.raises(PinError) as exc:
        m.require_complete()
    assert "harbor_revision" in str(exc.value)
    assert "student_base_model" not in str(exc.value)


def test_cross_tokenizer_pair_is_the_plans_pair_by_default():
    m = OPDRunManifest()
    assert m.teacher_model == CROSS_TEACHER == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert m.student_tokenizer == CROSS_STUDENT


def test_clip_eps_is_pinned_at_the_reference_default():
    """Recorded as a value, not left to a code default (user call #3)."""
    from vektori_trace.chunk_opd import DEFAULT_CLIP_EPS

    m = OPDRunManifest()
    assert m.clip_eps == DEFAULT_CLIP_EPS == 0.2
    assert m.to_dict()["clip_eps"] == 0.2
    assert m.large_chunk_threshold == 6


def test_adapter_hashes_are_required_and_distinct_fields():
    """Weights, config, and tokenizer are separate pins — one SHA cannot cover all."""
    m = _complete()
    m.student_adapter_config_sha256 = None
    assert "student_adapter_config_sha256" in m.missing_pins()

    m2 = _complete()
    m2.student_tokenizer_sha256 = None
    assert "student_tokenizer_sha256" in m2.missing_pins()


def test_advantage_clamp_none_is_a_real_value_not_a_missing_pin():
    """Upstream's default is no clamp; None must not read as 'unpinned'."""
    m = _complete()
    assert m.advantage_clamp is None
    assert "advantage_clamp" not in OPDRunManifest.REQUIRED
    m.require_complete()


def test_to_dict_carries_reference_hashes_and_paper_identity():
    d = _complete().to_dict()

    assert d["reference_sha256"] == REFERENCE_SHA256
    assert d["paper_arxiv_id"] == "2606.09456"
    assert d["paper_code_revision"] == PAPER_CODE_REVISION
    assert d["teacher_model"] == CROSS_TEACHER


def test_to_dict_is_json_serialisable():
    import json

    json.dumps(_complete().to_dict())


# ---------------------------------------------------------------------------
# Commit pin
# ---------------------------------------------------------------------------


def test_current_commit_is_a_sha_or_none():
    c = current_commit()
    if c is None:
        pytest.skip("not a git checkout")
    base = c[:-6] if c.endswith("-dirty") else c
    assert len(base) == 40
    assert all(ch in "0123456789abcdef" for ch in base)


def test_current_commit_outside_a_repo_is_none(tmp_path):
    assert current_commit(tmp_path) is None


class TestDerivedPins:
    """Hashes are read off disk, so a pin cannot disagree with what loaded."""

    def _adapter(self, tmp_path, weights=b"W", config=b'{"r": 32}'):
        d = tmp_path / "checkpoint-75"
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(weights)
        (d / "adapter_config.json").write_bytes(config)
        return d

    def test_weights_and_config_hashed_separately(self, tmp_path):
        """Two checkpoints sharing a config must not hash alike.

        This is the collision a single directory-level hash would permit, and
        it is the realistic one: Stage-B checkpoints differ in weights while
        their adapter_config.json is byte-identical.
        """
        from vektori_trace.opd_manifest import hash_adapter_dir

        a = hash_adapter_dir(self._adapter(tmp_path / "a", weights=b"ck75"))
        b = hash_adapter_dir(self._adapter(tmp_path / "b", weights=b"ck90"))

        assert a["student_adapter_config_sha256"] == b["student_adapter_config_sha256"]
        assert a["student_adapter_sha256"] != b["student_adapter_sha256"]

    def test_missing_files_are_none_not_raise(self, tmp_path):
        from vektori_trace.opd_manifest import hash_adapter_dir

        got = hash_adapter_dir(tmp_path / "nope")
        assert got == {
            "student_adapter_sha256": None,
            "student_adapter_config_sha256": None,
        }

    def test_tokenizer_hash_covers_chat_template(self, tmp_path):
        """A template change must move the tokenizer pin.

        CLAUDE.md: Qwen3's chat template decides where the empty-think wrapper
        lands. Hashing tokenizer.json alone would call two materially different
        renderers identical.
        """
        from vektori_trace.opd_manifest import hash_tokenizer_dir

        d = tmp_path / "tok"
        d.mkdir()
        (d / "tokenizer.json").write_bytes(b"vocab")
        (d / "tokenizer_config.json").write_bytes(b'{"chat_template": "A"}')
        first = hash_tokenizer_dir(d)

        (d / "tokenizer_config.json").write_bytes(b'{"chat_template": "B"}')
        assert hash_tokenizer_dir(d) != first

    def test_tokenizer_dir_absent_is_none(self, tmp_path):
        from vektori_trace.opd_manifest import hash_tokenizer_dir

        assert hash_tokenizer_dir(tmp_path / "missing") is None

    def test_build_manifest_derives_and_refuses(self, tmp_path):
        from vektori_trace.opd_manifest import PinError, build_run_manifest

        adapter = self._adapter(tmp_path)
        (adapter / "tokenizer.json").write_bytes(b"v")
        (adapter / "tokenizer_config.json").write_bytes(b"{}")

        m = build_run_manifest(
            base_model="Qwen/Qwen3-14B",
            adapter_path=str(adapter),
            task_corpus="/data/vektori-out/dsv4-corpus60",
            fireworks_model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
            harbor_revision="abc123",
        )
        assert m.student_adapter_sha256
        assert m.student_tokenizer_sha256
        assert m.vektori_trace_commit
        m.require_complete()  # all required pins derived or supplied

        # Without harbor_revision the gate refuses and names the field.
        m2 = build_run_manifest(
            base_model="Qwen/Qwen3-14B",
            adapter_path=str(adapter),
            task_corpus="/data/x",
            fireworks_model_id="fw",
        )
        with pytest.raises(PinError, match="harbor_revision"):
            m2.require_complete()

    def test_student_tokenizer_pin_is_the_14b_student(self):
        """CROSS_STUDENT aliased to the 8B pilot student until 2026-08-21.

        The manifest defaults `student_tokenizer` to it, so a run would have
        pinned Qwen3-8B while serving 14B.
        """
        from vektori_trace.opd_manifest import OPDRunManifest

        assert OPDRunManifest().student_tokenizer == "Qwen/Qwen3-14B"
