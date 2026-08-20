"""Phase-0 §6.1 pins — the manifest and the vendored paper-code hashes.

The point of these tests is that a pin nobody filled in must be loud. A manifest
that silently defaults its way to "complete" is worse than no manifest, because
a run report would then carry pins the run never actually verified.
"""

from __future__ import annotations

import pytest

from vektori_trace.opd_manifest import (
    PAPER_CODE_REVISION,
    VENDOR_SHA256,
    OPDRunManifest,
    PinError,
    current_commit,
    sha256_file,
    verify_vendor_pins,
)
from vektori_trace.tokenizer_check import CROSS_STUDENT, CROSS_TEACHER


def _complete() -> OPDRunManifest:
    return OPDRunManifest(
        student_base_model="Qwen/Qwen3-14B",
        student_adapter_path="/data/stage-b/checkpoint-75",
        student_adapter_sha256="0" * 64,
        fireworks_model_id="accounts/fireworks/models/deepseek-v4-flash-0731",
        harbor_revision="abc1234",
        task_corpus="cs/corpus50_v3",
        vektori_trace_commit="def5678",
    )


# ---------------------------------------------------------------------------
# Vendored paper code
# ---------------------------------------------------------------------------


def test_vendored_paper_code_matches_its_pin():
    """The port in chunk_opd.py is only meaningful against these exact bytes."""
    observed = verify_vendor_pins()
    assert observed == VENDOR_SHA256


def test_altered_vendor_file_is_detected(tmp_path):
    for name in VENDOR_SHA256:
        (tmp_path / name).write_text("# not the upstream file\n")

    with pytest.raises(PinError, match="does not match its pin"):
        verify_vendor_pins(tmp_path)


def test_missing_vendor_file_is_detected(tmp_path):
    with pytest.raises(PinError, match="missing"):
        verify_vendor_pins(tmp_path)


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


def test_advantage_clamp_none_is_a_real_value_not_a_missing_pin():
    """Upstream's default is no clamp; None must not read as 'unpinned'."""
    m = _complete()
    assert m.advantage_clamp is None
    assert "advantage_clamp" not in OPDRunManifest.REQUIRED
    m.require_complete()


def test_to_dict_carries_vendor_hashes_and_paper_identity():
    d = _complete().to_dict()

    assert d["vendor_sha256"] == VENDOR_SHA256
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
