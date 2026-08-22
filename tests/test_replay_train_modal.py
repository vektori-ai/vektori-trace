"""Persist contract for the replay training Modal wrapper.

These are source-level assertions, not a Modal run. They exist because a
$0.60 A100 job completed, reported `loss 0.506147, 560/560 tensors moved`, and
produced no durable adapter: `save_pretrained` wrote 490 MB into the
container's /tmp and a `< 200 MB` filter silently dropped it from the function
result. The banner was indistinguishable from a good run.

The rule these encode: **the artifact is the success criterion, not the
metrics.**
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "replay_train_modal.py"


@pytest.fixture(scope="module")
def src() -> str:
    return SCRIPT.read_text()


class TestPersistContract:
    def test_no_size_filter_on_returned_files(self, src: str):
        """The bug verbatim: a byte ceiling that drops the weights."""
        assert "200 * 1024 * 1024" not in src
        assert "adapter_files" not in src

    def test_output_dir_is_the_volume_not_tmp(self, src: str):
        """save_pretrained must write where the bytes survive the container."""
        assert "output_dir=dest_dir" in src
        assert 'dest_dir = Path(VOLUME_MOUNT) / V_REPLAY_IN_VOLUME' in src

    def test_weights_presence_is_asserted_before_success(self, src: str):
        assert "adapter_model.safetensors" in src
        assert "MIN_ADAPTER_BYTES" in src
        # and the floor is meaningful: ck75 is 513,877,864 bytes
        assert "400 * 1024 * 1024" in src

    def test_reload_is_verified_on_the_volume(self, src: str):
        """§8.4 wants the archived artifact reloadable, not a /tmp file."""
        assert "verify_adapter_reloadable(dest_dir)" in src

    def test_commit_happens_in_finally(self, src: str):
        """A late abort must not take the per-example rows with it."""
        i_finally = src.index("finally:")
        i_commit = src.index("vol.commit()", i_finally)
        assert i_commit > i_finally

    def test_entrypoint_refuses_a_missing_artifact(self, src: str):
        """The local side must fail rather than print a success banner."""
        tail = src[src.index("def main("):]
        assert "raise SystemExit" in tail
        assert "NO adapter_model.safetensors IS ON" in tail

    def test_refuses_a_non_empty_destination(self, src: str):
        assert "already exists and is not empty" in src


class TestBatchContract:
    def test_stored_teacher_actions_are_passed(self, src: str):
        """Omitting them makes assert_action_is_student_sampled a no-op (§8.4)."""
        assert "stored_teacher_actions=stored_actions or None" in src

    def test_sample_set_validated_before_gpu(self, src: str):
        i_val = src.index("validate_sample_set(")
        i_load = src.index("load_v0_for_training(")
        assert i_val < i_load, "completeness must be checked before weights load"

    def test_trace_share_default_matches_this_batch(self, src: str):
        """The documented command and the default must not disagree.

        click-3482@37 holds 43.01% of supervised tokens; a copy-pasted rerun at
        the 0.35 library default would fail closed on concentration.
        """
        tail = src[src.index("def main("):]
        assert "max_trace_share: float = 0.45" in tail
