"""`run_opd_training_modal` — the pre-GPU contract.

Every check here is about what must fail *before* a container exists. A Modal
run costs money from the moment the image pulls, so a missing key, a missing
corpus or a config that cannot fit the card must be caught on this side of the
call. Nothing in this file talks to Modal.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from vektori_trace.distill import (
    OPDTrainConfig,
    _student_size_b,
    force_gpu_safe_config,
    modal_timeout_for,
    run_opd_training_modal,
)


def test_student_size_parsed_from_hf_id():
    assert _student_size_b("Qwen/Qwen3-14B") == 14.0
    assert _student_size_b("Qwen/Qwen3-8B") == 8.0
    assert _student_size_b("Qwen/Qwen3-Coder-30B-A3B-Instruct") == 30.0


def test_unparseable_student_size_is_not_an_error():
    """A name we cannot read must not block a run — the parse only forces
    safety on, it never relaxes it."""
    assert _student_size_b("some-org/mystery-model") is None


def test_timeout_covers_the_work_requested():
    """A fixed timeout kills a default-length run after the GPU is paid for."""
    default = OPDTrainConfig(max_steps=200, examples_per_step=4)
    # 200 x 4 examples cannot fit in the 4h the SFT path hardcodes.
    assert modal_timeout_for(default) > 4 * 60 * 60
    short = OPDTrainConfig(max_steps=30, examples_per_step=4)
    assert modal_timeout_for(short) < modal_timeout_for(default)
    # Floor covers cold start and weight download even for a 1-step run.
    assert modal_timeout_for(OPDTrainConfig(max_steps=1, examples_per_step=1)) >= 3600
    # Modal's own ceiling.
    huge = OPDTrainConfig(max_steps=100_000, examples_per_step=8)
    assert modal_timeout_for(huge) <= 24 * 60 * 60


def test_missing_api_key_fails_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    prefixes = tmp_path / "prefixes"
    prefixes.mkdir()
    bridge = tmp_path / "bridge.json"
    bridge.write_text("{}")
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        run_opd_training_modal(
            OPDTrainConfig(output_dir=tmp_path / "out"),
            prefixes_dir=prefixes,
            bridge_path=bridge,
            fireworks_model="accounts/fireworks/models/x",
            teacher_tokenizer_id="org/tok",
        )


def test_missing_corpus_fails_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    bridge = tmp_path / "bridge.json"
    bridge.write_text("{}")
    with pytest.raises(FileNotFoundError, match="prefix corpus"):
        run_opd_training_modal(
            OPDTrainConfig(output_dir=tmp_path / "out"),
            prefixes_dir=tmp_path / "nope",
            bridge_path=bridge,
            fireworks_model="accounts/fireworks/models/x",
            teacher_tokenizer_id="org/tok",
        )


def test_missing_bridge_fails_before_any_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    prefixes = tmp_path / "prefixes"
    prefixes.mkdir()
    with pytest.raises(FileNotFoundError, match="bridge"):
        run_opd_training_modal(
            OPDTrainConfig(output_dir=tmp_path / "out"),
            prefixes_dir=prefixes,
            bridge_path=tmp_path / "nope.json",
            fireworks_model="accounts/fireworks/models/x",
            teacher_tokenizer_id="org/tok",
        )


def test_remote_function_is_serialized():
    """Modal rejects a locally-defined function unless it is pickled.

    `runtime/serve.py` already carries this fix; the SFT path did not, so its
    decorator raised before reaching a GPU. Both remote entry points are
    checked here because the failure is invisible until a real run.
    """
    from vektori_trace import train

    for fn in (run_opd_training_modal, train.train_lora_modal):
        src = inspect.getsource(fn)
        assert "serialized=True" in src, f"{fn.__name__} must pass serialized=True"


def test_teacher_model_and_tokenizer_are_separate_parameters():
    """A Fireworks resource path is not an HF repo id. Collapsing the two
    fails inside the container, after the image pull."""
    params = inspect.signature(run_opd_training_modal).parameters
    assert "fireworks_model" in params
    assert "teacher_tokenizer_id" in params


def test_upload_paths_are_volume_root_relative():
    """Volume upload paths are relative to the volume root, while reads are
    under the mount. Prefixing the upload with the mount lands files a
    directory deeper than they are read — the bug `stage_local_adapter_to_volume`
    has. Assert we do not copy it."""
    src = inspect.getsource(run_opd_training_modal)
    assert 'batch.put_directory(str(prefixes_dir), f"{inputs_root}/prefixes")' in src
    assert 'batch.put_file(str(bridge_path), f"{inputs_root}/bridge.json")' in src


def test_remote_commits_the_volume_even_on_failure():
    """Late aborts (alignment, clamp gate, granularity gate) happen after hours
    of GPU time, and the log is the only record of why. It must survive."""
    src = inspect.getsource(run_opd_training_modal)
    assert "finally:" in src
    assert "vol.commit()" in src
    finally_at = src.index("finally:")
    assert src.index("vol.commit()") > finally_at, "commit must be in the finally"


def test_large_student_forces_gradient_checkpointing():
    """14B bf16 weights plus un-checkpointed activations exceed a 48 GB card,
    and the CLI flag defaults to off."""
    cfg = OPDTrainConfig(student_model="Qwen/Qwen3-14B", gradient_checkpointing=False)
    assert force_gpu_safe_config(cfg).gradient_checkpointing is True


def test_small_student_keeps_the_callers_setting():
    """The rule only tightens; it must not silently change an 8B run."""
    cfg = OPDTrainConfig(student_model="Qwen/Qwen3-8B", gradient_checkpointing=False)
    assert force_gpu_safe_config(cfg).gradient_checkpointing is False


def test_unparseable_student_keeps_the_callers_setting():
    cfg = OPDTrainConfig(student_model="org/mystery", gradient_checkpointing=False)
    assert force_gpu_safe_config(cfg).gradient_checkpointing is False


def test_cli_modal_is_opt_in():
    """A bare `distill` must never allocate a GPU. Repo rule: GPU spend is a
    per-run decision, not a default."""
    import argparse

    from vektori_trace.cli.commands.distill import register_distill

    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    register_distill(sub)
    args = p.parse_args(["distill", "--teacher-traces", "x"])
    assert args.modal_gpu is None


def test_cli_accepts_modal_gpu():
    import argparse

    from vektori_trace.cli.commands.distill import register_distill

    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    register_distill(sub)
    args = p.parse_args(["distill", "--teacher-traces", "x", "--modal-gpu", "L40S"])
    assert args.modal_gpu == "L40S"


def test_local_path_unchanged_when_no_modal_gpu(tmp_path):
    """The in-process path is still the default code path, untouched."""
    src = inspect.getsource(
        __import__(
            "vektori_trace.cli.commands.distill", fromlist=["cmd_distill"]
        ).cmd_distill
    )
    assert "if modal_gpu:" in src
    assert "run_opd_training(" in src
    assert Path("vektori_trace/cli/commands/distill.py")


def test_remote_does_not_import_the_cli_package():
    """A GPU container must not need the CLI's dependencies.

    Importing anything under `vektori_trace.cli` executes that package's
    `__init__`, which pulls in every command module — including `llm.py` and
    its `openai` import. The training image carries torch/transformers/peft and
    no API client, so this surfaced as `ModuleNotFoundError: No module named
    'openai'` *after* the image pull, on a GPU billing by the second.
    """
    src = inspect.getsource(run_opd_training_modal)
    assert "vektori_trace.cli" not in src, (
        "remote imports the CLI package — that drags openai onto the GPU image"
    )
    assert "from vektori_trace.traces import load_teacher_trajectories" in src


def test_trace_loader_is_importable_without_the_cli():
    """The loader must live outside `cli/` so the container can reach it."""
    from vektori_trace.traces import load_teacher_trajectories

    assert callable(load_teacher_trajectories)
