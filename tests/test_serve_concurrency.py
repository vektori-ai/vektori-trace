"""The serving container must batch, not clone.

These are source-level assertions rather than live Modal calls, because the
failure they guard against is invisible locally and only shows up as a
multiplied GPU bill. On 2026-08-13 a sweep with two rollout workers and one
metrics poller ran three L40S containers at once — Modal's default is one
in-flight request per container, so each concurrent request booted another
copy of the model. See `docs/MODAL-CONCURRENCY.md`.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from vektori_trace.runtime import serve

REPO_ROOT = Path(__file__).resolve().parents[1]


def _serve_model_source() -> str:
    return inspect.getsource(serve.serve_model)


def test_serving_class_is_pinned_to_one_container():
    """Without this, every concurrent request is answered by a new GPU."""
    assert "max_containers=1" in _serve_model_source()


def test_serving_class_declares_input_concurrency():
    """`max_containers=1` alone would serialise the sweep: one container that
    accepts one input at a time is a queue, not a batch. The concurrency
    decorator is what lets vLLM see more than one request."""
    src = _serve_model_source()
    assert "modal.concurrent(" in src
    assert "max_inputs=" in src


def test_concurrency_decorator_is_applied_to_the_class():
    """Modal's docs: for the class pattern the decorator goes on the class, not
    on individual methods — all methods share the container."""
    src = _serve_model_source()
    at = src.index("@modal.concurrent(")
    cls_at = src.index("class VllmServer:")
    between = src[at:cls_at]
    assert "def " not in between, "@modal.concurrent must sit directly on the class"


def test_no_deprecated_concurrency_kwarg():
    """`allow_concurrent_inputs` was removed in Modal 1.0 and is not a valid
    `App.cls` parameter — passing it raises rather than configuring anything."""
    src = _serve_model_source()
    assert "allow_concurrent_inputs" not in src


def test_max_inputs_exceeds_target_inputs():
    """`target_inputs` is the autoscaler's aim, `max_inputs` the hard ceiling.
    Inverting them would cap bursts below the steady-state target."""
    src = _serve_model_source()
    max_i = int(src.split("max_inputs=")[1].split(",")[0].split(")")[0])
    target_i = int(src.split("target_inputs=")[1].split(",")[0].split(")")[0])
    assert max_i > target_i


def test_default_max_model_len_covers_observed_prompts():
    """Sized from real traffic, not a round number.

    Largest single agent prompt across run6's 24 rollouts was 11,304 tokens.
    A default below that rejects live requests mid-sweep; a very large one
    starves concurrency, because vLLM plans how many sequences fit against
    this value.
    """
    spec = (REPO_ROOT / "scripts" / "serve_student.py").read_text()
    ns = argparse.Namespace()
    # Parse the default out of the source rather than importing the script,
    # which pulls in modal at import time.
    chunk = spec.split('"--max-model-len"')[1]
    default = int(chunk.split("default=")[1].split(",")[0])
    ns.max_model_len = default
    assert default > 11_304, "below the largest observed prompt — would reject requests"
    assert default <= 20_480, "so large that the scheduler admits too few sequences"
