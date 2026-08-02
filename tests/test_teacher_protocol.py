"""Every teacher pool really implements `Teacher`.

No type checker runs in CI, so the Protocol in `providers.teacher.protocol` is
documentation on its own. This is the part with teeth. `runtime_checkable`
only checks that the methods exist, so signatures are compared separately —
otherwise a backend could rename a parameter and still pass.

Adding a teacher backend? Add it to `POOLS` and this will tell you what is
missing.
"""

from __future__ import annotations

import inspect

import pytest

from vektori_trace.providers.teacher.base import InMemoryIdScoringPool, VllmTeacherPool
from vektori_trace.providers.teacher.bedrock import BedrockTeacherPool
from vektori_trace.providers.teacher.cross import CrossTokenizerTeacherPool
from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
from vektori_trace.providers.teacher.protocol import (
    IdScoringPool,
    Teacher,
    TopKScoringPool,
)

POOLS = [
    InMemoryIdScoringPool,
    VllmTeacherPool,
    BedrockTeacherPool,
    FireworksTeacherPool,
    CrossTokenizerTeacherPool,
]

TEACHER_METHODS = ["score_ids", "score_ids_topk", "generate", "provenance"]


@pytest.mark.parametrize("pool", POOLS, ids=lambda c: c.__name__)
def test_pool_has_every_teacher_method(pool):
    missing = [m for m in TEACHER_METHODS if not callable(getattr(pool, m, None))]
    assert not missing, f"{pool.__name__} is missing {missing}"


@pytest.mark.parametrize("pool", POOLS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", TEACHER_METHODS)
def test_pool_method_signature_matches_protocol(pool, method):
    """Same parameter names, order, and kind as the Protocol declares.

    Return annotations are not compared — several pools legitimately narrow
    them — but the call shape has to match, because callers pass positionally
    and by keyword (`generate(prompt, max_tokens=...)`).
    """
    want = inspect.signature(getattr(Teacher, method))
    got = inspect.signature(getattr(pool, method))
    want_params = [(p.name, p.kind) for p in want.parameters.values()]
    got_params = [(p.name, p.kind) for p in got.parameters.values()]
    assert got_params == want_params, (
        f"{pool.__name__}.{method}{got} does not match Teacher.{method}{want}"
    )


@pytest.mark.parametrize("pool", POOLS, ids=lambda c: c.__name__)
def test_pool_satisfies_narrow_protocols(pool):
    """The two narrow Protocols are what `distill` annotates against."""
    assert callable(getattr(pool, "score_ids", None)), f"{pool.__name__} lacks score_ids"
    assert callable(getattr(pool, "score_ids_topk", None)), (
        f"{pool.__name__} lacks score_ids_topk"
    )


def test_narrow_protocols_are_runtime_checkable():
    """`isinstance` against these has to keep working — cross.py relies on it."""
    for proto in (IdScoringPool, TopKScoringPool, Teacher):
        assert isinstance(InMemoryIdScoringPool(), proto), proto.__name__


def test_cross_tokenizer_pool_has_no_prompt_logprobs():
    """The documented asymmetry, pinned.

    `CrossTokenizerTeacherPool` cannot offer `prompt_logprobs` — the prompt is
    in the student's tokenisation. If someone adds one, `Teacher` needs
    revisiting rather than the method quietly appearing.
    """
    assert not hasattr(CrossTokenizerTeacherPool, "prompt_logprobs")
    for pool in (InMemoryIdScoringPool, VllmTeacherPool, BedrockTeacherPool, FireworksTeacherPool):
        assert callable(getattr(pool, "prompt_logprobs", None)), pool.__name__
