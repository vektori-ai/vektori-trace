"""`FireworksTeacherPool` — request shape and, mostly, refusal to misalign.

The failure this guards against is not a crash. A one-token offset between the
tokens the student sampled and the logprobs the teacher returned produces a
perfectly finite loss that optimises the wrong thing, so every test here is
about the pool raising rather than returning something plausible.

No network: `_post_json` is patched and the recorded payload is asserted, which
is how the request shape stays pinned to `_request_teacher_echo` in
`fw-ai/cookbook` — the function Fireworks' own `sampled_reverse_kl` distillation
uses in production — rather than to a reading of the API reference.
"""

from __future__ import annotations

import pytest

from vektori_trace.providers.teacher import fireworks as tf
from vektori_trace.providers.teacher.base import TeacherScoringError


def _entry(token_id: int, logprob: float, alts: dict[int, float] | None = None) -> dict:
    return {
        "token": f"<{token_id}>",
        "token_id": token_id,
        "logprob": logprob,
        "sampling_logprob": logprob - 1.0,
        "bytes": [],
        "top_logprobs": [
            {"token": f"<{tid}>", "token_id": tid, "logprob": lp}
            for tid, lp in (alts or {}).items()
        ],
    }


def _response(entries: list[dict]) -> dict:
    return {"choices": [{"text": "", "logprobs": {"content": entries}}]}


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    return tf.FireworksTeacherPool(model="accounts/test/models/teacher")


def _patch(monkeypatch, response, sink: list | None = None):
    def fake_post(url, payload, *, timeout, headers=None):
        if sink is not None:
            sink.append({"url": url, "payload": payload, "headers": headers})
        return response

    monkeypatch.setattr(tf, "_post_json", fake_post)


def test_score_ids_sends_the_request_shape_fireworks_own_recipe_sends(pool, monkeypatch):
    """Pinned to `_request_teacher_echo` in fw-ai/cookbook, not to the API prose.

    That function is what Fireworks' `sampled_reverse_kl` distillation runs in
    production, so matching it is the closest thing to a guarantee available
    without an API key.
    """
    calls: list[dict] = []
    _patch(monkeypatch, _response([_entry(50, -0.5), _entry(60, -1.5)]), calls)

    pool.score_ids([10, 20, 30], [50, 60])

    payload = calls[0]["payload"]
    # Ids, not text: the whole reason this path exists.
    assert payload["prompt"] == [10, 20, 30, 50, 60]
    assert payload["echo"] is True
    assert payload["logprobs"] is True
    assert payload["raw_output"] is True
    assert payload["max_tokens"] == 1
    assert payload["temperature"] == 0.0
    # Omitted, not sent as 0 — the vendor recipe only sets it when K > 0.
    assert "top_logprobs" not in payload
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_echo_last_mode_sends_the_cheaper_documented_variant(monkeypatch):
    """`echo_last` ships far fewer logprobs for a 3.5k-token prefix, but the
    vendor's code does not use it, so it is opt-in rather than the default."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    pool = tf.FireworksTeacherPool(model="m", echo_mode="last")
    calls: list[dict] = []
    _patch(monkeypatch, _response([_entry(50, -0.5), _entry(60, -1.5)]), calls)

    pool.score_ids([10, 20, 30], [50, 60])

    assert calls[0]["payload"]["echo_last"] == 2
    assert "echo" not in calls[0]["payload"]


def test_an_unknown_echo_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    with pytest.raises(ValueError, match="echo_mode"):
        tf.FireworksTeacherPool(model="m", echo_mode="suffix")


def test_score_ids_returns_the_model_logprob_not_the_sampling_one(pool, monkeypatch):
    """`sampling_logprob` is renormalised by temperature and filters.

    Training against it would make the objective a KL to a temperature-warped
    teacher rather than to the teacher, and the two differ by a constant that is
    invisible in the loss curve.
    """
    _patch(monkeypatch, _response([_entry(50, -0.25), _entry(60, -2.0)]))

    assert pool.score_ids([10], [50, 60]) == [-0.25, -2.0]


def test_score_ids_finds_the_scored_run_after_a_leading_prefix_entry(pool, monkeypatch):
    """`echo_last` need not start the content array for the result to be usable."""
    _patch(monkeypatch, _response([_entry(30, -9.0), _entry(50, -0.5), _entry(60, -1.5)]))

    assert pool.score_ids([10, 20, 30], [50, 60]) == [-0.5, -1.5]


def test_score_ids_rejects_a_response_that_scored_other_tokens(pool, monkeypatch):
    """The ids we sent are not the ids it received — the silent-corruption case."""
    _patch(monkeypatch, _response([_entry(51, -0.5), _entry(61, -1.5)]))

    with pytest.raises(TeacherScoringError, match="do not appear"):
        pool.score_ids([10], [50, 60])


def test_a_repeated_run_starting_at_zero_is_not_ambiguous(pool, monkeypatch):
    """`echo_last` puts the scored tokens first, so a position-0 match is the
    answer, not one candidate among several. Only a *miss* at 0 needs a scan."""
    _patch(
        monkeypatch,
        _response([_entry(50, -0.5), _entry(50, -0.6), _entry(50, -0.7)]),
    )

    assert pool.score_ids([10], [50, 50]) == [-0.5, -0.6]


def test_score_ids_refuses_an_ambiguous_run(pool, monkeypatch):
    """Two candidate alignments and no reason to prefer either: raise.

    A one-token offset here is the exact failure `docs/OPD.md` describes — finite
    loss, wrong objective — so picking the first hit would be the bug.
    """
    _patch(
        monkeypatch,
        _response(
            [_entry(9, -9.0), _entry(50, -0.5), _entry(50, -0.6), _entry(50, -0.7)]
        ),
    )

    with pytest.raises(TeacherScoringError, match="appears 2 times"):
        pool.score_ids([10], [50, 50])


def test_score_ids_rejects_the_legacy_logprobs_format(pool, monkeypatch):
    """Legacy `top_logprobs` is keyed by token *string*, which cannot map to ids."""
    _patch(
        monkeypatch,
        {"choices": [{"logprobs": {"tokens": ["a"], "token_logprobs": [-0.5]}}]},
    )

    with pytest.raises(TeacherScoringError, match="legacy logprobs format"):
        pool.score_ids([10], [50])


def test_score_ids_rejects_a_response_with_no_logprobs(pool, monkeypatch):
    _patch(monkeypatch, {"choices": [{"text": "hello"}]})

    with pytest.raises(TeacherScoringError, match="no `logprobs`"):
        pool.score_ids([10], [50])


def test_empty_prompt_ids_is_an_error_not_an_empty_result(pool):
    with pytest.raises(TeacherScoringError, match="prompt_ids is empty"):
        pool.score_ids([], [50])


def test_no_tokens_is_an_empty_list_and_no_request(pool, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("should not have called the teacher")

    monkeypatch.setattr(tf, "_post_json", explode)
    assert pool.score_ids([10], []) == []


def test_topk_includes_the_sampled_token_even_when_outside_the_top_k(pool, monkeypatch):
    """vLLM guarantees this; Fireworks does not, so the pool merges it in.

    `distill.align_topk_rows` reads `row[token_id]` unconditionally, so a missing
    sampled token is a KeyError deep in the loss rather than a clear failure here.
    """
    _patch(
        monkeypatch,
        _response([_entry(50, -4.0, alts={70: -0.1, 80: -0.2})]),
    )

    rows = pool.score_ids_topk([10], [50], 2)

    assert rows[0][50] == -4.0
    assert rows[0][70] == -0.1


def test_topk_above_the_public_cap_raises_rather_than_clamping(pool):
    """K is part of the objective. Clamping 16 to 5 changes what was optimised."""
    with pytest.raises(TeacherScoringError, match="cap of 5"):
        pool.score_ids_topk([10], [50], 16)


def test_topk_rejects_alternatives_without_token_ids(pool, monkeypatch):
    entry = _entry(50, -0.5)
    entry["top_logprobs"] = [{"token": "hello", "logprob": -0.1}]
    _patch(monkeypatch, _response([entry]))

    with pytest.raises(TeacherScoringError, match="no token_id"):
        pool.score_ids_topk([10], [50], 1)


def test_text_prefix_scoring_is_refused_with_a_pointer_to_score_ids(pool):
    """No /tokenize endpoint, and a local tokenizer is the boundary-shift risk."""
    with pytest.raises(TeacherScoringError, match="score_ids"):
        pool.prompt_logprobs("some prefix", [50])


def _bare(token_id: int, logprob: float) -> dict:
    """An entry from a deployment that returns no `token_id` — the shape
    Fireworks' own code defends against by round-tripping the token string."""
    return {"token": f"<{token_id}>", "logprob": logprob, "bytes": []}


def test_positional_fallback_uses_the_full_length_layout(pool, monkeypatch):
    """No ids anywhere: `content[i]` scores `token[i]`, so the sampled tokens sit
    at `prompt_len:`. Verified by count, since nothing else can verify it."""
    _patch(
        monkeypatch,
        _response(
            [_bare(10, -9.0), _bare(20, -9.0), _bare(50, -0.5), _bare(60, -1.5)]
        ),
    )

    assert pool.score_ids([10, 20], [50, 60]) == [-0.5, -1.5]


def test_positional_fallback_uses_the_p_plus_c_minus_one_layout(pool, monkeypatch):
    """The "training-aligned" shape: `content[i]` scores `token[i+1]`, one entry
    shorter, because token 0 has nothing conditioning it."""
    _patch(
        monkeypatch,
        _response([_bare(20, -9.0), _bare(50, -0.5), _bare(60, -1.5)]),
    )

    assert pool.score_ids([10, 20], [50, 60]) == [-0.5, -1.5]


def test_positional_fallback_refuses_an_unrecognised_length(pool, monkeypatch):
    """Neither layout fits, and no ids to fall back on. Guessing here produces a
    finite loss against the wrong positions — the failure this module exists for."""
    _patch(monkeypatch, _response([_bare(99, -9.0), _bare(98, -9.0)]))

    with pytest.raises(TeacherScoringError, match="matches no echo_mode"):
        pool.score_ids([10, 20, 30], [50, 60])


def test_ids_are_still_verified_when_present_despite_the_fallback(pool, monkeypatch):
    """The fallback must not weaken the strict path: a response that carries ids
    and scored the wrong tokens still raises rather than aligning by count."""
    _patch(
        monkeypatch,
        _response([_entry(10, -9.0), _entry(20, -9.0), _entry(51, -0.5), _entry(61, -1.5)]),
    )

    with pytest.raises(TeacherScoringError, match="do not appear"):
        pool.score_ids([10, 20], [50, 60])


def test_missing_api_key_fails_at_construction(monkeypatch):
    """Not on the first scoring call, halfway into a run that has booked GPU time."""
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    with pytest.raises(TeacherScoringError, match="FIREWORKS_API_KEY"):
        tf.FireworksTeacherPool()


def test_provenance_records_the_quantisation(pool):
    """FP8 teacher logprobs are not comparable to a bf16 teacher's, so a report
    that omits this invites comparing two runs that differ in the teacher."""
    prov = pool.provenance()

    assert prov["teacher_host"] == "fireworks"
    assert "fp8" in prov["teacher_quantisation"]
