"""`BedrockTeacherPool` — request routing and both documented response shapes.

Bedrock's schema selection is implicit: `max_tokens` routes to OpenAICompletion,
which supports prompt logprobs, and `max_gen_len` routes to BedrockCompletion,
which supports generated-token logprobs only. Sending the wrong one gets a
successful response containing the wrong quantity, so the request shape is
asserted rather than assumed.

The response is parsed defensively for a documented reason: AWS's own CMI example
shows `prompt_logprobs` as a top-level field, while vLLM's OpenAI server nests it
under `choices[0]`. Both are tested because the docs do not say which an imported
model returns — `cli.py probe-teacher --backend bedrock` is what settles it
against real weights.

A stub client stands in for boto3; nothing here reaches AWS.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.providers.teacher.base import TeacherScoringError
from vektori_trace.providers.teacher.bedrock import BedrockTeacherPool, _find_prompt_logprobs


class StubClient:
    """Minimal `bedrock-runtime` stand-in: records the body, replays a response."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def invoke_model(self, *, modelId, body, accept, contentType):
        self.calls.append({"modelId": modelId, "body": json.loads(body)})

        class _Body:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode()

        return {"body": _Body(self.response)}


def _vllm_entries(*pairs: tuple[int, float]) -> list:
    """vLLM's prompt_logprobs shape: one dict per position, keyed by token id."""
    return [{str(tid): {"logprob": lp, "rank": 1}} for tid, lp in pairs]


def _pool(response: dict) -> BedrockTeacherPool:
    return BedrockTeacherPool(
        model_id="arn:aws:bedrock:us-east-1:1234:imported-model/abc",
        region="us-east-1",
        client=StubClient(response),
    )


def test_scoring_routes_to_the_openai_completion_schema():
    """`max_tokens`, never `max_gen_len` — the latter scores generated tokens only,
    which is the wrong quantity for OPD however successful the response looks."""
    # 3 prompt ids + 2 scored = 5 positions, the first unconditioned.
    pool = _pool(
        {"prompt_logprobs": [None, *_vllm_entries((20, -1.0), (30, -2.0), (50, -0.5), (60, -1.5))]}
    )

    pool.score_ids([10, 20, 30], [50, 60])

    body = pool.client.calls[0]["body"]
    assert "max_gen_len" not in body
    assert body["max_tokens"] == 1
    assert body["prompt"] == [10, 20, 30, 50, 60]
    assert body["prompt_logprobs"] == 1


def test_reads_prompt_logprobs_from_the_top_level():
    """The shape AWS's own Custom Model Import example documents."""
    pool = _pool(
        {"prompt_logprobs": [None, *_vllm_entries((20, -1.0), (50, -0.5), (60, -1.5))]}
    )

    assert pool.score_ids([10, 20], [50, 60]) == [-0.5, -1.5]


def test_reads_prompt_logprobs_nested_under_choices():
    """The shape vLLM's OpenAI server returns. The docs do not say which an
    imported model uses, so both work rather than one being guessed."""
    pool = _pool(
        {
            "choices": [
                {
                    "text": "",
                    "prompt_logprobs": [
                        None,
                        *_vllm_entries((20, -1.0), (50, -0.5), (60, -1.5)),
                    ],
                }
            ]
        }
    )

    assert pool.score_ids([10, 20], [50, 60]) == [-0.5, -1.5]


def test_missing_prompt_logprobs_names_both_causes():
    """Either the model predates logprob support or the request routed to the
    wrong schema. A reader hitting this should not have to guess which."""
    pool = _pool({"choices": [{"text": "hello"}], "usage": {}})

    with pytest.raises(TeacherScoringError, match=r"2025-11-11|BedrockCompletion"):
        pool.score_ids([10], [50])


def test_length_mismatch_refuses_to_guess_the_alignment():
    """Shorter-than-prompt logprobs have several plausible alignments and no
    correct one — the same discipline `teacher.VllmTeacherPool` applies."""
    pool = _pool({"prompt_logprobs": [None, *_vllm_entries((50, -0.5))]})

    with pytest.raises(TeacherScoringError, match="refusing to guess"):
        pool.score_ids([10, 20], [50, 60])


def test_a_teacher_that_scored_different_tokens_raises():
    pool = _pool(
        {"prompt_logprobs": [None, *_vllm_entries((20, -1.0), (51, -0.5), (61, -1.5))]}
    )

    with pytest.raises(TeacherScoringError, match="absent from prompt_logprobs"):
        pool.score_ids([10, 20], [50, 60])


def test_non_cmi_region_is_rejected_at_construction():
    """`ap-south-1` is the repo's usual default and is not a CMI region; the
    resulting invoke_model 404 reads like a missing model, not a missing feature."""
    with pytest.raises(TeacherScoringError, match="Custom Model Import"):
        BedrockTeacherPool(model_id="arn:x", region="ap-south-1", client=StubClient({}))


def test_empty_prompt_ids_is_an_error():
    pool = _pool({})

    with pytest.raises(TeacherScoringError, match="prompt_ids is empty"):
        pool.score_ids([], [50])


def test_topk_requires_the_sampled_token_to_be_present():
    """vLLM always includes the prompt's own token, so its absence means the
    response did not come from the tokens we sent."""
    pool = _pool(
        {
            "prompt_logprobs": [
                None,
                {"20": {"logprob": -1.0}},
                {"70": {"logprob": -0.1}, "80": {"logprob": -0.2}},
            ]
        }
    )

    with pytest.raises(TeacherScoringError, match="missing from the top-2"):
        pool.score_ids_topk([10, 20], [50], 2)


def test_find_prompt_logprobs_reports_where_it_looked():
    assert _find_prompt_logprobs({"prompt_logprobs": [1]}) == ([1], "top level")
    assert _find_prompt_logprobs({"choices": [{"prompt_logprobs": [1]}]}) == ([1], "choices[0]")
    assert _find_prompt_logprobs({"choices": [{}]}) == (None, "nowhere")
