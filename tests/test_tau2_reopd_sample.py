"""Capture validation for the ReOPD sampling path.

Every refusal here corresponds to a failure that yields a finite loss and a
clean-looking log if it is allowed through.
"""

from __future__ import annotations

import base64

import pytest

from vektori_trace.tau2.reopd_sample import (
    ReOPDSampleError,
    build_capture,
    capture_fingerprint,
    capture_to_sampled_action,
    sample_batch,
)


class FakeTokenizer:
    """One id -> one byte, so reconstruction is trivially checkable."""

    VOCAB = {1: "he", 2: "llo", 3: " wo", 4: "rld", 9: "<|im_end|>"}

    def decode(self, ids, **kw):
        return "".join(self.VOCAB.get(int(i), "?") for i in ids)


class FakePrefix:
    def __init__(self, prefix_id, prompt_token_ids):
        self.prefix_id = prefix_id
        self.prompt_token_ids = prompt_token_ids


def _body(token_ids=(1, 2), logprobs=(-0.1, -0.2), text="hello",
          prompt_ids=(50, 51), finish="stop"):
    return {
        "id": "req-1", "model": "ck35",
        "choices": [{
            "text": text,
            "token_ids": list(token_ids),
            "prompt_token_ids": list(prompt_ids),
            "logprobs": {"token_logprobs": list(logprobs)},
            "finish_reason": finish,
        }],
    }


def _capture(**kw):
    return build_capture("42#3", 0, _body(**kw), [50, 51], "fp",
                         FakeTokenizer(), policy_version="ck35")


# --- a good capture -------------------------------------------------------


def test_capture_carries_everything_the_loss_needs():
    c = _capture()
    assert c["key"] == "42#3#0"
    assert c["action_token_ids"] == [1, 2]
    assert c["behavior_logprobs"] == [-0.1, -0.2]
    assert base64.b64decode(c["action_bytes_b64"]) == b"hello"
    assert [base64.b64decode(b) for b in c["action_token_bytes_b64"]] == [b"he", b"llo"]


def test_trailing_stop_token_is_allowed():
    """vLLM returns the stop token among ids but omits it from `text`."""
    c = _capture(token_ids=(1, 2, 9), logprobs=(-0.1, -0.2, -0.3))
    assert base64.b64decode(c["action_bytes_b64"]) == b"hello<|im_end|>"


def test_capture_converts_to_sampled_action():
    a = capture_to_sampled_action(_capture())
    assert a.prefix_id == "42#3"
    assert a.action_bytes == b"hello"
    assert a.behavior_logprobs == [-0.1, -0.2]
    assert a.policy_version == "ck35"
    assert a.prompt_token_ids == [50, 51]


# --- refusals -------------------------------------------------------------


def test_refuses_missing_behavior_logprobs():
    with pytest.raises(ReOPDSampleError, match="log pi_old"):
        _capture(logprobs=())


def test_refuses_logprob_count_mismatch():
    with pytest.raises(ReOPDSampleError, match="one behavior logprob"):
        _capture(token_ids=(1, 2, 3), logprobs=(-0.1, -0.2))


def test_refuses_cap_hit():
    with pytest.raises(ReOPDSampleError, match="fragment"):
        _capture(finish="length")


def test_refuses_prompt_id_mismatch():
    """A truncated-from-the-front prompt samples an undescribable state."""
    with pytest.raises(ReOPDSampleError, match="different prompt ids"):
        _capture(prompt_ids=(50, 99))


def test_refuses_missing_prompt_ids():
    with pytest.raises(ReOPDSampleError, match="omitted prompt ids"):
        _capture(prompt_ids=())


def test_refuses_empty_action():
    with pytest.raises(ReOPDSampleError, match="empty sampled action"):
        _capture(text="")


def test_refuses_ids_that_disagree_with_text():
    with pytest.raises(ReOPDSampleError, match="do not reconstruct"):
        _capture(text="goodbye")


def test_refuses_content_beyond_the_reported_text():
    """Ids carrying real content the server did not report."""
    with pytest.raises(ReOPDSampleError, match="beyond the returned text"):
        _capture(token_ids=(1, 2, 3, 4), logprobs=(-0.1,) * 4, text="hello")


# --- fingerprints and resume ---------------------------------------------


def test_fingerprint_changes_with_policy_version():
    kw = dict(model="m", temperature=0.7, max_tokens=2048, prompt_ids=[1, 2])
    a = capture_fingerprint("42#3", 0, policy_version="v1", **kw)
    b = capture_fingerprint("42#3", 0, policy_version="v2", **kw)
    assert a != b


def test_fingerprint_changes_with_cap():
    kw = dict(model="m", policy_version="v1", temperature=0.7, prompt_ids=[1, 2])
    assert (capture_fingerprint("42#3", 0, max_tokens=512, **kw)
            != capture_fingerprint("42#3", 0, max_tokens=2048, **kw))


def test_sample_batch_reuses_matching_captures():
    calls = []

    def poster(url, payload, timeout):
        calls.append(payload)
        return 200, _body(prompt_ids=payload["prompt"])

    p = FakePrefix("42#3", [50, 51])
    first = sample_batch([p], api_base="http://x", model="ck35",
                         tokenizer=FakeTokenizer(), policy_version="ck35",
                         max_tokens=2048, temperature=0.7, poster=poster)
    assert len(calls) == 1

    second = sample_batch([p], api_base="http://x", model="ck35",
                          tokenizer=FakeTokenizer(), policy_version="ck35",
                          max_tokens=2048, temperature=0.7, poster=poster,
                          already={c["key"]: c for c in first})
    assert len(calls) == 1                       # nothing re-sampled
    assert second[0]["key"] == first[0]["key"]


def test_sample_batch_refuses_stale_capture():
    """A capture from an earlier policy version must not be resumed into."""
    def poster(url, payload, timeout):
        return 200, _body(prompt_ids=payload["prompt"])

    p = FakePrefix("42#3", [50, 51])
    old = sample_batch([p], api_base="http://x", model="ck35",
                       tokenizer=FakeTokenizer(), policy_version="update-2",
                       max_tokens=2048, temperature=0.7, poster=poster)

    with pytest.raises(ReOPDSampleError, match="different settings"):
        sample_batch([p], api_base="http://x", model="ck35",
                     tokenizer=FakeTokenizer(), policy_version="update-7",
                     max_tokens=2048, temperature=0.7, poster=poster,
                     already={c["key"]: c for c in old})


def test_sample_batch_sends_ids_not_a_string():
    """Re-rendering messages would retokenize under a drifting tokenizer."""
    seen = {}

    def poster(url, payload, timeout):
        seen.update(payload)
        return 200, _body(prompt_ids=payload["prompt"])

    sample_batch([FakePrefix("42#3", [50, 51])], api_base="http://x",
                 model="ck35", tokenizer=FakeTokenizer(),
                 policy_version="ck35", max_tokens=2048, temperature=0.7,
                 poster=poster)
    assert seen["prompt"] == [50, 51]
    assert isinstance(seen["prompt"], list)


def test_on_capture_fires_per_sample():
    """A crash at sample n must not discard the n-1 already generated."""
    landed = []

    def poster(url, payload, timeout):
        return 200, _body(prompt_ids=payload["prompt"])

    sample_batch([FakePrefix("1#0", [50, 51]), FakePrefix("2#0", [50, 51])],
                 api_base="http://x", model="ck35", tokenizer=FakeTokenizer(),
                 policy_version="ck35", max_tokens=2048, temperature=0.7,
                 poster=poster, on_capture=landed.append)
    assert len(landed) == 2


def test_http_error_is_raised():
    def poster(url, payload, timeout):
        return 503, {"error": "no capacity"}

    with pytest.raises(ReOPDSampleError, match="HTTP 503"):
        sample_batch([FakePrefix("1#0", [50, 51])], api_base="http://x",
                     model="ck35", tokenizer=FakeTokenizer(),
                     policy_version="ck35", max_tokens=2048, temperature=0.7,
                     poster=poster)


# --- transport failures are statuses, not exceptions ----------------------


def test_post_json_reports_a_timeout_as_status_zero(monkeypatch):
    """urlopen raises TimeoutError/URLError, neither of which is an HTTPError.

    Callers branch on the returned status -- verify_endpoint's long-prompt probe
    treats a non-200 as "this server is too short" -- so an uncaught transport
    error propagated raw out of the gate instead of being reported. 0 is not a
    real HTTP status, so it cannot be mistaken for a server-issued rejection.
    """
    import urllib.request

    import vektori_trace.tau2.reopd_sample as S

    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    st, body = S.post_json("http://x/v1/completions", {"prompt": [1]}, 1.0)
    assert st == 0
    assert "TimeoutError" in body["error"]


def test_post_json_reports_a_refused_connection_as_status_zero(monkeypatch):
    import urllib.error
    import urllib.request

    import vektori_trace.tau2.reopd_sample as S

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    st, body = S.post_json("http://x/v1/completions", {"prompt": [1]}, 1.0)
    assert st == 0
    assert "URLError" in body["error"]
