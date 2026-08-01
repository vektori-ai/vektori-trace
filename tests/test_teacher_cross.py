"""Tests for `teacher_cross` — offline, no network, no real tokenizer.

All tests use injected fakes so they run without FIREWORKS_API_KEY, GPU, or
the real DeepSeek/Qwen tokenizers. This matches the requirement "offline only".
"""

from __future__ import annotations

import pytest

from vektori_trace import teacher_cross as tc
from vektori_trace.encoding_dsv4 import (
    ASSISTANT_SP_TOKEN,
    ENCODING_DSV4_SHA256,
    USER_SP_TOKEN,
    bos_token,
    eos_token,
    thinking_end_token,
)
from vektori_trace.schema import Turn

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePool:
    """Records calls to score_ids and returns canned logprobs."""

    def __init__(self, logprobs: list[float] | None = None, error: Exception | None = None):
        self.calls: list[dict] = []
        self._logprobs = logprobs if logprobs is not None else []
        self._error = error

    def score_ids(self, prompt_ids: list[int], tokens: list[int]) -> list[float]:
        self.calls.append({"prompt_ids": list(prompt_ids), "tokens": list(tokens)})
        if self._error is not None:
            raise self._error
        return list(self._logprobs)

    def score_ids_topk(
        self, prompt_ids: list[int], tokens: list[int], top_k: int
    ) -> list[dict[int, float]]:
        self.calls.append({"prompt_ids": list(prompt_ids), "tokens": list(tokens), "top_k": top_k})
        return [{t: -0.5} for t in tokens]

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        return "fake output"

    def provenance(self) -> dict:
        return {"teacher_host": "fake", "teacher_model": "fake-model"}


class _FakeTokenizer:
    """Minimal tokenizer stub: encode splits by whitespace into fake ids."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        # Deterministic but clearly fake: ord of each char mod 200 + 1
        return [ord(c) % 200 + 1 for c in text]


# ---------------------------------------------------------------------------
# render_teacher_prefix
# ---------------------------------------------------------------------------


def test_render_teacher_prefix_contains_bos_in_chat_mode():
    """The BOS token must appear exactly once, at the start."""
    messages = [{"role": "user", "content": "Hello"}]
    result = tc.render_teacher_prefix(messages, thinking_mode="chat")
    assert result.startswith(bos_token)


def test_render_teacher_prefix_user_assistant_chat_mode():
    """A simple user/assistant exchange rendered in chat mode.

    Expected structure:
        BOS + USER_SP_TOKEN + content + ASSISTANT_SP_TOKEN + </think>
            + assistant_content + EOS
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = tc.render_teacher_prefix(messages, thinking_mode="chat")

    assert bos_token in result
    assert USER_SP_TOKEN in result
    assert ASSISTANT_SP_TOKEN in result
    assert eos_token in result
    # chat mode: thinking_end_token (</ think>) appears as the generation
    # prompt boundary, not thinking_start_token
    assert thinking_end_token in result
    # Content appears in order
    assert result.index(USER_SP_TOKEN) < result.index("Hello")
    assert result.index("Hello") < result.index(ASSISTANT_SP_TOKEN)
    assert result.index(ASSISTANT_SP_TOKEN) < result.index("Hi")
    assert result.index("Hi") < result.index(eos_token)


def test_render_teacher_prefix_exact_chat_structure():
    """Pin the exact rendered form for a minimal exchange in chat mode.

    This catches any silent change to encoding_dsv4 that would shift token
    boundaries without changing the SHA256 (which hashes the source file, not
    the output).
    """
    messages = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    result = tc.render_teacher_prefix(messages, thinking_mode="chat")
    expected = (
        bos_token
        + USER_SP_TOKEN
        + "A"
        + ASSISTANT_SP_TOKEN
        + thinking_end_token
        + "B"
        + eos_token
    )
    assert result == expected


def test_render_teacher_prefix_rejects_unknown_thinking_mode():
    """encoding_dsv4 asserts on an invalid thinking_mode."""
    messages = [{"role": "user", "content": "x"}]
    with pytest.raises((AssertionError, ValueError)):
        tc.render_teacher_prefix(messages, thinking_mode="weird")


# ---------------------------------------------------------------------------
# encode_teacher_ids
# ---------------------------------------------------------------------------


def test_encode_teacher_ids_delegates_to_tokenizer():
    tok = _FakeTokenizer()
    text = "Hello"
    ids = tc.encode_teacher_ids(text, tok)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert ids == tok.encode(text, add_special_tokens=False)


def test_encode_teacher_ids_returns_flat_list():
    """Always a plain list[int], even when tokenizer returns a tensor-like."""

    class _TensorLike:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    class _TensorTokenizer:
        def encode(self, text, *, add_special_tokens=False):
            return _TensorLike([10, 20, 30])

    ids = tc.encode_teacher_ids("any text", _TensorTokenizer())
    assert ids == [10, 20, 30]


# ---------------------------------------------------------------------------
# TeacherPrefixCache
# ---------------------------------------------------------------------------


def test_teacher_prefix_cache_miss_on_unknown_key():
    cache = tc.TeacherPrefixCache()
    result = cache.get("task-1", 0, thinking_mode="chat")
    assert result is None


def test_teacher_prefix_cache_hit_after_put():
    cache = tc.TeacherPrefixCache()
    ids = [1, 2, 3, 4]
    cache.put("task-1", 0, ids, thinking_mode="chat")
    assert cache.get("task-1", 0, thinking_mode="chat") == ids


def test_teacher_prefix_cache_hit_returns_exact_list():
    cache = tc.TeacherPrefixCache()
    ids = [10, 20, 30]
    cache.put("task-A", 5, ids)
    result = cache.get("task-A", 5)
    assert result == ids


def test_teacher_prefix_cache_different_keys_are_independent():
    cache = tc.TeacherPrefixCache()
    cache.put("task-1", 0, [1, 2], thinking_mode="chat")
    cache.put("task-2", 0, [3, 4], thinking_mode="chat")
    assert cache.get("task-1", 0, thinking_mode="chat") == [1, 2]
    assert cache.get("task-2", 0, thinking_mode="chat") == [3, 4]
    assert cache.get("task-1", 1, thinking_mode="chat") is None


def test_teacher_prefix_cache_same_ids_is_idempotent():
    """Putting the same ids twice for the same key is not an error."""
    cache = tc.TeacherPrefixCache()
    ids = [1, 2, 3]
    cache.put("task-1", 0, ids)
    cache.put("task-1", 0, list(ids))  # same content, different object
    assert cache.get("task-1", 0) == ids


def test_teacher_prefix_cache_conflict_is_hard_error():
    """Different ids for the same key → hard error, not a silent overwrite.

    This is the core invariant: the prefix must be deterministic. A cache
    conflict means the render path produced a different string at step 200
    than it did at step 5 — the exact bug the cache exists to surface.
    """
    cache = tc.TeacherPrefixCache()
    cache.put("task-1", 0, [1, 2, 3])
    with pytest.raises(ValueError, match="conflict"):
        cache.put("task-1", 0, [1, 2, 4])  # different last id


def test_teacher_prefix_cache_key_includes_encoding_sha():
    """The cache key embeds ENCODING_DSV4_SHA256, so a re-vendor bumps the hash
    and invalidates all cached entries automatically."""
    cache = tc.TeacherPrefixCache()
    ids = [5, 6, 7]
    cache.put("t", 0, ids, thinking_mode="chat")
    # The key contains the SHA; the public get() must use the same SHA
    assert cache.get("t", 0, thinking_mode="chat") == ids
    # Verify internal key shape
    key = next(iter(cache._store.keys()))
    assert ENCODING_DSV4_SHA256 in key


# ---------------------------------------------------------------------------
# CrossTokenizerTeacherPool — delegation
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pool():
    return _FakePool(logprobs=[-0.5, -1.0])


@pytest.fixture
def cross_pool(fake_pool):
    return tc.CrossTokenizerTeacherPool(
        pool=fake_pool,
        teacher_tokenizer=_FakeTokenizer(),
        thinking_mode="chat",
    )


def test_score_ids_delegates_to_underlying_pool(cross_pool, fake_pool):
    """Teacher-side ids are passed through unchanged to the underlying pool."""
    prompt_ids = [10, 20, 30]
    tokens = [50, 60]
    result = cross_pool.score_ids(prompt_ids, tokens)
    assert result == [-0.5, -1.0]
    assert len(fake_pool.calls) == 1
    assert fake_pool.calls[0]["prompt_ids"] == prompt_ids
    assert fake_pool.calls[0]["tokens"] == tokens


def test_score_ids_passes_ids_unchanged(cross_pool, fake_pool):
    """No conversion happens inside CrossTokenizerTeacherPool.score_ids."""
    prompt_ids = [100, 200, 300]
    tokens = [400, 500]
    cross_pool.score_ids(prompt_ids, tokens)
    assert fake_pool.calls[0]["prompt_ids"] == [100, 200, 300]
    assert fake_pool.calls[0]["tokens"] == [400, 500]


def test_score_ids_topk_delegates_to_underlying_pool(cross_pool, fake_pool):
    rows = cross_pool.score_ids_topk([10], [50, 60], top_k=3)
    assert len(rows) == 2
    assert 50 in rows[0]
    assert 60 in rows[1]


def test_generate_delegates_to_underlying_pool(cross_pool, fake_pool):
    result = cross_pool.generate("some prompt")
    assert result == "fake output"


def test_score_ids_topk_raises_when_pool_lacks_method():
    """A pool that only implements score_ids should fail clearly on topk."""

    class _MinimalPool:
        def score_ids(self, p, t):
            return []

        def provenance(self):
            return {}

    pool = tc.CrossTokenizerTeacherPool(
        pool=_MinimalPool(),
        teacher_tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(AttributeError, match="score_ids_topk"):
        pool.score_ids_topk([10], [50], top_k=1)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_provenance_includes_encoding_dsv4_sha256(cross_pool):
    prov = cross_pool.provenance()
    assert prov["encoding_dsv4_sha256"] == ENCODING_DSV4_SHA256


def test_provenance_includes_thinking_mode(cross_pool):
    prov = cross_pool.provenance()
    assert prov["thinking_mode"] == "chat"


def test_provenance_cross_tokenizer_flag(cross_pool):
    prov = cross_pool.provenance()
    assert prov["cross_tokenizer"] is True


def test_provenance_includes_fp8_note(cross_pool):
    prov = cross_pool.provenance()
    assert "fp8" in prov["fp8_note"].lower()


def test_provenance_merges_underlying_provenance(cross_pool):
    """The underlying pool's provenance appears in the merged result."""
    prov = cross_pool.provenance()
    assert prov["teacher_host"] == "fake"
    assert prov["teacher_model"] == "fake-model"


def test_provenance_meta_kwarg_is_included():
    pool = tc.CrossTokenizerTeacherPool(
        pool=_FakePool(),
        teacher_tokenizer=_FakeTokenizer(),
        meta={"experiment": "B1", "run_id": "abc"},
    )
    prov = pool.provenance()
    assert prov["experiment"] == "B1"
    assert prov["run_id"] == "abc"


# ---------------------------------------------------------------------------
# probe_echo_support
# ---------------------------------------------------------------------------


def test_probe_echo_support_ok_with_fake_pool():
    pool = tc.CrossTokenizerTeacherPool(
        pool=_FakePool(logprobs=[-0.4, -0.9]),
        teacher_tokenizer=_FakeTokenizer(),
    )
    result = pool.probe_echo_support(
        probe_prefix=[9707, 11],
        probe_tokens=[1879, 0],
    )
    assert result["ok"] is True
    assert result["n_returned"] == 2
    assert result["scored"] == [-0.4, -0.9]


def test_probe_echo_support_returns_false_when_pool_raises():
    from vektori_trace.teacher import TeacherScoringError

    error_pool = _FakePool(error=TeacherScoringError("connection failed"))
    pool = tc.CrossTokenizerTeacherPool(
        pool=error_pool,
        teacher_tokenizer=_FakeTokenizer(),
    )
    result = pool.probe_echo_support()
    assert result["ok"] is False
    assert "connection failed" in result["error"]


def test_probe_echo_support_returns_false_on_wrong_count():
    """Pool returns fewer logprobs than requested — suspect truncation or mis-echo."""
    pool_with_short = _FakePool(logprobs=[-0.5])  # only 1 value for 2 tokens
    pool = tc.CrossTokenizerTeacherPool(
        pool=pool_with_short,
        teacher_tokenizer=_FakeTokenizer(),
    )
    # 2 probe tokens, but pool returns 1 logprob
    result = pool.probe_echo_support(probe_prefix=[1, 2], probe_tokens=[3, 4])
    assert result["ok"] is False
    assert "expected 2" in result["error"]


def test_probe_echo_support_uses_default_probe_ids():
    """When no probe ids are specified, defaults are used and probe still runs."""
    pool = tc.CrossTokenizerTeacherPool(
        pool=_FakePool(logprobs=[-0.5, -0.7]),
        teacher_tokenizer=_FakeTokenizer(),
    )
    result = pool.probe_echo_support()
    assert result["ok"] is True
    assert result["probe_prefix"] == [9707, 11]
    assert result["probe_tokens"] == [1879, 0]


# ---------------------------------------------------------------------------
# turns_to_openai_messages
# ---------------------------------------------------------------------------


def test_turns_to_openai_messages_basic_roles():
    turns = [
        Turn(index=0, role="user", content="Hello"),
        Turn(index=1, role="assistant", content="Hi"),
    ]
    msgs = tc.turns_to_openai_messages(turns)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Hi"


def test_turns_to_openai_messages_drops_subagent_turns():
    turns = [
        Turn(index=0, role="user", content="hello", subagent_depth=0),
        Turn(index=1, role="assistant", content="from subagent", subagent_depth=1),
        Turn(index=2, role="assistant", content="real", subagent_depth=0),
    ]
    msgs = tc.turns_to_openai_messages(turns)
    contents = [m.get("content") for m in msgs]
    assert "from subagent" not in contents
    assert "real" in contents


def test_turns_to_openai_messages_thinking_as_reasoning_content():
    turns = [
        Turn(index=0, role="user", content="Q"),
        Turn(index=1, role="assistant", content="A", thinking="some reasoning"),
    ]
    msgs = tc.turns_to_openai_messages(turns)
    asst = msgs[1]
    assert asst.get("reasoning_content") == "some reasoning"
    assert asst["content"] == "A"


def test_probe_echo_support_rejects_non_finite_scores():
    """P0 shape check: NaN / non-numeric echo scores must fail the probe."""
    class _NanPool:
        def score_ids(self, prompt_ids, tokens):
            return [float("nan")] * len(tokens)

        def provenance(self):
            return {}

    pool = tc.CrossTokenizerTeacherPool(
        pool=_NanPool(),
        teacher_tokenizer=_FakeTokenizer(),
    )
    result = pool.probe_echo_support(probe_prefix=[1], probe_tokens=[2, 3])
    assert result["ok"] is False
    assert "shape check" in result["error"]


def test_probe_echo_support_rejects_string_scores():
    class _StrPool:
        def score_ids(self, prompt_ids, tokens):
            return ["nope"] * len(tokens)

        def provenance(self):
            return {}

    pool = tc.CrossTokenizerTeacherPool(
        pool=_StrPool(),
        teacher_tokenizer=_FakeTokenizer(),
    )
    result = pool.probe_echo_support(probe_prefix=[1], probe_tokens=[2])
    assert result["ok"] is False
    assert "shape check" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Cache keying — truncation depth is part of the key (§7 shared boundary)
# ─────────────────────────────────────────────────────────────────────────────


def test_prefix_cache_separates_truncation_depths():
    """Same (task, step) at two truncation depths are different prefixes."""
    from vektori_trace.teacher_cross import TeacherPrefixCache

    c = TeacherPrefixCache()
    c.put("t", 3, [1, 2, 3], n_dropped_turns=0)
    c.put("t", 3, [2, 3], n_dropped_turns=1)

    assert c.get("t", 3, n_dropped_turns=0) == [1, 2, 3]
    assert c.get("t", 3, n_dropped_turns=1) == [2, 3]


def test_prefix_cache_conflict_at_same_depth_is_a_hard_error():
    """A prefix that re-renders differently for one key is the bug to catch."""
    from vektori_trace.teacher_cross import TeacherPrefixCache

    c = TeacherPrefixCache()
    c.put("t", 3, [1, 2, 3])
    with pytest.raises(ValueError, match="re-rendered differently"):
        c.put("t", 3, [1, 2, 4])


def test_prefix_cache_reput_of_identical_ids_is_fine():
    from vektori_trace.teacher_cross import TeacherPrefixCache

    c = TeacherPrefixCache()
    c.put("t", 3, [1, 2, 3])
    c.put("t", 3, [1, 2, 3])
    assert c.get("t", 3) == [1, 2, 3]
