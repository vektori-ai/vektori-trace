"""Phase 0.5 token capture — sampled ids, not re-tokenized text."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from vektori_trace.dataset import IGNORE_INDEX, tokenize_from_captures, tokenize_from_ids
from vektori_trace.runtime.token_capture import (
    CAPTURE_FILENAME,
    RETURN_TOKEN_IDS_KEY,
    CapturedCompletion,
    TokenCaptureError,
    append_capture,
    capture_proxy,
    dump_capture_manifest,
    extract_captured_completion,
    litellm_extra_body,
    load_captures,
    token_capture_agent_kwargs,
)


def test_litellm_extra_body_requests_return_token_ids():
    assert litellm_extra_body() == {RETURN_TOKEN_IDS_KEY: True}
    assert litellm_extra_body(return_token_ids=False) == {}
    body = litellm_extra_body(logprobs=True, top_logprobs=5)
    assert body[RETURN_TOKEN_IDS_KEY] is True
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5


def test_token_capture_agent_kwargs_wraps_extra_body():
    ak = token_capture_agent_kwargs()
    assert ak == {"extra_body": {RETURN_TOKEN_IDS_KEY: True}}


def test_extract_from_vllm_chat_shape():
    resp = {
        "id": "cmpl-1",
        "model": "qwen3-8b",
        "created": 1_700_000_000,
        "prompt_token_ids": [10, 11, 12],
        "choices": [
            {
                "index": 0,
                "token_ids": [20, 21],
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -0.1, "token_id": 20},
                        {"token": "b", "logprob": -0.2, "token_id": 21},
                    ]
                },
            }
        ],
    }
    cap = extract_captured_completion(resp)
    assert cap.prompt_token_ids == [10, 11, 12]
    assert cap.token_ids == [20, 21]
    assert cap.logprobs == pytest.approx([-0.1, -0.2])
    assert cap.model == "qwen3-8b"
    assert cap.request_id == "cmpl-1"
    assert cap.input_ids == [10, 11, 12, 20, 21]
    # Ids and the text they decode to belong in the same record.
    assert cap.text == "hi"


def test_extract_text_from_completions_shape():
    """`/v1/completions` puts the emitted text on `choice.text`, not a message."""
    resp = {
        "prompt_token_ids": [1],
        "choices": [{"text": "def f():", "token_ids": [2, 3]}],
    }
    assert extract_captured_completion(resp).text == "def f():"


def test_extract_text_absent_is_none_not_empty():
    """No text field at all is distinct from a genuinely empty completion."""
    resp = {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]}
    assert extract_captured_completion(resp).text is None
    empty = {"prompt_token_ids": [1], "choices": [{"token_ids": [], "text": ""}]}
    assert extract_captured_completion(empty).text == ""


def test_extract_prompt_ids_on_choice():
    resp = {
        "choices": [
            {
                "prompt_token_ids": [1, 2],
                "token_ids": [3],
                "finish_reason": "length",
            }
        ]
    }
    cap = extract_captured_completion(resp)
    assert cap.prompt_token_ids == [1, 2]
    assert cap.token_ids == [3]


def test_extract_rejects_missing_ids():
    with pytest.raises(TokenCaptureError, match="prompt_token_ids"):
        extract_captured_completion({"choices": [{"token_ids": [1]}]})
    with pytest.raises(TokenCaptureError, match="token_ids"):
        extract_captured_completion(
            {"prompt_token_ids": [1], "choices": [{"message": {"content": "x"}}]}
        )


def test_append_and_load_roundtrip(tmp_path: Path):
    job = tmp_path / "job"
    caps = [
        CapturedCompletion(prompt_token_ids=[1, 2], token_ids=[3], model="m"),
        CapturedCompletion(prompt_token_ids=[1, 2, 3], token_ids=[4, 5], model="m"),
    ]
    for c in caps:
        append_capture(job, c)
    loaded = load_captures(job)
    assert len(loaded) == 2
    assert loaded[0].token_ids == [3]
    assert loaded[1].prompt_token_ids == [1, 2, 3]
    assert (job / CAPTURE_FILENAME).is_file()
    man = dump_capture_manifest(job, loaded, extra={"via": "test"})
    payload = json.loads(man.read_text())
    assert payload["n_completions"] == 2
    assert payload["n_completion_tokens"] == 3
    assert payload["via"] == "test"


def test_tokenize_from_ids_masks_prompt():
    ex = tokenize_from_ids([10, 11], [20, 21, 22], max_length=100)
    assert ex is not None
    assert ex.input_ids == [10, 11, 20, 21, 22]
    assert ex.labels == [IGNORE_INDEX, IGNORE_INDEX, 20, 21, 22]
    assert ex.attention_mask == [1, 1, 1, 1, 1]


def test_tokenize_from_ids_empty_completion():
    assert tokenize_from_ids([1, 2], []) is None


def test_tokenize_from_ids_truncates():
    ex = tokenize_from_ids([1, 2, 3], [4, 5, 6], max_length=4)
    assert ex is not None
    assert ex.input_ids == [1, 2, 3, 4]
    # prompt was 3 tokens → first three labels IGNORE, fourth is first completion
    assert ex.labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4]


def test_tokenize_from_captures_skips_empty():
    caps = [
        CapturedCompletion(prompt_token_ids=[1], token_ids=[]),
        CapturedCompletion(prompt_token_ids=[1], token_ids=[9]),
        {"prompt_token_ids": [2], "token_ids": [8, 7]},
    ]
    examples = tokenize_from_captures(caps)
    assert len(examples) == 2
    assert examples[0].input_ids == [1, 9]
    assert examples[1].input_ids == [2, 8, 7]


class _Upstream(BaseHTTPRequestHandler):
    last_body: dict | None = None

    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b"{}"
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "stub"}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        req = json.loads(raw.decode() or "{}")
        _Upstream.last_body = req
        # Every spelling, so a duplicated header is visible rather than masked
        # by dict lookup picking one of them.
        _Upstream.last_auth = [
            v for k, v in self.headers.items() if k.lower() == "authorization"
        ]
        assert req.get(RETURN_TOKEN_IDS_KEY) is True
        resp = {
            "id": "upstream-1",
            "model": "stub",
            "prompt_token_ids": [100, 101],
            "choices": [
                {
                    "token_ids": [200, 201],
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def upstream_server():
    _Upstream.last_body = None
    _Upstream.last_auth = None
    httpd = HTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}/v1"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_capture_proxy_injects_and_persists(upstream_server, tmp_path: Path):
    import urllib.request

    capture_dir = tmp_path / "caps"
    with capture_proxy(upstream_server, capture_dir) as proxy:
        payload = json.dumps(
            {
                "model": "stub",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            }
        ).encode()
        req = urllib.request.Request(
            proxy.api_base + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            out = json.loads(resp.read().decode())
        assert out["prompt_token_ids"] == [100, 101]
        assert out["choices"][0]["token_ids"] == [200, 201]

    assert _Upstream.last_body is not None
    assert _Upstream.last_body[RETURN_TOKEN_IDS_KEY] is True
    loaded = load_captures(capture_dir)
    assert len(loaded) == 1
    assert loaded[0].prompt_token_ids == [100, 101]
    assert loaded[0].token_ids == [200, 201]
    # Only the proxy sees both ends of the call, so only it can time one.
    # `created_at` is the upstream's whole-second `created` field and cannot.
    assert loaded[0].request_started_at is not None
    assert loaded[0].latency_ms is not None
    assert loaded[0].latency_ms >= 0.0


def test_capture_timing_survives_the_jsonl_roundtrip(tmp_path: Path):
    job = tmp_path / "job"
    append_capture(
        job,
        CapturedCompletion(
            prompt_token_ids=[1],
            token_ids=[2],
            text="ok",
            request_started_at=1_700_000_000.5,
            latency_ms=12.5,
        ),
    )
    (loaded,) = load_captures(job)
    assert loaded.text == "ok"
    assert loaded.request_started_at == pytest.approx(1_700_000_000.5)
    assert loaded.latency_ms == pytest.approx(12.5)


def test_served_to_harbor_kwargs_capture_flag():
    from vektori_trace.runtime.serve import ServedModel, served_to_harbor_kwargs

    served = ServedModel(
        api_base="http://127.0.0.1:8000/v1",
        model_name="qwen",
        base_model="Qwen/Qwen3-8B",
    )
    plain = served_to_harbor_kwargs(served)
    assert "agent_kwargs" not in plain
    capped = served_to_harbor_kwargs(served, capture_tokens=True)
    assert capped["agent_kwargs"]["extra_body"][RETURN_TOKEN_IDS_KEY] is True


def test_tokenize_rollouts_for_opd_requires_captures(tmp_path: Path):
    from vektori_trace.arms import tokenize_rollouts_for_opd
    from vektori_trace.runtime.rollout import CollectedRollout

    job = tmp_path / "job"
    job.mkdir()
    bare = CollectedRollout(
        task="t",
        passed=True,
        reward=1.0,
        jobs_dir=job,
        tokens_captured=False,
    )
    with pytest.raises(TokenCaptureError, match="no token captures"):
        tokenize_rollouts_for_opd([bare], require_captures=True)

    append_capture(
        job,
        CapturedCompletion(prompt_token_ids=[1, 2], token_ids=[3, 4]),
    )
    examples = tokenize_rollouts_for_opd([bare], require_captures=True)
    assert len(examples) == 1
    assert examples[0].input_ids == [1, 2, 3, 4]


def test_cli_registers_capture_proxy():
    from vektori_trace.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["capture-proxy", "--upstream", "http://127.0.0.1:8000/v1", "--out", "/tmp/x"]
    )
    assert args.command == "capture-proxy"
    assert args.upstream.endswith("/v1")


def test_cli_run_arms_capture_tokens_flag():
    from vektori_trace.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run-arms",
            "--selection",
            "s.json",
            "--diagnosis",
            "d.json",
            "--tasks-dir",
            "tasks",
            "--agent",
            "terminus-2",
            "--capture-tokens",
        ]
    )
    assert args.capture_tokens is True


def test_capture_proxy_overrides_client_authorization(upstream_server, tmp_path: Path):
    """`upstream_api_key` must actually reach the wire, replacing the client's.

    Regression test. The field was declared on `CaptureProxy` and hoisted into
    the request handler's closure, but the line applying it was lost in the move
    to `vektori_trace/runtime/`. Nothing failed loudly: the CLI still printed
    "auth overriding Authorization", so the proxy looked configured while the
    client's own credential went upstream unchanged. Against a real provider
    that is a 401 on every call, which reads as a bad key rather than as a bug —
    it cost a full agent rollout before anyone looked at the header.

    Asserting on the *list* of authorization headers, not a single lookup, is
    deliberate: HTTP header names are case-insensitive but a dict is not, so
    setting `Authorization` without removing an inbound `authorization` sends
    both and the upstream may honour either.
    """
    import urllib.request

    capture_dir = tmp_path / "caps"
    with capture_proxy(
        upstream_server, capture_dir, upstream_api_key="real-upstream-key"
    ) as proxy:
        payload = json.dumps(
            {"model": "stub", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        req = urllib.request.Request(
            proxy.api_base + "/chat/completions",
            data=payload,
            # Lowercase on purpose: urllib will title-case this, but a client
            # that does not is exactly the case a plain dict assignment misses.
            headers={"Content-Type": "application/json",
                     "authorization": "Bearer client-key-for-a-different-service"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    assert _Upstream.last_auth == ["Bearer real-upstream-key"], (
        f"upstream saw {_Upstream.last_auth!r}; expected exactly one header "
        "carrying the configured upstream key"
    )


def test_capture_proxy_leaves_authorization_alone_without_a_key(
    upstream_server, tmp_path: Path
):
    """No `upstream_api_key` means pass the client's credential through untouched.

    The override must not fire unconditionally — a proxy in front of a server
    that shares the client's auth (a local vLLM, say) would otherwise have its
    credential silently stripped.
    """
    import urllib.request

    with capture_proxy(upstream_server, tmp_path / "caps") as proxy:
        payload = json.dumps(
            {"model": "stub", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        req = urllib.request.Request(
            proxy.api_base + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer passthrough-key"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    assert _Upstream.last_auth == ["Bearer passthrough-key"]


def test_capture_proxy_helper_couples_top_logprobs_to_logprobs(
    upstream_server, tmp_path: Path
):
    """`top_logprobs=N` alone must still request logprobs.

    `CaptureProxy.start` only emits `top_logprobs` inside the `inject_logprobs`
    branch, so asking for alternatives without asking for logprobs returns
    neither and says nothing about it. `cmd_capture_proxy` couples the two; this
    pins the helper to the same meaning, so a test and the CLI cannot disagree
    about what identical arguments do.
    """
    import urllib.request

    with capture_proxy(upstream_server, tmp_path / "caps", top_logprobs=5) as proxy:
        req = urllib.request.Request(
            proxy.api_base + "/chat/completions",
            data=json.dumps(
                {"model": "stub", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

    assert _Upstream.last_body["logprobs"] is True
    assert _Upstream.last_body["top_logprobs"] == 5
