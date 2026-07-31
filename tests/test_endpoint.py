"""Endpoint attach + teacher scoring against a stub OpenAI-compatible server.

A real vLLM cannot run in unit tests, but the wire contract can: these serve the
exact JSON shapes vLLM returns (including `prompt_logprobs`, whose alignment is
the thing that would silently corrupt the OPD objective if it were wrong).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from vektori_trace.endpoint import (
    EndpointError,
    attach_endpoint,
    discover_model_name,
    endpoint_serve_cm,
    wait_for_health,
)
from vektori_trace.teacher import (
    TeacherScoringError,
    VllmTeacherPool,
    teacher_pool_from_endpoint,
)

SERVED_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
# The stub tokenizes by whitespace into these ids; scoring returns -0.5 per token.
PROMPT_IDS = [11, 12]
FIXED_LOGPROB = -0.5


class _Handler(BaseHTTPRequestHandler):
    # Class-level on purpose: the handler is instantiated per request, so an
    # instance attribute could not accumulate what the server was asked to load.
    loaded_loras: ClassVar[list[str]] = []

    def log_message(self, *args):
        return

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": SERVED_NAME}]})
        else:
            self._send({"error": "not found"}, code=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            n = len((req.get("prompt") or "").split())
            self._send({"tokens": PROMPT_IDS[:n] if n <= 2 else PROMPT_IDS + [13] * (n - 2)})
        elif self.path == "/v1/completions":
            prompt = req.get("prompt")
            if req.get("prompt_logprobs") is not None and isinstance(prompt, list):
                entries = [None] + [
                    {str(tid): {"logprob": FIXED_LOGPROB}} for tid in prompt[1:]
                ]
                self._send({"choices": [{"prompt_logprobs": entries}]})
            else:
                self._send({"choices": [{"text": " continued"}]})
        elif self.path == "/v1/load_lora_adapter":
            _Handler.loaded_loras.append(req["lora_name"])
            self._send({})
        elif self.path == "/v1/unload_lora_adapter":
            self._send({})
        else:
            self._send({"error": "not found"}, code=404)


@pytest.fixture()
def server():
    _Handler.loaded_loras = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_discover_model_name_reads_the_server(server):
    assert discover_model_name(server) == SERVED_NAME
    # The /v1 suffix is optional on input.
    assert discover_model_name(server + "/v1") == SERVED_NAME


def test_attach_endpoint_yields_a_harbor_ready_served_model(server):
    with attach_endpoint("Qwen/Qwen3-8B", api_base=server) as served:
        assert served.api_base == server + "/v1"
        assert served.model_name == SERVED_NAME
        assert served.harbor_model == f"hosted_vllm/{SERVED_NAME}"
        # model_info must be populated: harbor refuses hosted_vllm without it.
        assert served.model_info["max_input_tokens"] > 0


def test_attach_endpoint_registers_and_unregisters_an_adapter(server):
    with attach_endpoint(
        "Qwen/Qwen3-8B", api_base=server, adapter_path="/runs/A3/adapter"
    ) as served:
        # Routing happens on the LoRA name, not the base model name.
        assert served.model_name == "A3-lora"
        assert _Handler.loaded_loras == ["A3-lora"]
    assert served.adapter_path == "/runs/A3/adapter"


def test_endpoint_serve_cm_matches_the_serve_model_signature(server):
    cm = endpoint_serve_cm(server)
    # arms.run_arms calls serve_cm(model, gpu=...) and serve_cm(model, adapter_path=..., gpu=...)
    with cm("Qwen/Qwen3-8B", gpu="A10G") as served:
        assert served.model_name == SERVED_NAME
        # The GPU string is recorded, never invented: the hardware was fixed at
        # instance launch, so arms.json must not claim Modal chose it.
        assert served.gpu == "A10G"


def test_unreachable_endpoint_fails_fast_with_a_usable_message():
    with pytest.raises(EndpointError, match="no healthy"):
        # Port 1 is reserved and never listening.
        attach_endpoint("m", api_base="http://127.0.0.1:1", wait_seconds=0).__enter__()


def test_api_base_without_scheme_is_rejected():
    with pytest.raises(EndpointError, match="scheme"):
        discover_model_name("127.0.0.1:8000")


def test_prompt_logprobs_returns_one_logprob_per_supplied_token(server):
    pool = VllmTeacherPool(api_base=server, model=SERVED_NAME)
    scored = pool.prompt_logprobs("hello world", [21, 22, 23])
    assert scored == [FIXED_LOGPROB] * 3


def test_prompt_logprobs_is_empty_for_no_tokens(server):
    pool = VllmTeacherPool(api_base=server, model=SERVED_NAME)
    assert pool.prompt_logprobs("hello world", []) == []


def test_generate_returns_teacher_text(server):
    pool = VllmTeacherPool(api_base=server, model=SERVED_NAME)
    assert pool.generate("prefix") == " continued"


def test_teacher_pool_from_endpoint_probes_scoring(server):
    pool = teacher_pool_from_endpoint(server)
    assert pool.model == SERVED_NAME
    assert pool.provenance()["teacher_model"] == SERVED_NAME


def test_missing_prompt_logprobs_refuses_rather_than_returning_zeros():
    """An endpoint that drops the field must not read as "teacher agrees"."""

    class _NoLogprobs(VllmTeacherPool):
        def tokenize(self, text):
            return [1, 2]

    pool = _NoLogprobs(api_base="http://127.0.0.1:1")
    import vektori_trace.teacher as teacher_mod

    original = teacher_mod._post_json
    teacher_mod._post_json = lambda url, payload, timeout: {"choices": [{}]}
    try:
        with pytest.raises(TeacherScoringError, match="prompt_logprobs"):
            pool.prompt_logprobs("p", [7])
    finally:
        teacher_mod._post_json = original


def test_scored_token_mismatch_is_an_error_not_a_guess():
    """If the teacher scored a different id than we sent, alignment is gone."""
    import vektori_trace.teacher as teacher_mod

    pool = VllmTeacherPool(api_base="http://127.0.0.1:1")
    original = teacher_mod._post_json

    def fake(url, payload, timeout):
        if url.endswith("/tokenize"):
            return {"tokens": [1]}
        # Scored id 999 while we asked about 7.
        return {"choices": [{"prompt_logprobs": [None, {"999": {"logprob": -1.0}}]}]}

    teacher_mod._post_json = fake
    try:
        with pytest.raises(TeacherScoringError, match="absent from prompt_logprobs"):
            pool.prompt_logprobs("p", [7])
    finally:
        teacher_mod._post_json = original


def test_length_mismatch_is_an_error():
    import vektori_trace.teacher as teacher_mod

    pool = VllmTeacherPool(api_base="http://127.0.0.1:1")
    original = teacher_mod._post_json

    def fake(url, payload, timeout):
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2]}
        return {"choices": [{"prompt_logprobs": [None]}]}

    teacher_mod._post_json = fake
    try:
        with pytest.raises(TeacherScoringError, match="refusing to guess"):
            pool.prompt_logprobs("p", [7])
    finally:
        teacher_mod._post_json = original


def test_scored_logprobs_feed_the_opd_objective(server):
    """End-to-end shape check: teacher scores → reverse_kl_surrogate accepts them."""
    torch = pytest.importorskip("torch")
    from vektori_trace.opd import reverse_kl_surrogate

    pool = VllmTeacherPool(api_base=server, model=SERVED_NAME)
    teacher_lp = pool.prompt_logprobs("hello world", [21, 22, 23])
    student = torch.tensor([[-0.1, -0.2, -0.3]], requires_grad=True)
    teacher = torch.tensor([teacher_lp])
    loss = reverse_kl_surrogate(student, teacher)
    loss.backward()
    assert student.grad is not None


def test_html_at_v1_models_is_not_healthy():
    """A proxy answering 200 with an HTML page is not a model server.

    `/health` returning an empty 200 body is vLLM's documented behaviour and the
    reason a `JSONDecodeError` is tolerated there. Extending that tolerance to
    `/v1/models` makes an ALB error page pass the probe, so the failure resurfaces
    as a broken rollout mid-sweep instead of here.
    """

    class _Html(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path == "/health":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = b"<html>503 Service Unavailable</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), _Html)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(EndpointError):
            wait_for_health(f"http://127.0.0.1:{httpd.server_port}/v1", wait_seconds=0.0)
    finally:
        httpd.shutdown()
        httpd.server_close()
