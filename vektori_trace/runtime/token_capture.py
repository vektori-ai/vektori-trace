"""Phase 0.5 — capture the token ids the student *actually sampled*.

`docs/PILOT.md`: `dataset.py` re-tokenizes text. For SFT that is correct. For
OPD and GRPO it is not — both compare against the probability of the tokens
sampled at inference, and a re-tokenized sequence is close but not identical.
The number simply fails to move.

vLLM (≥0.10.2) accepts `"return_token_ids": true` on `/v1/chat/completions` and
returns `prompt_token_ids` + per-choice `token_ids` (and optional logprobs).
This module:

1. Builds the litellm / harbor kwargs that request those fields.
2. Parses them out of OpenAI-compatible responses (several nesting shapes).
3. Persists them next to a harbor job dir as JSONL.
4. Optionally sits as a reverse proxy in front of a self-hosted vLLM so harbor
   agents that we do not control still get the flag injected and the ids stored.

`dataset.tokenize_from_ids` consumes the captures — no re-tokenization.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

_log = logging.getLogger(__name__)

# Filename written under a harbor attempt dir (rollout.py / CaptureProxy).
CAPTURE_FILENAME = "token_captures.jsonl"

# Sibling ledger: one line per upstream success the proxy could *not* turn into a
# capture. Without it an empty token_captures.jsonl is indistinguishable from a
# run that made no model calls at all.
CAPTURE_FAILURE_FILENAME = "capture_failures.jsonl"

# Request body flag vLLM understands on chat + completions (v0.10.2+).
RETURN_TOKEN_IDS_KEY = "return_token_ids"


class TokenCaptureError(RuntimeError):
    """Response claimed token ids but the shape is unusable."""


@dataclass
class CapturedCompletion:
    """One model call's sampled ids — the unit OPD/GRPO train against."""

    prompt_token_ids: list[int]
    token_ids: list[int]
    logprobs: list[float] | None = None
    #: Per-position full logprob record, one dict per *generated* token:
    #: ``{"token_id": int|None, "token": str|None, "logprob": float,
    #:    "top": [{"token_id": int|None, "token": str|None, "logprob": float}, ...]}``
    #:
    #: `logprobs` above keeps only the chosen token's logprob, which is enough for
    #: plain reverse-KL but throws away the alternatives. The top-K rows are the
    #: input `align.align_topk_rows` consumes, and they cannot be reconstructed
    #: after the fact -- an un-captured alternative is gone for good. Kept as a
    #: separate field rather than replacing `logprobs` so existing readers
    #: (`dataset.tokenize_from_ids`, `scripts/verify_run_logs.py`) are unaffected.
    #:
    #: `token_id` is None when the provider returns the legacy string-keyed
    #: format. That is recorded rather than rejected here: the proxy must never
    #: fail a live agent turn over a logprob shape. `capture_failures.jsonl` and
    #: the manifest's `logprob_token_id_coverage` are where that becomes visible.
    logprob_detail: list[dict[str, Any]] | None = None
    model: str = ""
    request_id: str | None = None
    created_at: float = field(default_factory=time.time)
    choice_index: int = 0
    finish_reason: str | None = None
    # The text the model emitted, alongside the ids that produced it. Kept in the
    # same record deliberately: ids and text living in separate files (captures
    # vs the ATIF trajectory) means every later question needs a join, and a
    # silent disagreement between them is undetectable. Decoding `token_ids` must
    # reproduce this string — `scripts/verify_run_logs.py` asserts exactly that.
    text: str | None = None
    #: A reasoning model's chain of thought, which `message.content` excludes.
    #:
    #: DeepSeek-V4-Flash splits its output: everything before `</think>` lands in
    #: `message.reasoning_content` and only the part after it in
    #: `message.content`. `token_ids`, meanwhile, spans **both** plus the
    #: `</think>` token itself. So `text` alone accounts for a fraction of the
    #: ids -- one measured call emitted 16 ids for a `text` of "391" -- and the
    #: invariant this dataclass documents above (decoding `token_ids` reproduces
    #: `text`) is false for any reasoning model unless the CoT is kept too.
    #:
    #: This is not cosmetic for OPD: the reasoning tokens are the majority of what
    #: the student sampled and therefore the majority of what gets scored. Dropping
    #: them here would leave the ids trainable but make every downstream text-side
    #: check disagree with them, which reads as corruption rather than as a
    #: rendering convention.
    reasoning_text: str | None = None
    # Set by CaptureProxy, which is the only component that sees both ends of a
    # call. `created_at` comes from the upstream response's `created` field: whole
    # seconds, assigned by the server, and useless for measuring anything.
    request_started_at: float | None = None
    latency_ms: float | None = None

    @property
    def input_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + list(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapturedCompletion:
        return cls(
            prompt_token_ids=[int(x) for x in data["prompt_token_ids"]],
            token_ids=[int(x) for x in data["token_ids"]],
            logprobs=(
                [float(x) for x in data["logprobs"]]
                if data.get("logprobs") is not None
                else None
            ),
            logprob_detail=(
                [dict(x) for x in data["logprob_detail"]]
                if data.get("logprob_detail") is not None
                else None
            ),
            model=str(data.get("model") or ""),
            request_id=data.get("request_id"),
            created_at=float(data.get("created_at") or time.time()),
            choice_index=int(data.get("choice_index") or 0),
            finish_reason=data.get("finish_reason"),
            text=data.get("text"),
            reasoning_text=data.get("reasoning_text"),
            request_started_at=(
                float(data["request_started_at"])
                if data.get("request_started_at") is not None
                else None
            ),
            latency_ms=(
                float(data["latency_ms"]) if data.get("latency_ms") is not None else None
            ),
        )


def litellm_extra_body(
    *,
    return_token_ids: bool = True,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """Body fields forwarded to vLLM through litellm's `hosted_vllm` provider.

    Pass as `extra_body=...` on `litellm.completion`. Harbor agents that accept
    arbitrary litellm kwargs get the same dict via `token_capture_agent_kwargs`.
    """
    body: dict[str, Any] = {}
    if return_token_ids:
        body[RETURN_TOKEN_IDS_KEY] = True
    if logprobs:
        body["logprobs"] = True
        if top_logprobs is not None:
            body["top_logprobs"] = int(top_logprobs)
    return body


def token_capture_agent_kwargs(
    *,
    return_token_ids: bool = True,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    """`--ak` payload for harbor so litellm requests sampled token ids.

    terminus-2 / hosted_vllm forward unknown kwargs into litellm; `extra_body`
    is the documented seam for vLLM-only fields.
    """
    extra = litellm_extra_body(
        return_token_ids=return_token_ids,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
    if not extra:
        return {}
    return {"extra_body": extra}


def _as_int_list(value: Any, *, field_name: str) -> list[int]:
    if value is None:
        raise TokenCaptureError(f"missing {field_name}")
    if not isinstance(value, list):
        raise TokenCaptureError(f"{field_name} must be a list, got {type(value).__name__}")
    out: list[int] = []
    for i, item in enumerate(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError) as e:
            raise TokenCaptureError(
                f"{field_name}[{i}] is not an int: {item!r}"
            ) from e
    return out


def _completion_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Per-generated-token logprobs from the OpenAI chat/completions shape."""
    lp = choice.get("logprobs")
    if not lp:
        return None
    if isinstance(lp, dict):
        content = lp.get("content")
        if isinstance(content, list) and content:
            out: list[float] = []
            for entry in content:
                if not isinstance(entry, dict) or "logprob" not in entry:
                    return None
                out.append(float(entry["logprob"]))
            return out
        # /v1/completions legacy: {"token_logprobs": [...]}
        token_lps = lp.get("token_logprobs")
        if isinstance(token_lps, list):
            return [float(x) for x in token_lps if x is not None]
    return None


def _logprob_alt(entry: dict[str, Any]) -> dict[str, Any]:
    """One `{token_id, token, logprob}` row from an OpenAI-shaped logprobs entry.

    `token_id` is Fireworks' addition to the OpenAI schema (`NewLogProbsContent`
    marks it required) and is the only field that makes an alternative usable
    without re-tokenising. vLLM omits it. None here means "this provider did not
    say", never "id 0".
    """
    tid = entry.get("token_id")
    return {
        "token_id": int(tid) if isinstance(tid, int) else None,
        "token": entry.get("token") if isinstance(entry.get("token"), str) else None,
        "logprob": float(entry["logprob"]),
    }


def _completion_logprob_detail(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Full per-token logprob rows including the top-K alternatives.

    Returns None when the response carries no usable per-token logprobs at all.
    A position with logprobs but an empty `top_logprobs` yields `"top": []` --
    the distinction between "no alternatives requested" and "alternatives
    requested and none came back" is exactly what the coverage check needs.
    """
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None
    content = lp.get("content")
    if not isinstance(content, list) or not content:
        return None
    out: list[dict[str, Any]] = []
    for entry in content:
        if not isinstance(entry, dict) or "logprob" not in entry:
            return None
        row = _logprob_alt(entry)
        alts = entry.get("top_logprobs")
        top: list[dict[str, Any]] = []
        if isinstance(alts, list):
            for alt in alts:
                if isinstance(alt, dict) and "logprob" in alt:
                    top.append(_logprob_alt(alt))
        row["top"] = top
        out.append(row)
    return out


def _completion_text(choice: dict[str, Any]) -> str | None:
    """The emitted text, from either OpenAI response shape.

    `/v1/chat/completions` puts it at `choice.message.content`; `/v1/completions`
    at `choice.text`. Streaming deltas land at `choice.delta.content`. None means
    the response carried no text at all, which is different from an empty string
    (a real, if unusual, completion).
    """
    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if choice.get("text") is not None:
        return str(choice["text"])
    delta = choice.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta["content"])
    return None


def _completion_reasoning_text(choice: dict[str, Any]) -> str | None:
    """The chain of thought, when the provider returns it out-of-band.

    Fireworks/DeepSeek use `message.reasoning_content`; some providers use
    `reasoning`. None means this response had no separate CoT channel, which is
    the normal case for a non-reasoning model -- not an error.
    """
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    for key in ("reasoning_content", "reasoning"):
        val = message.get(key)
        if isinstance(val, str):
            return val
    return None


def detokenized_text(capture: CapturedCompletion) -> str | None:
    """The full emitted string, rebuilt from the captured logprob tokens.

    This is the only text in a capture that is guaranteed to correspond exactly
    to `token_ids`, because it *is* the per-token strings the provider returned
    alongside those ids -- no local tokenizer, no assumption about how the
    provider splits reasoning from answer, no guess at the `</think>` separator.
    `scripts/verify_run_logs.py`'s ids-vs-text check should compare against this
    rather than against `text` for any reasoning model.

    None when the capture carries no per-token detail, or when the provider
    returned ids without their token strings.
    """
    detail = capture.logprob_detail
    if not detail:
        return None
    parts = []
    for row in detail:
        tok = row.get("token")
        if not isinstance(tok, str):
            return None
        parts.append(tok)
    return "".join(parts)


def extract_captured_completion(
    response: dict[str, Any],
    *,
    choice_index: int = 0,
) -> CapturedCompletion:
    """Pull prompt + completion ids out of a vLLM OpenAI-compatible response.

    Accepts:
    - top-level `prompt_token_ids` (vLLM chat completions with return_token_ids)
    - `choices[i].token_ids` or `choices[i].prompt_token_ids` (streaming first chunk
      may put prompt ids on the choice)
    - dict-like objects with `.model_dump()` / attribute access (litellm ModelResponse)
    """
    data = _to_plain_dict(response)
    choices = data.get("choices") or []
    if not choices:
        raise TokenCaptureError("response has no choices")
    if choice_index < 0 or choice_index >= len(choices):
        raise TokenCaptureError(
            f"choice_index {choice_index} out of range for {len(choices)} choices"
        )
    choice = choices[choice_index]
    if not isinstance(choice, dict):
        choice = _to_plain_dict(choice)

    prompt_ids = data.get("prompt_token_ids")
    if prompt_ids is None:
        prompt_ids = choice.get("prompt_token_ids")
    prompt_token_ids = _as_int_list(prompt_ids, field_name="prompt_token_ids")

    token_ids_raw = choice.get("token_ids")
    if token_ids_raw is None:
        # Some builds nest under message.
        message = choice.get("message") or {}
        if isinstance(message, dict):
            token_ids_raw = message.get("token_ids")
    token_ids = _as_int_list(token_ids_raw, field_name="token_ids")

    return CapturedCompletion(
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
        logprobs=_completion_logprobs(choice),
        logprob_detail=_completion_logprob_detail(choice),
        model=str(data.get("model") or ""),
        request_id=data.get("id"),
        created_at=float(data.get("created") or time.time()),
        choice_index=choice_index,
        finish_reason=choice.get("finish_reason"),
        text=_completion_text(choice),
        reasoning_text=_completion_reasoning_text(choice),
    )


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    # litellm sometimes returns objects with __dict__ / attribute bags.
    if hasattr(obj, "__dict__"):
        raw = dict(obj.__dict__)
        # Prefer public fields; drop private.
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    raise TokenCaptureError(f"cannot interpret response of type {type(obj).__name__}")


def capture_path(job_dir: Path) -> Path:
    return Path(job_dir) / CAPTURE_FILENAME


def append_capture(job_dir: Path, completion: CapturedCompletion) -> Path:
    """Append one completion to `<job_dir>/token_captures.jsonl`."""
    path = capture_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(completion.to_dict()) + "\n")
    return path


def load_captures(job_dir: Path) -> list[CapturedCompletion]:
    """Load every captured completion for a harbor job (empty if none)."""
    path = capture_path(job_dir)
    if not path.is_file():
        return []
    out: list[CapturedCompletion] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CapturedCompletion.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise TokenCaptureError(
                f"{path}:{line_no}: corrupt capture line: {e}"
            ) from e
    return out


def _logprob_coverage(captures: list[CapturedCompletion]) -> dict[str, Any]:
    """Aggregate top-K logprob coverage across a run's captures."""
    positions = 0
    with_top = 0
    alts = 0
    alts_with_id = 0
    widths: set[int] = set()
    for cap in captures:
        for row in cap.logprob_detail or []:
            positions += 1
            top = row.get("top") or []
            if top:
                with_top += 1
                widths.add(len(top))
            for alt in top:
                alts += 1
                if alt.get("token_id") is not None:
                    alts_with_id += 1
    return {
        "logprob_positions": positions,
        "logprob_positions_with_topk": with_top,
        "logprob_topk_coverage": (with_top / positions) if positions else 0.0,
        "logprob_topk_widths": sorted(widths),
        "logprob_alternatives": alts,
        "logprob_token_id_coverage": (alts_with_id / alts) if alts else 0.0,
    }


def dump_capture_manifest(
    job_dir: Path,
    captures: list[CapturedCompletion],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Rewrite a summary JSON beside the JSONL (provenance / counts)."""
    path = Path(job_dir) / "token_captures.manifest.json"
    payload = {
        "n_completions": len(captures),
        "n_prompt_tokens": sum(len(c.prompt_token_ids) for c in captures),
        "n_completion_tokens": sum(len(c.token_ids) for c in captures),
        "models": sorted({c.model for c in captures if c.model}),
        "has_logprobs": any(c.logprobs is not None for c in captures),
        "has_logprob_detail": any(c.logprob_detail is not None for c in captures),
        "has_reasoning_text": any(c.reasoning_text is not None for c in captures),
        # How many generated positions actually carried top-K alternatives, and
        # how many of those alternatives were id-keyed. A run whose captures look
        # full but whose coverage is 0.0 is not trainable for the top-K objective,
        # and that has to be visible without re-reading 8 MB of JSONL.
        **_logprob_coverage(captures),
        "capture_file": CAPTURE_FILENAME,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Reverse proxy — inject return_token_ids for agents we do not control
# ---------------------------------------------------------------------------


@dataclass
class CaptureProxy:
    """Local OpenAI-compatible reverse proxy in front of a real vLLM server.

    Harbor agents call `api_base`; point that at this proxy. Every chat/completions
    (and completions) request gets `return_token_ids: true` injected, the upstream
    response is forwarded unchanged, and prompt/completion ids are written under
    `capture_dir` as JSONL.
    """

    upstream_api_base: str
    capture_dir: Path
    host: str = "127.0.0.1"
    port: int = 0  # 0 → ephemeral
    inject_logprobs: bool = False
    #: Injected as `top_logprobs` alongside `logprobs: true`. None leaves it off,
    #: which makes the provider return only the sampled token's logprob.
    #:
    #: Fireworks rejects any value above the serving deployment's `--max-logprobs`
    #: (**default 5** on serverless) with a 400. That is a deployment property, not
    #: a request one, so it is not clamped here: silently sending 5 when the run
    #: asked for 10 would change the objective's K without saying so, exactly the
    #: failure `teacher_fireworks.TOP_LOGPROBS_CAP` refuses. Probe first with
    #: `scripts/probe_fireworks_logprobs.py`.
    top_logprobs: int | None = None
    #: Overrides the client's `Authorization` header on the way upstream. Harbor
    #: hands litellm whatever key it was configured with; when the upstream is
    #: Fireworks rather than an unauthenticated local vLLM, that key is wrong.
    upstream_api_key: str | None = None
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        base = self.upstream_api_base.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise TokenCaptureError(
                f"upstream_api_base must include a scheme, got {self.upstream_api_base!r}"
            )
        self.upstream_api_base = base if base.endswith("/v1") else base + "/v1"
        self.capture_dir = Path(self.capture_dir)

    @property
    def api_base(self) -> str:
        if self._httpd is None:
            raise TokenCaptureError("proxy is not running")
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> str:
        """Bind and serve in a daemon thread. Returns the local `/v1` api_base."""
        if self._httpd is not None:
            return self.api_base
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        upstream = self.upstream_api_base
        capture_dir = self.capture_dir
        inject_logprobs = self.inject_logprobs
        top_logprobs = self.top_logprobs
        upstream_api_key = self.upstream_api_key
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length else b""

            def _forward(self, method: str) -> None:
                body = self._read_body() if method != "GET" else b""
                path = self.path
                # Strip hop that would confuse upstream.
                target = urljoin(upstream.rstrip("/") + "/", path.lstrip("/"))
                # Our api_base is .../v1; requests arrive as /v1/... or /health.
                # Rebuild against the upstream root for /health, against /v1 for rest.
                parsed_up = urlparse(upstream)
                up_root = f"{parsed_up.scheme}://{parsed_up.netloc}"
                if path == "/health" or path.startswith("/health?"):
                    target = up_root + path
                elif path.startswith("/v1"):
                    # upstream already ends in /v1 — drop the duplicate prefix.
                    suffix = path[len("/v1") :] or ""
                    target = upstream.rstrip("/") + suffix
                else:
                    target = up_root + path

                headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in {"host", "content-length", "transfer-encoding"}
                }
                if upstream_api_key:
                    # Whatever key the client was configured with is for a
                    # different service than the upstream this proxy actually
                    # talks to. Drop every spelling of the inbound header before
                    # setting ours -- HTTP header names are case-insensitive, a
                    # dict is not, and sending both is how you get a 401 that
                    # looks like a bad credential.
                    for _k in [k for k in headers if k.lower() == "authorization"]:
                        headers.pop(_k, None)
                    headers["Authorization"] = f"Bearer {upstream_api_key}"
                if method in {"POST", "PUT", "PATCH"} and body:
                    try:
                        payload = json.loads(body.decode("utf-8") or "{}")
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and (
                        "chat/completions" in path or path.rstrip("/").endswith("completions")
                    ):
                        payload[RETURN_TOKEN_IDS_KEY] = True
                        if inject_logprobs:
                            payload["logprobs"] = True
                            if top_logprobs is not None:
                                payload["top_logprobs"] = int(top_logprobs)
                        body = json.dumps(payload).encode("utf-8")
                        headers["Content-Type"] = "application/json"
                    headers["Content-Length"] = str(len(body))
                elif method in {"POST", "PUT", "PATCH"}:
                    headers["Content-Length"] = str(len(body))

                req = urllib.request.Request(
                    target,
                    data=body if method != "GET" else None,
                    headers=headers,
                    method=method,
                )
                # Wall clock for joining against gpu_log.jsonl / passk_log.jsonl;
                # perf_counter for the duration, because wall clock can step.
                request_started_at = time.time()
                t0 = time.perf_counter()
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp:
                        resp_body = resp.read()
                        code = resp.status
                        content_type = resp.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as e:
                    resp_body = e.read()
                    code = e.code
                    content_type = e.headers.get("Content-Type", "application/json")
                except urllib.error.URLError as e:
                    err = json.dumps({"error": f"upstream unreachable: {e}"}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                # Capture token ids from successful generation responses.
                if (
                    200 <= code < 300
                    and method == "POST"
                    and (
                        "chat/completions" in path
                        or path.rstrip("/").endswith("/completions")
                    )
                ):
                    try:
                        parsed = json.loads(resp_body.decode("utf-8"))
                        cap = extract_captured_completion(parsed)
                        cap.request_started_at = request_started_at
                        cap.latency_ms = (time.perf_counter() - t0) * 1000.0
                        with lock:
                            append_capture(capture_dir, cap)
                    except (
                        TokenCaptureError,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                    ) as exc:
                        # Never fail the client because capture failed — OPD will
                        # notice the missing file and refuse to train on re-tokenized
                        # text when capture was requested. But say so: a silently
                        # empty capture file looks identical to a run that made no
                        # model calls, and the whole point of the proxy is that
                        # nothing goes unrecorded.
                        _log.warning(
                            "token capture failed for %s (%s): %s",
                            path,
                            type(exc).__name__,
                            exc,
                        )
                        # A warning on stderr is not a record. Every call that
                        # reached the model but produced no capture is written
                        # here, so "N captures" can always be reconciled against
                        # "M upstream 2xx responses" instead of being taken on
                        # trust. Best-effort: a failure to log a failure must
                        # still not break the agent's turn.
                        try:
                            with lock:
                                fail_path = Path(capture_dir) / CAPTURE_FAILURE_FILENAME
                                fail_path.parent.mkdir(parents=True, exist_ok=True)
                                with fail_path.open("a", encoding="utf-8") as fh:
                                    fh.write(
                                        json.dumps(
                                            {
                                                "at": request_started_at,
                                                "path": path,
                                                "status": code,
                                                "error_type": type(exc).__name__,
                                                "error": str(exc),
                                                "body_head": resp_body[:600].decode(
                                                    "utf-8", "replace"
                                                ),
                                            }
                                        )
                                        + "\n"
                                    )
                        except OSError:
                            _log.warning("could not append to %s", CAPTURE_FAILURE_FILENAME)

                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

            def do_GET(self):
                self._forward("GET")

            def do_POST(self):
                self._forward("POST")

            def do_PUT(self):
                self._forward("PUT")

            def do_DELETE(self):
                self._forward("DELETE")

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd = httpd
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread = thread
        thread.start()
        return self.api_base

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


@contextmanager
def capture_proxy(
    upstream_api_base: str,
    capture_dir: Path | str,
    *,
    inject_logprobs: bool = False,
    top_logprobs: int | None = None,
    upstream_api_key: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Iterator[CaptureProxy]:
    """Context manager around `CaptureProxy.start` / `stop`.

    Mirrors every `CaptureProxy` field that changes what goes on the wire.
    A helper that silently drops one is worse than no helper: tests written
    against it then exercise a configuration the CLI never runs.
    """
    proxy = CaptureProxy(
        upstream_api_base=upstream_api_base,
        capture_dir=Path(capture_dir),
        top_logprobs=top_logprobs,
        upstream_api_key=upstream_api_key,
        host=host,
        port=port,
        inject_logprobs=inject_logprobs,
    )
    proxy.start()
    try:
        yield proxy
    finally:
        proxy.stop()


__all__ = [
    "CAPTURE_FAILURE_FILENAME",
    "CAPTURE_FILENAME",
    "RETURN_TOKEN_IDS_KEY",
    "CaptureProxy",
    "CapturedCompletion",
    "TokenCaptureError",
    "append_capture",
    "capture_path",
    "capture_proxy",
    "detokenized_text",
    "dump_capture_manifest",
    "extract_captured_completion",
    "litellm_extra_body",
    "load_captures",
    "token_capture_agent_kwargs",
]
