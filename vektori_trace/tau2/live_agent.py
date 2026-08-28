"""A Tau2 agent that captures what `generate()` throws away.

Why this exists
---------------
Tau2's `LLMAgent` calls `utils.llm_utils.generate`, which posts to
`/chat/completions` and keeps exactly two things off the response:

    content    = response_choice.message.content
    tool_calls = response_choice.message.tool_calls

Both are *post-parse*. vLLM runs `--reasoning-parser qwen3` and
`--tool-call-parser hermes` server-side, so by the time litellm returns, the
`<think>` block has been moved to a separate field and the `<tool_call>` block
has been restructured into objects. The raw text is gone, and `generate` never
asked for `logprobs` or `return_token_ids` in the first place.

Those three -- **token ids, behaviour logprobs, raw bytes** -- are the training
signal. `log pi_old` cannot be recovered after the fact: a later forward pass
gives `log pi_current`, which is a different quantity. A turn captured without
them is unusable, so this agent raises rather than returning one.

What it does instead
--------------------
Subclass `LLMAgent` and override `_generate_next_message` only. Tau2 keeps its
orchestrator, environments, tools, user simulator and grader; this class owns
one turn:

    Tau2 state  -> canonical messages   (the corpus's head: system+tools)
                -> prompt token ids     (dataset.render, the pinned renderer)
                -> POST /completions    (logprobs + return_token_ids)
                -> capture              (ids, logprobs, raw bytes)
                -> split think/tool     (client-side; the server did not)
                -> AssistantMessage     (handed back to Tau2 unchanged)

Rendering parity is the point of `canonical_messages`
-----------------------------------------------------
`export.build_row` rendered every frozen prefix from
`[{"role": "system", "content": policy}] + prompt`, with
`template_kwargs={"tools": tools, "enable_thinking": True}`. Two corpus bugs
were caught only by re-rendering and comparing bytes: the policy was omitted
entirely, and the tools never reached the teacher. **Neither is caught by any
hash, and both produce a finite loss, a successful alignment and clean logs.**
So this module reconstructs the same semantic head with the same renderer. The
boundary intentionally differs by the frozen rows' four-token *empty* think
wrapper: those rows supervise visible actions only, while live OPD must stop
before that wrapper so Qwen can generate real reasoning and we can capture its
tokens and behavior logprobs. The distinction is explicit in the run manifest.

Note that Tau2's own system prompt (`LLMAgent.system_prompt`) wraps the domain
policy in `SYSTEM_PROMPT`/`AGENT_INSTRUCTION`. The corpus used the bare policy
string. Which one the student is served must match which one it was trained on,
so `system_content` is an explicit constructor argument with no default -- a
silent choice here is a train/serve skew that nothing downstream would show.

What it refuses
---------------
Every one of these has a matching silent failure:

- a response whose ids and logprobs differ in length -- there is no way to know
  which token lost its score;
- `finish_reason == "length"` -- a fragment is not a decision, and scoring one
  as a completed action is what produced the 0/13 OPD run;
- ids that do not reconstruct the text the server reported -- the server
  consumed a different prompt than the one rendered;
- a prompt longer than the endpoint's input budget -- vLLM truncates from the
  *front*, silently sampling a state the run cannot describe.
- a reasoning-inclusive run whose response has no non-empty `<think>` span.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..replay_sample import token_bytes_from_ids


class LiveCaptureError(RuntimeError):
    """A turn cannot be trusted as a scorable action."""


#: Only trailing template/whitespace tokens may separate the reconstructed
#: bytes from the text the server reported. Anything else is a real
#: id/text disagreement. Mirrors `reopd_sample._ONLY_SPECIALS`.
_ONLY_SPECIALS = re.compile(r"(<\|[^|>]+\|>|\s)+")

#: Qwen3 reasoning block. vLLM's `qwen3` reasoning parser strips exactly this;
#: on the `/completions` path it arrives inline and we split it ourselves.
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

#: Hermes tool-call block. `--tool-call-parser hermes` produces structured
#: calls from these; on `/completions` we parse them client-side.
#: `tau2_parser_parity_modal.py` verified 25/25 targets round-trip through
#: vLLM's `Hermes2ProToolParser`; `live_parser_parity.py` runs the same corpus
#: through *this* regex so the two splitters are known to agree.
_TOOLCALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class TurnCapture:
    """Everything one assistant turn must carry to be trainable.

    `behavior_logprobs` is `log pi_old` in the importance ratio. It is captured
    here or it does not exist.
    """

    episode_id: str
    task_id: str
    turn_index: int
    policy_version: str

    prompt_token_ids: list[int]
    sampled_token_ids: list[int]
    behavior_logprobs: list[float]
    action_token_bytes: list[bytes]

    raw_text: str
    finish_reason: str

    reasoning: str | None
    content: str | None
    tool_calls: list[dict[str, Any]]

    request_payload: dict[str, Any] = field(default_factory=dict)
    provider_response: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        """JSON-safe form. Bytes are hex; a lossy `.decode()` here would
        discard exactly the UTF-8 splits that commits `1e63b6e`/`f3a1983`
        were about."""
        generated_bytes = b"".join(self.action_token_bytes)
        reasoning_span = _reasoning_byte_span(self.raw_text)
        d = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "turn_index": self.turn_index,
            "policy_version": self.policy_version,
            "prompt_token_ids": self.prompt_token_ids,
            "sampled_token_ids": self.sampled_token_ids,
            "behavior_logprobs": self.behavior_logprobs,
            "action_token_bytes_hex": [b.hex() for b in self.action_token_bytes],
            "raw_text": self.raw_text,
            "raw_bytes_hex": self.raw_text.encode("utf-8").hex(),
            "raw_sha256": hashlib.sha256(
                self.raw_text.encode("utf-8")
            ).hexdigest(),
            "generated_bytes_hex": generated_bytes.hex(),
            "generated_bytes_sha256": hashlib.sha256(generated_bytes).hexdigest(),
            "raw_is_exact_generated_bytes": (
                generated_bytes == self.raw_text.encode("utf-8")
            ),
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
            "reasoning_byte_span": reasoning_span,
            "reasoning_token_indices": _tokens_overlapping_span(
                self.action_token_bytes, reasoning_span
            ),
            "content": self.content,
            "tool_calls": self.tool_calls,
            "request_payload": self.request_payload,
            "provider_response": self.provider_response,
            "usage": self.usage,
            "timestamp": self.timestamp,
        }
        return d


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    """POST and parse, returning `(status, body)` without raising on 4xx/5xx.

    Injected in tests; kept identical in shape to `reopd_sample.post_json` so
    the two sampling paths fail the same way.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": raw}


def split_generation(raw: str) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Split one raw generation into `(reasoning, content, tool_calls)`.

    This is the work vLLM's `qwen3` and `hermes` parsers do server-side. On the
    `/completions` path they do not run, so it happens here.

    Tool arguments are parsed as JSON because that is what Tau2's `ToolCall`
    expects -- `generate` does `json.loads(tool_call.function.arguments)`. A
    malformed block raises: a tool call the student emitted but that cannot be
    executed is a *student mistake* worth training on, but it is the caller's
    job to record it as such, not this function's job to guess a repair.
    """
    reasoning = None
    m = _THINK.search(raw)
    rest = raw
    if m:
        reasoning = m.group(1)
        rest = raw[: m.start()] + raw[m.end() :]

    tool_calls: list[dict[str, Any]] = []
    for i, tc in enumerate(_TOOLCALL.finditer(rest)):
        blob = tc.group(1)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError as e:
            raise LiveCaptureError(
                f"tool_call block {i} is not JSON: {e}. Raw block: {blob[:200]!r}"
            ) from e
        name = obj.get("name")
        if not name:
            raise LiveCaptureError(f"tool_call block {i} has no 'name': {obj!r}")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            # Hermes usually emits an object, but a string payload is legal
            # JSON and must not silently become a string-valued argument dict.
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise LiveCaptureError(
                    f"tool_call block {i} 'arguments' is a string that is not "
                    f"JSON: {e}"
                ) from e
        if not isinstance(args, dict):
            raise LiveCaptureError(
                f"tool_call block {i} 'arguments' is {type(args).__name__}, "
                "expected object"
            )
        tool_calls.append({"name": str(name), "arguments": args})

    content = _TOOLCALL.sub("", rest).strip()
    return reasoning, (content or None), tool_calls


def _reasoning_byte_span(raw: str) -> list[int] | None:
    """UTF-8 byte offsets of the reasoning payload inside the raw action."""
    match = _THINK.search(raw)
    if match is None:
        return None
    return [
        len(raw[: match.start(1)].encode("utf-8")),
        len(raw[: match.end(1)].encode("utf-8")),
    ]


def _tokens_overlapping_span(
    token_bytes: list[bytes], span: list[int] | None
) -> list[int]:
    """Token indices whose byte intervals overlap ``span``."""
    if span is None:
        return []
    start, end = span
    position = 0
    indices: list[int] = []
    for index, piece in enumerate(token_bytes):
        next_position = position + len(piece)
        if position < end and next_position > start:
            indices.append(index)
        position = next_position
    return indices


def verify_ids_reconstruct_text(
    tokenizer: Any, token_ids: list[int], text: str
) -> list[bytes]:
    """Per-token bytes, proven to reconstruct what the server reported.

    Returns the byte list `align.align_by_bytes` consumes. Raises if the ids
    and the text disagree by anything other than trailing template tokens or
    whitespace -- which means the server did not generate from the prompt that
    was sent.
    """
    tok_bytes = token_bytes_from_ids(tokenizer, token_ids)
    joined = b"".join(tok_bytes)
    want = text.encode("utf-8")
    if joined == want:
        return tok_bytes
    if joined.startswith(want):
        tail = joined[len(want) :].decode("utf-8", "replace")
        if _ONLY_SPECIALS.fullmatch(tail):
            return tok_bytes
    raise LiveCaptureError(
        "returned token ids do not reconstruct the returned text; the server "
        f"consumed a different prompt. ids->{joined[:120]!r} text->{want[:120]!r}"
    )


def build_capture(
    body: dict,
    *,
    episode_id: str,
    task_id: str,
    turn_index: int,
    policy_version: str,
    prompt_ids: list[int],
    tokenizer: Any,
    max_tokens: int,
    require_reasoning: bool = False,
) -> TurnCapture:
    """Turn a `/completions` response into a trainable capture, or raise."""
    choices = body.get("choices") or []
    if not choices:
        raise LiveCaptureError(f"response has no choices: {str(body)[:300]}")
    choice = choices[0]

    text = choice.get("text")
    if text is None:
        raise LiveCaptureError("response choice has no 'text'")

    finish = str(choice.get("finish_reason") or "")
    if finish == "length":
        raise LiveCaptureError(
            f"turn {turn_index} hit the {max_tokens}-token cap "
            "(finish_reason=length). A truncated action is a fragment, not a "
            "decision; scoring one as a completed action is the failure that "
            "produced the 0/13 OPD run. Fail closed."
        )

    token_ids = choice.get("token_ids") or []
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    if not token_ids:
        raise LiveCaptureError(
            "response carried no token ids. The request must set "
            "`return_token_ids: true`; without ids there is nothing to align."
        )
    if len(token_ids) != len(logprobs):
        raise LiveCaptureError(
            f"turn {turn_index}: {len(token_ids)} ids but {len(logprobs)} "
            "logprobs. These are log pi_old per sampled token; a length "
            "mismatch means an unknown token lost its score."
        )
    if any(lp is None for lp in logprobs):
        raise LiveCaptureError(
            f"turn {turn_index}: a behaviour logprob is null. log pi_old cannot "
            "be recovered by a later forward pass."
        )

    tok_bytes = verify_ids_reconstruct_text(tokenizer, list(token_ids), text)
    reasoning, content, tool_calls = split_generation(text)
    if require_reasoning and not (reasoning and reasoning.strip()):
        raise LiveCaptureError(
            f"turn {turn_index}: generation has no non-empty <think> reasoning "
            "span; reasoning-inclusive live OPD refuses an action whose "
            "reasoning tokens were not generated and captured"
        )

    return TurnCapture(
        episode_id=episode_id,
        task_id=task_id,
        turn_index=turn_index,
        policy_version=policy_version,
        prompt_token_ids=list(prompt_ids),
        sampled_token_ids=[int(t) for t in token_ids],
        behavior_logprobs=[float(x) for x in logprobs],
        action_token_bytes=tok_bytes,
        raw_text=text,
        finish_reason=finish,
        reasoning=reasoning,
        content=content,
        tool_calls=tool_calls,
        provider_response=dict(body),
        usage=dict(body.get("usage") or {}),
    )


# ---------------------------------------------------------------------------
# Rendering: Tau2 state -> the exact ids the corpus was built with
# ---------------------------------------------------------------------------


def canonical_head(system_content: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """The system turn, shaped the way the corpus shaped it.

    `c30_loader` builds `[{"role": "system", "content": policy, "tools": schemas}]`
    and prepends it. Tools ride *on* the system message because
    `encoding_dsv4._render_system` reads `msg["tools"]` off that turn; carrying
    them beside the message would leave the teacher unconditioned on tools
    while the student's ids contain the full schema block. That defect produced
    a finite loss and clean logs.
    """
    return {"role": "system", "content": system_content, "tools": list(tools)}


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[int]:
    """Render the state immediately before Qwen generates its complete turn.

    ``enable_thinking=True`` is explicit and the renderer stops at the normal
    generation prompt. It must not append the frozen action-only corpus's
    masked empty-think wrapper: that would close ``</think>`` before sampling
    and make reasoning capture impossible. The sampled continuation therefore
    contains reasoning, visible content/tool calls, and behavior logprobs for
    every generated token.
    """
    from ..dataset import _encode_messages

    ids = _encode_messages(
        tokenizer,
        messages,
        {"tools": list(tools), "enable_thinking": True},
        add_generation_prompt=True,
    )
    # Do not append the frozen corpus's masked empty-think wrapper here. That
    # boundary is correct for action-only SFT labels but would close reasoning
    # before live sampling begins.
    return list(ids)


def tau2_messages_to_canonical(
    system_content: str,
    tools: list[dict[str, Any]],
    tau2_messages: list[Any],
) -> list[dict[str, Any]]:
    """`LLMAgentState.messages` -> the corpus's message shape.

    Tau2 carries pydantic messages; the renderer wants plain dicts in the same
    shape `rows.semantic.jsonl` stored. Assistant tool calls are re-emitted in
    OpenAI form because that is what the chat template consumes.

    A message this function does not recognise raises rather than being
    dropped: a silently skipped turn renders a prompt describing a
    conversation that never happened.
    """
    out: list[dict[str, Any]] = [canonical_head(system_content, tools)]
    for m in tau2_messages:
        role = getattr(m, "role", None)
        if role == "user":
            out.append({"role": "user", "content": m.content})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tcs
                ]
            out.append(msg)
        elif role == "tool":
            out.append({
                "role": "tool",
                "content": m.content,
                "tool_call_id": getattr(m, "id", None),
            })
        else:
            raise LiveCaptureError(
                f"unhandled Tau2 message role {role!r}; refusing to render a "
                "prompt that silently omits a turn"
            )
    return out


# ---------------------------------------------------------------------------
# The Tau2 seam
# ---------------------------------------------------------------------------


class CapturingLLMAgent:
    """Drop-in for Tau2's `LLMAgent` that captures the training signal.

    Constructed with the same `(tools, domain_policy, llm, llm_args)` Tau2 hands
    an agent factory, plus the pieces `/completions` needs. Tau2 calls
    `get_init_state` then `generate_next_message` per turn; both are delegated
    to the real `LLMAgent` except for the one generation call.

    This is a composition wrapper rather than a subclass so the module imports
    without Tau2 installed -- the unit tests for splitting, capture validation
    and byte reconstruction run on any machine, and only the live path needs
    the benchmark on the path.
    """

    def __init__(
        self,
        *,
        tools: list[Any],
        domain_policy: str,
        llm: str,
        api_base: str,
        tokenizer: Any,
        tool_schemas: list[dict[str, Any]],
        system_content: str,
        policy_version: str,
        max_tokens: int,
        max_input_tokens: int,
        temperature: float = 0.0,
        timeout: float = 600.0,
        require_reasoning: bool = True,
        llm_args: dict | None = None,
        on_capture: Callable[[TurnCapture], None] | None = None,
        on_turn: (
            Callable[[TurnCapture, list[dict[str, Any]], dict[str, Any]], None]
            | None
        ) = None,
        on_failure: Callable[..., None] | None = None,
        poster: Callable[..., tuple[int, dict]] | None = None,
    ) -> None:
        from tau2.agent.llm_agent import LLMAgent  # imported lazily; see above

        self._inner = LLMAgent(
            tools=tools, domain_policy=domain_policy, llm=llm,
            llm_args=dict(llm_args or {}),
        )
        self.api_base = api_base.rstrip("/")
        self.model = llm.split("/", 1)[-1] if "/" in llm else llm
        self.tokenizer = tokenizer
        self.tool_schemas = list(tool_schemas)
        # No default. Tau2 wraps the policy in SYSTEM_PROMPT/AGENT_INSTRUCTION;
        # the corpus used the bare policy. Guessing here is a train/serve skew
        # that nothing downstream would surface.
        self.system_content = system_content
        self.policy_version = policy_version
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.require_reasoning = require_reasoning
        self._post = poster or post_json
        self._on_capture = on_capture
        # Higher-level archive hook. `on_capture` predates the live episode
        # archive and intentionally remains compatible; this hook additionally
        # carries the semantic pre-action history and parsed Tau2 message.
        self._on_turn = on_turn
        # Called with the raw response body before a `build_capture` failure
        # propagates. The generation was paid for; its ids and behaviour
        # logprobs exist only in that body and are gone once it is discarded.
        #
        # Scope is exactly that: a 200 response whose body did not yield a
        # trainable capture -- malformed Hermes, a cap termination, missing
        # ids, an unusable logprob. It does NOT cover failures with no
        # generation to salvage: a non-200, a transport exception, or the
        # prompt-budget refusal above, all of which raise before any tokens
        # were produced. Those are the driver's to record at episode level.
        self._on_failure = on_failure

        self.episode_id: str = "unset"
        self.task_id: str = "unset"
        self.turn_index: int = 0
        self.captures: list[TurnCapture] = []

    # -- Tau2 delegation ----------------------------------------------------

    @property
    def tools(self):
        return self._inner.tools

    @property
    def system_prompt(self) -> str:
        return self._inner.system_prompt

    def get_init_state(self, message_history=None):
        self.turn_index = 0
        return self._inner.get_init_state(message_history)

    def set_episode(self, episode_id: str, task_id: str) -> None:
        """Tag subsequent captures. Called by the driver before each episode."""
        self.episode_id = episode_id
        self.task_id = task_id
        self.turn_index = 0

    def set_seed(self, seed: int) -> None:
        """Delegate the method Tau2's orchestrator calls at initialization."""
        self._inner.set_seed(seed)

    def is_stop(self, message: Any) -> bool:
        """Delegate Tau2's agent-stop predicate."""
        return self._inner.is_stop(message)

    # -- the one method that differs ---------------------------------------

    def generate_next_message(self, message, state):
        """Mirror `LLMAgent.generate_next_message`, but capture the generation.

        Tau2's version appends the assistant message to `state.messages` after
        `_generate_next_message` returns; that bookkeeping is reproduced here
        exactly, because the next turn's prompt is rendered from this state.
        """
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(self, message, state):
        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

        # Same state bookkeeping as `LLMAgent._generate_next_message`. Doing it
        # differently would render a prompt describing a conversation Tau2 does
        # not think it is in.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        canonical = tau2_messages_to_canonical(
            self.system_content, self.tool_schemas, list(state.messages)
        )
        prompt_ids = render_prompt_ids(self.tokenizer, canonical, self.tool_schemas)

        # vLLM truncates an over-long prompt from the *front*, which samples a
        # state the run cannot describe while every downstream assertion still
        # passes. Refuse instead.
        if len(prompt_ids) > self.max_input_tokens:
            raise LiveCaptureError(
                f"turn {self.turn_index}: prompt is {len(prompt_ids)} tokens "
                f"against an input budget of {self.max_input_tokens}. vLLM "
                "would silently drop tokens from the front of the prompt."
            )

        request_payload = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": 0,
            "return_token_ids": True,
        }
        status, body = self._post(
            f"{self.api_base}/completions",
            request_payload,
            self.timeout,
        )
        if status != 200:
            raise LiveCaptureError(
                f"turn {self.turn_index}: HTTP {status}: {str(body)[:300]}"
            )
        try:
            cap = build_capture(
                body,
                episode_id=self.episode_id,
                task_id=self.task_id,
                turn_index=self.turn_index,
                policy_version=self.policy_version,
                prompt_ids=prompt_ids,
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
                require_reasoning=self.require_reasoning,
            )
            cap.request_payload = dict(request_payload)
        except LiveCaptureError as exc:
            # Archive before re-raising. Malformed Hermes, an empty generation
            # and a cap termination are all real measurements of the policy --
            # they belong in the validity metrics -- and this is the only
            # moment their raw ids and behaviour logprobs exist.
            if self._on_failure is not None:
                self._on_failure(
                    body=body,
                    episode_id=self.episode_id,
                    task_id=self.task_id,
                    turn_index=self.turn_index,
                    policy_version=self.policy_version,
                    prompt_ids=prompt_ids,
                    request_payload=request_payload,
                    semantic_history=canonical,
                    error=exc,
                )
            raise
        assistant_message = AssistantMessage(
            role="assistant",
            content=cap.content,
            tool_calls=[
                ToolCall(
                    id=f"{cap.episode_id}-{cap.turn_index}-{i}",
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for i, tc in enumerate(cap.tool_calls)
            ] or None,
            cost=0.0,
            usage=cap.usage,
            raw_data={"live_capture": True, "finish_reason": cap.finish_reason},
        )
        self.captures.append(cap)
        if self._on_capture is not None:
            # Persist as it lands. A crash on turn 6 must not discard the
            # GPU-generated turns before it -- their behaviour logprobs cannot
            # be recreated.
            self._on_capture(cap)
        if self._on_turn is not None:
            parsed = assistant_message.model_dump(mode="json")
            self._on_turn(cap, canonical, parsed)
        self.turn_index += 1
        return assistant_message


__all__ = [
    "CapturingLLMAgent",
    "LiveCaptureError",
    "TurnCapture",
    "build_capture",
    "canonical_head",
    "post_json",
    "render_prompt_ids",
    "split_generation",
    "tau2_messages_to_canonical",
    "verify_ids_reconstruct_text",
]
