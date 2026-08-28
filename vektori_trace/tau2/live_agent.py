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
So this module reconstructs the same head, from the same renderer, and
`live_render_parity.py` asserts it against the frozen ids before any episode is
paid for.

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
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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

    usage: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        """JSON-safe form. Bytes are hex; a lossy `.decode()` here would
        discard exactly the UTF-8 splits that commits `1e63b6e`/`f3a1983`
        were about."""
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
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
            "content": self.content,
            "tool_calls": self.tool_calls,
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
    """Prompt ids for the *next* assistant turn.

    Uses `dataset._encode_messages` -- the same call `tokenize_messages` makes
    to build the prefix half of every frozen C30 row -- with the template kwargs
    `export.build_row` pinned: `{"tools": tools, "enable_thinking": True}`, and
    `add_generation_prompt=True`.

    That last flag is what puts the render exactly where the model will be asked
    to generate: under `enable_thinking=True` it ends at
    `<|im_start|>assistant\\n`. `build_row` relies on the same property to place
    its supervised span, so matching it here is what makes a live prompt and a
    frozen prefix comparable at all.

    `_encode_messages` is private, and deliberately reached for rather than
    reimplemented: a second renderer is a second thing to keep in sync, and the
    whole point of `live_render_parity` is that this path and the corpus path
    are the *same* code. Relying on tokenizer defaults instead of naming these
    kwargs is how a corpus ends up rendered differently from the way it is
    served; Qwen3's template has already cost this repo one silent label bug.

    The `<think>\\n\\n</think>\\n\\n` wrapper
    ----------------------------------------
    `add_generation_prompt=True` ends the render at `<|im_start|>assistant\\n`.
    The frozen prefixes go **four tokens further**, and that is not an accident
    of the corpus build -- it is a consequence of how the boundary is defined.

    `prompt_ids_from_row` cuts at the first *unmasked* label, and
    `tokenize_messages(..., mask_think_wrapper=True)` masks exactly those four
    wrapper tokens so the model is never trained to emit an empty reasoning
    block. Masked means "not supervised", which puts the wrapper on the prompt
    side of the cut:

        ... <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n | No problem! I can ...
                                  ^^^^ in the prompt ^^^^      ^ first supervised token

    So a live render that stops at `add_generation_prompt` is four tokens short
    of the state the corpus sampled from -- and, worse, would ask the model to
    open its own `<think>` block while the corpus had already closed one. That
    is a train/serve skew that produces a finite loss and clean logs.

    Measured: without the wrapper, **0/289** prefixes re-render
    (`tau2_live_render_parity.py`, 2026-08-28); every failure diverged at
    exactly the prompt's last token with the frozen side reading
    `<think>\\n\\n</think>\\n\\n`.

    The ids are appended rather than hardcoded, and the caller-visible failure
    is an assert in `dataset._think_wrapper_ids` on a tokenizer that disagrees
    -- a CPU failure instead of a silently different prompt.
    """
    from ..dataset import _encode_messages, _think_wrapper_ids

    ids = _encode_messages(
        tokenizer,
        messages,
        {"tools": list(tools), "enable_thinking": True},
        add_generation_prompt=True,
    )
    return list(ids) + list(_think_wrapper_ids(tokenizer))


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
        llm_args: Optional[dict] = None,
        on_capture: Optional[Callable[[TurnCapture], None]] = None,
        poster: Optional[Callable[..., tuple[int, dict]]] = None,
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
        self._post = poster or post_json
        self._on_capture = on_capture

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

        status, body = self._post(
            f"{self.api_base}/completions",
            {
                "model": self.model,
                "prompt": prompt_ids,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "logprobs": 0,
                "return_token_ids": True,
            },
            self.timeout,
        )
        if status != 200:
            raise LiveCaptureError(
                f"turn {self.turn_index}: HTTP {status}: {str(body)[:300]}"
            )

        cap = build_capture(
            body,
            episode_id=self.episode_id,
            task_id=self.task_id,
            turn_index=self.turn_index,
            policy_version=self.policy_version,
            prompt_ids=prompt_ids,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
        )
        self.captures.append(cap)
        if self._on_capture is not None:
            # Persist as it lands. A crash on turn 6 must not discard the
            # GPU-generated turns before it -- their behaviour logprobs cannot
            # be recreated.
            self._on_capture(cap)
        self.turn_index += 1

        return AssistantMessage(
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
