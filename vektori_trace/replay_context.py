"""Prefix context-budget checks (plan §4, §11).

§4 makes provider-side silent truncation a hard failure, and §11 lists "silent
history truncation" as a stop condition. The action side of that is already
guarded: `replay_sample` refuses a capture whose `finish_reason` is "length".
The *prefix* side is not, and it fails differently and more quietly.

The asymmetry is the point. A too-long action comes back flagged — the provider
says "length" and the capture is refused. A too-long prefix does not: vLLM and
most OpenAI-compatible servers drop tokens from the front of the prompt, or
error, depending on configuration. When they drop, sampling succeeds, the
action looks well-formed, `log pi_old` is finite, alignment passes, and the
loss is a real number — computed at a state that is not the replay state the
run claims to have sampled. Every §8.4 assertion still passes.

Worse for this experiment specifically: the states most likely to overflow are
long-horizon and post-compaction prefixes, which are exactly the ones §8.3
exists to stress. A silent overflow would therefore bias the batch toward the
short, easy states while reporting the intended stratification.

So the budget is checked locally, before the request, against the pinned
tokenizer — not inferred from the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContextBudgetError(RuntimeError):
    """A replay prefix does not fit the serving context window."""


#: Qwen3-14B's native context, and what SOL-HANDOFF pins the L40S server to
#: (`--max-model-len 40960`). Not a default to be quietly raised: the serving
#: side must be changed with it or the check becomes a lie.
DEFAULT_MAX_MODEL_LEN = 40960


@dataclass(frozen=True)
class ContextBudget:
    """What one prefix costs against the window, and whether it fits."""

    prefix_id: str
    prefix_tokens: int
    max_new_tokens: int
    max_model_len: int

    @property
    def required(self) -> int:
        return self.prefix_tokens + self.max_new_tokens

    @property
    def headroom(self) -> int:
        return self.max_model_len - self.required

    @property
    def fits(self) -> bool:
        return self.headroom >= 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_id": self.prefix_id,
            "prefix_tokens": self.prefix_tokens,
            "max_new_tokens": self.max_new_tokens,
            "max_model_len": self.max_model_len,
            "required": self.required,
            "headroom": self.headroom,
            "fits": self.fits,
        }


def measure_prefix(
    prefix_id: str,
    prompt_text: str,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> ContextBudget:
    """Token cost of one rendered prefix, measured with the pinned tokenizer.

    `add_special_tokens=False` because `prompt_text` is already a fully rendered
    chat prompt: letting the tokenizer add its own would count tokens the server
    will not see, making the budget wrong in the safe direction but wrong.
    """
    ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return ContextBudget(
        prefix_id=prefix_id,
        prefix_tokens=len(ids),
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
    )


def assert_prefix_fits(budget: ContextBudget) -> None:
    """Refuse a prefix that cannot be sampled without truncation.

    Fails closed, and before the request. The alternative — letting the server
    decide — produces a finite loss at an unknown state, which is the failure
    mode §4 calls a hard failure precisely because nothing downstream reveals it.
    """
    if not budget.fits:
        raise ContextBudgetError(
            f"{budget.prefix_id}: prefix is {budget.prefix_tokens} tokens and the "
            f"action cap is {budget.max_new_tokens}, needing {budget.required} of "
            f"{budget.max_model_len}. Over by {-budget.headroom}. The server would "
            "silently drop leading tokens and sampling would succeed at a state "
            "this run cannot describe (§4, §11)."
        )


def summarize_budgets(budgets: list[ContextBudget]) -> dict[str, Any]:
    """Batch-level context report for §10.

    `min_headroom` is the number worth watching: a batch that fits with 200
    tokens to spare fits by accident, and any change to the renderer or cap
    breaks it.
    """
    if not budgets:
        return {"n_prefixes": 0}
    toks = sorted(b.prefix_tokens for b in budgets)
    over = [b for b in budgets if not b.fits]
    return {
        "n_prefixes": len(budgets),
        "n_overflow": len(over),
        "overflow_prefix_ids": [b.prefix_id for b in over],
        "min_prefix_tokens": toks[0],
        "median_prefix_tokens": toks[len(toks) // 2],
        "max_prefix_tokens": toks[-1],
        "min_headroom": min(b.headroom for b in budgets),
        "max_model_len": budgets[0].max_model_len,
        "max_new_tokens": budgets[0].max_new_tokens,
    }


__all__ = [
    "ContextBudget",
    "ContextBudgetError",
    "DEFAULT_MAX_MODEL_LEN",
    "assert_prefix_fits",
    "measure_prefix",
    "summarize_budgets",
]
