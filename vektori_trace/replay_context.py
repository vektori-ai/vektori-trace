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


def filter_candidates_by_budget(
    candidates: list[Any],
    render: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    progress: Any = None,
    allow_render_errors: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Drop candidates that cannot be sampled, *before* selection runs.

    Filtering has to happen here rather than after selection or inside the
    sampling loop, and the reason is about what "stratified" means. Selection
    picks eight prefixes spread across tasks and trace stages; if 38% of the
    pool cannot actually be sampled, a post-hoc rejection re-runs selection on
    an unknown subset, and an in-loop rejection kills the batch after some
    actions are already paid for. Only a pre-filter lets selection spread
    across the states that are genuinely reachable.

    The report deliberately records the stage and post-compaction distribution
    **before and after** filtering. Overflow is not uniform — long-horizon
    prefixes are exactly the ones that overflow — so the filter can quietly
    convert a late-stage stratum into an early-stage one while every count
    still looks right.

    `render` maps one candidate to its prompt string.
    """
    kept: list[Any] = []
    dropped: list[dict[str, Any]] = []
    budgets: list[ContextBudget] = []

    def _stage(c: Any) -> str:
        return str(getattr(c, "step_index", "?"))

    before_steps: dict[str, int] = {}
    after_steps: dict[str, int] = {}
    before_pc = after_pc = 0

    for i, c in enumerate(candidates):
        if progress is not None and i % 500 == 0:
            progress(i, len(candidates))
        before_steps[_stage(c)] = before_steps.get(_stage(c), 0) + 1
        if getattr(c, "post_compaction", False):
            before_pc += 1
        try:
            budget = measure_prefix(
                getattr(c, "prefix_id", str(i)),
                render(c),
                tokenizer,
                max_new_tokens=max_new_tokens,
                max_model_len=max_model_len,
            )
        except Exception as e:
            dropped.append(
                {"prefix_id": getattr(c, "prefix_id", str(i)),
                 "reason": f"render/measure failed: {type(e).__name__}: {e}"}
            )
            continue
        budgets.append(budget)
        if budget.fits:
            kept.append(c)
            after_steps[_stage(c)] = after_steps.get(_stage(c), 0) + 1
            if getattr(c, "post_compaction", False):
                after_pc += 1
        else:
            dropped.append(
                {"prefix_id": budget.prefix_id,
                 "prefix_tokens": budget.prefix_tokens,
                 "over_by": -budget.headroom,
                 "reason": "context overflow"}
            )

    fitting = [b for b in budgets if b.fits]
    n_render_errors = len(dropped) - (len(budgets) - len(fitting))
    if n_render_errors and not allow_render_errors:
        # Overflow is a legitimate exclusion; a render failure is not. It means
        # a candidate could not be built at all — a code or data fault — and
        # continuing would select from a subset biased by whatever broke, while
        # every downstream count still looks consistent.
        first = next(
            (d for d in dropped if d.get("reason", "").startswith("render/measure")),
            {},
        )
        raise ContextBudgetError(
            f"{n_render_errors} candidate(s) failed to render or tokenize; "
            "refusing to select from a pool shaped by an unexplained failure. "
            f"First: {first.get('prefix_id')}: {first.get('reason')}. "
            "Pass allow_render_errors=True only for exploratory reporting."
        )
    report = {
        "n_candidates": len(candidates),
        "n_fitting": len(kept),
        "n_overflow": len(budgets) - len(fitting),
        "n_render_errors": n_render_errors,
        "overflow_rate": (
            round(1 - len(fitting) / len(budgets), 4) if budgets else None
        ),
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "budget_for_prefix": max_model_len - max_new_tokens,
        "eligible_tasks_before": len({getattr(c, "task", None) for c in candidates}),
        "eligible_tasks_after": len({getattr(c, "task", None) for c in kept}),
        "eligible_traces_before": len({getattr(c, "trace_id", None) for c in candidates}),
        "eligible_traces_after": len({getattr(c, "trace_id", None) for c in kept}),
        "post_compaction_before": before_pc,
        "post_compaction_after": after_pc,
        "stage_distribution_before": dict(
            sorted(before_steps.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
        ),
        "stage_distribution_after": dict(
            sorted(after_steps.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
        ),
        "prefix_tokens_fitting": summarize_budgets(fitting) if fitting else {},
        "dropped": dropped[:200],
        "n_dropped_recorded": min(len(dropped), 200),
        "n_dropped_total": len(dropped),
    }
    return kept, report


def assert_prompt_ids_match(
    prefix_id: str,
    local_ids: list[int],
    server_ids: list[int] | None,
    *,
    max_report: int = 6,
) -> dict[str, Any]:
    """The server must have consumed exactly the prompt we measured.

    The local budget check proves our *rendering* fits. It cannot prove the
    server saw that rendering: a chat-template difference, a tokenizer revision
    skew, or front-truncation all leave the local count intact while changing
    what was actually conditioned on. `log pi_old` would then be captured under
    a prompt we cannot reproduce, and every downstream assertion still passes.

    The returned `prompt_token_ids` make this checkable for free at sampling
    time, so it is checked rather than assumed — and it matters most exactly
    where headroom is thin.
    """
    if not server_ids:
        raise ContextBudgetError(
            f"{prefix_id}: the server returned no prompt_token_ids, so the "
            "prompt it consumed cannot be reconciled with the one measured "
            "locally. Request them (`return_token_ids`) rather than assuming "
            "the renderings agree."
        )
    if len(local_ids) != len(server_ids):
        raise ContextBudgetError(
            f"{prefix_id}: local rendering is {len(local_ids)} tokens but the "
            f"server consumed {len(server_ids)}. A shorter server prompt means "
            "silent front-truncation; a different length at all means the two "
            "renderings disagree (§4). Difference: "
            f"{len(server_ids) - len(local_ids):+d}."
        )
    mismatches = [i for i, (a, b) in enumerate(zip(local_ids, server_ids)) if a != b]
    if mismatches:
        head = mismatches[:max_report]
        detail = ", ".join(
            f"pos {i}: local {local_ids[i]} != server {server_ids[i]}" for i in head
        )
        raise ContextBudgetError(
            f"{prefix_id}: {len(mismatches)} prompt token id(s) differ between "
            f"the local rendering and the server's. First: {detail}. The "
            "template or tokenizer has drifted; log pi_old would be captured "
            "under a prompt this run cannot reproduce."
        )
    return {"prefix_id": prefix_id, "n_prompt_tokens": len(local_ids), "exact_match": True}


__all__ = [
    "ContextBudget",
    "assert_prompt_ids_match",
    "filter_candidates_by_budget",
    "ContextBudgetError",
    "DEFAULT_MAX_MODEL_LEN",
    "assert_prefix_fits",
    "measure_prefix",
    "summarize_budgets",
]
