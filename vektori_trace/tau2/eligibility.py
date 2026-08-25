"""Section 5 eligibility gates over normalized Tau2 traces.

A gate that cannot fail is worse than no gate: it reports compliance that was
never checked. The first version of this module had one — confirmation was
satisfied by *any* prior assistant speech, and since the scripted greeting is in
every prompt, it passed unconditionally on all 73 traces. That result meant
nothing. Every gate here is written so that a plausible violation would fail it,
and the ones that cannot be checked from the transcript say so rather than
returning True.

Structural gates prove the recording is well formed. Policy gates prove the
episode obeyed the retail policy's hard requirements. Rendering eligibility is
separate again and lives in `export.py`, because it needs the tokenizer; a trace
is *fully* eligible only when all three have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .normalize import NormalizedTrace

AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})

# Only these are treated as proof the recording ended on purpose. An absent
# termination_reason is *not* clean: it means the field is missing, which is a
# defect in the recording, not evidence of a tidy stop.
CLEAN_TERMINATIONS = frozenset({"user_stop", "agent_stop"})

# Substrings that mark a tool result as an error rather than data.
ERROR_MARKERS = ("error", "not found", "invalid", "cannot", "unable to",
                 "permission", "denied", "failed")

# Bare assent. These words alone do not establish *what* was assented to:
# "yes, that's my email" matches every one of them. They are necessary, never
# sufficient — a mutation is confirmed only when assent follows a proposal that
# names the same action, and the outcome is `uncertain` whenever that link
# cannot be established from the transcript.
ASSENT = ("yes", "yeah", "yep", "correct", "confirm", "confirmed", "go ahead",
          "please do", "sounds good", "that's right", "thats right", "sure",
          "ok", "okay", "proceed", "do it", "agreed")

DISSENT = ("no", "don't", "dont", "stop", "wait", "cancel that", "not yet",
           "hold on", "actually", "instead")

# Confirmation outcomes. Only PASS and NOT_APPLICABLE are eligible when
# confirmation is mandatory; UNCERTAIN is never silently promoted to a pass.
CONFIRM_PASS = "pass"
CONFIRM_FAIL = "fail"
CONFIRM_UNCERTAIN = "uncertain"
CONFIRM_NA = "not_applicable"

# Manual review outcomes. `not_reviewed` is the honest default: no human has
# looked at the trace, which is different from a human finding it acceptable.
MANUAL_PASS = "pass"
MANUAL_FAIL = "fail"
MANUAL_UNCERTAIN = "uncertain"
MANUAL_NOT_REVIEWED = "not_reviewed"

# Surface forms by which an agent proposal names the mutation it is about to
# make. Matching is on the proposal text preceding the call.
PROPOSAL_TERMS = {
    "cancel_pending_order": ("cancel",),
    "modify_pending_order_items": ("modify", "change", "swap", "exchange the item",
                                   "replace"),
    "modify_pending_order_address": ("address",),
    "modify_pending_order_payment": ("payment", "card", "gift card"),
    "modify_user_address": ("address",),
    "return_delivered_order_items": ("return",),
    "exchange_delivered_order_items": ("exchange", "swap"),
}


@dataclass
class Verdict:
    task_id: str
    gates: dict[str, bool] = field(default_factory=dict)
    unaudited: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Confirmation is a DIAGNOSTIC, never a gate. The `diagnostic_` prefix is
    # deliberate: an earlier version gated on this classifier and rejected
    # 64/73 traces, and manual inspection showed it was rejecting clearly
    # compliant sequences -- its proposal taxonomy conflated overlapping
    # mutation categories, so one agent sentence saying "modification" opened
    # four proposals at once and made every subsequent assent "ambiguous".
    # Those outcomes measure classifier uncertainty, not trace non-compliance.
    # Confirmation is evaluated primarily through Tau2's official communication
    # checks, supplemented by targeted manual review.
    diagnostic_confirmation: str = CONFIRM_NA
    diagnostic_confirmation_notes: list[str] = field(default_factory=list)

    # Filled in by human review, never by this module. `not_reviewed` is the
    # honest default: nobody has looked, which is not the same as approval.
    manual_confirmation: str = MANUAL_NOT_REVIEWED
    manual_reviewer: str | None = None
    manual_review_reason: str | None = None

    # Id-level proof for any gate that excluded this trace, so an exclusion is
    # independently auditable from the report alone.
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def structural(self) -> bool:
        return all(v for k, v in self.gates.items() if not k.startswith("policy_"))

    def eligible(self) -> bool:
        """Every gate passes.

        Deliberately does NOT consult `diagnostic_confirmation`. Eligibility is
        decided by structural integrity, the official Tau2 reward,
        authentication, precondition discipline, and rendering. Confirmation is
        reported alongside and audited by hand.
        """
        return all(self.gates.values())

    def needs_manual_review(self) -> str | None:
        """Why a human should look at this trace, in priority order."""
        if any("declined" in d for d in self.diagnostic_confirmation_notes):
            return "executed_after_decline"
        if self.diagnostic_confirmation == CONFIRM_FAIL:
            return "heuristic_fail"
        if self.diagnostic_confirmation == CONFIRM_UNCERTAIN:
            return "heuristic_uncertain"
        return None

    def failed(self) -> list[str]:
        return sorted(k for k, v in self.gates.items() if not v)


def _is_error_result(content: str) -> bool:
    body = (content or "").strip().lower()
    if not body:
        return False
    if body.startswith("error"):
        return True
    if '"error"' in body or "'error'" in body:
        return True
    return any(m in body[:160] for m in ERROR_MARKERS)


def _assented(text: str) -> bool | None:
    """True on assent, False on dissent, None when the turn says neither."""
    t = (text or "").strip().lower()
    if not t:
        return None
    head = t[:200]
    if any(d in head for d in DISSENT) and not any(
            head.startswith(a) for a in ("yes", "yeah", "correct", "confirm")):
        return False
    if any(a in head for a in ASSENT):
        return True
    return None


def _identifiers(args: dict[str, Any]) -> list[str]:
    """Argument values specific enough that a proposal should have named them.

    Order ids, item ids and addresses identify *which* thing is being changed.
    Booleans, enums and short codes do not discriminate and would produce noise.
    """
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str):
            if len(v) >= 6 and any(c.isdigit() for c in v):
                out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(args)
    return out


def _proposes(text: str, tool_name: str) -> bool:
    """Does this agent turn name the mutation `tool_name` is about to perform?

    Bare acknowledgement ("let me check that") is not a proposal. The turn must
    contain a term specific to the action, which is what lets a later "yes" be
    attributed to *this* mutation rather than to an unrelated question.
    """
    body = (text or "").lower()
    if not body:
        return False
    terms = PROPOSAL_TERMS.get(tool_name)
    if not terms:
        return False
    return any(t in body for t in terms)


def _confirmation_outcome(msgs: list[dict[str, Any]],
                          mutating_tools: frozenset[str] | set[str],
                          result_by_id: dict[str, str]) -> tuple[str, list[str]]:
    """Classify the trace's confirmation discipline as pass/fail/uncertain/na.

    The sequence a pass requires:

        assistant proposes a specific mutation
            -> user assents to that proposal
            -> assistant executes the matching mutation

    Anything that breaks the chain is a fail. Anything that cannot be resolved
    — bare assent with more than one proposal open, assent whose referent is
    ambiguous, a mutation with no term this module knows how to match — is
    `uncertain`, and the caller must not treat it as a pass.
    """
    detail: list[str] = []
    outcome = CONFIRM_NA

    def worsen(new: str) -> str:
        rank = {CONFIRM_NA: 0, CONFIRM_PASS: 1, CONFIRM_UNCERTAIN: 2, CONFIRM_FAIL: 3}
        return new if rank[new] > rank[outcome] else outcome

    pending: list[str] = []      # mutations proposed since the user last spoke
    assent_for: list[str] = []   # proposals the user's last turn assented to
    last_user: str | None = None
    pending_text = ""            # agent text since the user last spoke
    proposal_text = ""           # the text the user's assent applied to

    for m in msgs:
        role = m.get("role")

        if role == "user":
            last_user = (m.get("content") or "")
            verdict = _assented(last_user)
            if verdict is True:
                if len(pending) == 1:
                    assent_for = list(pending)
                elif len(pending) > 1:
                    assent_for = []
                    detail.append(
                        f"bare assent with {len(pending)} proposals open: "
                        f"{pending} — referent ambiguous"
                    )
                    outcome = worsen(CONFIRM_UNCERTAIN)
                else:
                    assent_for = []
            elif verdict is False:
                assent_for = []
            else:
                assent_for = []
            proposal_text = (pending_text + " " + last_user).lower() if assent_for else ""
            pending = []
            pending_text = ""
            continue

        if role != "assistant":
            continue

        for tc in (m.get("tool_calls") or []):
            name = tc.get("name")
            if name not in mutating_tools:
                continue
            outcome = worsen(CONFIRM_PASS) if outcome == CONFIRM_NA else outcome

            if name not in PROPOSAL_TERMS:
                detail.append(f"{name}: no proposal terms defined, cannot match")
                outcome = worsen(CONFIRM_UNCERTAIN)
                continue
            if name in assent_for:
                # The action was proposed and assented to. Check that what is
                # being executed is what was described: identifiers in the call
                # (order ids, item ids, addresses) should appear in the text the
                # user agreed to. An argument the proposal never mentioned means
                # the executed mutation and the confirmed one may differ.
                ids = _identifiers(tc.get("arguments") or {})
                unstated = [i for i in ids if i.lower() not in proposal_text]
                if unstated:
                    detail.append(
                        f"{name}: arguments {unstated[:4]} do not appear in the "
                        "proposal the user assented to"
                    )
                    outcome = worsen(CONFIRM_UNCERTAIN)
                continue
            if _assented(last_user or "") is False:
                detail.append(f"{name}: executed after the user declined")
                outcome = worsen(CONFIRM_FAIL)
            elif _assented(last_user or "") is None:
                detail.append(
                    f"{name}: no user assent between the proposal and the mutation"
                )
                outcome = worsen(CONFIRM_FAIL)
            else:
                detail.append(
                    f"{name}: user assented but not to a proposal naming this action"
                )
                outcome = worsen(CONFIRM_UNCERTAIN)

        text = (m.get("content") or "").strip()
        if text:
            pending_text += " " + text
            # sorted(): `mutating_tools` is a set, and iterating it unsorted made
            # the diagnostic note text vary between runs. Nothing downstream
            # gates on it, but a report that will not rebuild byte-identically
            # cannot be used to prove the corpus reproduces.
            for tool in sorted(mutating_tools):
                if _proposes(text, tool) and tool not in pending:
                    pending.append(tool)

    return outcome, detail


def audit(trace: NormalizedTrace, sim: dict[str, Any],
          mutating_tools: frozenset[str] | set[str]) -> Verdict:
    """Run every gate. `mutating_tools` comes from the live Tau2 schema."""
    v = Verdict(task_id=trace.task_id)
    ri = sim.get("reward_info") or {}
    msgs = sim.get("messages") or []
    g = v.gates

    # ---- structural ------------------------------------------------------
    g["reward_pass"] = ri.get("reward") in (1, 1.0)

    term = (sim.get("termination_reason") or "").strip().lower()
    g["terminated_clean"] = term in CLEAN_TERMINATIONS
    if not term:
        v.notes.append("termination_reason is absent")
    g["auditable_components"] = any(
        ri.get(k) is not None
        for k in ("db_check", "communicate_checks", "action_checks", "nl_assertions")
    )
    g["has_decisions"] = len(trace.decisions) >= 3
    g["no_empty_actions"] = all(
        d.tool_names or (d.target.get("content") or "").strip()
        for d in trace.decisions
    )

    # Pair calls to results by id: every call answered exactly once, every
    # result answering a real call. A count comparison misses duplicates and
    # crossed ids entirely.
    call_ids: list[str] = []
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                call_ids.append(tc.get("id"))
    result_ids = [(m.get("id") or m.get("tool_call_id"))
                  for m in msgs if m.get("role") == "tool"]

    g["tool_ids_present"] = all(i is not None for i in call_ids + result_ids)
    if not g["tool_ids_present"]:
        v.notes.append(
            f"{sum(1 for i in call_ids if i is None)} calls and "
            f"{sum(1 for i in result_ids if i is None)} results lack ids"
        )
    g["no_duplicate_call_ids"] = len(call_ids) == len(set(call_ids))
    g["no_duplicate_results"] = len(result_ids) == len(set(result_ids))
    g["no_orphan_tool_results"] = set(result_ids) <= set(call_ids)
    g["no_unanswered_calls"] = set(call_ids) <= set(result_ids)
    if not g["no_orphan_tool_results"]:
        v.notes.append(f"orphan result ids: {sorted(set(result_ids) - set(call_ids))[:4]}")
    if not g["no_unanswered_calls"]:
        v.notes.append(f"unanswered call ids: {sorted(set(call_ids) - set(result_ids))[:4]}")

    # ---- policy ----------------------------------------------------------
    # Walk the transcript once, tracking authentication, open errors, and what
    # the agent has proposed but not yet had confirmed.
    result_by_id = {(m.get("id") or m.get("tool_call_id")): (m.get("content") or "")
                    for m in msgs if m.get("role") == "tool"}

    authenticated = False          # an auth tool RETURNED a user, not merely called
    auth_ok = True
    precondition_ok = True
    open_errors: set[str] = set()  # tool names whose last result was an error
    open_error_ids: dict[str, str] = {}  # tool name -> the call id that failed

    for m in msgs:
        if m.get("role") != "assistant":
            continue

        for tc in (m.get("tool_calls") or []):
            name, cid = tc.get("name"), tc.get("id")
            res = result_by_id.get(cid, "")

            if name in mutating_tools:
                if not authenticated:
                    auth_ok = False
                    v.notes.append(f"mutation {name} before a successful authentication")
                if open_errors:
                    precondition_ok = False
                    # Record call/result/mutation ids so an exclusion on this
                    # gate can be re-checked by hand from the report alone.
                    v.evidence.append({
                        "gate": "policy_no_mutation_after_error",
                        "mutation_tool": name,
                        "mutation_call_id": cid,
                        "unresolved": [
                            {"tool": et, "call_id": open_error_ids.get(et),
                             "result": (result_by_id.get(open_error_ids.get(et)) or "")[:200]}
                            for et in sorted(open_errors)
                        ],
                    })
                    v.notes.append(
                        f"mutation {name} (call {cid}) while {sorted(open_errors)} "
                        "remain unresolved"
                    )

            # An error stays open until the SAME tool returns successfully.
            # Any-next-call clearing let an unrelated lookup mask the failure.
            if _is_error_result(res):
                open_errors.add(name)
                open_error_ids[name] = cid
            elif name in open_errors:
                open_errors.discard(name)
                open_error_ids.pop(name, None)

            if name in AUTH_TOOLS and res and not _is_error_result(res):
                authenticated = True

    g["policy_auth_before_mutation"] = auth_ok
    g["policy_no_mutation_after_error"] = precondition_ok

    # Confirmation is deliberately NOT a gate — not even a four-valued one.
    # See the note on Verdict.diagnostic_confirmation for what happened when it
    # was one.
    v.diagnostic_confirmation, v.diagnostic_confirmation_notes = _confirmation_outcome(
        msgs, mutating_tools, result_by_id
    )

    # What no transcript-level check can settle: whether the user understood the
    # specific terms (price difference, item ids, payment method) they assented
    # to. Tau2's own communication checks cover part of this and are reported
    # alongside the official reward.
    v.unaudited.append("user_understood_the_specific_terms_assented_to")

    return v
