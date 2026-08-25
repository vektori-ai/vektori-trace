"""Tau2 normalization and eligibility gates.

Each test targets a defect that was actually shipped and had to be found by
review: a confirmation gate that could not fail, orphan detection that only
counted, an error flag cleared by any unrelated call.
"""
import copy
import pytest

from vektori_trace.tau2.eligibility import (
    CONFIRM_FAIL, CONFIRM_NA, CONFIRM_PASS, CONFIRM_UNCERTAIN, audit,
)
from vektori_trace.tau2.normalize import (
    GreetingProvenanceError, SCRIPTED_GREETING, normalize_trace, select_trace,
)

MUT = frozenset({"cancel_pending_order", "modify_pending_order_items"})


def msg(role, content="", tool_calls=None, raw=True, mid=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    if role == "assistant" and raw:
        m["raw_data"] = {"message": {"reasoning_content": "private thoughts"}}
    if role == "tool":
        m["id"] = mid
    return m


def call(name, args, cid):
    return {"id": cid, "name": name, "arguments": args, "requestor": "assistant"}


def sim(messages, reward=1.0, term="user_stop"):
    return {"task_id": "T", "messages": messages, "trial": 0, "seed": 1,
            "termination_reason": term,
            "reward_info": {"reward": reward, "db_check": {}}}


def base_messages():
    return [
        msg("assistant", SCRIPTED_GREETING, raw=False),
        msg("user", "cancel my order #W123456"),
        msg("assistant", "", [call("find_user_id_by_email", {"email": "a@b.c"}, "c1")]),
        msg("tool", "user_42", mid="c1"),
        msg("assistant", "I can cancel order #W123456 for you. Confirm?"),
        msg("user", "yes please go ahead"),
        msg("assistant", "", [call("cancel_pending_order", {"order_id": "#W123456"}, "c2")]),
        msg("tool", '{"order_id": "#W123456", "status": "cancelled"}', mid="c2"),
        msg("assistant", "Done, your order is cancelled."),
    ]


# ---------------- greeting ----------------

def test_greeting_is_not_a_target_but_stays_in_prompts():
    tr = normalize_trace(sim(base_messages()), "f.json")
    assert all(d.target["content"] != SCRIPTED_GREETING for d in tr.decisions)
    later = [d for d in tr.decisions if d.position > 0]
    assert all(any(m["content"] == SCRIPTED_GREETING for m in d.prompt) for m in [1] for d in later)


def test_second_assistant_without_raw_data_fails_closed():
    m = base_messages()
    m[4].pop("raw_data")
    with pytest.raises(GreetingProvenanceError, match="exactly one"):
        normalize_trace(sim(m), "f.json")


def test_greeting_not_at_index_zero_fails():
    m = [msg("user", "hi")] + base_messages()
    with pytest.raises(GreetingProvenanceError):
        normalize_trace(sim(m), "f.json")


def test_genuine_action_colliding_with_greeting_text_fails():
    m = base_messages()
    m[4]["content"] = SCRIPTED_GREETING
    with pytest.raises(GreetingProvenanceError, match="collides"):
        normalize_trace(sim(m), "f.json")


# ---------------- private reasoning ----------------

def test_reasoning_never_reaches_prompt_or_target():
    tr = normalize_trace(sim(base_messages()), "f.json")
    for d in tr.decisions:
        assert "private thoughts" not in str(d.prompt)
        assert "private thoughts" not in str(d.target)
        assert "raw_data" not in d.target
    assert any(d.reasoning_content == "private thoughts" for d in tr.decisions)


# ---------------- confirmation ----------------

def test_confirmation_passes_on_proposal_assent_mutation():
    s = sim(base_messages())
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation == CONFIRM_PASS, v.diagnostic_confirmation_notes


def test_greeting_alone_cannot_confirm_a_mutation():
    """The defect that made the first gate unfailable."""
    m = [
        msg("assistant", SCRIPTED_GREETING, raw=False),
        msg("user", "cancel order #W123456"),
        msg("assistant", "", [call("find_user_id_by_email", {"email": "a@b.c"}, "c1")]),
        msg("tool", "user_42", mid="c1"),
        msg("assistant", "", [call("cancel_pending_order", {"order_id": "#W123456"}, "c2")]),
        msg("tool", "ok", mid="c2"),
        msg("assistant", "Your order is cancelled."),
    ]
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    # The greeting is the only assistant text before the mutation, and it must
    # not be readable as confirmation.
    assert v.diagnostic_confirmation == CONFIRM_FAIL
    assert v.needs_manual_review() == "heuristic_fail"
    # But confirmation contributes no gate, so the trace stays eligible and
    # goes to the manual queue instead of being silently dropped.
    assert all(not k.startswith("confirm") for k in v.gates)
    assert v.eligible()


def test_assent_to_an_unrelated_question_is_not_confirmation():
    """"yes, that's my email" must not authorize a cancellation."""
    m = base_messages()
    m[4] = msg("assistant", "Is a@b.c your email?")
    m[5] = msg("user", "yes, that's my email")
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation in (CONFIRM_FAIL, CONFIRM_UNCERTAIN)
    assert v.eligible()          # diagnostic does not gate
    assert v.needs_manual_review() is not None


def test_ambiguous_assent_with_two_proposals_is_uncertain_not_pass():
    m = base_messages()
    m[4] = msg("assistant", "I can cancel #W123456 or modify the items. Proceed?")
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation == CONFIRM_UNCERTAIN
    assert v.eligible()          # diagnostic does not gate
    assert v.needs_manual_review() == "heuristic_uncertain"


def test_user_declining_then_mutation_fails():
    m = base_messages()
    m[5] = msg("user", "no, don't cancel it")
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation == CONFIRM_FAIL
    # Highest manual-review priority: a mutation after an apparent decline.
    assert v.needs_manual_review() == "executed_after_decline"


def test_no_mutation_is_not_applicable():
    m = [
        msg("assistant", SCRIPTED_GREETING, raw=False),
        msg("user", "where is my order?"),
        msg("assistant", "Let me look that up."),
        msg("assistant", "", [call("get_order_details", {"order_id": "#W1"}, "c1")]),
        msg("tool", "{}", mid="c1"),
        msg("assistant", "It ships tomorrow."),
    ]
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation == CONFIRM_NA
    assert v.eligible()


# ---------------- auth, orphans, preconditions ----------------

def test_auth_tool_that_errored_does_not_authenticate():
    m = base_messages()
    m[3] = msg("tool", "Error: user not found", mid="c1")
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["policy_auth_before_mutation"] is False


def test_orphan_result_detected_by_id_not_count():
    m = base_messages()
    m[7]["id"] = "c99"
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["no_orphan_tool_results"] is False


def test_duplicate_results_detected():
    m = base_messages()
    m.insert(8, msg("tool", "again", mid="c2"))
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["no_duplicate_results"] is False


def test_error_is_not_cleared_by_an_unrelated_tool():
    m = base_messages()
    m.insert(6, msg("assistant", "", [call("cancel_pending_order", {"order_id": "#W9"}, "e1")]))
    m.insert(7, msg("tool", "Error: order not pending", mid="e1"))
    m.insert(8, msg("assistant", "", [call("get_order_details", {"order_id": "#W1"}, "e2")]))
    m.insert(9, msg("tool", "{}", mid="e2"))
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["policy_no_mutation_after_error"] is False


def test_missing_termination_reason_is_not_clean():
    s = sim(base_messages(), term="")
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["terminated_clean"] is False


def test_trace_selection_is_deterministic():
    a = normalize_trace({**sim(base_messages()), "trial": 1, "seed": 5}, "f")
    b = normalize_trace({**sim(base_messages()), "trial": 0, "seed": 9}, "f")
    assert select_trace([a, b]).trial == 0
    assert select_trace([b, a]).trial == 0


def test_confirmation_never_gates_eligibility():
    """The regression that mattered: a classifier must not decide the corpus."""
    m = base_messages()
    m[4] = msg("assistant", "Is a@b.c your email?")
    m[5] = msg("user", "yes, that's my email")
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.diagnostic_confirmation != CONFIRM_PASS
    assert v.eligible(), "diagnostic_confirmation must not affect eligibility"
    assert all(not k.startswith("confirm") for k in v.gates)


def test_precondition_exclusion_preserves_auditable_ids():
    m = base_messages()
    m.insert(6, msg("assistant", "", [call("cancel_pending_order", {"order_id": "#W9"}, "e1")]))
    m.insert(7, msg("tool", "Error: order not pending", mid="e1"))
    s = sim(m)
    v = audit(normalize_trace(s, "f"), s, MUT)
    assert v.gates["policy_no_mutation_after_error"] is False
    ev = [e for e in v.evidence if e["gate"] == "policy_no_mutation_after_error"]
    assert ev and ev[0]["mutation_call_id"]
    assert ev[0]["unresolved"][0]["call_id"] == "e1"
