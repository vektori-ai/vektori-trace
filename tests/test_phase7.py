"""The Phase 7 grader, validated before it is pointed at a GPU.

Positives are the shape the teacher actually emitted; negatives are the shape
v1 actually emitted. A grader that has not been shown to separate those two is
not evidence about a checkpoint, it is an untested predicate — and the whole
value of Phase 7 is that it is the first honest measurement in the sequence.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.evaluate.phase7 import (
    GateResult,
    grade,
    select_checkpoint,
    summarize,
)

FORMAT_GATES = ("native_json", "required_fields", "no_legacy_envelope")


def action(commands, *, analysis="a", plan="p", complete=False):
    return json.dumps(
        {
            "analysis": analysis,
            "plan": plan,
            "commands": [{"keystrokes": k, "duration": 1.0} for k in commands],
            "task_complete": complete,
        }
    )


def g(text, **kw):
    kw.setdefault("prefix_id", "x")
    kw.setdefault("checkpoint", "ck10")
    kw.setdefault("category", "orientation")
    kw.setdefault("suite", "acquisition")
    return grade(text, **kw)


# --------------------------------------------------------------------------
# Positives: what DeepSeek emitted
# --------------------------------------------------------------------------


def test_teacher_shaped_action_passes_every_format_gate():
    res = g(action(["ls -la\n"]), finish_reason="stop")
    assert res.parser_error is None
    for gate in FORMAT_GATES:
        assert res.gates[gate] is True, gate
    assert res.gates["command_structure"] is True
    assert res.gates["eos_before_limit"] is True
    assert res.n_commands == 1
    assert res.first_command == "ls -la"


def test_task_complete_is_not_a_command():
    res = g(action([], complete=True), finish_reason="stop")
    assert res.gates["required_fields"] is True
    assert res.n_commands == 0


# --------------------------------------------------------------------------
# Negatives: what v1 emitted — the exact failure this run repairs
# --------------------------------------------------------------------------


def test_v1_tool_call_envelope_fails():
    """305 consecutive `Missing required fields` came from precisely this."""
    v1 = (
        '<tool_call>\n{"name": "bash_command", "arguments": '
        '{"keystrokes": "ls -la\\n", "duration": 1.0}}\n</tool_call>'
    )
    res = g(v1, finish_reason="stop")
    assert res.parser_error is not None
    assert res.gates["harbor_accepts"] is False
    assert res.gates["native_json"] is False
    assert res.gates["no_legacy_envelope"] is False


def test_prose_with_bash_fence_fails():
    """The 18 teacher turns terminus's parser rejected and v1 trained on."""
    res = g("I'll start by listing files.\n```bash\nls -la\n```", finish_reason="stop")
    assert res.gates["harbor_accepts"] is False
    assert res.gates["native_json"] is False


def test_missing_plan_field_fails_required_fields():
    res = g(json.dumps({"analysis": "a", "commands": []}), finish_reason="stop")
    assert res.gates["required_fields"] is False


# --------------------------------------------------------------------------
# The two-tier distinction: harbor is lenient, the strict gate is not
# --------------------------------------------------------------------------


def test_fenced_json_runs_but_is_not_native():
    """harbor salvages a fenced object with only a warning.

    Scoring this as a plain pass would hide a checkpoint that is nearly
    repaired but still wrapping its output; scoring it as a plain failure would
    claim a rollout breaks when it would in fact proceed.
    """
    res = g("```json\n" + action(["ls\n"]) + "\n```", finish_reason="stop")
    assert res.gates["harbor_accepts"] is True
    assert res.gates["native_json"] is False
    assert res.parser_warning


def test_thinking_block_is_not_native_json():
    """`enable_thinking=False` is pinned in the dataset; a <think> block is
    off-protocol even though harbor would dig the object out."""
    res = g("<think>hmm</think>" + action(["ls\n"]), finish_reason="stop")
    assert res.gates["harbor_accepts"] is True
    assert res.gates["native_json"] is False
    assert res.gates["no_legacy_envelope"] is False


# --------------------------------------------------------------------------
# Structural gates
# --------------------------------------------------------------------------


def test_invented_top_level_field_fails():
    obj = json.loads(action(["ls\n"]))
    obj["tool"] = "bash"
    res = g(json.dumps(obj), finish_reason="stop")
    assert res.gates["no_invented_fields"] is False


def test_invented_command_field_fails_structure():
    """`is_blocking`/`timeout_sec` are dropped by harbor with a warning, so a
    model relying on them issues commands whose semantics it does not get."""
    text = json.dumps(
        {
            "analysis": "a",
            "plan": "p",
            "commands": [{"keystrokes": "ls\n", "is_blocking": True}],
            "task_complete": False,
        }
    )
    res = g(text, finish_reason="stop")
    assert res.gates["command_structure"] is False


def test_truncation_fails_eos_gate():
    res = g(action(["ls\n"]), finish_reason="length")
    assert res.gates["eos_before_limit"] is False


def test_repeated_command_fails_non_repetition():
    res = g(action(["ls\n"] * 3), finish_reason="stop")
    assert res.gates["non_repetition"] is False
    assert g(action(["ls\n"] * 2), finish_reason="stop").gates["non_repetition"] is True


# --------------------------------------------------------------------------
# Behavior gates apply only where the prefix can test them
# --------------------------------------------------------------------------


def test_behavior_gates_are_category_conditional():
    res = g(action(["ls -la\n"]), category="orientation", finish_reason="stop")
    assert "orientation" in res.gates
    assert "edit_emission" not in res.gates
    assert "test_emission" not in res.gates


def test_edit_category_requires_an_edit():
    assert g(action(["ls\n"]), category="first_edit").gates["edit_emission"] is False
    ok = g(action(["sed -i 's/a/b/' x.py\n"]), category="first_edit")
    assert ok.gates["edit_emission"] is True


def test_test_category_requires_a_test():
    assert g(action(["ls\n"]), category="test_exec").gates["test_emission"] is False
    assert g(action(["pytest -q\n"]), category="test_exec").gates["test_emission"] is True


def test_clone_gate_only_binds_when_the_repo_is_present():
    text = action(["git clone https://github.com/pallets/click\n"])
    assert g(text, git_present=True).gates["no_clone_when_git_exists"] is False
    assert "no_clone_when_git_exists" not in g(text, git_present=False).gates


def test_orientation_gate_wants_an_inspection_not_a_clone():
    assert g(action(["git clone x\n"]), category="orientation").gates["orientation"] is False
    assert g(action(["ls -la\n"]), category="orientation").gates["orientation"] is True


def test_recovery_is_producing_a_parseable_action():
    bad = g("sorry, here is the fix:\n```bash\nls\n```", category="parse_error_recovery")
    assert bad.gates["recovery"] is False
    good = g(action(["ls -la\n"]), category="parse_error_recovery")
    assert good.gates["recovery"] is True


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _res(ck, suite, ok):
    r = GateResult(prefix_id="p", checkpoint=ck, category="orientation",
                   suite=suite, completion="")
    r.gates = dict.fromkeys(FORMAT_GATES, ok)
    return r


def test_selection_takes_the_earliest_passing_not_the_last():
    order = ["ck10", "ck20", "ck30"]
    results = [
        _res("ck10", "generalization", False),
        _res("ck20", "generalization", True),
        _res("ck30", "generalization", True),
    ]
    chosen, trace = select_checkpoint(results, order=order)
    assert chosen == "ck20"
    assert trace["ck10"]["passed"] is False


def test_selection_returns_none_when_nothing_passes():
    order = ["ck10", "ck20"]
    results = [_res("ck10", "generalization", False), _res("ck20", "generalization", False)]
    chosen, _ = select_checkpoint(results, order=order)
    assert chosen is None


def test_selection_ignores_the_acquisition_suite():
    """Acquisition prefixes are training inputs. A checkpoint can reproduce a
    memorised continuation there, so selection reads the held-out suite."""
    order = ["ck10"]
    results = [
        _res("ck10", "acquisition", True),
        _res("ck10", "generalization", False),
    ]
    chosen, _ = select_checkpoint(results, order=order)
    assert chosen is None


def test_untested_checkpoint_is_not_recorded_as_failing():
    order = ["ck10", "ck20"]
    results = [_res("ck10", "generalization", True)]
    chosen, trace = select_checkpoint(results, order=order)
    assert chosen == "ck10"
    assert trace["ck20"] == {"status": "no results"}


def test_summary_splits_by_checkpoint_and_suite():
    results = [
        _res("ck10", "acquisition", True),
        _res("ck10", "generalization", False),
        _res("ck20", "generalization", True),
    ]
    s = summarize(results)
    assert s["cells"]["ck10|acquisition"]["passed"] == 1
    assert s["cells"]["ck10|generalization"]["passed"] == 0
    assert s["cells"]["ck20|generalization"]["n"] == 1


@pytest.mark.parametrize("bad", ["", "   ", "null", "[]", "{}"])
def test_degenerate_completions_do_not_crash_the_grader(bad):
    res = g(bad, finish_reason="stop")
    assert res.gates["native_json"] is False
