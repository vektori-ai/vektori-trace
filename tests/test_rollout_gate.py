"""What `scripts/rollout_gate.py` may and may not conclude from a trajectory.

The first version of this grader read harbor's `message` as the model's raw
completion and scored six correct turns as total format failures. These tests
pin the corrected reading: a trajectory is harbor's *parse*, so acceptance and
parser loops are measurable and `native_json` is not.
"""

from __future__ import annotations

import json

from scripts import rollout_gate as rg


def _traj(*turns) -> dict:
    """A trajectory in harbor's ATIF shape. Each turn is (message, keystrokes)."""
    steps = [{"step_id": 1, "source": "user", "message": "system prompt"}]
    for i, (msg, keystrokes) in enumerate(turns):
        steps.append({
            "step_id": i + 2,
            "source": "agent",
            "message": msg,
            "reasoning_content": "",
            "tool_calls": [
                {"tool_call_id": f"call_{i}_{j}", "function_name": "bash_command",
                 "arguments": {"keystrokes": k, "duration": 0.1}}
                for j, k in enumerate(keystrokes)
            ],
            "metrics": {"completion_tokens": 100},
        })
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def _write(tmp_path, traj) -> "object":
    d = tmp_path / "job" / "trial" / "agent"
    d.mkdir(parents=True)
    (d / "trajectory.json").write_text(json.dumps(traj))
    return d / "trajectory.json"


def test_commands_mean_the_parser_accepted_the_turn(tmp_path):
    """This is the whole inversion: harbor only produces ParsedCommands from a
    response its parser could use, so their presence is the evidence."""
    p = _write(tmp_path, _traj(("Analysis: x\nPlan: y", ["ls -la\n"])))
    r = rg.judge(p)
    assert r["summary"]["n_accepted"] == 1
    assert r["summary"]["turn_1_accepted"] is True
    assert r["turns"][0]["n_commands"] == 1


def test_prose_with_no_commands_is_unparsed(tmp_path):
    """A turn harbor could not turn into an action. `message` alone never
    decides this — the absence of commands does."""
    p = _write(tmp_path, _traj(("I think the file is in src/", [])))
    r = rg.judge(p)
    assert r["summary"]["n_accepted"] == 0
    assert r["turns"][0]["accepted"] is False


def test_two_unparsed_turns_in_a_row_are_a_parser_loop(tmp_path):
    p = _write(tmp_path, _traj(("a", []), ("b", []), ("c", ["ls\n"])))
    r = rg.judge(p)
    assert r["summary"]["parser_loop"] is True
    assert r["summary"]["longest_unparsed_run"] == 2


def test_one_stumble_between_good_turns_is_not_a_loop(tmp_path):
    """The corpus contains recoveries; a single unparsed turn is one of them."""
    p = _write(tmp_path, _traj(("a", ["ls\n"]), ("b", []), ("c", ["pwd\n"])))
    r = rg.judge(p)
    assert r["summary"]["parser_loop"] is False
    assert r["summary"]["longest_unparsed_run"] == 1


def test_a_trailing_unparsed_run_still_counts(tmp_path):
    """The run does not have to be followed by a good turn to be a loop — a
    rollout that ends by looping ends by looping."""
    p = _write(tmp_path, _traj(("a", ["ls\n"]), ("b", []), ("c", [])))
    r = rg.judge(p)
    assert r["summary"]["longest_unparsed_run"] == 2
    assert r["summary"]["parser_loop"] is True


def test_the_six_turn_ck84_shape_is_all_accepted(tmp_path):
    """The step 7 rollout: six turns, 16 commands, no loop."""
    counts = [2, 3, 3, 3, 4, 1]
    p = _write(tmp_path, _traj(*[("Analysis: a\nPlan: b", ["ls\n"] * c) for c in counts]))
    r = rg.judge(p)
    assert r["summary"]["n_turns"] == 6
    assert r["summary"]["n_accepted"] == 6
    assert r["summary"]["n_commands"] == 16
    assert r["summary"]["parser_loop"] is False


def test_user_steps_are_not_graded_as_model_turns(tmp_path):
    p = _write(tmp_path, _traj(("a", ["ls\n"])))
    r = rg.judge(p)
    assert r["summary"]["n_turns"] == 1


def test_it_reports_tokens_so_discarded_text_is_visible(tmp_path):
    """58-94% of the real rollout's tokens were outside the JSON. The count has
    to survive into the report or that is invisible."""
    p = _write(tmp_path, _traj(("a", ["ls\n"])))
    r = rg.judge(p)
    assert r["summary"]["total_completion_tokens"] == 100
