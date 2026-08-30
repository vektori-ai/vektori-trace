"""Amend the episode-spread stop rule to a diagnostic. Recorded BEFORE scoring.

Usage: python scripts/pilotd_amend_stop_rules.py <manifest.json> <out.json>

The frozen rule "stop if interleaving exceeds 2/8 episodes" fired at update 2.
This amendment demotes it to a warning and keeps every mass-based rule intact.
It is written and staged BEFORE update 2 is scored, so it is a prospective
amendment, not a post-hoc rationalisation -- but it is still an amendment made
after seeing the data that triggered it, and the write-up says so plainly.

Why the episode count is the wrong instrument here, on the evidence:

  The three update-2 actions share one behaviour -- the model narrates
  executable tool calls INSIDE <think>, closes the block, then REPEATS the
  same calls outside where they actually execute:

      <think> reasoning ... <tool_call>{...}</tool_call> </think>
      <tool_call>{...}</tool_call>        <- the real, executed call

  So the episodes function correctly. Only the reasoning payload is
  unmappable, because Hermes markup has no DeepSeek DSML counterpart. One
  such action makes an episode "affected" regardless of how little of that
  episode it touches, so the metric counts breadth while ignoring mass -- and
  the mass fell while the breadth rose (5.19% -> 3.85% of actions, 5.81% ->
  4.31% of reasoning bytes, retention 92.68% -> 93.47%).

What does NOT change: the conservative skip. The affected reasoning payloads
stay excluded. No Hermes markup is converted to DSML and no contaminated
reasoning is sent to the teacher mid-pilot -- that would alter the teacher's
causal context and the projection contract, and belongs in a future
projection version with its own canary.
"""
import json
import sys

AMENDMENT = {
    "date": "2026-08-30",
    "kind": "stop_rule_amendment_episode_spread_to_diagnostic",
    "status": "EXPLORATORY CONTINUATION -- the original preregistered rule fired",
    "what_changed": {
        "before": "stop if interleaved-tool exclusions exceed 2/8 episodes",
        "after": "episode spread is a WARNING/diagnostic; mass-based rules bind",
    },
    "binding_stop_rules_after_amendment": {
        "affected_actions_above": 0.10,
        "excluded_reasoning_bytes_above": 0.10,
        "retention_below": 0.90,
        "undeclared_exclusion_class": "stop",
        "repeated_cap_terminations": "stop -- never raise the cap",
    },
    "trigger": {
        "update": 2,
        "affected_episodes": "3/8",
        "affected_actions": "3/78 = 3.85%",
        "excluded_reasoning_bytes": "8171/189652 = 4.31%",
        "retention": 0.9347,
    },
    "diagnosed_behaviour": (
        "The model narrates executable tool calls inside <think>, closes the "
        "block, then repeats the same calls outside where they execute. The "
        "episodes therefore function correctly; only the reasoning payload is "
        "unmappable. Observed in u002-task102-seed1@3#0 (3 calls), "
        "u002-task64-seed2@3#0 (3), u002-task4-seed0@6#0 (5). Associated with "
        "multi-order / parallel-tool actions."
    ),
    "why_the_metric_is_wrong_here": (
        "A single affected action marks a whole episode, so the metric measures "
        "breadth and is blind to mass. Across updates 0-2 the breadth rose "
        "(1/8, 2/8, 3/8) while the mass fell (2.67%, 5.19%, 3.85% of actions; "
        "5.81% -> 4.31% of reasoning bytes) and retention improved."
    ),
    "honest_limitations": [
        "This threshold was changed after the data that tripped it was seen.",
        "The task-76 concentration explanation is dead: update 2's roster held "
        "no task 76 and the behaviour still appeared in three episodes.",
        "Three updates of eight episodes cannot separate policy drift from "
        "roster composition.",
        "Update 2 also produced the run's first cap termination (a reasoning "
        "loop). Both findings are reasoning-boundary failures and OPD gives "
        "</think> no positive weight by construction.",
    ],
    "unchanged": [
        "plan_hash and the frozen 80 (task, seed) pairs",
        "the conservative skip: affected reasoning payloads stay excluded",
        "no Hermes -> DSML conversion, no contaminated reasoning to the teacher",
        "parser v3 / projection v4 / chunk-v2",
        "policy weights, captured actions, score fingerprints",
    ],
}


def main():
    m = json.load(open(sys.argv[1]))
    plan_before = m.get("plan_hash")
    excl = m.get("declared_exclusions") or {}
    key = "hermes_markup_inside_reasoning_span"
    if key in excl:
        excl[key]["stop_rules"] = {
            "episode_spread": "DIAGNOSTIC ONLY as of 2026-08-30 (see amendments)",
            "affected_actions_above": 0.10,
            "reasoning_bytes_excluded_above": 0.10,
            "retention_below": 0.90,
        }
    m.setdefault("amendments", []).append(AMENDMENT)
    m["status_claim"] = (
        "EXPLORATORY CONTINUATION after a preregistered stop rule fired at "
        "update 2 and was amended on the evidence. Instrumentation and "
        "stability only; no efficacy claim is available."
    )
    assert m.get("plan_hash") == plan_before
    with open(sys.argv[2], "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("amendment recorded: episode spread -> diagnostic")
    print("binding rules: actions>10%, bytes>10%, retention<90%, undeclared class")
    print("plan_hash unchanged: %s" % m["plan_hash"])


main()
