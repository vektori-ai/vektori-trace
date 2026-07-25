"""Generate a Harbor-spec task (task.toml/instruction.md/environment/tests/solution)
for a diagnosed capability deficit, isolating it as a concrete, verifiable task."""

from __future__ import annotations

import re
from pathlib import Path

from .diagnose import DeficitScore
from .llm import call_json
from .mining.emitter import HarborTask, write_harbor_task

_TASKGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "task_description": {
            "type": "string",
            "description": "One-line description for task.toml [task].description",
        },
        "instruction_md": {
            "type": "string",
            "description": "Full contents of instruction.md given to the agent",
        },
        "dockerfile": {
            "type": "string",
            "description": "Full contents of environment/Dockerfile setting up the initial state",
        },
        "test_outputs_py": {
            "type": "string",
            "description": (
                "Full contents of tests/test_outputs.py: pytest function(s) named "
                "test_* that check the filesystem/output state left by the agent, "
                "isolating exactly the diagnosed capability."
            ),
        },
        "solve_sh": {
            "type": "string",
            "description": (
                "Full contents of solution/solve.sh: an oracle bash script that "
                "correctly exercises the capability and makes the tests pass."
            ),
        },
    },
    "required": [
        "task_description",
        "instruction_md",
        "dockerfile",
        "test_outputs_py",
        "solve_sh",
    ],
    "additionalProperties": False,
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "deficit"


def generate_task_files(deficit: DeficitScore, model: str | None = None) -> dict:
    cap = deficit.capability
    examples = "\n\n".join(
        f"--- example failing trajectory ({t.run_id}) ---\n{t.condensed(max_turns=40)}"
        for t in deficit.lacking_loss_traces[:2]
    )
    user = (
        f"Capability deficit to isolate: {cap.name}\n"
        f"Description: {cap.description}\n\n"
        f"Evidence from real failing trajectories where this was lacking:\n{examples}\n\n"
        "Design a small, self-contained task (runnable in a single Ubuntu 24.04 Docker "
        "container, no network access, no real external services — simulate any needed "
        "state/files/APIs as local files or a tiny local script) that isolates exactly "
        "this capability. It must be solvable by a correct agent and the pytest checks "
        "must fail for an agent that has the same deficit as in the evidence above. "
        "Keep it minimal: one clear objective, one or two pytest checks."
    )
    result = call_json(
        system=(
            "You synthesize small, verifiable coding/agentic tasks (Harbor spec) that "
            "isolate a single capability deficit observed in real agent failures."
        ),
        user=user,
        schema_name="harbor_task",
        json_schema=_TASKGEN_SCHEMA,
        model=model,
    )
    return result


# Harbor's verifier contract: whatever tests/test.sh leaves in
# /logs/verifier/reward.txt is the reward. Mined tasks score F2P/P2P via the
# shipped verifier.py; a synthesized task has no F2P set, so its checks are
# pass/fail on pytest's exit code.
_SYNTHETIC_TEST_SH = """#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
( cd /workspace && python3 -m pytest -v /tests/test_outputs.py ) \\
  > /logs/verifier/test_output.log 2>&1
TEST_EXIT_CODE=$?
cat /logs/verifier/test_output.log
[ "$TEST_EXIT_CODE" -eq 0 ] \\
  && echo "1.0" > /logs/verifier/reward.txt \\
  || echo "0.0" > /logs/verifier/reward.txt
exit 0
"""


def scaffold_task(
    deficit: DeficitScore, tasks_dir: Path, org: str = "vektori", model: str | None = None
) -> Path:
    """Write a Harbor task dir filled with LLM-generated content targeting the
    given deficit. Returns the task dir path.

    Uses `emitter.write_harbor_task` rather than shelling out to
    `harbor task init`: that command's `--tasks-dir`/`--description`/`--author`
    flags don't exist, so the call could only ever fail, and it made the CLI
    depend on a harbor binary being installed to produce files we already know
    how to write. The emitter is the one task writer, and mined and synthesized
    tasks now come out of it identically.
    """
    name = _slugify(deficit.capability.name)
    generated = generate_task_files(deficit, model=model)

    task = HarborTask(
        name=name,
        org=org,
        description=generated["task_description"],
        instruction=generated["instruction_md"],
        # Synthesized tasks have no gold diff — the oracle is solve.sh, written
        # below over the emitter's git-apply shim.
        oracle_diff="",
        repo2env={
            "source": "taskgen",
            "capability_id": deficit.capability.id,
            "capability_name": deficit.capability.name,
            "gap": deficit.gap,
            "prevalence": deficit.prevalence,
            "n_relevant_wins": deficit.n_relevant_wins,
            "n_relevant_losses": deficit.n_relevant_losses,
            "evidence_run_ids": [t.run_id for t in deficit.lacking_loss_traces],
            "reward_kinds": ["test_execution"],
        },
        category="capability_probe",
        environment_dockerfile=generated["dockerfile"],
        test_script=_SYNTHETIC_TEST_SH,
        aux_files={"tests/test_outputs.py": generated["test_outputs_py"]},
    )
    task_dir = write_harbor_task(task, tasks_dir)

    solve_path = task_dir / "solution" / "solve.sh"
    solve_path.write_text(generated["solve_sh"], encoding="utf-8")
    solve_path.chmod(0o755)
    # The emitter writes an empty patch.diff for the no-gold-diff case; drop it
    # so nothing downstream mistakes it for a real oracle patch.
    (task_dir / "solution" / "patch.diff").unlink(missing_ok=True)

    return task_dir
