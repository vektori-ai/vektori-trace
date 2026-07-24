"""Generate a Harbor-spec task (task.toml/instruction.md/environment/tests/solution)
for a diagnosed capability deficit, isolating it as a concrete, verifiable task."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .diagnose import DeficitScore
from .llm import call_json

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


def scaffold_task(
    deficit: DeficitScore, tasks_dir: Path, org: str = "vektori", model: str | None = None
) -> Path:
    """Scaffold a Harbor task dir via `harbor task init`, then fill it in with
    LLM-generated content targeting the given deficit. Returns the task dir path."""
    name = _slugify(deficit.capability.name)
    generated = generate_task_files(deficit, model=model)

    subprocess.run(
        [
            "harbor",
            "task",
            "init",
            f"{org}/{name}",
            "--tasks-dir",
            str(tasks_dir),
            "--description",
            generated["task_description"],
            "--author",
            "vektori-trace",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    task_dir = tasks_dir / name
    (task_dir / "instruction.md").write_text(generated["instruction_md"])
    (task_dir / "environment" / "Dockerfile").write_text(generated["dockerfile"])
    (task_dir / "tests" / "test_outputs.py").write_text(generated["test_outputs_py"])
    solve_path = task_dir / "solution" / "solve.sh"
    solve_path.write_text(generated["solve_sh"])
    solve_path.chmod(0o755)

    return task_dir
