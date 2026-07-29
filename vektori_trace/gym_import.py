"""Step B — gym import adapter (PLAN.md).

Import R2E-Gym / SWE-smith style instances behind the existing harbor task
interface, reusing the *same* verifier the mined tasks use — `build_eval_script`
plus the graded `tests/verifier.py` + `f2p.json` / `p2p.json` artifacts.

A gym task without a runnable F2P/P2P verifier is **not importable**. Emitting a
placeholder that always exits non-zero would make every rollout on it fail, which
at the pass@k gate is indistinguishable from "outside student support" — it would
manufacture exactly the regime separation the gate exists to test for. Such
records are skipped with a reason instead.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .mining.emitter import HarborTask, write_harbor_task
from .mining.pipeline import (
    MODEL_PATCH_PATH,
    _runtime_aux_files,
    build_environment_dockerfile,
    build_eval_script,
)

# Per-language defaults. The verifier auto-detects its log parser from the test
# command (`mining/verifier.py:_detect_runner`), so handing a Go or Rust task a
# pytest command produces a parser that matches nothing and grades every rollout
# 0.0 — indistinguishable, at the pass@k gate, from a task outside the student's
# support. A language with no default is not importable rather than guessed at.
DEFAULT_TEST_CMDS = {
    "python": "python -m pytest -rA --tb=short",
    "go": "go test ./... -v",
    "rust": "cargo test",
    "javascript": "npx jest --verbose",
    "typescript": "npx jest --verbose",
}
DEFAULT_TEST_CMD = DEFAULT_TEST_CMDS["python"]


@dataclass
class GymInstance:
    """Minimal fields shared by R2E-Gym / SWE-smith style records."""

    instance_id: str
    problem_statement: str
    image: str | None = None
    base_commit: str | None = None
    test_patch: str | None = None
    gold_patch: str | None = None
    fail_to_pass: list[str] | None = None
    pass_to_pass: list[str] | None = None
    repo: str | None = None
    language: str | None = None
    test_cmds: list[str] = field(default_factory=list)
    extra: dict[str, Any] | None = None

    def unverifiable_reason(self) -> str | None:
        """Why this record cannot be graded, or None when it can be."""
        if not self.image:
            return "no container image: nothing to run the suite in"
        if not self.fail_to_pass:
            return "no FAIL_TO_PASS tests: the task has no pass/fail oracle"
        if not self.base_commit:
            return "no base_commit: test files cannot be reset for anti-tamper"
        if not self.test_patch:
            return "no test_patch: F2P tests are not present in the base tree"
        if not self.test_cmds and self.default_test_cmd() is None:
            lang = self.language or "unset"
            return (
                f"no test_cmds and no default for language {lang!r}: the verifier "
                "would auto-detect the wrong log parser and grade every rollout 0.0"
            )
        return None

    def default_test_cmd(self) -> str | None:
        """The language default, or None when there is no safe guess.

        An unset language defaults to Python only when the F2P names look like
        pytest node ids — guessing a runner is how a task silently grades 0.0.
        """
        lang = (self.language or "").strip().lower()
        if lang in DEFAULT_TEST_CMDS:
            return DEFAULT_TEST_CMDS[lang]
        if lang:
            return None
        f2p = self.fail_to_pass or []
        if f2p and all(t.endswith(".py") or "::" in t or t.startswith("test") for t in f2p):
            return DEFAULT_TEST_CMDS["python"]
        return None


def load_jsonl_instances(path: Path) -> list[GymInstance]:
    """Load a JSONL dump of gym instances (R2E-Gym / SWE-smith shaped)."""
    rows: list[GymInstance] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(parse_gym_record(json.loads(line)))
    return rows


def parse_gym_record(d: dict[str, Any]) -> GymInstance:
    """Normalize common key aliases from R2E-Gym / SWE-smith / SWE-bench."""
    instance_id = (
        d.get("instance_id")
        or d.get("id")
        or d.get("task_id")
        or d.get("instanceId")
    )
    if not instance_id:
        raise ValueError(f"gym record missing instance_id: keys={sorted(d)}")
    problem = (
        d.get("problem_statement")
        or d.get("problem")
        or d.get("instruction")
        or d.get("issue")
        or ""
    )
    f2p = d.get("FAIL_TO_PASS") or d.get("fail_to_pass") or d.get("f2p") or []
    p2p = d.get("PASS_TO_PASS") or d.get("pass_to_pass") or d.get("p2p") or []
    if isinstance(f2p, str):
        f2p = json.loads(f2p) if f2p.startswith("[") else [f2p]
    if isinstance(p2p, str):
        p2p = json.loads(p2p) if p2p.startswith("[") else [p2p]
    cmds = d.get("test_cmds") or d.get("test_cmd") or d.get("test_command") or []
    if isinstance(cmds, str):
        cmds = [cmds]
    return GymInstance(
        instance_id=str(instance_id),
        problem_statement=str(problem),
        image=d.get("image") or d.get("docker_image") or d.get("instance_image"),
        base_commit=d.get("base_commit") or d.get("baseCommit"),
        test_patch=d.get("test_patch") or d.get("testPatch"),
        gold_patch=d.get("patch") or d.get("gold_patch") or d.get("fix_patch"),
        fail_to_pass=list(f2p) if f2p else [],
        pass_to_pass=list(p2p) if p2p else [],
        repo=d.get("repo") or d.get("repo_name"),
        language=d.get("language") or d.get("lang"),
        test_cmds=[str(c) for c in cmds],
        extra={k: v for k, v in d.items() if k not in {
            "test_cmds", "test_cmd", "test_command",
            "instance_id", "id", "task_id", "instanceId",
            "problem_statement", "problem", "instruction", "issue",
            "image", "docker_image", "instance_image",
            "base_commit", "baseCommit",
            "test_patch", "testPatch",
            "patch", "gold_patch", "fix_patch",
            "FAIL_TO_PASS", "fail_to_pass", "f2p",
            "PASS_TO_PASS", "pass_to_pass", "p2p",
            "repo", "repo_name", "language", "lang",
        }},
    )


class UnverifiableGymTask(ValueError):
    """The record carries no runnable pass/fail oracle — it cannot be imported."""


def build_harbor_task(instance: GymInstance, *, org: str = "gym") -> HarborTask:
    """Build a `HarborTask` with the same graded verifier the miner emits."""
    reason = instance.unverifiable_reason()
    if reason is not None:
        raise UnverifiableGymTask(f"{instance.instance_id}: {reason}")

    assert instance.image and instance.base_commit and instance.test_patch
    test_cmds = list(instance.test_cmds)
    if not test_cmds:
        # `unverifiable_reason` already refused records with neither an explicit
        # command nor a language default.
        default_cmd = instance.default_test_cmd()
        assert default_cmd is not None
        test_cmds = [f"{default_cmd} {' '.join(instance.fail_to_pass or [])}".strip()]
    eval_script = build_eval_script(
        base_commit=instance.base_commit,
        test_patch=instance.test_patch,
        test_cmds=test_cmds,
        language=instance.language,
        fail_to_pass=instance.fail_to_pass,
        pass_to_pass=instance.pass_to_pass,
        model_patch_path=MODEL_PATCH_PATH,
    )
    return HarborTask(
        name=_safe_name(instance.instance_id),
        org=org,
        description=f"{instance.repo or 'gym'}: {instance.instance_id}",
        instruction=instance.problem_statement.strip() + "\n",
        oracle_diff=instance.gold_patch or "",
        repo2env={
            "pipeline": "gym_import",
            "pipeline_version": "0.1.0",
            "repo": instance.repo or "",
            "ref": instance.base_commit,
            "source_access": "gym",
            "reward_kinds": ["test_execution"],
            "gym_import": {
                "instance_id": instance.instance_id,
                "image": instance.image,
                "base_commit": instance.base_commit,
                "fail_to_pass": instance.fail_to_pass or [],
                "pass_to_pass": instance.pass_to_pass or [],
                "test_cmds": test_cmds,
                "reward_mode": "graded",
            },
            "reward_calibration": {
                "f2p_count": len(instance.fail_to_pass or []),
                "p2p_count": len(instance.pass_to_pass or []),
            },
        },
        difficulty="medium",
        category="bugfix",
        keywords=[instance.repo or "gym", "gym_import"],
        environment_dockerfile=build_environment_dockerfile(
            bootstrap_image=instance.image,
            base_commit=instance.base_commit,
        ),
        test_script=eval_script,
        aux_files=_runtime_aux_files(
            instance.fail_to_pass or [], instance.pass_to_pass or []
        ),
        verifier_network_mode="no-network",
    )


def emit_harbor_task(instance: GymInstance, out_dir: Path) -> Path:
    """Write one harbor task directory for a gym instance."""
    task = build_harbor_task(instance)
    if (out_dir / task.name).exists():
        shutil.rmtree(out_dir / task.name)
    # `write_harbor_task` appends `task.name` itself — hand it the parent.
    task_dir = write_harbor_task(task, out_dir)
    (task_dir / "gym_source.json").write_text(
        json.dumps(
            {
                "instance_id": instance.instance_id,
                "image": instance.image,
                "base_commit": instance.base_commit,
                "extra": instance.extra or {},
            },
            indent=2,
        )
        + "\n"
    )
    return task_dir


@dataclass
class GymImportResult:
    tasks: list[Path]
    skipped: dict[str, str]  # instance_id -> reason

    def __len__(self) -> int:  # keeps `len(import_gym(...))` meaning "imported"
        return len(self.tasks)


def import_gym(
    source: Path,
    out_dir: Path,
    *,
    limit: int | None = None,
) -> GymImportResult:
    """Import a JSONL gym dump into `out_dir` as harbor tasks.

    Records with no runnable oracle are skipped with a reason, never emitted as
    always-failing placeholders.
    """
    instances = load_jsonl_instances(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[Path] = []
    skipped: dict[str, str] = {}
    for inst in instances:
        # `limit` counts *imported* tasks, not records read. Slicing the input
        # first makes a short result ambiguous: corpus property or limit artefact?
        if limit is not None and len(tasks) >= limit:
            break
        try:
            tasks.append(emit_harbor_task(inst, out_dir))
        except UnverifiableGymTask as e:
            skipped[inst.instance_id] = str(e).split(": ", 1)[-1]
    return GymImportResult(tasks=tasks, skipped=skipped)


def _safe_name(instance_id: str) -> str:
    return instance_id.replace("/", "__").replace(" ", "_")


__all__ = [
    "DEFAULT_TEST_CMD",
    "DEFAULT_TEST_CMDS",
    "GymImportResult",
    "GymInstance",
    "UnverifiableGymTask",
    "build_harbor_task",
    "emit_harbor_task",
    "import_gym",
    "load_jsonl_instances",
    "parse_gym_record",
]
