"""Unit tests for mining/miner.py — no Docker/network/harbor involved.

`mine_tasks` itself (real PR mining) needs Docker + a real repo, so it's left
for manual/integration testing. This covers the part that's pure and fast:
turning agent-run results into Run/Turn traces + a manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from vektori_trace.mining.miner import MinedRun, collect_traces
from vektori_trace.schema import Trace, Turn, load_manifest


class FakeTraceRunner:
    def __init__(self, results: dict[str, MinedRun]):
        self.results = results

    def run(self, task_dir: Path) -> MinedRun:
        return self.results[task_dir.name]


def test_collect_traces_writes_win_and_loss(tmp_path: Path) -> None:
    task_a = tmp_path / "tasks" / "vektori__widget-1"
    task_b = tmp_path / "tasks" / "vektori__widget-2"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)

    runner = FakeTraceRunner(
        {
            "vektori__widget-1": MinedRun(
                turns=[Turn(index=0, role="assistant", content="fixed it")], passed=True
            ),
            "vektori__widget-2": MinedRun(
                turns=[Turn(index=0, role="assistant", content="gave up")], passed=False
            ),
        }
    )

    traces_dir = tmp_path / "traces"
    manifest = collect_traces([task_a, task_b], runner, traces_dir)

    assert {m["outcome"] for m in manifest} == {"win", "loss"}
    assert len(list(traces_dir.glob("*.json"))) == 2

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    entries = load_manifest(manifest_path)
    traces = [Trace.load(e.path, outcome=e.outcome) for e in entries]
    outcomes = {t.outcome for t in traces}
    assert outcomes == {"win", "loss"}
    win_trace = next(t for t in traces if t.outcome == "win")
    assert win_trace.turns[0].content == "fixed it"
