"""`retry_slot` exercised against a real directory tree.

The first version of these tests inspected source text and AST. They passed
while the implementation bulk-copied `events.jsonl` and the whole
`simulations/` directory, carrying the FAILED episode's material into a
destination whose episode row had been deliberately removed. Source
inspection cannot see that; running the copy can.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "scripts" / "tau2_live_opd_modal.py"

KEPT = ["u000-task44-seed1", "u000-task53-seed2", "u000-task68-seed0",
        "u000-task71-seed1", "u000-task76-seed0", "u000-task108-seed1",
        "u000-task109-seed0"]
BAD = "u000-task95-seed1"
ALL = KEPT + [BAD]


def _load_retry_slot():
    """Import the function body without Modal decorators running."""
    src = SRC.read_text()
    start = src.index("def retry_slot(")
    # Stop at the NEXT top-level decorator or def, whichever comes first --
    # slicing to "\ndef " alone drags in the following @app.function line.
    cand = [i for i in (src.find("\n@app.function(", start + 10),
                        src.find("\ndef ", start + 10)) if i != -1]
    end = min(cand)
    ns = {}
    exec(compile("import json, os, shutil\n" + src[start:end], "<r>", "exec"), ns)
    return ns["retry_slot"]


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A source run with 7 sampled + 1 failed, and a staged destination."""
    base = tmp_path / "runs"
    src_u = base / "src" / "update-000"
    arch = src_u / "live_archive"
    (arch / "turns").mkdir(parents=True)
    (arch / "simulations").mkdir()

    with (arch / "episodes.jsonl").open("w") as fh:
        for e in ALL:
            fh.write(json.dumps({
                "episode_id": e, "task_id": e.split("task")[1].split("-")[0],
                "status": "failed" if e == BAD else "sampled",
                "adapter_hash": "3869b147ab7ce5d2",
                "policy_version": "live-u000", "gen_config_hash": "",
                "require_reasoning": True, "num_turns": 8,
            }) + "\n")
    for e in ALL:
        (arch / "turns" / f"{e}.jsonl").write_text(
            json.dumps({"kind": "turn", "episode_id": e}) + "\n")
        (arch / "simulations" / f"{e}.json").write_text('{"sim": true}')
    with (arch / "events.jsonl").open("w") as fh:
        for e in ALL:
            # Top-level id (older shape)
            fh.write(json.dumps({"episode_id": e, "event": "turn"}) + "\n")
            # REAL shape: id nested in payload, nothing top-level.
            fh.write(json.dumps({
                "event": "turn_sampled", "event_id": f"{e}-1",
                "schema_version": 1, "recorded_at": 0,
                "payload": {"episode_id": e, "turn_index": 0},
            }) + "\n")
        fh.write(json.dumps({"event": "update_start"}) + "\n")  # no id at all
    (src_u / ".PLANNED").write_text("{}")

    manifest = {
        "plan_hash": "aa9251ccb6d566fa",
        "plans_by_update": [[{"episode_id": e, "task_id": "1", "seed": 0}
                             for e in ALL]],
    }
    (base / "src").joinpath("manifest.json").write_text(json.dumps(manifest))
    (base / "dst").mkdir(parents=True)
    (base / "dst" / "manifest.json").write_text(json.dumps(manifest))

    fn = _load_retry_slot()
    fn.__globals__["VOLUME_MOUNT"] = str(tmp_path)
    fn.__globals__["RUNS_IN_VOLUME"] = "runs"
    fn.__globals__["vol"] = type("V", (), {"commit": staticmethod(lambda: None)})()
    return fn, base


class TestNoContamination:
    def test_failed_episode_absent_from_every_file(self, tree):
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        dst = base / "dst" / "update-000"
        for path in dst.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "retry_provenance.json":
                # Provenance MUST name the retried slot -- that is its job.
                continue
            assert BAD not in path.name, f"{path} names the failed episode"
            if path.suffix in (".json", ".jsonl"):
                assert BAD not in path.read_text(), f"{path} mentions it"

    def test_events_are_filtered_not_bulk_copied(self, tree):
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        ev = (base / "dst" / "update-000" / "live_archive" / "events.jsonl")
        rows = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
        def _eid(r):
            if r.get("episode_id"):
                return r["episode_id"]
            pl = r.get("payload")
            return pl.get("episode_id") if isinstance(pl, dict) else None
        ids = {_eid(r) for r in rows} - {None}
        assert ids == set(KEPT), f"unexpected ids carried: {ids}"
        assert BAD not in ev.read_text()
        # the episode-less row survives
        assert any(r.get("event") == "update_start" for r in rows)

    def test_simulations_are_filtered(self, tree):
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        sims = (base / "dst" / "update-000" / "live_archive" / "simulations")
        assert {p.stem for p in sims.iterdir()} == set(KEPT)

    def test_seven_turn_files_carried(self, tree):
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        turns = (base / "dst" / "update-000" / "live_archive" / "turns")
        assert {p.stem for p in turns.iterdir()} == set(KEPT)


class TestValidation:
    def test_refuses_a_sampled_episode(self, tree):
        fn, base = tree
        with pytest.raises(ValueError, match="must never be resampled"):
            fn(source_run_id="src", dest_run_id="dst", update=0,
               episode_id=KEPT[0])

    def test_refuses_a_third_attempt(self, tree):
        fn, base = tree
        with pytest.raises(ValueError, match="budget"):
            fn(source_run_id="src", dest_run_id="dst", update=0,
               episode_id=BAD, attempt=3)

    def test_refuses_missing_destination_manifest(self, tree):
        fn, base = tree
        (base / "dst" / "manifest.json").unlink()
        with pytest.raises(FileNotFoundError, match="manifest"):
            fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)

    def test_refuses_plan_hash_mismatch(self, tree):
        fn, base = tree
        m = json.loads((base / "dst" / "manifest.json").read_text())
        m["plan_hash"] = "deadbeefdeadbeef"
        (base / "dst" / "manifest.json").write_text(json.dumps(m))
        with pytest.raises(ValueError, match="plan_hash"):
            fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)

    def test_refuses_incomplete_roster(self, tree):
        """A missing survivor must not silently shrink the update."""
        fn, base = tree
        arch = base / "src" / "update-000" / "live_archive"
        rows = [json.loads(l) for l in
                (arch / "episodes.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["episode_id"] != KEPT[0]]
        (arch / "episodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="planned roster"):
            fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)

    def test_refuses_missing_turn_file(self, tree):
        fn, base = tree
        (base / "src" / "update-000" / "live_archive" / "turns"
         / f"{KEPT[0]}.jsonl").unlink()
        with pytest.raises(FileNotFoundError, match="no turn file"):
            fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)

    def test_refuses_same_source_and_dest(self, tree):
        fn, base = tree
        with pytest.raises(ValueError, match="differ"):
            fn(source_run_id="src", dest_run_id="src", update=0, episode_id=BAD)


class TestProvenance:
    def test_written_and_complete(self, tree):
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        prov = json.loads(
            (base / "dst" / "update-000" / "retry_provenance.json").read_text())
        # multi-slot retry (2026-08-30): now a sorted list of ids
        assert prov["retried_episode_id"] == [BAD]
        assert prov["retry_attempt"] == 2
        assert sorted(prov["carried_episodes"]) == sorted(KEPT)
        assert prov["n_carried"] == 7
        assert "CONDITIONAL SAMPLING" in prov["declared"]
        assert prov["events_dropped_for_retried_slot"] >= 1
        assert sorted(prov["simulations_copied"]) == sorted(
            f"{e}.json" for e in KEPT)

    def test_retried_slot_is_left_empty(self, tree):
        """An absent slot is what makes capture_live_update resample it."""
        fn, base = tree
        fn(source_run_id="src", dest_run_id="dst", update=0, episode_id=BAD)
        eps = (base / "dst" / "update-000" / "live_archive" / "episodes.jsonl")
        ids = {json.loads(l)["episode_id"]
               for l in eps.read_text().splitlines() if l.strip()}
        assert BAD not in ids
        assert ids == set(KEPT)
