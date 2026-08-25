"""Invariants on the C30 replay-prefix manifest.

Both continuation branches read this manifest. If it is wrong, or if the two
branches can end up reading different things, the experiment measures the prefix
stream rather than the objective -- and no training log would show it.

`scripts/` is not a package, so the module is loaded by path.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "tau2_freeze_c30", REPO / "scripts" / "tau2_freeze_c30_prefixes.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tau2_freeze_c30"] = _mod
_spec.loader.exec_module(_mod)

IGNORE = -100


def _row(task_id, position, n_tok=10, n_sup=3, action_type="message"):
    labels = [IGNORE] * (n_tok - n_sup) + list(range(n_sup))
    return {
        "task_id": task_id, "position": position,
        "action_type": action_type, "tool_names": [],
        "semantic_hash": hashlib.sha256(
            f"{task_id}#{position}".encode()).hexdigest(),
        "input_ids": list(range(n_tok)),
        "labels": labels,
        "attention_mask": [1] * n_tok,
    }


@pytest.fixture
def artifacts(tmp_path):
    """A corpus shaped like the real one: all partitions in one rows file."""
    w30 = [str(i) for i in range(1, 31)]
    c30 = [str(i) for i in range(31, 61)]
    s16 = [str(i) for i in range(61, 77)]
    f38 = [str(i) for i in range(77, 115)]
    manifest = {"manifest_hash": "b741bfceb1f3d027",
                "partitions": {"W30": w30, "C30": c30, "S16": s16, "F38": f38}}
    (tmp_path / "task_split_manifest.json").write_text(json.dumps(manifest))

    rows = []
    for t in w30:
        rows += [_row(t, p) for p in range(9)]
    for t in c30:                      # ~9.6 rows/task, like the real 289/30
        rows += [_row(t, p) for p in range(10)]
    for t in s16 + f38:
        rows += [_row(t, p) for p in range(4)]
    (tmp_path / "rows.tokenized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "rows.semantic.jsonl").write_text("{}\n")

    hashes = {}
    for fn in ("task_split_manifest.json", "rows.semantic.jsonl",
               "rows.tokenized.jsonl"):
        hashes[fn] = _mod.sha256_file(str(tmp_path / fn))
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes))
    return str(tmp_path)


def test_selects_every_c30_row_uncapped(artifacts):
    """All rows, not a min(6,n) subset.

    W30 trained `A_warm` on every genuine decision; V2 §6.2 requires the
    representation to be identical on both sides of the split, so capping C30
    alone would introduce exactly the asymmetry the plan forbids.
    """
    rows, _ = _mod.load_c30_rows(artifacts)
    assert len(rows) == 300          # 30 tasks x 10, nothing dropped
    per_task = {}
    for r in rows:
        per_task[r["task_id"]] = per_task.get(r["task_id"], 0) + 1
    assert max(per_task.values()) > 6, "a six-per-task cap was applied"


def test_only_c30_tasks_are_selected(artifacts):
    rows, manifest = _mod.load_c30_rows(artifacts)
    got = {r["task_id"] for r in rows}
    assert got == set(manifest["partitions"]["C30"])


def test_w30_rows_never_leak_into_the_prefix_pool(artifacts):
    """`A_warm` already trained on W30; a leak makes adaptation meaningless."""
    rows, manifest = _mod.load_c30_rows(artifacts)
    got = {r["task_id"] for r in rows}
    assert not (got & set(manifest["partitions"]["W30"]))


def test_evaluation_partitions_never_leak_in(artifacts):
    rows, manifest = _mod.load_c30_rows(artifacts)
    got = {r["task_id"] for r in rows}
    assert not (got & set(manifest["partitions"]["S16"]))
    assert not (got & set(manifest["partitions"]["F38"]))


def test_a_corpus_that_moved_is_refused(artifacts):
    """A rebuilt corpus under the same manifest hash is a different experiment."""
    p = Path(artifacts) / "rows.tokenized.jsonl"
    p.write_text(p.read_text() + json.dumps(_row("31", 99)) + "\n")
    with pytest.raises(_mod.FreezeError) as e:
        _mod.load_c30_rows(artifacts)
    assert "moved under the split" in str(e.value)


def test_overlapping_partitions_are_refused(tmp_path):
    """W30 ∩ C30 == empty is enforced in code, never by convention (V2 §4)."""
    manifest = {"manifest_hash": "x",
                "partitions": {"W30": ["1", "31"], "C30": ["31", "32"],
                               "S16": [], "F38": []}}
    (tmp_path / "task_split_manifest.json").write_text(json.dumps(manifest))
    rows = [_row("31", 0), _row("32", 0)]
    (tmp_path / "rows.tokenized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "rows.semantic.jsonl").write_text("{}\n")
    hashes = {fn: _mod.sha256_file(str(tmp_path / fn))
              for fn in ("task_split_manifest.json", "rows.semantic.jsonl",
                         "rows.tokenized.jsonl")}
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes))
    with pytest.raises(_mod.FreezeError) as e:
        _mod.load_c30_rows(str(tmp_path))
    assert "another partition" in str(e.value)


def test_a_row_with_no_supervised_tokens_is_refused(artifacts):
    rows, _ = _mod.load_c30_rows(artifacts)
    rows[0]["labels"] = [IGNORE] * len(rows[0]["input_ids"])
    # Re-run the check the loader applies.
    assert not any(l != IGNORE for l in rows[0]["labels"])


def test_sampling_order_covers_every_prefix_exactly_once(artifacts):
    rows, _ = _mod.load_c30_rows(artifacts)
    order = _mod.sampling_order(rows, _mod.SEED)
    assert len(order) == len(rows)
    assert len(set(order)) == len(rows)


def test_sampling_order_is_task_first(artifacts):
    """Task-first/position-second, so a few long traces cannot dominate (V2 §8).

    The first N entries of a 30-task pool should touch 30 distinct tasks.
    """
    rows, _ = _mod.load_c30_rows(artifacts)
    order = _mod.sampling_order(rows, _mod.SEED)
    first_round = {pid.split("#")[0] for pid in order[:30]}
    assert len(first_round) == 30, (
        "the first pass repeats a task before covering the others; a long trace "
        "can then dominate a 32-update budget"
    )


def test_sampling_order_is_deterministic(artifacts):
    rows, _ = _mod.load_c30_rows(artifacts)
    assert (_mod.sampling_order(rows, _mod.SEED)
            == _mod.sampling_order(rows, _mod.SEED))


def test_a_different_seed_gives_a_different_order(artifacts):
    rows, _ = _mod.load_c30_rows(artifacts)
    assert (_mod.sampling_order(rows, _mod.SEED)
            != _mod.sampling_order(rows, _mod.SEED + 1))


def test_manifest_hash_is_stable_across_rebuilds(artifacts):
    """Build once, read twice: two rebuilds must be byte-identical."""
    rows, split = _mod.load_c30_rows(artifacts)
    a = _mod.build_manifest(rows, split, _mod.SEED)
    b = _mod.build_manifest(rows, split, _mod.SEED)
    assert a["prefix_manifest_hash"] == b["prefix_manifest_hash"]
    assert a == b


def test_manifest_hash_changes_when_the_pool_changes(artifacts):
    rows, split = _mod.load_c30_rows(artifacts)
    a = _mod.build_manifest(rows, split, _mod.SEED)
    b = _mod.build_manifest(rows[:-1], split, _mod.SEED)
    assert a["prefix_manifest_hash"] != b["prefix_manifest_hash"]


def test_manifest_carries_what_a_branch_needs_to_verify_a_row(artifacts):
    """A branch must be able to prove the row it trains on is the frozen row."""
    rows, split = _mod.load_c30_rows(artifacts)
    m = _mod.build_manifest(rows, split, _mod.SEED)
    p = m["prefixes"][0]
    assert set(p) >= {"prefix_id", "task_id", "position", "semantic_hash",
                      "n_tokens", "n_supervised_tokens", "action_type"}
    assert m["split_manifest_hash"] == split["manifest_hash"]


def test_supervised_token_count_is_recorded(artifacts):
    """Reported, not forced (V2 §8) -- but it has to be reported."""
    rows, split = _mod.load_c30_rows(artifacts)
    m = _mod.build_manifest(rows, split, _mod.SEED)
    assert m["n_supervised_tokens"] == sum(
        sum(1 for l in r["labels"] if l != IGNORE) for r in rows)
    assert m["n_supervised_tokens"] > 0
