"""The C30 loader's refusals, which is most of what it is for.

A join across three files fails silently by default: pair the wrong tokenized
row with the wrong messages and everything downstream still produces a finite
loss. So the tests that matter here are the ones proving each mis-join is
caught, not the happy path.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from vektori_trace.tau2.c30_loader import (
    C30LoadError,
    cycle_updates,
    load_c30_prefixes,
    prompt_ids_from_row,
)

IGNORE = -100

#: The corpus was rendered with a system policy that `rows.semantic.jsonl` does
#: not store. Tests supply it explicitly, as the real loader requires.
POLICY = "retail policy v1"
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def _sem_hash(prompt, target):
    return hashlib.sha256(
        json.dumps({"prompt": prompt, "target": target}, sort_keys=True,
                   default=str).encode()
    ).hexdigest()


def _build(tmp_path, n_tasks=3, n_pos=2):
    """A miniature frozen corpus with the same shape as the real one."""
    art = tmp_path / "artifacts"
    art.mkdir()

    prefixes, tok_rows, sem_rows = [], [], []
    for t in range(n_tasks):
        for p in range(n_pos):
            task_id = str(10 + t)
            # No system message: `rows.semantic.jsonl` stores only
            # `decision.prompt`, and the policy is prepended by the loader.
            prompt = [
                {"role": "user", "content": f"request {task_id}/{p}"},
                {"role": "assistant", "content": f"prior {task_id}/{p}"},
            ]
            target = {"role": "assistant", "content": f"action {task_id}/{p}"}
            sh = _sem_hash(prompt, target)
            pid = f"{task_id}#{p}"

            n_prompt, n_target = 5 + p, 3
            input_ids = list(range(100, 100 + n_prompt + n_target))
            labels = [IGNORE] * n_prompt + input_ids[n_prompt:]

            prefixes.append({
                "prefix_id": pid, "task_id": task_id, "position": p,
                "action_type": "message", "tool_names": [],
                "semantic_hash": sh,
                "n_tokens": len(input_ids), "n_supervised_tokens": n_target,
            })
            tok_rows.append({
                "task_id": task_id, "position": p, "action_type": "message",
                "tool_names": [], "semantic_hash": sh,
                "input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids),
            })
            sem_rows.append({
                "task_id": task_id, "position": p, "message_index": p * 2,
                "action_type": "message", "tool_names": [],
                "prompt": prompt, "target": target, "semantic_hash": sh,
            })

    def write_jsonl(name, rows):
        path = art / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    tok_path = write_jsonl("rows.tokenized.jsonl", tok_rows)
    sem_path = write_jsonl("rows.semantic.jsonl", sem_rows)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (art / "artifact_hashes.json").write_text(json.dumps({
        "rows.tokenized.jsonl": sha(tok_path),
        "rows.semantic.jsonl": sha(sem_path),
    }))

    order = [p["prefix_id"] for p in prefixes]
    manifest = {
        "partition": "C30", "seed": 1, "split_manifest_hash": "split-abc",
        "n_tasks": n_tasks, "n_prefixes": len(prefixes),
        "n_supervised_tokens": sum(p["n_supervised_tokens"] for p in prefixes),
        "prefixes": prefixes, "sampling_order": order,
        "prefix_manifest_hash": "deadbeefcafe0000",
    }
    (art / "c30_prefix_manifest.json").write_text(json.dumps(manifest))
    return art


def _rehash(art):
    """Recompute artifact_hashes after a test mutates a corpus file."""
    h = {}
    for fn in ("rows.tokenized.jsonl", "rows.semantic.jsonl"):
        h[fn] = hashlib.sha256((art / fn).read_bytes()).hexdigest()
    (art / "artifact_hashes.json").write_text(json.dumps(h))


# --- the join ------------------------------------------------------------


def test_joins_all_three_files(tmp_path):
    art = _build(tmp_path)
    prefixes, report = load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)

    assert len(prefixes) == 6
    assert report["n_tasks"] == 3
    assert report["prefix_manifest_hash"] == "deadbeefcafe0000"

    p = prefixes[0]
    assert p.prefix_id == "10#0"
    assert p.canonical_messages[0]["role"] == "system"
    assert p.canonical_messages[0]["content"] == POLICY
    assert p.stored_teacher_action["content"] == "action 10/0"
    # prompt ids stop at the label boundary, not at the end of the sequence
    assert p.n_prompt_tokens == 5
    assert p.prompt_token_ids == list(range(100, 105))


def test_returns_frozen_sampling_order_not_corpus_order(tmp_path):
    art = _build(tmp_path)
    man = json.loads((art / "c30_prefix_manifest.json").read_text())
    man["sampling_order"] = list(reversed(man["sampling_order"]))
    (art / "c30_prefix_manifest.json").write_text(json.dumps(man))

    prefixes, _ = load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)
    assert [p.prefix_id for p in prefixes] == man["sampling_order"]


def test_step_index_mirrors_position(tmp_path):
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)), system_policy=POLICY, tools=TOOLS)
    assert all(p.step_index == p.position for p in prefixes)


# --- refusals ------------------------------------------------------------


def test_refuses_wrong_manifest_hash(tmp_path):
    art = _build(tmp_path)
    with pytest.raises(C30LoadError, match="different prefix pool"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS,
                          expect_manifest_hash="0000000000000000")


def test_refuses_corpus_that_moved_under_the_split(tmp_path):
    art = _build(tmp_path)
    rows = (art / "rows.tokenized.jsonl").read_text().splitlines()
    r = json.loads(rows[0])
    r["input_ids"][0] = 999            # same semantic_hash, different ids
    rows[0] = json.dumps(r)
    (art / "rows.tokenized.jsonl").write_text("\n".join(rows) + "\n")
    # deliberately do NOT rehash: this is the "corpus moved" case
    with pytest.raises(C30LoadError, match="moved under the split"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_refuses_semantic_hash_mismatch(tmp_path):
    """The mis-join this module exists to prevent."""
    art = _build(tmp_path)
    rows = (art / "rows.semantic.jsonl").read_text().splitlines()
    r = json.loads(rows[0])
    r["semantic_hash"] = "f" * 64
    rows[0] = json.dumps(r)
    (art / "rows.semantic.jsonl").write_text("\n".join(rows) + "\n")
    _rehash(art)
    with pytest.raises(C30LoadError, match="semantic semantic_hash"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_refuses_missing_semantic_row(tmp_path):
    art = _build(tmp_path)
    rows = (art / "rows.semantic.jsonl").read_text().splitlines()
    (art / "rows.semantic.jsonl").write_text("\n".join(rows[1:]) + "\n")
    _rehash(art)
    with pytest.raises(C30LoadError, match="absent from rows.semantic"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_refuses_non_c30_partition(tmp_path):
    art = _build(tmp_path)
    man = json.loads((art / "c30_prefix_manifest.json").read_text())
    man["partition"] = "W30"
    (art / "c30_prefix_manifest.json").write_text(json.dumps(man))
    with pytest.raises(C30LoadError, match="not 'C30'"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_refuses_fully_masked_row():
    with pytest.raises(C30LoadError, match="every label is masked"):
        prompt_ids_from_row({"task_id": "1", "position": 0,
                             "input_ids": [1, 2, 3], "labels": [IGNORE] * 3})


def test_refuses_row_with_no_prompt():
    with pytest.raises(C30LoadError, match="no prompt to sample from"):
        prompt_ids_from_row({"task_id": "1", "position": 0,
                             "input_ids": [1, 2, 3], "labels": [1, 2, 3]})


# --- update cycling ------------------------------------------------------


def test_cycle_gives_fixed_size_batches(tmp_path):
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)), system_policy=POLICY, tools=TOOLS)
    batches = cycle_updates(prefixes, n_per_update=4, n_updates=3)
    assert [len(b) for b in batches] == [4, 4, 4]


def test_cycle_wraps_without_restarting_at_the_head(tmp_path):
    """A pool that does not divide evenly must not over-weight its head."""
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)), system_policy=POLICY, tools=TOOLS)   # 6 prefixes
    batches = cycle_updates(prefixes, n_per_update=4, n_updates=3)

    ids = [p.prefix_id for b in batches for p in b]
    assert ids[:6] == [p.prefix_id for p in prefixes]
    # second pass continues from index 4, not from 0
    assert batches[1][0].prefix_id == prefixes[4].prefix_id

    # 12 draws over 6 prefixes: every prefix exactly twice
    from collections import Counter
    assert set(Counter(ids).values()) == {2}


def test_cycle_rejects_bad_shape(tmp_path):
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)), system_policy=POLICY, tools=TOOLS)
    with pytest.raises(C30LoadError):
        cycle_updates(prefixes, n_per_update=0, n_updates=3)
    with pytest.raises(C30LoadError):
        cycle_updates([], n_per_update=4, n_updates=3)


# --- teacher context reconstruction --------------------------------------


def test_system_policy_is_prepended_for_the_teacher(tmp_path):
    """The student's ids were rendered with it; the teacher must see it too."""
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)),
                                    system_policy=POLICY, tools=TOOLS)
    m = prefixes[0].canonical_messages
    assert m[0]["role"] == "system"
    assert m[0]["content"] == POLICY
    # and the semantic prompt still follows, unmodified
    assert m[1]["role"] == "user"
    assert len(m) == 3


def test_tools_are_carried_for_the_teacher_render(tmp_path):
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)),
                                    system_policy=POLICY, tools=TOOLS)
    assert prefixes[0].tools == TOOLS


def test_tools_ride_on_the_system_message(tmp_path):
    """encoding_dsv4 reads msg['tools']; a separate field reaches nothing.

    `render_teacher_prefix` takes only (messages, thinking_mode), so tools that
    live solely on the prefix object never enter DeepSeek's context while the
    student's ids carry the full schema block.
    """
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)),
                                    system_policy=POLICY, tools=TOOLS)
    sysmsg = prefixes[0].canonical_messages[0]
    assert sysmsg["role"] == "system"
    assert sysmsg["tools"] == TOOLS


def test_empty_policy_is_refused(tmp_path):
    with pytest.raises(C30LoadError, match="system_policy is required"):
        load_c30_prefixes(str(_build(tmp_path)), system_policy="  ", tools=TOOLS)


def test_tool_hash_mismatch_is_refused(tmp_path):
    """Rendering against a different tool set compares two different states."""
    art = _build(tmp_path)
    (art / "eligibility.json").write_text(json.dumps({"tools_hash": "0" * 16}))
    with pytest.raises(C30LoadError, match="tool schema hash"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_duplicate_system_message_is_refused(tmp_path):
    art = _build(tmp_path)
    rows = (art / "rows.semantic.jsonl").read_text().splitlines()
    r = json.loads(rows[0])
    r["prompt"] = [{"role": "system", "content": "x"}] + r["prompt"]
    r["semantic_hash"] = _sem_hash(r["prompt"], r["target"])
    rows[0] = json.dumps(r)
    (art / "rows.semantic.jsonl").write_text("\n".join(rows) + "\n")
    _rehash(art)

    man = json.loads((art / "c30_prefix_manifest.json").read_text())
    for pr in man["prefixes"]:
        if pr["prefix_id"] == f"{r['task_id']}#{r['position']}":
            pr["semantic_hash"] = r["semantic_hash"]
    (art / "c30_prefix_manifest.json").write_text(json.dumps(man))

    tok = (art / "rows.tokenized.jsonl").read_text().splitlines()
    t0 = json.loads(tok[0]); t0["semantic_hash"] = r["semantic_hash"]
    tok[0] = json.dumps(t0)
    (art / "rows.tokenized.jsonl").write_text("\n".join(tok) + "\n")
    _rehash(art)

    with pytest.raises(C30LoadError, match="already begins with a system"):
        load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)


def test_trace_id_comes_from_the_eligibility_record(tmp_path):
    art = _build(tmp_path)
    (art / "eligibility.json").write_text(json.dumps({
        "selected_traces": {"10": {"trace_hash": "abc123"}},
    }))
    prefixes, _ = load_c30_prefixes(str(art), system_policy=POLICY, tools=TOOLS)
    by_task = {p.task_id: p for p in prefixes}
    assert by_task["10"].trace_id == "abc123"
    # no record for task 11 -> falls back to the task id, never a fabrication
    assert by_task["11"].trace_id == "11"


def test_task_property_for_replay_archive(tmp_path):
    prefixes, _ = load_c30_prefixes(str(_build(tmp_path)),
                                    system_policy=POLICY, tools=TOOLS)
    assert prefixes[0].task == prefixes[0].task_id
