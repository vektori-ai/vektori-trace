"""Canonicalization and refusals for the ck35 diversity canary.

Two things are worth testing here and they are different:

- the canonical form, because a wrong one either hides collapse (by merging
  distinct decisions) or invents diversity (by splitting one decision on
  whitespace or a generated id);
- the refusals, because each guards a failure that is silent -- a wrong prefix
  pool, an adapter that was never applied, a resumed run that mixes two
  configurations.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.tau2.action_canon import (  # noqa: E402
    PARSE_FAILURE, canonical_action, diversity,
)

_spec = importlib.util.spec_from_file_location(
    "tau2_swr_canary", REPO / "scripts" / "tau2_swr_canary.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tau2_swr_canary"] = _mod
_spec.loader.exec_module(_mod)


def _call(name, args, call_id="chatcmpl-tool-abc123"):
    return ('<tool_call>' + json.dumps(
        {"id": call_id, "name": name, "arguments": args}) + '</tool_call>')


# --- canonical form ------------------------------------------------------

def test_key_order_does_not_create_diversity():
    """{"a":1,"b":2} and {"b":2,"a":1} are the same arguments."""
    a, _ = canonical_action(_call("f", {"a": 1, "b": 2}))
    b, _ = canonical_action(_call("f", {"b": 2, "a": 1}))
    assert a == b


def test_generated_call_ids_do_not_create_diversity():
    """Ids are assigned per request, so keeping them makes everything distinct."""
    a, _ = canonical_action(_call("f", {"x": 1}, call_id="chatcmpl-tool-aaa"))
    b, _ = canonical_action(_call("f", {"x": 1}, call_id="chatcmpl-tool-bbb"))
    assert a == b


def test_whitespace_does_not_create_diversity():
    a, _ = canonical_action("I will  check\n\nthat for you.")
    b, _ = canonical_action("I will check that for you.")
    assert a == b


def test_list_order_is_preserved():
    """item_ids [x,y] is not the same request as [y,x]; graders compare order."""
    a, _ = canonical_action(_call("ret", {"item_ids": ["x", "y"]}))
    b, _ = canonical_action(_call("ret", {"item_ids": ["y", "x"]}))
    assert a != b


def test_multi_call_order_is_preserved():
    """authenticate-then-mutate is a different trajectory from the reverse."""
    a, _ = canonical_action(_call("auth", {}) + _call("mutate", {}))
    b, _ = canonical_action(_call("mutate", {}) + _call("auth", {}))
    assert a != b


def test_different_argument_values_stay_distinct():
    a, _ = canonical_action(_call("f", {"order_id": "#W1"}))
    b, _ = canonical_action(_call("f", {"order_id": "#W2"}))
    assert a != b


def test_different_tool_names_stay_distinct():
    a, _ = canonical_action(_call("get_order", {}))
    b, _ = canonical_action(_call("cancel_order", {}))
    assert a != b


def test_arguments_as_a_json_string_normalize_too():
    """Providers often deliver `arguments` as a string, with varying spacing."""
    a, _ = canonical_action(_call("f", '{"a": 1, "b": 2}'))
    b, _ = canonical_action(_call("f", '{"b":2,"a":1}'))
    assert a == b


def test_prose_alongside_a_call_is_part_of_the_action():
    """Saying "let me check" versus saying nothing is a different choice."""
    a, _ = canonical_action("Let me check. " + _call("f", {}))
    b, _ = canonical_action(_call("f", {}))
    assert a != b


def test_a_malformed_tool_call_is_its_own_value():
    """Not folded into the text bucket -- a broken sampler must not read diverse."""
    canon, ok = canonical_action("<tool_call>{not json</tool_call>")
    assert canon == PARSE_FAILURE
    assert ok is False


def test_a_plain_message_is_a_successful_parse():
    _, ok = canonical_action("Your order has been cancelled.")
    assert ok is True


# --- the two rates -------------------------------------------------------

def test_formatting_only_variation_is_visible():
    """The case the canonical rate exists for: one decision, cosmetic jitter.

    Byte-distinct says 3/3 -- fully diverse. Canonical says 1/3 -- collapsed.
    Reporting only the first would pass a policy that emits one action.
    """
    actions = [
        _call("f", {"a": 1, "b": 2}, call_id="chatcmpl-tool-1"),
        _call("f", {"b": 2, "a": 1}, call_id="chatcmpl-tool-2"),
        _call("f", {"a": 1, "b": 2}, call_id="chatcmpl-tool-3"),
    ]
    d = diversity(actions)
    assert d["n_byte_distinct"] == 3
    assert d["n_canonical_distinct"] == 1
    assert d["formatting_only_variation"] == 2
    assert d["largest_canonical_fraction"] == 1.0


def test_genuinely_diverse_actions_score_high_on_both():
    actions = [_call("f", {"order_id": f"#W{i}"}) for i in range(4)]
    d = diversity(actions)
    assert d["byte_distinct_rate"] == 1.0
    assert d["canonical_distinct_rate"] == 1.0
    assert d["formatting_only_variation"] == 0


def test_unparseable_rate_is_reported_separately():
    d = diversity(["<tool_call>{bad</tool_call>", _call("f", {}), "hello"])
    assert d["n_unparseable"] == 1
    assert d["unparseable_rate"] == pytest.approx(1 / 3)


# --- refusals ------------------------------------------------------------

@pytest.fixture
def frozen(tmp_path):
    """A corpus + prefix manifest matching what the freezer emits."""
    rows, prefixes, order = [], [], []
    for task in ("31", "32", "33", "34"):
        for pos in range(2):
            ids = [1, 2, 3, int(task), pos]
            sem = hashlib.sha256(f"{task}#{pos}".encode()).hexdigest()
            rows.append({"task_id": task, "position": pos,
                         "action_type": "message", "tool_names": [],
                         "semantic_hash": sem, "input_ids": ids,
                         "labels": [-100] * 4 + [7],
                         "attention_mask": [1] * 5})
            prefixes.append({"prefix_id": f"{task}#{pos}", "task_id": task,
                             "position": pos, "action_type": "message",
                             "tool_names": [], "semantic_hash": sem,
                             "n_tokens": 5, "n_supervised_tokens": 1})
    # task-first order, as the freezer produces
    for pos in range(2):
        for task in ("31", "32", "33", "34"):
            order.append(f"{task}#{pos}")
    (tmp_path / "rows.tokenized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    manifest = {"partition": "C30", "split_manifest_hash": "b741bfceb1f3d027",
                "seed": 20260824, "n_tasks": 4, "n_prefixes": len(prefixes),
                "n_supervised_tokens": len(prefixes),
                "prefixes": prefixes, "sampling_order": order,
                "prefix_manifest_hash": "deadbeefdeadbeef"}
    (tmp_path / "c30_prefix_manifest.json").write_text(json.dumps(manifest))
    # The corpus byte hashes the loader verifies: semantic hashes do not cover
    # tokenization, so a retokenized corpus would otherwise pass unnoticed.
    (tmp_path / "artifact_hashes.json").write_text(json.dumps({
        "rows.tokenized.jsonl": hashlib.sha256(
            (tmp_path / "rows.tokenized.jsonl").read_bytes()).hexdigest()}))
    return tmp_path


def test_a_foreign_prefix_manifest_is_refused(frozen):
    """Captures against a different pool are not comparable to anything."""
    with pytest.raises(_mod.CanaryError) as e:
        _mod.load_frozen_prefixes(
            frozen, frozen / "c30_prefix_manifest.json",
            _mod.FROZEN_PREFIX_MANIFEST_HASH)
    assert "not comparable" in str(e.value)


def test_the_frozen_hash_is_the_real_one():
    """Pinned to the manifest actually frozen on the box."""
    assert _mod.FROZEN_PREFIX_MANIFEST_HASH == "8e78c7b96161d024"


def test_a_retokenized_corpus_is_refused(frozen):
    """Byte hashes catch what semantic hashes cannot.

    `semantic_hash` covers messages and target, not tokenization, so a corpus
    retokenized under a different tokenizer or template keeps every semantic
    hash while every `input_ids` changes -- and the prompts would silently stop
    matching the ones the manifest was frozen against.
    """
    p = frozen / "rows.tokenized.jsonl"
    lines = p.read_text().splitlines()
    row = json.loads(lines[0])
    row["input_ids"] = [999, 998, 997, 996, 995]   # same semantics, new tokens
    lines[0] = json.dumps(row)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(_mod.CanaryError) as e:
        _mod.load_frozen_prefixes(frozen, frozen / "c30_prefix_manifest.json",
                                  "deadbeefdeadbeef")
    assert "moved under the prefix manifest" in str(e.value)


def test_a_corpus_row_whose_identity_moved_is_refused(frozen):
    """A position id can be reused; the semantic hash is what pins identity.

    Byte hashes are refreshed here so the *semantic* check is what fires.
    """
    p = frozen / "rows.tokenized.jsonl"
    lines = p.read_text().splitlines()
    row = json.loads(lines[0])
    row["semantic_hash"] = "0" * 64
    lines[0] = json.dumps(row)
    p.write_text("\n".join(lines) + "\n")
    (frozen / "artifact_hashes.json").write_text(json.dumps({
        "rows.tokenized.jsonl": hashlib.sha256(p.read_bytes()).hexdigest()}))
    with pytest.raises(_mod.CanaryError) as e:
        _mod.load_frozen_prefixes(frozen, frozen / "c30_prefix_manifest.json",
                                  "deadbeefdeadbeef")
    assert "moved under the manifest" in str(e.value)


def test_prefixes_are_task_balanced(frozen):
    manifest, _ = _mod.load_frozen_prefixes(
        frozen, frozen / "c30_prefix_manifest.json", "deadbeefdeadbeef")
    picked = _mod.choose_prefixes(manifest, 4)
    assert len({p.split("#")[0] for p in picked}) == 4


def test_asking_for_more_prefixes_than_exist_is_refused(frozen):
    manifest, _ = _mod.load_frozen_prefixes(
        frozen, frozen / "c30_prefix_manifest.json", "deadbeefdeadbeef")
    with pytest.raises(_mod.CanaryError):
        _mod.choose_prefixes(manifest, 999)


def test_fingerprint_binds_model_temperature_and_prompt():
    """A stale capture from another configuration must not be resumed into."""
    base = _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2, 3])
    assert base != _mod.capture_fingerprint("31#0", 0, "ck70", 1.0, 2048, [1, 2, 3])
    assert base != _mod.capture_fingerprint("31#0", 0, "ck35", 0.7, 2048, [1, 2, 3])
    assert base != _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2, 4])
    assert base != _mod.capture_fingerprint("31#1", 0, "ck35", 1.0, 2048, [1, 2, 3])
    assert base == _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2, 3])


def test_resume_keeps_matching_captures(tmp_path):
    fp = _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2])
    p = tmp_path / "captures.jsonl"
    p.write_text(json.dumps(
        {"prefix_id": "31#0", "sample_index": 0, "fingerprint": fp}) + "\n")
    assert list(_mod.load_captures(p, {"31#0#0": fp})) == ["31#0#0"]


def test_resume_drops_captures_from_another_run(tmp_path):
    """A transient failure must not let a ck70 capture into a ck35 run."""
    stale = _mod.capture_fingerprint("31#0", 0, "ck70", 1.0, 2048, [1, 2])
    want = _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2])
    p = tmp_path / "captures.jsonl"
    p.write_text(json.dumps(
        {"prefix_id": "31#0", "sample_index": 0, "fingerprint": stale}) + "\n")
    assert _mod.load_captures(p, {"31#0#0": want}) == {}


def test_resume_tolerates_a_truncated_final_line(tmp_path):
    """A crash mid-write must not discard the samples already paid for."""
    fp = _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2])
    p = tmp_path / "captures.jsonl"
    p.write_text(json.dumps(
        {"prefix_id": "31#0", "sample_index": 0, "fingerprint": fp})
        + "\n{\"prefix_id\": \"31#0\", \"sample_ind")
    assert list(_mod.load_captures(p, {"31#0#0": fp})) == ["31#0#0"]


class _FakeTok:
    """Decodes one id at a time, matching `token_bytes_from_ids`'s contract."""

    def __init__(self, table):
        self._t = table

    def decode(self, ids, **_kw):
        return "".join(self._t[int(i)] for i in ids)


def _tok(table=None):
    return _FakeTok(table or {1: "he", 2: "llo", 9: "P"})


def test_a_truncated_action_is_refused():
    """A fragment is not a decision; scoring one repeats the 0/13 OPD failure."""
    body = {"choices": [{"text": "partial", "token_ids": [1],
                         "prompt_token_ids": [9], "finish_reason": "length",
                         "logprobs": {"token_logprobs": [-0.1]}}]}
    with pytest.raises(_mod.CanaryError) as e:
        _mod._capture("31#0", 0, body, [9], "fp", _tok())
    assert "max_tokens" in str(e.value)


def test_a_different_prompt_than_the_manifest_is_refused():
    body = {"choices": [{"text": "ok", "token_ids": [1],
                         "prompt_token_ids": [7], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1]}}]}
    with pytest.raises(_mod.CanaryError) as e:
        _mod._capture("31#0", 0, body, [9], "fp", _tok())
    assert "different prompt ids" in str(e.value)


def test_a_capture_records_byte_exact_identity():
    """Bytes come from the TOKENS, and the per-token split is persisted.

    `text.encode()` would assert nothing: it produces bytes for whatever string
    the server rendered, which does not prove the returned ids reconstruct them.
    A cross-tokenizer scorer needs the per-token bytes anyway.
    """
    import base64
    body = {"choices": [{"text": "hello", "token_ids": [1, 2],
                         "prompt_token_ids": [9], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1, -0.2]}}]}
    cap = _mod._capture("31#0", 0, body, [9], "fp", _tok())
    assert cap["action_sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert cap["n_action_bytes"] == 5
    assert cap["behavior_logprobs"] == [-0.1, -0.2]
    assert base64.b64decode(cap["action_bytes_b64"]) == b"hello"
    assert [base64.b64decode(b) for b in cap["action_token_bytes_b64"]] == [
        b"he", b"llo"]


def test_a_trailing_eos_token_is_accepted():
    """The real shape of a healthy vLLM capture.

    vLLM returns the sampled ids INCLUDING the stop token but renders `text`
    without it, so a good capture reconstructs to `text + "<|im_end|>"`.
    Demanding exact equality rejected the first sample of a perfectly good run
    (2026-08-25); `token_bytes_from_ids` keeps special tokens on purpose.
    """
    tok = _FakeTok({1: "he", 2: "llo", 3: "<|im_end|>", 9: "P"})
    body = {"choices": [{"text": "hello", "token_ids": [1, 2, 3],
                         "prompt_token_ids": [9], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3]}}]}
    cap = _mod._capture("31#0", 0, body, [9], "fp", tok)
    # The EOS byte stays in the capture: a scorer needs the tokens the model
    # actually emitted, not a cleaned-up rendering.
    assert cap["n_action_bytes"] == len(b"hello<|im_end|>")


def test_trailing_content_that_is_not_special_is_refused():
    """A trailer that is real content means the capture carries what the server
    never reported, which is a genuine disagreement rather than an EOS."""
    tok = _FakeTok({1: "he", 2: "llo", 3: " and more", 9: "P"})
    body = {"choices": [{"text": "hello", "token_ids": [1, 2, 3],
                         "prompt_token_ids": [9], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3]}}]}
    with pytest.raises(_mod.CanaryError) as e:
        _mod._capture("31#0", 0, body, [9], "fp", tok)
    assert "beyond the returned text" in str(e.value)


def test_ids_that_do_not_reconstruct_the_text_are_refused():
    """The check `text.encode()` could never make.

    If the server's ids and its rendered text disagree, one of them is wrong and
    the capture cannot be reproduced from its own ids -- so it is not scorable,
    however plausible it looks.
    """
    body = {"choices": [{"text": "goodbye", "token_ids": [1, 2],
                         "prompt_token_ids": [9], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1, -0.2]}}]}
    with pytest.raises(_mod.CanaryError) as e:
        _mod._capture("31#0", 0, body, [9], "fp", _tok())
    assert "do not reconstruct" in str(e.value)


def test_missing_behavior_logprobs_are_refused():
    body = {"choices": [{"text": "hello", "token_ids": [1, 2],
                         "prompt_token_ids": [9], "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1]}}]}
    with pytest.raises(_mod.CanaryError):
        _mod._capture("31#0", 0, body, [9], "fp", _tok())


def test_report_fails_a_collapsed_policy():
    caps = [{"prefix_id": "31#0", "sample_index": i,
             "action_text": _call("f", {"a": 1}, call_id=f"chatcmpl-tool-{i}")}
            for i in range(4)]
    r = _mod.build_report(caps, ["31#0"], 4, 0.25, 0.75, 0.20)
    assert r["pass"] is False
    assert r["per_prefix"][0]["n_byte_distinct"] == 4     # looks diverse
    assert r["per_prefix"][0]["n_canonical_distinct"] == 1  # is not


def test_report_passes_a_diverse_policy():
    caps = [{"prefix_id": "31#0", "sample_index": i,
             "action_text": _call("f", {"order_id": f"#W{i}"})}
            for i in range(4)]
    r = _mod.build_report(caps, ["31#0"], 4, 0.25, 0.75, 0.20)
    assert r["pass"] is True


def test_report_fails_on_too_many_unparseable_actions():
    """Diversity numbers over garbage are not evidence of anything."""
    caps = [{"prefix_id": "31#0", "sample_index": i,
             "action_text": f"<tool_call>{{bad{i}</tool_call>"}
            for i in range(4)]
    r = _mod.build_report(caps, ["31#0"], 4, 0.25, 0.75, 0.20)
    assert r["unparseable_rate"] == 1.0
    assert r["pass"] is False


def test_report_says_it_is_not_the_v2_gate():
    caps = [{"prefix_id": "31#0", "sample_index": i,
             "action_text": _call("f", {"order_id": f"#W{i}"})}
            for i in range(4)]
    r = _mod.build_report(caps, ["31#0"], 4, 0.25, 0.75, 0.20)
    assert "7.1a" in r["note"]


# --- the prefix/target boundary ------------------------------------------

def test_prompt_excludes_the_supervised_target(frozen):
    """The bug this test exists for.

    A corpus row is prefix + DeepSeek's recorded action, with labels masked to
    -100 everywhere but the action. Sending the whole row hands the student the
    answer and asks it to continue AFTER it -- so the measured "diversity" is
    diversity of continuation-after-the-target, a plausible number describing
    the wrong quantity. On the real corpus a row is ~4,840 tokens of which the
    last ~54 are supervised, so length alone never reveals it.
    """
    row = {"task_id": "31", "position": 0,
           "input_ids": [10, 11, 12, 13, 14],
           "labels": [-100, -100, -100, 13, 14]}
    assert _mod.prompt_ids_for(row) == [10, 11, 12]


def test_prompt_stops_at_the_first_supervised_token(frozen):
    row = {"task_id": "31", "position": 0,
           "input_ids": [1, 2, 3, 4],
           "labels": [-100, 2, 3, 4]}
    assert _mod.prompt_ids_for(row) == [1]


def test_a_row_with_no_prefix_is_refused():
    row = {"task_id": "31", "position": 0,
           "input_ids": [1, 2], "labels": [1, 2]}
    with pytest.raises(_mod.CanaryError) as e:
        _mod.prompt_ids_for(row)
    assert "no prefix" in str(e.value)


def test_a_row_with_no_supervision_is_refused():
    row = {"task_id": "31", "position": 0,
           "input_ids": [1, 2], "labels": [-100, -100]}
    with pytest.raises(_mod.CanaryError):
        _mod.prompt_ids_for(row)


def test_the_real_corpus_shape_would_have_leaked_the_answer(frozen):
    """Guard the specific ratio: a tiny target inside a long row."""
    row = {"task_id": "31", "position": 0,
           "input_ids": list(range(4840)),
           "labels": [-100] * 4786 + list(range(54))}
    prompt = _mod.prompt_ids_for(row)
    assert len(prompt) == 4786
    assert len(prompt) < len(row["input_ids"])


def test_fingerprint_binds_max_tokens_and_policy_hash():
    base = _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2], "h1")
    assert base != _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 512, [1, 2], "h1")
    assert base != _mod.capture_fingerprint("31#0", 0, "ck35", 1.0, 2048, [1, 2], "h2")


def test_an_unclosed_tool_call_is_not_an_ordinary_message():
    """Otherwise broken output reads as prose, and every truncation looks unique."""
    canon, ok = canonical_action('<tool_call>{"name": "f", "argum')
    assert canon == PARSE_FAILURE
    assert ok is False


def test_truncated_calls_do_not_inflate_diversity():
    d = diversity(['<tool_call>{"name":"f","a',
                   '<tool_call>{"name":"f","ab',
                   '<tool_call>{"name":"f","abc'])
    assert d["n_canonical_distinct"] == 1     # all one bucket
    assert d["unparseable_rate"] == 1.0
