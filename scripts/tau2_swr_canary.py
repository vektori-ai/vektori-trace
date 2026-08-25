#!/usr/bin/env python3
"""Sample ck35 at frozen C30 prefixes; stop if action diversity has collapsed.

The cheap preflight before any replay training. It calls no teacher and trains
nothing: it samples the student at prefixes drawn from the frozen C30 manifest,
saves byte-exact captures, and reports how much the sampled actions actually
vary.

Why it exists
-------------
`A_warm` is a launchpad, not a deliverable. If warm SFT collapsed the action
distribution onto one memorized continuation per state, the student samples the
same action every time, the teacher scores the same thing every time, and the
replay gradient is flat. Replay would then fail for a reason that has nothing to
do with the objective under test, and nothing in a training log would show it.
Finding that out here costs one endpoint; finding it out later costs the
experiment.

**This is a heuristic stop check, not V2 §7.1a.** That gate is *A0-relative*:
it compares the checkpoint's diversity against the frozen base, because "low
diversity" only means collapse relative to where the model started. This
measures the checkpoint alone. A pass here does not discharge §7.1a; a failure
here is decisive regardless, since a near-deterministic sampler cannot produce a
replay gradient whatever the base does.

Two diversity numbers, both reported
------------------------------------
- **byte-distinct**: literally different bytes -- raw sampling variation.
- **canonical-distinct**: different *decisions*, after sorting JSON keys,
  dropping generated call ids, and normalizing whitespace.

Byte-distinct alone overstates: a collapsed policy emitting one action with
cosmetic jitter reads as fully diverse. Canonical alone hides what the sampler
is doing. The gap between them is itself the interesting signal. See
`vektori_trace/tau2/action_canon.py` for the rules and why each was chosen.

What it refuses
---------------
Every one of these has a matching failure that would otherwise be silent:

- a prefix manifest whose hash is not the frozen one -- captures against a
  different prefix pool are not comparable to anything;
- an endpoint that does not advertise the requested model -- vLLM resolves an
  unknown name against the base, so the adapter silently does nothing;
- a server that consumed different prompt token ids than the manifest holds;
- captures on disk whose fingerprint belongs to a different prefix, model,
  temperature or prompt -- resuming into those would mix two runs;
- a truncated action (`finish_reason == "length"`), because a fragment is not a
  decision and scoring one later would repeat the mistake that produced the
  0/13 OPD run.

Resumable by design. Captures are appended one line at a time and fsync'd, so a
transient failure keeps every sample already paid for; rerunning fills only the
gaps.

What it shares with the replay path, and what it does not
---------------------------------------------------------
Byte reconstruction is delegated to `replay_sample.token_bytes_from_ids`, so
this path and the training path derive action bytes identically. The HTTP
request, the resume bookkeeping and the fingerprint are written here rather than
reused: `run_replay_opd` samples through Harbor's corpus objects, which is a
different input shape from a frozen C30 row. Calling that "reusing the capture
pipeline" would overstate it -- the byte-exactness is shared, the plumbing is
not.

    python scripts/tau2_swr_canary.py --dry-run
    python scripts/tau2_swr_canary.py --api-base "$STUDENT_API_BASE"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.tau2.action_canon import diversity  # noqa: E402
from vektori_trace.replay_sample import token_bytes_from_ids  # noqa: E402

# The frozen C30 prefix manifest (scripts/tau2_freeze_c30_prefixes.py).
# Pinned, not discovered: a canary run against a different prefix pool produces
# numbers that cannot be compared with anything else.
FROZEN_PREFIX_MANIFEST_HASH = "8e78c7b96161d024"

BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL = "Qwen3-4B-ck35"
# `A_warm` as decided 2026-08-25 (handoff §7 step 2): ck35, taken as a tiebreak
# against ck70 after both measured equal on three selection tasks.
POLICY_VERSION = "tau2-a_warm-ck35"

# Label value marking an unsupervised position.
IGNORE = -100


class CanaryError(RuntimeError):
    """A refusal. Never caught -- every one of these invalidates the run."""


def _post(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:1000]}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def require_endpoint_model(api_base: str, model: str, timeout: float) -> None:
    """Fail before spending anything if the adapter is not actually served.

    vLLM resolves a request's `model` against the base name before consulting
    the LoRA table, so an unadvertised name falls through to base weights. The
    run completes, the adapter is never applied, and the result reads as "the
    checkpoint changed nothing".
    """
    try:
        with urllib.request.urlopen(
                api_base.rstrip("/") + "/models", timeout=timeout) as r:
            body = json.loads(r.read().decode())
    except Exception as exc:
        raise CanaryError(f"endpoint readiness failed: {exc}") from exc
    advertised = [str(row.get("id")) for row in (body.get("data") or [])]
    if model not in advertised:
        raise CanaryError(
            f"endpoint does not advertise {model!r}; available={advertised}. "
            f"Requests would resolve to base weights and the adapter would "
            f"silently do nothing.")


def load_frozen_prefixes(artifacts: Path, manifest_path: Path,
                         expect_hash: str | None) -> tuple[dict, dict]:
    """The frozen manifest plus the tokenized rows it names, keyed by prefix id."""
    if not manifest_path.exists():
        raise CanaryError(
            f"missing {manifest_path}. Build it with "
            f"scripts/tau2_freeze_c30_prefixes.py; this script never creates a "
            f"prefix pool, because a pool built under a canary is not the pool "
            f"the branches will train on.")
    manifest = json.loads(manifest_path.read_text())

    got = manifest.get("prefix_manifest_hash")
    if expect_hash and got != expect_hash:
        raise CanaryError(
            f"prefix manifest hash {got!r} != frozen {expect_hash!r}. Captures "
            f"against a different prefix pool are not comparable. Pass "
            f"--expect-manifest-hash {got} deliberately if the pool genuinely "
            f"changed.")

    rows_path = artifacts / "rows.tokenized.jsonl"
    if not rows_path.exists():
        raise CanaryError(f"missing {rows_path}")

    # Verify the corpus bytes, not just per-row semantic hashes.
    #
    # `semantic_hash` covers the *semantic* row -- messages and target. It does
    # not cover the tokenization, so a corpus retokenized under a different
    # tokenizer or template keeps every semantic hash while every `input_ids`
    # changes. Prompts would then differ from the ones the manifest was frozen
    # against, silently.
    hash_path = artifacts / "artifact_hashes.json"
    if not hash_path.exists():
        raise CanaryError(
            f"missing {hash_path}; the corpus cannot be proven unchanged")
    for fn, want in json.loads(hash_path.read_text()).items():
        fp = artifacts / fn
        if not fp.exists():
            raise CanaryError(f"hash manifest names {fn}, which is missing")
        got = hashlib.sha256(fp.read_bytes()).hexdigest()
        if got != want:
            raise CanaryError(
                f"{fn} hash {got[:16]} != frozen {want[:16]}; the corpus moved "
                f"under the prefix manifest, so the tokenized prompts may "
                f"differ from the ones it was frozen against")

    wanted = {p["prefix_id"]: p for p in manifest["prefixes"]}
    rows: dict[str, dict] = {}
    for line in rows_path.open():
        r = json.loads(line)
        pid = f"{r['task_id']}#{r['position']}"
        if pid not in wanted:
            continue
        want = wanted[pid]
        # The manifest recorded this row's identity; prove the corpus still
        # holds the same row rather than one that merely shares a position id.
        if r["semantic_hash"] != want["semantic_hash"]:
            raise CanaryError(
                f"{pid}: semantic hash {r['semantic_hash'][:16]} != frozen "
                f"{want['semantic_hash'][:16]}; the corpus moved under the "
                f"manifest")
        rows[pid] = r
    missing = set(wanted) - set(rows)
    if missing:
        raise CanaryError(f"{len(missing)} frozen prefixes absent from the "
                          f"corpus: {sorted(missing)[:5]}")
    return manifest, rows


def prompt_ids_for(row: dict) -> list[int]:
    """The prefix ONLY -- everything before the supervised action.

    A corpus row's `input_ids` is the whole training sequence: prefix followed
    by DeepSeek's recorded action, with `labels` masked to -100 everywhere
    except that action. Sending the whole row would hand the student the answer
    and ask it to continue *after* it, so the "diversity" measured would be
    diversity of continuation-after-the-target -- a plausible number describing
    the wrong quantity entirely.

    Measured on the real corpus: a row is ~4,840 tokens of which the last ~54
    are supervised, so the mistake is ~99% invisible by length alone.

    The split point is the first non-masked label, which is exactly where the
    replay branch will sample from.
    """
    labels = row["labels"]
    target_start = next((i for i, l in enumerate(labels) if l != IGNORE), None)
    if target_start is None:
        raise CanaryError(
            f"{row['task_id']}#{row['position']}: no supervised tokens, so "
            f"there is no prefix/target boundary to sample at")
    if target_start == 0:
        raise CanaryError(
            f"{row['task_id']}#{row['position']}: supervision starts at token "
            f"0, leaving no prefix to condition on")
    return [int(x) for x in row["input_ids"][:target_start]]


def choose_prefixes(manifest: dict, n: int) -> list[str]:
    """Task-balanced prefixes, taken in the manifest's own frozen order.

    The manifest's `sampling_order` is already task-first/position-second, so
    its head touches n distinct tasks. Reusing it rather than sampling here
    means the canary looks at the same states the branches will see first, and
    needs no seed of its own.
    """
    order = manifest["sampling_order"]
    if n > len(order):
        raise CanaryError(f"asked for {n} prefixes; pool holds {len(order)}")
    picked = order[:n]
    tasks = {p.split("#")[0] for p in picked}
    if len(tasks) != len(picked):
        raise CanaryError(
            f"the frozen sampling order repeats a task within its first {n} "
            f"entries ({len(tasks)} distinct); it should be task-first")
    return picked


def capture_fingerprint(prefix_id: str, sample_index: int, model: str,
                        temperature: float, max_tokens: int,
                        prompt_ids: list[int],
                        policy_hash: str = "") -> str:
    """What an existing capture is valid for.

    Keying on `prefix#sample` alone is not enough -- those ids are positional
    and get reused the moment the pool or the model changes, so a stale file
    would be silently accepted for a different run.

    Every sampling parameter that changes the distribution is bound in, not just
    the obvious ones. `max_tokens` matters because it decides whether an action
    could be truncated, so captures taken under different caps are not
    interchangeable. `policy_hash` binds the adapter and tokenizer bytes: two
    runs can name the same served model while the volume underneath it holds a
    different adapter, and a served *name* is not a policy identity.
    """
    h = hashlib.sha256()
    for part in (prefix_id, str(sample_index), model, f"{temperature:.6f}",
                 str(max_tokens), policy_hash):
        h.update(part.encode())
        h.update(b"\0")
    h.update(json.dumps(prompt_ids).encode())
    return h.hexdigest()[:32]


def _policy_hash(adapter_dir: str | None) -> str:
    """Hash of the adapter weights and tokenizer, or "" when unpinned.

    A served *name* is not a policy identity: the same name can front a
    different adapter after a redeploy, so captures keyed on the name alone
    could be resumed across two different policies. Hashing the files makes
    that impossible. Empty when the adapter is not reachable locally -- the run
    still works, it simply cannot prove which weights produced it, and the
    report says so.
    """
    if not adapter_dir:
        return ""
    d = Path(adapter_dir)
    if not d.is_dir():
        raise CanaryError(f"--adapter-dir {adapter_dir} is not a directory")
    h = hashlib.sha256()
    for name in ("adapter_model.safetensors", "adapter_config.json",
                 "tokenizer.json", "chat_template.jinja"):
        fp = d / name
        if fp.exists():
            h.update(name.encode())
            h.update(hashlib.sha256(fp.read_bytes()).digest())
    return h.hexdigest()[:16]


def load_captures(path: Path, expected: dict[str, str]) -> dict[str, dict]:
    """Captures already on disk, dropping any whose fingerprint no longer fits.

    Tolerates a truncated final line: a crash mid-write must not invalidate the
    samples before it, which is the whole reason for appending one at a time.
    """
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    stale = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break            # truncated tail; everything before it stands
        key = f"{row.get('prefix_id')}#{row.get('sample_index')}"
        if expected.get(key) != row.get("fingerprint"):
            stale += 1
            continue
        out[key] = row
    if stale:
        print(f"  ignored {stale} capture(s) whose fingerprint belongs to "
              f"another prefix/model/temperature", flush=True)
    return out


def _capture(prefix_id: str, sample_index: int, body: dict,
             prompt_ids: list[int], fingerprint: str, tokenizer) -> dict:
    choice = (body.get("choices") or [{}])[0]
    token_ids = choice.get("token_ids") or body.get("token_ids") or []
    got_prompt = choice.get("prompt_token_ids") or body.get("prompt_token_ids") or []
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
    text = choice.get("text")
    tag = f"{prefix_id}#{sample_index}"

    if not isinstance(text, str) or not text:
        raise CanaryError(f"{tag}: empty sampled action")
    if not token_ids or len(token_ids) != len(logprobs):
        raise CanaryError(f"{tag}: need one behavior logprob per action token "
                          f"(got {len(token_ids)} ids, {len(logprobs)} logprobs)")
    if not got_prompt:
        raise CanaryError(f"{tag}: server omitted prompt ids, so the prompt it "
                          f"actually consumed cannot be verified")
    if [int(x) for x in got_prompt] != prompt_ids:
        raise CanaryError(f"{tag}: server consumed different prompt ids than the "
                          f"frozen manifest holds")
    if choice.get("finish_reason") == "length":
        # A truncated action is a fragment, not a decision. Scoring one as if it
        # were complete is the failure that produced the 0/13 OPD run.
        raise CanaryError(f"{tag}: action hit max_tokens; raise --max-tokens "
                          f"rather than scoring a fragment")

    # Byte-exact identity comes from the TOKENS, not from re-encoding the text.
    #
    # `text.encode("utf-8")` asserts nothing: it produces bytes for whatever
    # string the server rendered, which does not prove the returned token ids
    # reconstruct those bytes. A scorer downstream needs per-token bytes anyway
    # (cross-tokenizer alignment operates on them), and deriving them here from
    # the pinned tokenizer is what makes the capture usable rather than merely
    # plausible. Delegated to the repo's existing capture adapter so this path
    # and the replay path cannot drift.
    token_bytes = token_bytes_from_ids(tokenizer, [int(x) for x in token_ids])
    action_bytes = b"".join(token_bytes)
    if action_bytes.decode("utf-8", "replace") != text:
        raise CanaryError(
            f"{tag}: the returned token ids do not reconstruct the returned "
            f"text. One of them is wrong, and a capture that cannot be "
            f"reproduced from its own ids is not scorable.")

    return {
        "prefix_id": prefix_id,
        "sample_index": sample_index,
        "fingerprint": fingerprint,
        "policy_version": POLICY_VERSION,
        "served_model": body.get("model"),
        "request_id": body.get("id"),
        "prompt_token_ids": prompt_ids,
        "action_token_ids": [int(x) for x in token_ids],
        "behavior_logprobs": [float(x) for x in logprobs],
        "action_text": text,
        # Persisted the way the replay path expects: whole-action bytes plus the
        # per-token split, both base64'd because JSONL cannot hold raw bytes.
        "action_bytes_b64": base64.b64encode(action_bytes).decode(),
        "action_token_bytes_b64": [base64.b64encode(b).decode()
                                   for b in token_bytes],
        "action_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "n_action_bytes": len(action_bytes),
        "finish_reason": choice.get("finish_reason"),
    }


def build_report(captures: list[dict], prefix_ids: list[str], n_samples: int,
                 min_canonical_rate: float, min_diverse_fraction: float,
                 max_unparseable_rate: float) -> dict:
    grouped: dict[str, list[dict]] = {p: [] for p in prefix_ids}
    for c in captures:
        grouped[c["prefix_id"]].append(c)

    per_prefix = []
    for pid in prefix_ids:
        rows = sorted(grouped[pid], key=lambda r: r["sample_index"])
        if len(rows) != n_samples:
            raise CanaryError(f"{pid}: expected {n_samples} samples, "
                              f"found {len(rows)}")
        d = diversity([r["action_text"] for r in rows])
        d["prefix_id"] = pid
        # The verdict is CANONICAL diversity: byte variation that survives
        # canonicalization is formatting jitter, not a different decision.
        #
        # One distinct action is collapse at ANY sample count, and the rate
        # alone does not say so: 1 of 4 is 0.25, which clears a 0.25 floor while
        # describing a policy that emitted the same decision four times. That is
        # precisely the state this canary exists to refuse, so it is failed
        # explicitly rather than left to the threshold.
        d["pass"] = (d["n_canonical_distinct"] > 1
                     and d["canonical_distinct_rate"] >= min_canonical_rate)
        per_prefix.append(d)

    diverse_fraction = sum(p["pass"] for p in per_prefix) / len(per_prefix)
    unparseable = sum(p["n_unparseable"] for p in per_prefix)
    total = sum(p["n_samples"] for p in per_prefix)
    unparseable_rate = unparseable / total

    passed = (diverse_fraction >= min_diverse_fraction
              and unparseable_rate <= max_unparseable_rate)
    return {
        "metric": "byte_and_canonical_action_diversity",
        "note": ("heuristic stop check, NOT the V2 §7.1a A0-relative entropy "
                 "gate: it measures this checkpoint alone, so a pass here does "
                 "not discharge §7.1a"),
        "policy_version": POLICY_VERSION,
        "n_prefixes": len(prefix_ids),
        "n_samples_per_prefix": n_samples,
        "min_canonical_distinct_rate": min_canonical_rate,
        "min_diverse_prefix_fraction": min_diverse_fraction,
        "max_unparseable_rate": max_unparseable_rate,
        "diverse_prefix_fraction": diverse_fraction,
        "unparseable_rate": unparseable_rate,
        "per_prefix": per_prefix,
        "pass": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--manifest", default=None,
                    help="default: <artifacts>/c30_prefix_manifest.json")
    ap.add_argument("--expect-manifest-hash", default=FROZEN_PREFIX_MANIFEST_HASH,
                    help="refuse any other prefix pool; pass a different value "
                         "deliberately if the pool genuinely changed")
    ap.add_argument("--api-base", default=os.environ.get("STUDENT_API_BASE"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-prefixes", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="serving-time sampling temperature. The point is to "
                         "measure the distribution replay would sample from, so "
                         "this must match what the replay branch will use.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="measured Tau2 action p99 is 442 tokens and the max is "
                         "592, so 2048 is ~3.5x the longest stored action; a hit "
                         "is refused rather than scored as a fragment")
    ap.add_argument("--min-canonical-rate", type=float, default=0.25)
    ap.add_argument("--min-diverse-prefix-fraction", type=float, default=0.75)
    ap.add_argument("--max-unparseable-rate", type=float, default=0.20)
    ap.add_argument("--adapter-dir", default=None,
                    help="local path to the served adapter. Used to hash the "
                         "policy into the capture fingerprint and to load the "
                         "adapter's own tokenizer; without it the run cannot "
                         "prove which weights produced the samples.")
    ap.add_argument("--tokenizer", default=None,
                    help="override the tokenizer path (default: --adapter-dir, "
                         "else the base model)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="/data/tau2/swr_ck35_canary")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the frozen pool and print the plan; calls "
                         "nothing and costs nothing")
    a = ap.parse_args()

    artifacts = Path(a.artifacts)
    manifest_path = Path(a.manifest or artifacts / "c30_prefix_manifest.json")
    manifest, rows = load_frozen_prefixes(artifacts, manifest_path,
                                          a.expect_manifest_hash)
    prefix_ids = choose_prefixes(manifest, a.n_prefixes)

    plan = {
        "policy_version": POLICY_VERSION,
        "model": a.model,
        "prefix_manifest_hash": manifest["prefix_manifest_hash"],
        "split_manifest_hash": manifest.get("split_manifest_hash"),
        "prefix_ids": prefix_ids,
        "tasks": sorted({p.split("#")[0] for p in prefix_ids},
                        key=lambda t: int(t) if t.isdigit() else 0),
        "n_samples_per_prefix": a.n_samples,
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
        "n_requests": len(prefix_ids) * a.n_samples,
    }
    print(json.dumps(plan, indent=2))

    if a.dry_run:
        print("\n--dry-run: frozen C30 pool verified against the corpus; "
              "no endpoint called")
        return 0
    if not a.api_base:
        print("set --api-base or STUDENT_API_BASE", file=sys.stderr)
        return 2

    require_endpoint_model(a.api_base, a.model, a.timeout)
    print(f"\nendpoint advertises {a.model!r}", flush=True)

    # The tokenizer that turns returned ids back into bytes. Pinned to the
    # adapter's own directory, not the base model's: the adapter ships its
    # tokenizer and chat template precisely so serve and score cannot drift.
    from vektori_trace.vocab_bridge import load_tokenizer
    tokenizer = load_tokenizer(a.tokenizer or a.adapter_dir or BASE_MODEL)

    # Policy identity is the adapter bytes, not the served name. Two runs can
    # name the same model while the volume underneath holds a different
    # adapter; binding the hash means captures from those runs cannot be mixed.
    policy_hash = _policy_hash(a.adapter_dir)
    print(f"policy hash {policy_hash or '(unpinned)'}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    captures_path = out / "captures.jsonl"

    expected = {
        f"{pid}#{i}": capture_fingerprint(
            pid, i, a.model, a.temperature, a.max_tokens,
            prompt_ids_for(rows[pid]), policy_hash)
        for pid in prefix_ids for i in range(a.n_samples)
    }
    have = load_captures(captures_path, expected)
    if have:
        print(f"  resuming: {len(have)}/{len(expected)} samples already "
              f"captured", flush=True)

    with captures_path.open("a", encoding="utf-8") as fh:
        for pid in prefix_ids:
            prompt_ids = prompt_ids_for(rows[pid])
            for i in range(a.n_samples):
                key = f"{pid}#{i}"
                if key in have:
                    continue
                status, body = _post(
                    a.api_base.rstrip("/") + "/completions",
                    {"model": a.model, "prompt": prompt_ids,
                     "max_tokens": a.max_tokens, "temperature": a.temperature,
                     "logprobs": 0, "return_token_ids": True},
                    a.timeout)
                if status != 200:
                    raise CanaryError(f"{key}: HTTP {status}: {body}")
                cap = _capture(pid, i, body, prompt_ids, expected[key], tokenizer)
                have[key] = cap
                fh.write(json.dumps(cap) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  {pid}: {a.n_samples} actions", flush=True)

    report = build_report(list(have.values()), prefix_ids, a.n_samples,
                          a.min_canonical_rate, a.min_diverse_prefix_fraction,
                          a.max_unparseable_rate)
    report["plan"] = plan
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print()
    for p in report["per_prefix"]:
        print(f"  {p['prefix_id']:>10}  byte {p['n_byte_distinct']}/{p['n_samples']}"
              f"  canonical {p['n_canonical_distinct']}/{p['n_samples']}"
              f"  formatting-only {p['formatting_only_variation']}"
              f"  unparseable {p['n_unparseable']}"
              f"  {'ok' if p['pass'] else 'COLLAPSED'}")
    print(f"\n  diverse prefixes   {report['diverse_prefix_fraction']:.2f} "
          f"(need {a.min_diverse_prefix_fraction})")
    print(f"  unparseable rate   {report['unparseable_rate']:.2f} "
          f"(max {a.max_unparseable_rate})")
    print(f"  captures           {captures_path}")

    if not report["pass"]:
        print(f"\nSTOP: {a.model} failed the diversity canary. Replay OPD would "
              f"sample near-identical actions, the teacher would score the same "
              f"thing every time, and the gradient would be flat -- a data "
              f"problem that would read as an objective failure. The remedy is "
              f"an earlier checkpoint, not more replay updates.", file=sys.stderr)
        return 1
    print(f"\nPASS: {a.model} still samples varied actions. This is not V2 "
          f"§7.1a -- that gate compares against frozen A0 and remains open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
