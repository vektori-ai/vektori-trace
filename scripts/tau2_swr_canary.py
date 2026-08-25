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

    python scripts/tau2_swr_canary.py --dry-run
    python scripts/tau2_swr_canary.py --api-base "$STUDENT_API_BASE"
"""

from __future__ import annotations

import argparse
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

# The frozen C30 prefix manifest (scripts/tau2_freeze_c30_prefixes.py).
# Pinned, not discovered: a canary run against a different prefix pool produces
# numbers that cannot be compared with anything else.
FROZEN_PREFIX_MANIFEST_HASH = "8e78c7b96161d024"

DEFAULT_MODEL = "Qwen3-4B-ck35"
# `A_warm` as decided 2026-08-25 (handoff §7 step 2): ck35, taken as a tiebreak
# against ck70 after both measured equal on three selection tasks.
POLICY_VERSION = "tau2-a_warm-ck35"


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
                        temperature: float, prompt_ids: list[int]) -> str:
    """What an existing capture is valid for.

    Keying on `prefix#sample` alone is not enough -- those ids are positional
    and get reused the moment the pool or the model changes, so a stale file
    would be silently accepted for a different run. Bind the capture to the
    policy, the sampling temperature and the exact prompt it was conditioned on.
    """
    h = hashlib.sha256()
    for part in (prefix_id, str(sample_index), model, f"{temperature:.6f}"):
        h.update(part.encode())
        h.update(b"\0")
    h.update(json.dumps(prompt_ids).encode())
    return h.hexdigest()[:32]


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
             prompt_ids: list[int], fingerprint: str) -> dict:
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

    action_bytes = text.encode("utf-8")
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
        # Byte-exact, so a later scorer never has to re-derive bytes from text.
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

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    captures_path = out / "captures.jsonl"

    expected = {
        f"{pid}#{i}": capture_fingerprint(
            pid, i, a.model, a.temperature, rows[pid]["input_ids"])
        for pid in prefix_ids for i in range(a.n_samples)
    }
    have = load_captures(captures_path, expected)
    if have:
        print(f"  resuming: {len(have)}/{len(expected)} samples already "
              f"captured", flush=True)

    with captures_path.open("a", encoding="utf-8") as fh:
        for pid in prefix_ids:
            prompt_ids = rows[pid]["input_ids"]
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
                cap = _capture(pid, i, body, prompt_ids, expected[key])
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
