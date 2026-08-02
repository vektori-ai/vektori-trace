#!/usr/bin/env python3
"""Prove a sweep's logs are complete and that they join. Exit 0 = trustworthy.

A pass@k run produces four artifacts, written by four different processes:

    passk_log.jsonl          one line per rollout          (passk.py)
    <job_dir>/token_captures.jsonl   one line per model call   (CaptureProxy)
    <job_dir>/...            harbor's ATIF trajectory      (harbor)
    gpu_log.jsonl            NVML samples                  (serve_student.py)

Each can fail independently and silently. An empty capture file looks exactly
like a rollout that made no model calls; a gap in the GPU log looks exactly like
a quiet GPU. This script is what distinguishes those, so that "the logging
worked" is a checked claim rather than an assumption made after the GPU is gone.

The join check is the one that matters. Four logs with plausible contents but
disagreeing clocks answer no question worth asking, and you only find that out
when you try to use them.

    # before the sweep, against a single throwaway completion:
    uv run python scripts/verify_run_logs.py --out ./vektori-out --preflight

    # after the sweep:
    uv run python scripts/verify_run_logs.py --out ./vektori-out --expect-rollouts 4
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vektori_trace.passk import PASSK_LOG_FILENAME
from vektori_trace.token_capture import CAPTURE_FILENAME

# A GPU sample every `interval` seconds; flag a hole larger than this multiple.
# Generous on purpose — one slow NVML call is not a fault, a minute of silence is.
GAP_TOLERANCE = 4.0


class Check:
    """A named assertion with its evidence, so a failure says what it saw."""

    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def add(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((ok, name, detail))
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.rows if not ok)

    def render(self) -> None:
        for ok, name, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line is what a killed process leaves behind.
            # Everything before it is still good; say so rather than discarding.
            print(f"  warning: unparseable line in {path}", file=sys.stderr)
    return out


def check_captures(chk: Check, capture_files: list[Path], tokenizer_id: str | None) -> None:
    """Ids and text present, and — if a tokenizer is available — in agreement."""
    total = 0
    missing_text = 0
    missing_timing = 0
    empty_ids = 0
    samples: list[tuple[list[int], str]] = []
    for f in capture_files:
        for rec in read_jsonl(f):
            total += 1
            ids = rec.get("token_ids") or []
            text = rec.get("text")
            if not ids:
                empty_ids += 1
            if text is None:
                missing_text += 1
            elif ids:
                samples.append((ids, text))
            if rec.get("latency_ms") is None or rec.get("request_started_at") is None:
                missing_timing += 1

    chk.add(total > 0, "token captures exist", f"{total} model calls recorded")
    chk.add(missing_text == 0, "every capture carries emitted text",
            f"{missing_text}/{total} missing")
    chk.add(missing_timing == 0, "every capture carries proxy timing",
            f"{missing_timing}/{total} missing latency/request_started_at")
    chk.add(empty_ids == 0, "no capture has empty token_ids",
            f"{empty_ids}/{total} empty")

    if tokenizer_id is None:
        print("  [skip] ids-decode-to-text (pass --tokenizer to enable)")
        return
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception as e:
        chk.add(False, "tokenizer loadable for decode check", f"{type(e).__name__}: {e}")
        return

    # The check that catches a silently broken capture: ids and text that
    # disagree are worse than no capture at all, because they look fine.
    bad = []
    for ids, text in samples[:50]:
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != text and decoded.strip() != text.strip():
            bad.append((decoded[:60], text[:60]))
    chk.add(not bad, "decoding token_ids reproduces text",
            "" if not bad else f"{len(bad)} mismatch, first: {bad[0]!r}")


def check_gpu_log(chk: Check, path: Path, interval: float,
                  window: tuple[float, float] | None) -> None:
    records = read_jsonl(path)
    chk.add(bool(records), "gpu log exists", f"{len(records)} samples")
    if not records:
        return

    errored = [r for r in records if r.get("error")]
    chk.add(len(errored) < len(records), "gpu log has real samples",
            f"{len(errored)}/{len(records)} are errors")

    good = [r for r in records if not r.get("error")]
    if good:
        utils = [r.get("gpu_util_pct") for r in good if r.get("gpu_util_pct") is not None]
        mems = [r.get("mem_used_mib") for r in good if r.get("mem_used_mib") is not None]
        chk.add(bool(utils), "gpu utilisation is recorded",
                f"max {max(utils)}%" if utils else "no gpu_util_pct field")
        # Weights alone are ~15.3 GiB; a served model reporting near-zero memory
        # means NVML is reading a different device than the one doing the work.
        chk.add(bool(mems) and max(mems) > 1024, "gpu memory looks like a loaded model",
                f"max {max(mems)} MiB" if mems else "no mem_used_mib field")

    stamps = sorted(r["logged_at"] for r in records if r.get("logged_at"))
    gaps = [
        (b - a) for a, b in pairwise(stamps)
        if (b - a) > interval * GAP_TOLERANCE
    ]
    chk.add(not gaps, "gpu log has no sampling holes",
            "" if not gaps else f"{len(gaps)} gaps, largest {max(gaps):.0f}s")

    if window and stamps:
        start, end = window
        chk.add(stamps[0] <= start and stamps[-1] >= end,
                "gpu log spans the whole sweep",
                f"log {stamps[0]:.0f}..{stamps[-1]:.0f} vs sweep {start:.0f}..{end:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the sweep's --out directory")
    ap.add_argument("--gpu-log", default=None,
                    help="gpu_log.jsonl (default: <out>/gpu_log.jsonl)")
    ap.add_argument("--gpu-log-interval", type=float, default=5.0,
                    help="interval serve_student.py sampled at, for gap detection")
    ap.add_argument("--expect-rollouts", type=int, default=None,
                    help="fail unless exactly this many rollouts are logged")
    ap.add_argument("--tokenizer", default=None,
                    help="HF id (e.g. Qwen/Qwen3-8B) to verify ids decode to text")
    ap.add_argument("--preflight", action="store_true",
                    help="only check captures + gpu log; no rollouts expected yet")
    args = ap.parse_args()

    out = Path(args.out)
    gpu_log = Path(args.gpu_log) if args.gpu_log else out / "gpu_log.jsonl"
    chk = Check()

    capture_files = sorted(out.rglob(CAPTURE_FILENAME))
    print(f"\nrun dir  {out}")
    print(f"captures {len(capture_files)} file(s)\n")

    if args.preflight:
        print("preflight — proving logging works before spending rollouts:")
        check_captures(chk, capture_files, args.tokenizer)
        check_gpu_log(chk, gpu_log, args.gpu_log_interval, window=None)
        chk.render()
        print()
        return 1 if chk.failed else 0

    rollouts = read_jsonl(out / PASSK_LOG_FILENAME)
    print("rollout log:")
    chk.add(bool(rollouts), "passk_log.jsonl exists", f"{len(rollouts)} rollouts")
    if args.expect_rollouts is not None:
        chk.add(len(rollouts) == args.expect_rollouts, "rollout count as expected",
                f"{len(rollouts)} vs {args.expect_rollouts}")

    job_dirs = [Path(r["jobs_dir"]) for r in rollouts if r.get("jobs_dir")]
    chk.add(len(set(job_dirs)) == len(job_dirs),
            "every rollout has its own job dir",
            f"{len(set(job_dirs))} distinct of {len(job_dirs)}")

    timed = [r for r in rollouts if r.get("elapsed_sec") is not None
             and r.get("started_at") is not None]
    chk.add(len(timed) == len(rollouts), "every rollout is timed",
            f"{len(timed)}/{len(rollouts)}")

    infra = [r for r in rollouts if r.get("infra_failure")]
    if infra:
        print(f"  note: {len(infra)}/{len(rollouts)} were infra failures — these are "
              "excluded from the pass rate, not counted as model failures")

    print("\ntoken captures:")
    check_captures(chk, capture_files, args.tokenizer)

    window = None
    if timed:
        window = (min(r["started_at"] for r in timed),
                  max(r["started_at"] + r["elapsed_sec"] for r in timed))
    print("\ngpu telemetry:")
    check_gpu_log(chk, gpu_log, args.gpu_log_interval, window)

    # The join. Four logs that each look fine but do not line up in time answer
    # nothing, and that only surfaces when you try to use them.
    print("\njoin (do the logs line up in time?):")
    gpu_stamps = sorted(r["logged_at"] for r in read_jsonl(gpu_log)
                        if r.get("logged_at"))
    uncovered = []
    for r in timed:
        start, end = r["started_at"], r["started_at"] + r["elapsed_sec"]
        if not any(start <= s <= end for s in gpu_stamps):
            uncovered.append(f"{r['task']}#{r.get('rollout_index')}")
    chk.add(not uncovered, "every rollout window contains gpu samples",
            "" if not uncovered else f"{len(uncovered)} uncovered: {uncovered[:3]}")

    caps_in_window = 0
    for f in capture_files:
        for rec in read_jsonl(f):
            ts = rec.get("request_started_at")
            if ts and any(r["started_at"] <= ts <= r["started_at"] + r["elapsed_sec"]
                          for r in timed):
                caps_in_window += 1
    chk.add(caps_in_window > 0 or not timed,
            "token captures fall inside rollout windows",
            f"{caps_in_window} captures matched a rollout")

    print()
    chk.render()
    print()
    if chk.failed:
        print(f"{chk.failed} check(s) failed — the run's logs are not trustworthy.")
        return 1
    print("all checks passed — logs are complete and joinable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
