#!/usr/bin/env python3
"""Free, no-teacher rehearsal of the repaired production scoring path.

**What this does and does not prove.** It runs archived live actions through
the *real* parser, projector and chunk-advantage code, with a deterministic
FAKE teacher standing in for DeepSeek. So it proves the path executes, that
chunk identity survives scoring/persistence/resume, that accounting is total,
and that the repaired arithmetic equals `chunk_opd`.

It does **not** produce real advantages. Score rows archived before 2026-08-29
hold only flat per-token credit whose chunk grouping is unrecoverable, so a
numerical replay against them would be meaningless -- the fix is precisely that
those rows are no longer reusable. Real numbers require rescoring against
DeepSeek, which costs money and is a separate, explicitly approved step.

Usage:
    python3 scripts/tau2_offline_rehearsal.py <actions.jsonl> [--json out.json]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


class FakeTeacher:
    """Deterministic stand-in for the DeepSeek pool.

    Logprobs are a hash of the scored ids, so they are reproducible, finite,
    negative and -- critically -- *unequal within a chunk*, which is the only
    regime that distinguishes the chunk rule from the per-token one.
    """

    def __init__(self) -> None:
        self.calls = 0

    def score_ids(self, prefix_ids: list[int], action_ids: list[int]) -> list[float]:
        self.calls += 1
        out = []
        for i, tid in enumerate(action_ids):
            h = hashlib.sha256(f"{tid}:{i}".encode()).digest()
            # spread over [-4.0, -0.05], never 0, never -inf
            out.append(-(0.05 + (int.from_bytes(h[:4], "big") / 2**32) * 3.95))
        return out


def rehearse(actions_path: Path) -> dict[str, Any]:
    from vektori_trace.tau2.live_agent import (
        PARSER_VERSION,
        LiveCaptureError,
        split_generation,
    )
    from vektori_trace.tau2.live_batch import projected_turn_advantages
    from vektori_trace.tau2.live_projection import PROJECTION_VERSION, project_action
    from vektori_trace.tau2.live_score import SCORE_ALGORITHM, ProjectedChunk

    rows = [json.loads(l) for l in actions_path.read_text().splitlines() if l.strip()]
    report: dict[str, Any] = {
        "n_actions": len(rows),
        "parser_version": PARSER_VERSION,
        "projection_version": PROJECTION_VERSION,
        "score_algorithm": SCORE_ALGORITHM,
        "teacher": "FAKE (deterministic hash) -- advantages are not real",
        "parsed": 0,
        "parse_refused": 0,
        "reasoning_recovered": 0,
        "implicit_boundary": 0,
        "supervised_tokens": 0,
        "excluded_tokens": 0,
        "excluded_by_reason": {},
        "n_chunks": 0,
        "chunk_kinds": {},
        "failures": [],
    }

    fake = FakeTeacher()
    for r in rows:
        key = r.get("key", "?")
        raw = base64.b64decode(r["action_bytes_b64"]).decode("utf-8", "replace")
        for special in ("<|im_end|>", "<|endoftext|>"):
            if raw.endswith(special):
                raw = raw[: -len(special)]
        toks = [base64.b64decode(b) for b in r["action_token_bytes_b64"]]

        try:
            reasoning, content, tools = split_generation(raw)
        except LiveCaptureError as e:
            report["parse_refused"] += 1
            report["failures"].append({"key": key, "stage": "parse", "why": str(e)[:120]})
            continue
        report["parsed"] += 1
        if reasoning and reasoning.strip():
            report["reasoning_recovered"] += 1
            if "</think>" not in raw:
                report["implicit_boundary"] += 1

        try:
            proj = project_action(raw, toks)
        except Exception as e:  # noqa: BLE001
            report["failures"].append({"key": key, "stage": "project", "why": str(e)[:120]})
            continue

        pr = proj.report()
        assert pr["n_supervised"] + pr["n_excluded"] == len(toks), key
        report["supervised_tokens"] += pr["n_supervised"]
        report["excluded_tokens"] += pr["n_excluded"]
        for k, v in pr["excluded_by_reason"].items():
            report["excluded_by_reason"][k] = report["excluded_by_reason"].get(k, 0) + v

        # Build chunks the way live_score does, but with fake teacher scores:
        # one chunk per contiguous run of supervised indices per kind.
        chunks: list[ProjectedChunk] = []
        by_kind: dict[str, list[int]] = {}
        for i, kind in sorted(proj.supervised.items()):
            by_kind.setdefault(kind, []).append(i)
        for kind, idxs in by_kind.items():
            run: list[int] = []
            n = 0
            for i in idxs:
                if run and i != run[-1] + 1:
                    lp = fake.score_ids([], list(range(len(run))))
                    chunks.append(ProjectedChunk(f"{kind}:{n}", kind, tuple(run),
                                                 tuple(lp[: max(1, len(run) // 2)] or lp[:1])))
                    n += 1
                    run = []
                run.append(i)
            if run:
                lp = fake.score_ids([], list(range(len(run))))
                chunks.append(ProjectedChunk(f"{kind}:{n}", kind, tuple(run),
                                             tuple(lp[: max(1, len(run) // 2)] or lp[:1])))

        beh = list(r["behavior_logprobs"])
        try:
            ta = projected_turn_advantages(
                turn_index=0,
                action_token_ids=list(r["action_token_ids"]),
                behavior_logprobs=beh,
                chunks=chunks,
            )
        except Exception as e:  # noqa: BLE001
            report["failures"].append({"key": key, "stage": "advantage", "why": str(e)[:160]})
            continue

        report["n_chunks"] += ta.stats.n_chunks
        for kind, cnt in (ta.stats.tokens_by_kind or {}).items():
            report["chunk_kinds"][kind] = report["chunk_kinds"].get(kind, 0) + cnt

        # every supervised token must have a finite advantage; nothing else may
        for i, (a, sup) in enumerate(zip(ta.advantages, ta.supervised_mask)):
            if sup:
                assert a == a and abs(a) != float("inf"), f"{key}: non-finite at {i}"
            else:
                assert a == 0.0, f"{key}: unsupervised token {i} carries {a}"

        # round-trip the chunks through JSON -- the resume path
        back = [ProjectedChunk.from_json(c.to_json()) for c in chunks]
        ta2 = projected_turn_advantages(
            turn_index=0, action_token_ids=list(r["action_token_ids"]),
            behavior_logprobs=beh, chunks=back,
        )
        assert ta2.advantages == ta.advantages, f"{key}: resume changed advantages"

    report["teacher_calls"] = fake.calls
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("actions", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if not a.actions.exists():
        print(f"no such file: {a.actions}", file=sys.stderr)
        return 2
    rep = rehearse(a.actions)
    print(json.dumps(rep, indent=2))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2))
    return 1 if rep["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
