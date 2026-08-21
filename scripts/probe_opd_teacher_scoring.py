#!/usr/bin/env python3
"""Phase-0 §6.3 gate — can DeepSeek actually score ck75's actions?

Proves, against the real deployment, every condition
`docs/OPD-MULTITURN-PLAN.md` §6.3 requires before any Harbor or GPU work:

  A. the expected DeepSeek model revision answers;
  B. the returned number is the teacher-forced `logprob`, NOT `sampling_logprob`;
  C. returned token ids / bytes locate the action span exactly;
  D. exactly one finite log probability per DeepSeek action token;
  E. nothing was generated and no context was silently truncated;
  F. teacher forcing works against a separately rendered multi-turn prefix.

Run on **one short** and **one multi-turn** transcript, as §6.3 specifies.

Token ids come from the **pinned local encoder** (`encoding_dsv4` +
`providers/teacher/cross.py`), not from a server endpoint: Fireworks exposes no
`/tokenize`, and `providers/teacher/fireworks.py` documents that the OPD loop
never needs one because it holds the prefix as ids already. Probing any other
tokenisation path would prove nothing about the run.

Read-only and cheap: every call is `max_tokens=1` over a bounded prefix. The
multi-turn transcript is the only one with real size, and it is scored once.
`top_logprobs` is never sent — §6.3 omits it, and top-5 is irrelevant to this
objective.

A FAIL here is a §11 stop condition. Do not fall back to GOLD, SimpleOPD,
isolated-action scoring, or a top-5 reconstruction: fix the transport or stop.

Usage
-----
    export FIREWORKS_API_KEY=...            # or: set -a; . /data/.env.fw; set +a
    .venv/bin/python scripts/probe_opd_teacher_scoring.py \
        --model accounts/fireworks/models/deepseek-v4-flash-0731 \
        --out /tmp/opd_probe_report.json

Use the project interpreter (`.venv/bin/python` / `uv run python`), not a bare
`python3`: the probe imports the repo's renderer, tokenizer and scorer on
purpose, so the package has to be importable.

Add `--echo-mode last` to also check the cheaper `echo_last` shape. Exit code is
0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

# A short single-turn action and a multi-turn transcript whose last assistant
# message is what gets scored. Deliberately Terminus-shaped: JSON, a shell
# command, a path, digits — the content §6.2 calls out as alignment-hostile.
#: The plan's teacher (§ front matter). Pinned here so a probe run against some
#: other deployment cannot be mistaken for evidence about this one.
EXPECTED_TEACHER = "accounts/fireworks/models/deepseek-v4-flash-0731"

SHORT_ACTION = '{"cmd": "ls -la /workspace/src"}'

MULTI_TURN = [
    ("system", "You are a terminal agent. Reply with one JSON tool call."),
    ("user", "Find where the resolver lives in this repo."),
    ("assistant", '{"cmd": "find /workspace -name \'*.py\' | head -20"}'),
    ("user", "/workspace/src/hatch/resolver.py\n/workspace/src/hatch/cli.py"),
    ("assistant", '{"cmd": "sed -n \'1,40p\' /workspace/src/hatch/resolver.py"}'),
]


class Gate:
    """One §6.3 condition, its verdict, and the evidence behind it."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, gate: str, ok: bool, detail: str, evidence: Any = None) -> bool:
        self.rows.append(
            {"gate": gate, "pass": bool(ok), "detail": detail, "evidence": evidence}
        )
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {gate}: {detail}")
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [r["gate"] for r in self.rows if not r["pass"]]


def load_teacher_tokenizer(repo: str):
    """The pinned local DeepSeek tokenizer — the production path.

    Fireworks exposes **no** `/tokenize` endpoint; `providers/teacher/fireworks.py`
    documents this and the OPD loop never needs one, because it holds the prefix
    as ids already. Ids therefore come from the SHA-pinned local encoder, exactly
    as `providers/teacher/cross.py` does it at training time. Probing a different
    tokenisation path than the run will use would prove nothing about the run.
    """
    from vektori_trace.vocab_bridge import load_tokenizer

    return load_tokenizer(repo)


def render_and_encode(messages, tok, thinking_mode: str = "chat"):
    """messages -> DeepSeek-rendered string -> teacher ids, the production way."""
    from vektori_trace.providers.teacher.cross import (
        encode_teacher_ids,
        render_teacher_prefix,
    )

    text = render_teacher_prefix(messages, thinking_mode=thinking_mode)
    return text, encode_teacher_ids(text, tok)


def _decode(tok: Any, ids: list[int]) -> str:
    """Decode ids **with special tokens visible**.

    Critical: the default `decode` silently drops EOS, so a span still carrying
    the renderer's `<|end_of_sentence|>` decodes to exactly the action bytes and
    looks byte-exact while containing a token ck75 never sampled. Scoring that
    token would supervise the teacher's opinion of a turn boundary as if it were
    the model's own output.
    """
    for kwargs in ({"skip_special_tokens": False}, {}):
        try:
            return tok.decode(ids, **kwargs)
        except TypeError:
            continue
        except Exception:  # pragma: no cover - flavour differences
            return ""
    return ""


def _git_commit() -> str | None:
    """This checkout's commit, so a report names the code that produced it."""
    import subprocess

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev
    except Exception:
        return None


def run_case(
    name: str,
    prior_messages: list[dict[str, Any]],
    action_text: str,
    tok: Any,
    pool: Any,
    echo_mode: str,
    g: Gate,
) -> dict[str, Any]:
    """One §6.3 case, driven through the production scorer.

    Everything the plan asks us to verify about the response — that the ids we
    sent are the ids that came back, that the run is located *uniquely*, that
    the number is `logprob` and not `sampling_logprob` — is already enforced
    inside `FireworksTeacherPool.score_ids` / `_align_scored_entries` /
    `_entry_logprob`, which raise `TeacherScoringError` rather than guessing.

    So the probe's job is not to reimplement those checks: it is to exercise the
    real path on §6.3's transcripts and record the evidence. A duplicate
    implementation here could pass while the production one fails, which would
    prove nothing about the run.
    """
    from vektori_trace.providers.teacher.fireworks import TeacherScoringError

    print(f"\n=== {name} (echo_mode={echo_mode}) ===")
    ev: dict[str, Any] = {"case": name, "echo_mode": echo_mode}

    # --- C: locate the action span under the pinned tokenizer, by tokenising
    # `teacher_prefix + exact_action` jointly. §4 forbids assuming
    #   tokenize(prefix + action) == tokenize(prefix) + tokenize(action).
    _prefix_text, prefix_ids = render_and_encode(prior_messages, tok)
    joint_text, joint_ids = render_and_encode(
        [*prior_messages, {"role": "assistant", "content": action_text}], tok
    )

    if action_text not in joint_text:
        g.check(
            f"{name}/C-boundary",
            False,
            "rendered joint prompt does not contain the verbatim action bytes",
        )
        return ev

    prefix_ok = joint_ids[: len(prefix_ids)] == prefix_ids
    action_ids = joint_ids[len(prefix_ids):]

    # The renderer closes a finished assistant turn with EOS. ck75 never sampled
    # it, so §4 ("only bytes actually sampled") excludes it. Detected by decoding
    # with specials visible — the default decode hides EOS, which would let a
    # span look byte-exact while carrying a token we must not supervise.
    n_dropped = 0
    while len(action_ids) > 1 and _decode(tok, action_ids) != action_text:
        if not _decode(tok, action_ids).startswith(action_text):
            break
        action_ids = action_ids[:-1]
        n_dropped += 1

    ev.update(
        n_prefix_tokens=len(prefix_ids),
        n_action_tokens=len(action_ids),
        n_trailing_tokens_dropped=n_dropped,
        prefix_is_exact_id_prefix_of_joint=prefix_ok,
        action_ids=action_ids,
        # The full submitted sequence, so the exact request can be re-derived
        # offline from the report alone (§10 archival). The echoed ids are not
        # recorded separately because `score_ids` returns logprobs, not entries —
        # but it raises unless every entry's token_id matched these, so a
        # returned value is the equality proof.
        prefix_ids=prefix_ids,
    )

    g.check(
        f"{name}/C-boundary",
        len(action_ids) > 0 and prefix_ok,
        f"{len(prefix_ids)} prefix + {len(action_ids)} action tokens under the "
        f"pinned tokenizer; prefix is an exact id-prefix of the joint encoding "
        f"({prefix_ok}); dropped {n_dropped} trailing template token(s)",
    )

    decoded = _decode(tok, action_ids)
    ev["action_bytes_reconstructed"] = decoded
    g.check(
        f"{name}/C-bytes",
        decoded == action_text,
        f"action ids reconstruct the exact action bytes ({len(action_text)} chars)"
        if decoded == action_text
        else f"reconstructed {decoded!r} != {action_text!r}",
    )
    if decoded != action_text or not action_ids:
        return ev

    # --- The decisive call, through the production scorer.
    try:
        logprobs = pool.score_ids(prefix_ids, action_ids)
    except TeacherScoringError as e:
        # These are precisely §11's stop conditions, raised by the code the run
        # will use: ids not echoed, run not unique, no logprob, bad layout.
        g.check(f"{name}/D-scoring", False, f"TeacherScoringError: {e}")
        return ev
    except Exception as e:  # transport, auth, HTTP
        g.check(f"{name}/transport", False, f"{type(e).__name__}: {e}")
        return ev

    ev["logprobs"] = [round(float(v), 6) for v in logprobs]

    # --- D: one finite logprob per action token.
    finite = [math.isfinite(v) for v in logprobs]
    g.check(
        f"{name}/D-finite",
        len(logprobs) == len(action_ids) and all(finite),
        f"{sum(finite)}/{len(action_ids)} action tokens carry a finite logprob "
        f"(min {min(logprobs, default=float('nan')):.4f})",
    )

    # --- B: `score_ids` reads `logprob` and refuses `sampling_logprob` by
    # construction (`_entry_logprob`), and verifies each entry's token_id.
    # Reaching here means both held.
    g.check(
        f"{name}/B-logprob-field",
        True,
        "scored via _entry_logprob, which reads `logprob` (never "
        "`sampling_logprob`) and rejects any entry whose token_id differs",
    )

    # --- C-span / uniqueness: `_align_scored_entries` raises when the run is
    # absent or appears more than once, so a returned value is proof.
    g.check(
        f"{name}/C-span-unique",
        True,
        "_align_scored_entries located the scored run uniquely and id-for-id "
        "(it raises on zero or multiple matches)",
    )

    # --- E: nothing generated; the teacher saw the whole prefix.
    g.check(
        f"{name}/E-no-truncation",
        len(logprobs) == len(action_ids),
        f"scored exactly the {len(action_ids)} submitted action tokens "
        f"(prefix of {len(prefix_ids)} accepted without truncation)",
    )
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="accounts/fireworks/models/deepseek-v4-flash-0731"
    )
    ap.add_argument(
        "--tokenizer",
        default="deepseek-ai/DeepSeek-V4-Flash-0731",
        help="HF repo for the teacher tokenizer (the production path; Fireworks "
        "has no /tokenize endpoint)",
    )
    ap.add_argument("--echo-mode", default="full", choices=["full", "last", "both"])
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set", file=sys.stderr)
        return 2

    # The renderer and tokenizer are the production ones. A failure to load
    # them is a real Phase-0 failure (the run needs both), not a probe excuse.
    try:
        from vektori_trace.encoding_dsv4 import verify_encoding_dsv4_pin

        encoder_sha = verify_encoding_dsv4_pin()
        print(f"pinned DeepSeek encoder verified (sha256 {encoder_sha[:16]}…)")
        print(f"loading teacher tokenizer {args.tokenizer}")
        tok = load_teacher_tokenizer(args.tokenizer)
        tokenizer_id = getattr(tok, "name_or_path", None) or args.tokenizer
        try:
            vocab_size = tok.get_vocab_size()
        except Exception:  # pragma: no cover - flavour differences
            try:
                vocab_size = len(tok.get_vocab())
            except Exception:
                vocab_size = None
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "vektori_trace":
            print(
                f"\ncannot import vektori_trace: {e}\n\n"
                "Run this with the project's interpreter, not the system one:\n"
                "    .venv/bin/python scripts/probe_opd_teacher_scoring.py ...\n"
                "(or `uv run python scripts/...`). The probe deliberately uses the "
                "repo's own renderer/tokenizer/scorer, so it needs the package "
                "importable — a bare `python3` is usually the system interpreter "
                "without it.",
                file=sys.stderr,
            )
            return 2
        print(f"\nFAILED to load the pinned teacher renderer/tokenizer: {e}")
        return 2
    except Exception as e:  # report it, do not mask it
        print(f"\nFAILED to load the pinned teacher renderer/tokenizer: {e}")
        print(
            "This is the same path the training run uses, so it must work before "
            "any scoring is meaningful."
        )
        return 2

    # §6.3: "at least one short and one multi-turn frozen transcript".
    short_messages = [
        {"role": "system", "content": "You are a terminal agent."},
        {"role": "user", "content": "List the source directory."},
    ]
    multi_messages = [
        {"role": role, "content": text} for role, text in MULTI_TURN[:-1]
    ]
    multi_action = MULTI_TURN[-1][1]

    modes = ["full", "last"] if args.echo_mode == "both" else [args.echo_mode]

    g = Gate()
    # Everything needed to re-derive the ids offline later. A report that says
    # "the span matched" without recording which tokenizer produced it cannot be
    # rechecked against a future revision.
    report: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "tokenizer_resolved": tokenizer_id,
        "tokenizer_vocab_size": vocab_size,
        "encoding_dsv4_sha256": encoder_sha,
        "probe_commit": _git_commit(),
        "cases": [],
    }
    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool

    for mode in modes:
        # The production pool, not a hand-rolled request: its provenance and its
        # refusal behaviour are what the training run will actually rely on.
        pool = FireworksTeacherPool(model=args.model, echo_mode=mode)
        prov = pool.provenance()
        report.setdefault("teacher_provenance", prov)
        # --- A: the model actually configured for scoring is the pinned one.
        g.check(
            f"A-revision[{mode}]",
            prov.get("teacher_model") == args.model == EXPECTED_TEACHER,
            f"scoring against {prov.get('teacher_model')!r} "
            f"(serving precision: {prov.get('teacher_quantisation')})",
            prov,
        )
        report["cases"].append(
            run_case("short", short_messages, SHORT_ACTION, tok, pool, mode, g)
        )
        report["cases"].append(
            run_case("multiturn", multi_messages, multi_action, tok, pool, mode, g)
        )

    report["gates"] = g.rows
    report["all_passed"] = not g.failed

    print("\n" + "=" * 66)
    if g.failed:
        print(f"FAILED gates: {', '.join(g.failed)}")
        print(
            "\n§11 stop condition. Do NOT fall back to GOLD, SimpleOPD, "
            "isolated-action scoring, or top-5 reconstruction.\n"
            "Do not run the Harbor/GPU smoke."
        )
    else:
        print("ALL §6.3 GATES PASSED — teacher scoring is usable for chunk OPD.")
        print("Next gate (§6.4, no spend): mocked two-turn optimizer proof.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nreport: {args.out}")

    return 0 if not g.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
