"""The argparse tree.

`build_parser` used to be one 844-line function holding every subcommand's
flags. Each block now lives in the command module that implements it, next to
its `cmd_*` handler, exposed as `register_*(sub)`.

The call order below is load-bearing: argparse lists subcommands in
registration order, so reordering these lines changes `--help` output.
"""

from __future__ import annotations

import argparse

from .commands import (
    capture,
    diagnose,
    distill,
    env,
    evaluate,
    ground,
    mine,
    replay,
    resume,
    route,
    select,
    teacher,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, separated from running it so argument parsing —
    notably the threshold validators — is testable without dispatching."""
    parser = argparse.ArgumentParser(prog="vektori-trace")
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose.register_diagnose(sub)
    select.register_select(sub)
    env.register_selftest(sub)
    env.register_checkenv(sub)
    env.register_prove(sub)
    mine.register_mine(sub)
    mine.register_mine_commits(sub)
    replay.register_replay(sub)
    train.register_train(sub)
    train.register_run_arms(sub)
    distill.register_distill(sub)
    teacher.register_check_tokenizers(sub)
    teacher.register_build_bridge(sub)
    teacher.register_align_report(sub)
    evaluate.register_passk(sub)
    evaluate.register_import_gym(sub)
    route.register_route(sub)
    route.register_plan_b_arms(sub)
    resume.register_resume_check(sub)
    resume.register_bisect(sub)
    ground.register_ground(sub)
    teacher.register_probe_teacher(sub)
    capture.register_capture_proxy(sub)

    return parser
