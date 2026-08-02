"""Subcommand registration order is pinned.

argparse lists subcommands in the order they were registered, so the order of
the `register_*(sub)` calls in `cli/parser.py` is what `vektori-trace --help`
prints. Those calls now span twelve command modules, which makes the order easy
to disturb by accident — reordering an import, or moving a register function to
a module that is called earlier.

Nothing else would catch that: every subcommand still works, the suite still
passes, only the help output silently reshuffles. Hence this.

`probe-teacher` sits after `ground` rather than next to the other teacher-side
commands, and `capture-proxy` trails both. That is how they were registered;
pinned as-is rather than tidied, so this test keeps documenting reality —
reordering them would change `--help`.
"""

from __future__ import annotations

from vektori_trace.cli import build_parser

EXPECTED_ORDER = [
    "diagnose",
    "select",
    "selftest",
    "check-env",
    "prove",
    "mine",
    "mine-commits",
    "replay",
    "train",
    "run-arms",
    "distill",
    "check-tokenizers",
    "build-bridge",
    "align-report",
    "passk",
    "import-gym",
    "route",
    "plan-b-arms",
    "resume-check",
    "bisect",
    "ground",
    "probe-teacher",
    "capture-proxy",
]


def _subcommands():
    parser = build_parser()
    for action in parser._actions:
        if getattr(action, "choices", None):
            return list(action.choices)
    raise AssertionError("no subparsers found on the root parser")


def test_subcommand_order_is_unchanged():
    assert _subcommands() == EXPECTED_ORDER


def test_every_subcommand_has_a_handler():
    """A register_* that forgets set_defaults(func=...) would make `main` crash
    with AttributeError at dispatch, long after parsing succeeded."""
    parser = build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    missing = [name for name, p in sub.choices.items() if p.get_default("func") is None]
    assert not missing, f"subcommands with no func= wired: {missing}"


def test_handler_names_match_their_subcommand():
    """Guards against a register_* block being copied and its func= left
    pointing at the command it was copied from."""
    parser = build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None))
    mismatches = {}
    for name, p in sub.choices.items():
        want = "cmd_" + name.replace("-", "_")
        got = p.get_default("func").__name__
        # A few handlers are named for what they do rather than for the flag.
        alias = {
            "check-env": "cmd_checkenv",
        }
        if got != alias.get(name, want):
            mismatches[name] = got
    assert not mismatches, f"subcommand -> handler mismatches: {mismatches}"
