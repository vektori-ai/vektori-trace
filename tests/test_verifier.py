"""Differential tests: the in-container verifier vs. the library log parsers.

`vektori_trace/mining/verifier.py` is baked into every emitted task and runs
inside the container with nothing but the stdlib, so it carries its own
condensed copy of the four parsers in `vektori_trace/mining/log_parsers/*`.
The two copies are duplicated by necessity, agree today, and nothing enforced
it — so on drift the mining side computes one F2P set and the shipped verifier
computes another, and every task built in between silently scores 0.0.

These tests are that enforcement. They are also the only tests over the
parsers at all, so the fixtures double as their unit tests: each corpus entry
pins the real output shape of a runner, and `test_*_differential` asserts both
implementations read it identically.
"""

from __future__ import annotations

import random

from vektori_trace.mining import verifier as v
from vektori_trace.mining.log_parsers import (
    parse_cargo_test,
    parse_go_test,
    parse_jest,
    parse_logs,
    parse_pytest,
)

# --------------------------------------------------------------------------
# Fixture corpora — real runner output shapes, one entry per format quirk.
# --------------------------------------------------------------------------

PYTEST_LOGS = [
    "",
    # verbose progress
    """
tests/test_foo.py::test_a PASSED                                    [ 25%]
tests/test_foo.py::test_b FAILED                                    [ 50%]
tests/test_foo.py::test_c SKIPPED                                   [ 75%]
tests/test_bar.py::test_d ERROR                                     [100%]
""",
    # short summary
    """
=========================== short test summary info ============================
PASSED tests/test_foo.py::test_a
FAILED tests/test_foo.py::test_b - AssertionError: assert 1 == 2
SKIPPED [1] tests/test_foo.py:42: needs network
ERROR tests/test_bar.py::test_d
""",
    # both, summary last (last-write-wins)
    """
tests/test_foo.py::test_a PASSED                                    [ 50%]
tests/test_foo.py::test_b PASSED                                    [100%]
=========================== short test summary info ============================
FAILED tests/test_foo.py::test_b - ValueError: boom
""",
    # parametrized ids, class-scoped ids, dashes and brackets in names
    """
tests/test_p.py::test_x[1-2] PASSED
tests/test_p.py::TestKlass::test_y[a-b-c] FAILED
tests/test_p.py::test_z[case-with-dash] SKIPPED
""",
    # noise that must NOT be read as a test
    """
Some unrelated line PASSED something else
collecting ... collected 3 items
Requirement already satisfied: pytest
tests/test_ok.py::test_real PASSED
""",
    # a traceback body, no per-test statuses at all
    """
Traceback (most recent call last):
  File "conftest.py", line 1, in <module>
    import missing_module
ModuleNotFoundError: No module named 'missing_module'
""",
]

GO_LOGS = [
    "",
    """
=== RUN   TestAlpha
--- PASS: TestAlpha (0.00s)
=== RUN   TestBeta
    beta_test.go:14: mismatch
--- FAIL: TestBeta (0.01s)
=== RUN   TestGamma
--- SKIP: TestGamma (0.00s)
FAIL
exit status 1
""",
    # subtests are indented
    """
--- PASS: TestOuter (0.00s)
    --- PASS: TestOuter/sub_one (0.00s)
    --- FAIL: TestOuter/sub_two (0.00s)
""",
    "ok  \tgithub.com/x/y\t0.012s\n",
]

CARGO_LOGS = [
    "",
    """
running 4 tests
test tests::alpha ... ok
test tests::beta ... FAILED
test tests::gamma ... ignored
test module::nested::delta ... ok

failures:
    tests::beta

test result: FAILED. 2 passed; 1 failed; 1 ignored; 0 measured
""",
    "running 0 tests\n\ntest result: ok. 0 passed; 0 failed\n",
]

JEST_LOGS = [
    "",
    """
PASS src/foo.test.ts
  Widget
    ✓ renders (12 ms)
    ✕ explodes (3 ms)
    ○ skipped: pending case
FAIL src/bar.test.js
  ✓ standalone passes (1 ms)
  ✗ standalone fails

Tests:       2 failed, 2 passed, 1 skipped, 5 total
""",
    # nested describes, windows glyphs
    """
PASS test/nested.spec.tsx
  Outer
    Inner
      √ deep case (5 ms)
      × deep failure
""",
]

ALL_LOGS = PYTEST_LOGS + GO_LOGS + CARGO_LOGS + JEST_LOGS


# --------------------------------------------------------------------------
# Differentials
# --------------------------------------------------------------------------


def _differential(lib_fn, ver_fn, logs) -> None:
    for log in logs:
        assert lib_fn(log) == ver_fn(log), f"parser drift on log:\n{log!r}"


def test_pytest_differential() -> None:
    _differential(parse_pytest, v.parse_pytest, ALL_LOGS)


def test_go_differential() -> None:
    _differential(parse_go_test, v.parse_go_test, ALL_LOGS)


def test_cargo_differential() -> None:
    _differential(parse_cargo_test, v.parse_cargo_test, ALL_LOGS)


def test_jest_differential() -> None:
    _differential(parse_jest, v.parse_jest, ALL_LOGS)


def test_every_parser_against_every_corpus() -> None:
    """Cross-feed each parser the other runners' logs.

    Drift shows up in the garbage cases first: one copy tightens a regex, and
    the two disagree on output neither was written for — which is exactly the
    situation where mining and the shipped verifier diverge in the field.
    """
    for lib_fn, ver_fn in (
        (parse_pytest, v.parse_pytest),
        (parse_go_test, v.parse_go_test),
        (parse_cargo_test, v.parse_cargo_test),
        (parse_jest, v.parse_jest),
    ):
        _differential(lib_fn, ver_fn, ALL_LOGS)


def test_differential_on_shuffled_line_soup() -> None:
    """Interleave lines from every corpus, seeded, to hit orderings the
    hand-written fixtures don't cover (a status line inside a describe block,
    a summary before its progress line, and so on)."""
    lines = [ln for log in ALL_LOGS for ln in log.split("\n") if ln.strip()]
    rng = random.Random(1729)
    for _ in range(200):
        soup = "\n".join(rng.sample(lines, k=min(12, len(lines))))
        assert parse_pytest(soup) == v.parse_pytest(soup), soup
        assert parse_go_test(soup) == v.parse_go_test(soup), soup
        assert parse_cargo_test(soup) == v.parse_cargo_test(soup), soup
        assert parse_jest(soup) == v.parse_jest(soup), soup


def test_runner_dispatch_agrees() -> None:
    """The two dispatchers take different arguments — the library takes the
    command list, the verifier takes an already-detected runner name — so the
    detection step is compared separately from the routing step."""
    cases = {
        "pytest": ["python -m pytest -v"],
        "go": ["go test ./..."],
        "cargo": ["cargo test --all"],
        "jest": ["npx jest --ci"],
        "unknown": ["make check"],
    }
    for expected, cmds in cases.items():
        assert v._detect_runner(" ".join(cmds)) == expected
        for log in ALL_LOGS:
            assert parse_logs(cmds, log) == v.parse_logs(expected, log)


# --------------------------------------------------------------------------
# grade() — the reward the whole pipeline is thresholded on
# --------------------------------------------------------------------------


def test_grade_all_f2p_and_p2p_pass_is_reward_one() -> None:
    status = {"t_a": "PASSED", "t_b": "PASSED", "t_keep": "PASSED"}
    out = v.grade(["t_a", "t_b"], ["t_keep"], status)
    assert out["f2p_rate"] == 1.0
    assert out["p2p_rate"] == 1.0
    assert out["reward"] == 1.0
    assert out["resolved"] is True


def test_grade_partial_f2p_is_not_resolved() -> None:
    status = {"t_a": "PASSED", "t_b": "FAILED", "t_keep": "PASSED"}
    out = v.grade(["t_a", "t_b"], ["t_keep"], status)
    assert out["f2p_rate"] == 0.5
    assert out["reward"] < 1.0
    assert out["resolved"] is False


def test_grade_regression_in_p2p_zeroes_the_reward() -> None:
    """A fix that breaks a previously-passing test is not a fix."""
    status = {"t_a": "PASSED", "t_keep": "FAILED"}
    out = v.grade(["t_a"], ["t_keep"], status)
    assert out["f2p_rate"] == 1.0
    assert out["p2p_rate"] == 0.0
    assert out["reward"] == 0.0
    assert out["resolved"] is False


def test_grade_missing_test_is_not_a_pass() -> None:
    """A test absent from the log never ran; absence must not read as PASSED."""
    out = v.grade(["t_a"], [], {})
    assert out["f2p_rate"] == 0.0
    assert out["reward"] == 0.0
    assert out["resolved"] is False


def test_grade_no_p2p_gives_p2p_rate_one() -> None:
    out = v.grade(["t_a"], [], {"t_a": "PASSED"})
    assert out["p2p_rate"] == 1.0
    assert out["reward"] == 1.0
