"""Yield bugs in the text/path handling that silently cost tasks.

Two distinct failures, both invisible in the output:
  - the leak stripper deleted any `word#number` text, so `C#9` / `RFC#7231`
    were cut out of the problem statement mid-sentence
  - the diff-header regex required non-whitespace paths, so a PR touching a
    file whose name contains a space parsed as zero changed files and was
    dropped as `empty_source_patch`
"""

from __future__ import annotations

import pytest

from vektori_trace.mining.pipeline import (
    _files_in_patch,
    _path_is_test,
    _strip_info_leak,
    split_patch_and_test_patch,
)

# ----- leak stripper ---------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Targeting C#9 records here.",
        "Per RFC#7231 the header is optional.",
        "Broken since ES2020#3 landed.",
        "The F#5 build also fails.",
        "Regression from Python3#2 style naming.",
    ],
)
def test_identifiers_shaped_like_refs_survive(text):
    """`<short-word>#<number>` is a version/spec identifier, not an issue ref."""
    assert _strip_info_leak(text) == text


def test_real_cross_repo_issue_refs_are_still_stripped():
    assert "gorilla#739" not in _strip_info_leak("see also gorilla#739 for context")
    assert "#1234" not in _strip_info_leak("Fix the parser (#1234)")


# ----- test-path classification ----------------------------------------------


def test_conftest_is_test_infrastructure():
    assert _path_is_test("conftest.py") is True
    assert _path_is_test("src/pkg/conftest.py") is True
    # Already covered by the test-directory rule, and must stay covered.
    assert _path_is_test("tests/conftest.py") is True


def test_go_test_files_are_tests():
    assert _path_is_test("pkg/foo/bar_test.go") is True


def test_source_files_are_not_tests():
    assert _path_is_test("src/app.py") is False
    assert _path_is_test("docs/testing.md") is False


# ----- diff paths with spaces ------------------------------------------------

SPACED_DIFF = """diff --git a/src/my module.py b/src/my module.py
--- a/src/my module.py
+++ b/src/my module.py
@@ -1 +1 @@
-a
+b
diff --git a/tests/test my thing.py b/tests/test my thing.py
--- a/tests/test my thing.py
+++ b/tests/test my thing.py
@@ -0,0 +1,2 @@
+def test_x():
+    pass
"""

QUOTED_DIFF = '''diff --git "a/src/na\\303\\257ve.py" "b/src/na\\303\\257ve.py"
--- "a/src/na\\303\\257ve.py"
+++ "b/src/na\\303\\257ve.py"
@@ -1 +1 @@
-a
+b
'''


def test_spaced_paths_are_extracted():
    assert _files_in_patch(SPACED_DIFF) == ["src/my module.py", "tests/test my thing.py"]


def test_spaced_paths_still_split_source_from_tests():
    patch, test_patch = split_patch_and_test_patch(SPACED_DIFF)
    assert "src/my module.py" in patch
    assert "tests/test my thing.py" in test_patch
    assert "tests/test my thing.py" not in patch


def test_quoted_path_is_unquoted():
    assert _files_in_patch(QUOTED_DIFF) == ["src/naïve.py"]


def test_ordinary_paths_are_unaffected():
    plain = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    assert _files_in_patch(plain) == ["src/app.py"]


def test_spaced_test_paths_are_shell_quoted_in_the_eval_script():
    """Surviving the parser is only half of it — the path also has to reach
    `git checkout` as one argument."""
    from vektori_trace.mining.pipeline import build_eval_script

    _, test_patch = split_patch_and_test_patch(SPACED_DIFF)
    script = build_eval_script("a" * 40, test_patch, ["pytest"])
    assert "'tests/test my thing.py'" in script


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
