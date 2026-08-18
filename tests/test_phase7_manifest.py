"""The gate manifest — what gets frozen, and what must refuse to freeze.

The manifest is written once and never regenerated, so every defect in it is
permanent for the run it gates. These cover the two that would not look like
defects in the output: an op miscounted into the wrong category, and a
selection set that is short or correlated but still prints a sha.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

from vektori_trace.evaluate.phase7 import SELECTION_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "phase7_manifest", ROOT / "scripts" / "phase7_manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase7_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


man = _load()


def act(*keystrokes: str) -> str:
    return json.dumps(
        {
            "analysis": "a",
            "plan": "p",
            "commands": [{"keystrokes": k, "duration": 1.0} for k in keystrokes],
        }
    )


# --------------------------------------------------------------------------
# Op classification, one command at a time
# --------------------------------------------------------------------------


def test_an_edit_after_a_read_is_an_edit():
    """The joined-blob bug: `^`-anchored regexes without MULTILINE only saw
    command 0, so this action was classified read-only and its turn never
    became a `first_edit` prefix."""
    assert man._ops(act("ls -la\n", "sed -i 's/a/b/' x.py\n")) == (True, True, False)


def test_a_test_after_a_cd_is_a_test():
    assert man._ops(act("cd /workspace\n", "pytest -q\n"))[2] is True


def test_a_heredoc_write_is_an_edit_not_a_read():
    _read, edit, _ = man._ops(act("cat > x.py <<'EOF'\nprint(1)\nEOF\n"))
    assert edit is True


def test_unparseable_content_is_no_ops_not_a_crash():
    assert man._ops("not json") == (False, False, False)


# --------------------------------------------------------------------------
# The 45: distinct tasks, spread across repos, or no freeze
# --------------------------------------------------------------------------


def _cand(cat, repo, task, i=0):
    return {
        "category": cat,
        "repo": repo,
        "task": f"org__{task}",
        "prefix_chars": 100 + i,
        "line_no": i,
        "message_index": i,
    }


REPOS = ("click", "jinja", "anyio", "hatch", "prefect")


def _pool(cat, per_repo=3):
    return [
        _cand(cat, repo, f"{repo}-{n}", n)
        for repo in REPOS
        for n in range(per_repo)
    ]


def test_five_per_selection_category_lands_one_per_repo():
    cands = _pool("orientation")
    got, short = man.pick(cands, counts={"orientation": 5}, rng=random.Random(0))
    assert short == {}
    assert len(got) == 5
    assert sorted(e["repo"] for e in got) == sorted(REPOS)


def test_a_task_is_never_used_twice():
    """The old fallback (`pickable = fresh or bucket`) padded a short category
    by reusing a task at another turn — two correlated draws counted as two
    prefixes."""
    cands = [_cand("orientation", "click", "click-1", i) for i in range(4)]
    got, short = man.pick(cands, counts={"orientation": 5}, rng=random.Random(0))
    assert len(got) == 1
    assert short == {"orientation": 4}


def test_a_short_category_is_reported_not_padded():
    cands = _pool("post_compaction", per_repo=1)[:3]
    got, short = man.pick(cands, counts={"post_compaction": 5}, rng=random.Random(0))
    assert len(got) == 3
    assert short == {"post_compaction": 2}


def test_tripwire_categories_take_one_each():
    cands = _pool("orientation") + _pool("first_edit") + _pool("test_exec")
    counts = {"orientation": 5, "first_edit": 1, "test_exec": 1}
    got, short = man.pick(cands, counts=counts, rng=random.Random(0))
    assert short == {}
    by_cat = {c: sum(1 for e in got if e["category"] == c) for c in counts}
    assert by_cat == {"orientation": 5, "first_edit": 1, "test_exec": 1}
    # and no task appears twice across categories either
    assert len({e["task"] for e in got}) == len(got)


def test_categories_cover_exactly_the_selection_and_tripwire_split():
    assert set(SELECTION_CATEGORIES) <= set(man.CATEGORIES)
    assert len(SELECTION_CATEGORIES) * 5 * 3 == 45
