"""Alignment guards on the train/serve logprob probe.

This probe exists to catch tokenizer misalignment between the trainer and the
serving engine. Its own failure mode is therefore the interesting one: a check
that *passes* while the two sides scored different things. Two ways that
happened, both now fatal:

- different token counts, previously WARNed and compared on a truncated prefix;
- equal counts but different token ids, previously not checked at all, because
  SGLang's token ids were discarded and only lengths were compared.

The heavy path (`trainer_logprobs`) needs torch and a real checkpoint, so these
drive `main()` with both scorers stubbed.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "probe_logprob_agreement", REPO / "scripts" / "probe_logprob_agreement.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["probe_logprob_agreement"] = _mod
_spec.loader.exec_module(_mod)


class _FakeTok:
    """Tokenizer stub: text -> ids, and a decode that never raises."""

    def __init__(self, ids):
        self._ids = ids

    def __call__(self, _text):
        return {"input_ids": list(self._ids)}

    def decode(self, ids):
        return f"<{','.join(str(i) for i in ids)}>"


@pytest.fixture
def stub(monkeypatch):
    """Drive main() with both scorers replaced. Returns a config dict to edit."""
    cfg = {
        "prompt_ids": [1, 2, 3, 4, 5],
        # Scored ids are prompt_ids[1:]: the first token is unconditioned.
        "trainer": ([-0.10, -0.20, -0.30, -0.40], [2, 3, 4, 5]),
        "sglang": ([-0.10, -0.20, -0.30, -0.40], [2, 3, 4, 5]),
    }

    monkeypatch.setattr(
        _mod, "load_tokenizer", lambda *_a, **_k: _FakeTok(cfg["prompt_ids"]),
        raising=False)
    # The module resolves its tokenizer through transformers; patch whichever
    # symbol it actually uses.
    import types
    fake_tf = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *_a, **_k: _FakeTok(cfg["prompt_ids"])))
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    monkeypatch.setattr(_mod, "trainer_logprobs",
                        lambda *_a, **_k: cfg["trainer"])
    monkeypatch.setattr(_mod, "sglang_logprobs",
                        lambda *_a, **_k: cfg["sglang"])
    return cfg


def _run(tmp_path, *extra):
    out = tmp_path / "probe.json"
    argv = ["probe_logprob_agreement.py", "--model", "fake/model",
            "--json-out", str(out), *extra]
    sys.argv = argv
    return _mod.main(), out


def test_matching_ids_and_close_logprobs_pass(stub, tmp_path):
    rc, out = _run(tmp_path)
    assert rc == 0
    assert json.loads(out.read_text())["pass"] is True


def test_equal_length_but_different_token_ids_is_fatal(stub, tmp_path):
    """The gap this test exists for.

    Both sides return 4 logprobs, so every length check passes -- but they
    scored different tokens, so the logprobs describe different distributions
    and any agreement between them is coincidental. Before the fix SGLang's ids
    were discarded, so this scored a clean PASS.
    """
    stub["sglang"] = ([-0.10, -0.20, -0.30, -0.40], [2, 3, 99, 5])
    rc, _ = _run(tmp_path)
    assert rc == 2


def test_identical_logprobs_with_different_ids_still_fails(stub, tmp_path):
    """The worst case: perfect numeric agreement over the wrong tokens.

    Deltas are all exactly zero, so mean and max clear any threshold. Only the
    ids reveal that the two sides never scored the same sequence.
    """
    stub["sglang"] = ([-0.10, -0.20, -0.30, -0.40], [7, 8, 9, 10])
    rc, _ = _run(tmp_path)
    assert rc == 2


def test_length_mismatch_is_fatal(stub, tmp_path):
    """Previously a WARN followed by comparison on a truncated prefix."""
    stub["sglang"] = ([-0.10, -0.20, -0.30], [2, 3, 4])
    rc, _ = _run(tmp_path)
    assert rc == 2


def test_length_mismatch_is_not_rescued_by_close_values(stub, tmp_path):
    stub["sglang"] = ([-0.10, -0.20, -0.30, -0.40, -0.50], [2, 3, 4, 5, 6])
    rc, _ = _run(tmp_path)
    assert rc == 2


def test_diverging_logprobs_on_matching_ids_fail_the_verdict(stub, tmp_path):
    """Same tokens, genuinely different distributions -> a real FAIL, not fatal."""
    stub["sglang"] = ([-5.0, -6.0, -7.0, -8.0], [2, 3, 4, 5])
    rc, out = _run(tmp_path)
    assert rc == 1                     # verdict failure, not an alignment abort
    assert json.loads(out.read_text())["pass"] is False


def test_scored_token_ids_are_recorded(stub, tmp_path):
    """A later reader must be able to check the sequence without a rerun."""
    _, out = _run(tmp_path)
    assert json.loads(out.read_text())["scored_token_ids"] == [2, 3, 4, 5]


def test_the_invalid_kl_metric_is_gone(stub, tmp_path):
    """It was a signed realized-token statistic, not a KL divergence.

    Terms could be negative, so a divergent pair could sit under an upper
    threshold and pass.
    """
    _, out = _run(tmp_path)
    assert "kl" not in json.loads(out.read_text())


def test_max_kl_flag_still_parses_but_is_ignored(stub, tmp_path):
    """Kept so an existing invocation does not break."""
    rc, out = _run(tmp_path, "--max-kl", "0.0")
    assert rc == 0                     # would have failed if still gating
    assert json.loads(out.read_text())["pass"] is True
