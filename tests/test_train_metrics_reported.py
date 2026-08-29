"""A paid update must report its own loss and grad_norm.

The one-episode canary printed `loss: None` / `grad_norm: None` while the
trainer had just logged `loss=0.5415`: the metrics live in the report's
"optimizer" dict, but callers receive the checkpoint `state`. Over a 10-update
run that silently disables the only per-update signal a stop condition could
watch, so a missing metric is a hard failure rather than a None.
"""

from __future__ import annotations

import inspect

import pytest

from vektori_trace.tau2 import live_train
from vektori_trace.tau2.live_train import LiveTrainError


def _src():
    return inspect.getsource(live_train.run_projected_train_stage)


class TestMetricsTravelWithState:
    def test_loss_and_grad_norm_are_copied_onto_state(self):
        src = _src()
        assert 'for k in ("loss", "grad_norm"):' in src
        assert "state[k] = v" in src

    def test_missing_metric_raises_rather_than_reporting_none(self):
        src = _src()
        i = src.index('for k in ("loss", "grad_norm")')
        window = src[i:i + 600]
        assert "raise LiveTrainError" in window
        assert "if v is None" in window

    def test_supervised_token_count_is_reported(self):
        assert '"global_supervised_tokens"' in _src()

    def test_the_guard_precedes_return(self):
        src = _src()
        assert src.index('for k in ("loss", "grad_norm")') < src.rindex("return state")


class TestGuardBehaviour:
    """Exercise the rule itself, independent of the Modal-bound stage."""

    @staticmethod
    def _apply(opt, state, index=0):
        for k in ("loss", "grad_norm"):
            v = (opt or {}).get(k)
            if v is None:
                raise LiveTrainError(f"trainer returned no {k!r} for update {index}")
            state[k] = v
        return state

    def test_real_metrics_land_on_state(self):
        st = self._apply({"loss": 0.5415, "grad_norm": 1.2}, {"adapter_hash": "x"})
        assert st["loss"] == 0.5415
        assert st["grad_norm"] == 1.2
        assert st["adapter_hash"] == "x"

    def test_zero_loss_is_valid_and_not_treated_as_missing(self):
        """0.0 is falsy; it must not be mistaken for absent."""
        st = self._apply({"loss": 0.0, "grad_norm": 0.0}, {})
        assert st["loss"] == 0.0
        assert st["grad_norm"] == 0.0

    @pytest.mark.parametrize("opt", [
        None, {}, {"loss": 0.5}, {"grad_norm": 1.0}, {"loss": None, "grad_norm": 1.0},
    ], ids=["none", "empty", "no-grad", "no-loss", "explicit-none"])
    def test_missing_metrics_refused(self, opt):
        with pytest.raises(LiveTrainError, match="returned no"):
            self._apply(opt, {})

    def test_the_canary_regression(self):
        """The exact shape observed: trainer had a loss, report showed None."""
        trainer_result = {"loss": 0.5415114539049634, "grad_norm": 1.0,
                          "n_examples": 11}
        st = self._apply(trainer_result, {"adapter_hash": "90e4511f26247f33"})
        assert st["loss"] == pytest.approx(0.5415114539049634)
        assert st["loss"] is not None and st["grad_norm"] is not None
