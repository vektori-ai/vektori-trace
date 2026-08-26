"""Swapping the served adapter, and refusing to believe a swap that did not happen.

The failure this guards is the quietest one in the run: the endpoint keeps
serving CK35 while the trainer moves on, so `log pi_old` comes from a policy
that never sampled the action. The loss stays finite and every other metric
looks correct.
"""

from __future__ import annotations

import json

import pytest

from vektori_trace.tau2.reopd_refresh import (
    RefreshError,
    load_adapter,
    probe_logprobs,
    refresh_policy,
    reload_serving_volume,
    served_models,
)


class FakeServer:
    """Minimal vLLM stand-in: tracks loaded adapters and their probe logprobs."""

    def __init__(self, *, has_route=True, models=("ck35",), lp=None,
                 register_on_load=True):
        self.has_route = has_route
        self.models = list(models)
        self.lp = lp or {"ck35": [-0.1, -0.2, -0.3]}
        self.register_on_load = register_on_load
        self.loaded = []
        self.unloaded = []

    def post(self, url, payload, timeout):
        if url.endswith("/reload-volume"):
            return 200, json.dumps({"ok": True, "seconds": 0.1})
        if url.endswith("/load_lora_adapter"):
            if not self.has_route:
                return 404, "Not Found"
            name = payload["lora_name"]
            self.loaded.append((name, payload["lora_path"]))
            if self.register_on_load:
                self.models.append(name)
                self.lp.setdefault(name, [-0.4, -0.5, -0.6])
            return 200, "{}"
        if url.endswith("/unload_lora_adapter"):
            self.unloaded.append(payload["lora_name"])
            return 200, "{}"
        if url.endswith("/completions"):
            m = payload["model"]
            return 200, json.dumps({"choices": [
                {"logprobs": {"token_logprobs": self.lp.get(m, [])}}]})
        return 500, "unexpected"

    def get(self, url, timeout):
        return {"data": [{"id": m} for m in self.models]}


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()
    import vektori_trace.tau2.reopd_refresh as R
    monkeypatch.setattr(R, "_post", srv.post)
    monkeypatch.setattr(R, "_get", srv.get)
    return srv


# --- the happy path -------------------------------------------------------


def test_refresh_loads_verifies_and_unloads(server):
    rep = refresh_policy(
        "http://x", new_name="ck35-u000", new_path="/adapters/run/u0/checkpoint",
        probe_prompt_ids=[1, 2, 3],
        previous_logprobs=[-0.1, -0.2, -0.3], previous_name="ck35",
    )
    assert server.loaded == [("ck35-u000", "/adapters/run/u0/checkpoint")]
    assert rep["name"] == "ck35-u000"
    assert rep["max_logprob_delta"] > 0
    assert server.unloaded == ["ck35"]          # stale adapter released
    assert rep["unload"]["ok"] is True


def test_refresh_reloads_volume_before_loading(server):
    rep = refresh_policy(
        "http://x", new_name="ck35-u000", new_path="/new/checkpoint",
        probe_prompt_ids=[1, 2, 3], previous_logprobs=[-0.1, -0.2, -0.3],
        previous_name="ck35", reload_url="http://control/reload-volume",
    )
    assert rep["volume_reload"]["ok"] is True


def test_failed_volume_reload_is_fatal(monkeypatch):
    import vektori_trace.tau2.reopd_refresh as R
    monkeypatch.setattr(R, "_post", lambda *a, **k: (500, "busy volume"))
    with pytest.raises(RefreshError, match="reload failed HTTP 500"):
        reload_serving_volume("http://control/reload-volume")


def test_first_refresh_needs_no_previous(server):
    rep = refresh_policy("http://x", new_name="ck35-u000",
                         new_path="/p", probe_prompt_ids=[1, 2, 3])
    assert "max_logprob_delta" not in rep
    assert rep["probe_logprobs"]


def test_served_models_lists_ids(server):
    assert served_models("http://x") == ["ck35"]


# --- refusals -------------------------------------------------------------


def test_identical_logprobs_are_fatal(server):
    """Two adapters one step apart must differ somewhere on a greedy probe."""
    server.lp["ck35-u000"] = [-0.1, -0.2, -0.3]     # same as previous
    with pytest.raises(RefreshError, match="identical to the previous policy"):
        refresh_policy("http://x", new_name="ck35-u000", new_path="/p",
                       probe_prompt_ids=[1, 2, 3],
                       previous_logprobs=[-0.1, -0.2, -0.3])


def test_adapter_absent_from_models_is_fatal(server):
    """vLLM resolves an unknown name against the base weights, silently."""
    server.register_on_load = False
    with pytest.raises(RefreshError, match="not in /v1/models"):
        refresh_policy("http://x", new_name="ck35-u000", new_path="/p",
                       probe_prompt_ids=[1, 2, 3])


def test_missing_route_names_the_env_flag(server):
    """A server started without runtime LoRA updating has no such route."""
    server.has_route = False
    with pytest.raises(RefreshError, match="VLLM_ALLOW_RUNTIME_LORA_UPDATING"):
        load_adapter("http://x", "n", "/p")


def test_failed_unload_is_recorded_without_invalidating_swap(server, monkeypatch):
    original = server.post

    def fail_unload(url, payload, timeout):
        if url.endswith("/unload_lora_adapter"):
            return 500, "nope"
        return original(url, payload, timeout)

    import vektori_trace.tau2.reopd_refresh as R
    monkeypatch.setattr(R, "_post", fail_unload)
    rep = refresh_policy(
        "http://x", new_name="ck35-u000", new_path="/p",
        probe_prompt_ids=[1, 2, 3],
        previous_logprobs=[-0.1, -0.2, -0.3], previous_name="ck35",
    )
    assert rep["unload"] == {"name": "ck35", "status": 500,
                              "ok": False, "body": "nope"}


def test_probe_without_logprobs_is_fatal(server):
    server.lp["ck35"] = []
    with pytest.raises(RefreshError, match="no logprobs"):
        probe_logprobs("http://x", "ck35", [1, 2, 3])


def test_probe_uses_greedy_sampling(server):
    """Temperature 0, or the fingerprint measures the sampler not the weights."""
    seen = {}

    def spy(url, payload, timeout):
        seen.update(payload)
        return server.post(url, payload, timeout)

    import vektori_trace.tau2.reopd_refresh as R
    R._post = spy
    probe_logprobs("http://x", "ck35", [1, 2, 3])
    assert seen["temperature"] == 0.0


# --- the serve module must enable the route ------------------------------


def test_serve_enables_runtime_lora_updating():
    src = open("vektori_trace/runtime/serve.py").read()
    assert '"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"' in src


def test_serve_exposes_volume_reload_control_plane():
    src = open("vektori_trace/runtime/serve.py").read()
    assert "def reload_volume(self)" in src
    assert "vol.reload()" in src
    assert "reload_volume_url=reload_url" in src


def test_driver_refreshes_before_sampling_and_skips_update_0():
    src = open("scripts/tau2_reopd_train.py").read()
    i_refresh = src.index("refresh_serving_policy(args, idx")
    i_sample = src.index("captures = sample_batch(")
    assert i_refresh < i_sample, "refresh must precede any paid sampling"
    assert "if idx > 0 and not u.reached(\"SAMPLED\")" in src
