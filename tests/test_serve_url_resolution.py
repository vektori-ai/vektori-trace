"""Endpoint URL resolution, without Modal, a GPU or a network.

2026-08-28: two endpoint starts logged "UP in 183s" and then 404'd every
request. vLLM was healthy; no URL had been obtained and the caller fabricated
`https://{app_name}--openai-compat.modal.run`, which is missing the workspace
prefix, the class segment and an ephemeral app's `-dev` suffix. A fabricated
address is worse than an error -- the endpoint looks up until the first real
request, and the failure reads as the model's.
"""

from __future__ import annotations

import pytest

from vektori_trace.runtime.serve import resolve_web_url

REAL = "https://workspace--vektori-trace-serve-vllmserver-openai-compat-dev.modal.run"


class _Getter:
    def __init__(self, url):
        self._url = url

    def get_web_url(self):
        return self._url


class _Attr:
    def __init__(self, url):
        self.web_url = url


class _Raises:
    def get_web_url(self):
        raise RuntimeError("modal client error")


class TestResolution:
    def test_class_function_getter(self):
        assert resolve_web_url(_Getter(REAL)) == REAL

    def test_falls_back_to_web_url_attribute(self):
        assert resolve_web_url(_Attr(REAL)) == REAL

    def test_first_candidate_wins(self):
        other = REAL.replace("openai-compat", "reload-volume")
        assert resolve_web_url(_Getter(REAL), _Getter(other)) == REAL

    def test_skips_none_candidates(self):
        assert resolve_web_url(None, None, _Getter(REAL)) == REAL

    def test_skips_a_candidate_that_raises(self):
        """A failing client call must not abort the search."""
        assert resolve_web_url(_Raises(), _Getter(REAL)) == REAL

    def test_bound_method_is_acceptable(self):
        """The class Function is tried first, but the instance is valid too.

        Whether `get_web_url()` exists on a bound method was never established;
        both probes returned a correct URL from it. The resolver does not
        depend on which is true.
        """
        assert resolve_web_url(None, _Getter(REAL)) == REAL


class TestFailsClosed:
    def test_no_candidate_raises(self):
        with pytest.raises(RuntimeError, match="Refusing to guess"):
            resolve_web_url()

    def test_all_none_raises(self):
        with pytest.raises(RuntimeError, match="Refusing to guess"):
            resolve_web_url(None, None)

    def test_all_empty_raises(self):
        with pytest.raises(RuntimeError, match="Refusing to guess"):
            resolve_web_url(_Getter(None), _Attr(None))

    def test_never_fabricates(self):
        """The exact 2026-08-28 failure: no URL available."""
        with pytest.raises(RuntimeError) as exc:
            resolve_web_url(_Raises(), _Attr(None), what="OpenAI-compatible endpoint")
        assert "modal.run" not in str(exc.value), "must not invent an address"
        assert "OpenAI-compatible endpoint" in str(exc.value)

    def test_non_https_refused(self):
        with pytest.raises(RuntimeError, match="not an https"):
            resolve_web_url(_Getter("http://insecure.example"))

    def test_garbage_refused(self):
        with pytest.raises(RuntimeError, match="not an https"):
            resolve_web_url(_Getter("not-a-url"))


def test_the_fabricated_url_shape_is_not_produced():
    """Pin the specific bad address so a regression is obvious."""
    fabricated = "https://vektori-trace-serve--openai-compat.modal.run"
    with pytest.raises(RuntimeError) as exc:
        resolve_web_url()
    assert fabricated not in str(exc.value)
