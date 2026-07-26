"""RepoSpec URL handling — the shapes humans actually paste."""

from __future__ import annotations

import pytest

from vektori_trace.mining.spec import RepoSpec


@pytest.mark.parametrize(
    "url",
    [
        "psf/requests",
        "https://github.com/psf/requests",
        "https://github.com/psf/requests/",
        "https://github.com/psf/requests.git",
        "http://github.com/psf/requests",
        # Copied out of a browser address bar — the common case, and the one
        # that used to parse as owner='tree' / owner='pull'.
        "https://github.com/psf/requests/tree/main",
        "https://github.com/psf/requests/tree/v2.31.0/src",
        "https://github.com/psf/requests/pull/1234",
        "https://github.com/psf/requests/issues/42",
        "https://github.com/psf/requests/blob/main/setup.py",
        "  https://github.com/psf/requests/pull/1234  ",
    ],
)
def test_owner_name_survives_browser_url_tails(url: str) -> None:
    assert RepoSpec(url=url).owner_name == ("psf", "requests")


def test_url_is_canonicalized_to_the_clone_root() -> None:
    """`url` is handed straight to `git clone`, so the tail has to be gone
    from the stored value, not just from `owner_name`."""
    assert RepoSpec(url="https://github.com/psf/requests/pull/1234").url == (
        "https://github.com/psf/requests"
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:psf/requests.git", "git@github.com:psf/requests"),
        ("ssh://git@github.com/psf/requests", "git@github.com:psf/requests"),
    ],
)
def test_ssh_urls_keep_their_scheme(url: str, expected: str) -> None:
    spec = RepoSpec(url=url)
    assert spec.url == expected
    assert spec.owner_name == ("psf", "requests")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "requests",
        "https://github.com/psf",
        "https://gitlab.com/psf/requests",
    ],
)
def test_unparseable_urls_raise(url: str) -> None:
    with pytest.raises(ValueError):
        RepoSpec(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://github.com/psf/requests",
        "file:///etc/passwd",
        "svn+ssh://github.com/psf/requests",
        "git://github.com/psf/requests",
    ],
)
def test_unsupported_schemes_are_rejected_not_reinterpreted(url: str) -> None:
    """Any scheme we don't handle used to fall through to the bare
    `owner/name` branch, where splitting on "/" made the scheme the owner:
    `ftp://github.com/psf/requests` canonicalized to
    `https://github.com/ftp:/github.com` — and `url` is documented as safe to
    hand straight to `git clone`."""
    with pytest.raises(ValueError, match=r"only github\.com repos are supported"):
        RepoSpec(url=url)
