"""Filesystem cache for bootstrap results.

Layout under cache_dir (defaults to ./envs):

  <cache_dir>/<owner>__<name>/<short_commit>[__<opts_hash>]/
    bootstrap.json         # BootstrapResult, serialized
    Dockerfile             # reconstructed from agent commands
    transcript.jsonl       # full agent trace

The opts_hash is appended when the spec deviates from defaults along axes
that change image identity (platform, base_image, user_dockerfile,
image_registry). Without it, a prior `linux/amd64` build would silently
satisfy a later `--platform linux/arm64` request from the same SHA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from vektori_trace.mining.bootstrap.spec import BootstrapResult, LanguageHint

logger = logging.getLogger(__name__)


def _options_hash(opts: dict[str, Any] | None) -> str:
    """Stable 8-char hash of spec options that affect image identity.

    None / empty → returns "" so existing single-config caches keep their
    short-commit-only path (backwards compatible with v0.2 caches).
    """
    if not opts:
        return ""
    # Sort + JSON-serialize for stability; ignore None values so a
    # default-everywhere spec hashes the same as one with explicit defaults.
    filtered = {k: v for k, v in opts.items() if v is not None}
    if not filtered:
        return ""
    payload = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def cache_key(
    repo: str,
    ref: str,
    cache_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> Path:
    """Return the cache directory for a (repo, ref, options) tuple."""
    owner, _, name = repo.partition("/")
    if not name:
        name = owner
        owner = "_"
    short = (ref or "head")[:12]
    opts_hash = _options_hash(options)
    slug = f"{short}__{opts_hash}" if opts_hash else short
    return cache_dir / f"{owner}__{name}" / slug


def save(
    result: BootstrapResult, cache_dir: Path, *, options: dict[str, Any] | None = None
) -> Path:
    """Write a BootstrapResult to its cache slot. Returns the dir."""
    slot = cache_key(result.repo, result.ref, cache_dir, options=options)
    slot.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    # Pathlib + enum aren't JSON-serializable by default
    payload["language"] = result.language.value
    if result.transcript_path is not None:
        payload["transcript_path"] = str(result.transcript_path)
    (slot / "bootstrap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if result.dockerfile_reconstruction:
        (slot / "Dockerfile").write_text(result.dockerfile_reconstruction, encoding="utf-8")

    return slot


def load(
    repo: str,
    ref: str,
    cache_dir: Path,
    *,
    options: dict[str, Any] | None = None,
) -> BootstrapResult | None:
    """Return a cached BootstrapResult, or None if not present / unparseable."""
    slot = cache_key(repo, ref, cache_dir, options=options)
    f = slot / "bootstrap.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cache load failed for %s: %s", f, exc)
        return None

    # Coerce LanguageHint back from string
    if isinstance(data.get("language"), str):
        try:
            data["language"] = LanguageHint(data["language"])
        except ValueError:
            data["language"] = LanguageHint.UNKNOWN

    if isinstance(data.get("transcript_path"), str):
        data["transcript_path"] = Path(data["transcript_path"])

    # Filter to known fields so future BootstrapResult additions don't break cached loads
    if is_dataclass(BootstrapResult):
        known = {f.name for f in fields(BootstrapResult)}
        data = {k: v for k, v in data.items() if k in known}
    result = BootstrapResult(**data)

    # The JSON is not evidence the image still exists. `docker image prune`,
    # a disk cleanup or a new machine all leave the metadata behind, and the
    # tag is local-only — so the sandbox tries to *pull* it and every task
    # fails with "pull access denied … may require 'docker login'", which
    # reads as a registry auth problem rather than a missing local image.
    # Treat a vanished image as a cache miss so it is simply rebuilt.
    if not result.pushed_to_registry and not _image_exists_locally(result.image_tag):
        logger.warning(
            "cache slot %s names image %s, which is not present locally — rebuilding",
            slot,
            result.image_tag,
        )
        return None
    return result


def _image_exists_locally(image_tag: str) -> bool:
    """True when `docker image inspect` can resolve the tag on this host."""
    if not image_tag:
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Docker being unreachable is not the same as the image being absent.
        # Assume present and let the caller fail with the real Docker error
        # rather than silently triggering an expensive rebuild.
        logger.debug("docker image inspect failed for %s: %s", image_tag, exc)
        return True
    return proc.returncode == 0
