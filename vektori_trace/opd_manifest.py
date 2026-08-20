"""Phase-0 §6.1 pin manifest — what a cross-tokenizer OPD run is allowed to claim.

`docs/OPD-MULTITURN-PLAN.md` §6.1 requires every artifact to be pinned by hash or
immutable revision *before* Harbor or GPU work, and §11 makes a stale adapter or
teacher revision a stop condition. This module is where those pins live, so a run
report carries them without anyone retyping a revision string.

Two deliberate design points:

- **Unknown is `None`, never a plausible default.** A pin nobody has filled in
  reads as `None` and `missing_pins()` names it. Defaulting to "whatever
  `tokenizer_check` says" would let a run report a pin it never checked.
- **The paper-code pin is verified against bytes on disk**, not trusted. The
  vendored files under `vendor/opd_paper/` are what the port in `chunk_opd.py`
  claims to implement; `verify_vendor_pins()` re-hashes them so an edit to
  "just fix a typo" in vendored code shows up as a failed pin.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .tokenizer_check import CROSS_STUDENT, CROSS_TEACHER

#: Repo of record for the loss, and the revision `chunk_opd.py` was ported from.
PAPER_ARXIV_ID = "2606.09456"
PAPER_REPO = "https://github.com/ivanniu/On-Policy-Distill"
PAPER_CODE_REVISION = "927a8264f2e303b7f82c2d331a58fd4240c8805a"

#: sha256 of each vendored upstream file, recorded when it was vendored.
VENDOR_SHA256 = {
    "reward_manager_opd.py": (
        "a24be172575818e3b44ac6287d68e9984a956c467f2e3baaa90d6c2ad365fe3e"
    ),
    "core_algos.py": (
        "dd0d93e9ff039b36fa2ca565d730c4f30690dc5b04078ac3dfb28ba2c442ccd2"
    ),
}

VENDOR_DIR = Path(__file__).parent / "vendor" / "opd_paper"


class PinError(RuntimeError):
    """A pinned artifact does not match what is actually on disk or configured."""


@dataclass
class OPDRunManifest:
    """Every §6.1 pin for one cross-tokenizer OPD run.

    Fields left `None` are *unpinned*, which `missing_pins()` reports and
    `require_complete()` refuses. Nothing here is inferred at runtime — a pin is
    something a human recorded, or it is absent.
    """

    # -- student (ck75) ------------------------------------------------------
    student_base_model: str | None = None
    #: Stage-B checkpoint-75; the adapter actually loaded, not the family.
    student_adapter_path: str | None = None
    student_adapter_sha256: str | None = None
    student_tokenizer: str = CROSS_STUDENT
    #: The serving-side chat template. Qwen3's think-wrapper behaviour is
    #: template-dependent (see CLAUDE.md), so the renderer is a pin, not a detail.
    student_renderer_sha256: str | None = None

    # -- teacher (DeepSeek) --------------------------------------------------
    teacher_model: str = CROSS_TEACHER
    teacher_tokenizer: str = CROSS_TEACHER
    #: Fireworks' served model id and the revision it reports back.
    fireworks_model_id: str | None = None
    fireworks_served_revision: str | None = None
    teacher_serving_precision: str | None = None
    teacher_renderer_sha256: str | None = None

    # -- method --------------------------------------------------------------
    paper_arxiv_id: str = PAPER_ARXIV_ID
    paper_code_revision: str = PAPER_CODE_REVISION

    # -- environment ---------------------------------------------------------
    harbor_revision: str | None = None
    terminus_revision: str | None = None
    task_corpus: str | None = None
    environment_image: str | None = None
    parser_revision: str | None = None

    # -- generation / loss knobs --------------------------------------------
    temperature: float | None = None
    top_p: float | None = None
    max_new_tokens: int | None = None
    max_context_tokens: int | None = None
    compaction_policy: str | None = None
    clip_eps: float | None = None
    large_chunk_threshold: int | None = None
    advantage_clamp: float | None = None

    # -- our own code --------------------------------------------------------
    vektori_trace_commit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    #: Pins that must be present before any paid or GPU phase. Deliberately does
    #: not include every field — `advantage_clamp=None` is a real, meaningful
    #: value (upstream's default: no clamp), and generation knobs are set per
    #: phase. These are the ones whose absence makes a result unattributable.
    REQUIRED = (
        "student_base_model",
        "student_adapter_path",
        "student_adapter_sha256",
        "fireworks_model_id",
        "harbor_revision",
        "task_corpus",
        "vektori_trace_commit",
    )

    def missing_pins(self) -> list[str]:
        """Required pins that are still `None` or empty."""
        return [f for f in self.REQUIRED if not getattr(self, f)]

    def require_complete(self) -> None:
        """Raise unless every required pin is filled in.

        Call this before spending money or GPU time, not after: an unpinned run
        produces numbers that cannot be attributed to a configuration, which
        §11 treats as a stop condition rather than a reporting gap.
        """
        missing = self.missing_pins()
        if missing:
            raise PinError(
                "unpinned artifacts, refusing to proceed: "
                + ", ".join(missing)
                + " (docs/OPD-MULTITURN-PLAN.md §6.1)"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["vendor_sha256"] = dict(VENDOR_SHA256)
        d["paper_repo"] = PAPER_REPO
        return d


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_vendor_pins(vendor_dir: Path | None = None) -> dict[str, str]:
    """Re-hash the vendored upstream sources against `VENDOR_SHA256`.

    Returns the observed hashes on success. Raises `PinError` naming every file
    that is missing or altered — the port in `chunk_opd.py` is only meaningful
    relative to these exact bytes.
    """
    root = vendor_dir or VENDOR_DIR
    observed: dict[str, str] = {}
    problems: list[str] = []

    for name, expected in VENDOR_SHA256.items():
        path = root / name
        if not path.is_file():
            problems.append(f"{name}: missing at {path}")
            continue
        got = sha256_file(path)
        observed[name] = got
        if got != expected:
            problems.append(f"{name}: expected {expected}, got {got}")

    if problems:
        raise PinError(
            "vendored paper code does not match its pin: "
            + "; ".join(problems)
            + f" (revision {PAPER_CODE_REVISION}; see {PAPER_REPO})"
        )
    return observed


def current_commit(repo_root: Path | None = None) -> str | None:
    """This checkout's git commit, or `None` outside a repo.

    Returned with a `-dirty` suffix when the tree has uncommitted changes, so a
    manifest cannot claim a clean commit for code that was edited after it.
    """
    root = repo_root or Path(__file__).parent.parent
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return f"{rev}-dirty" if dirty else rev


__all__ = [
    "PAPER_ARXIV_ID",
    "PAPER_CODE_REVISION",
    "PAPER_REPO",
    "VENDOR_DIR",
    "VENDOR_SHA256",
    "OPDRunManifest",
    "PinError",
    "current_commit",
    "sha256_file",
    "verify_vendor_pins",
]
