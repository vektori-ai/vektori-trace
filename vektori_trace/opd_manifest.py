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
  files under `opd_reference/` are what the port in `chunk_opd.py` claims to
  implement; `verify_reference_pins()` re-hashes them so an edit to "just fix a
  typo" in reference code shows up as a failed pin.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chunk_opd import DEFAULT_CLIP_EPS, DEFAULT_LARGE_CHUNK_THRESHOLD
from .tokenizer_check import CROSS_STUDENT, CROSS_TEACHER

#: Repo of record for the loss, and the revision `chunk_opd.py` was ported from.
PAPER_ARXIV_ID = "2606.09456"
PAPER_REPO = "https://github.com/ivanniu/On-Policy-Distill"
PAPER_CODE_REVISION = "927a8264f2e303b7f82c2d331a58fd4240c8805a"

#: sha256 of each upstream reference file, recorded when it was copied in.
REFERENCE_SHA256 = {
    "reward_manager_opd.py": (
        "a24be172575818e3b44ac6287d68e9984a956c467f2e3baaa90d6c2ad365fe3e"
    ),
    "core_algos.py": (
        "dd0d93e9ff039b36fa2ca565d730c4f30690dc5b04078ac3dfb28ba2c442ccd2"
    ),
}

REFERENCE_DIR = Path(__file__).parent / "opd_reference"


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
    #: Base model *revision*, not just the family — a re-tagged base silently
    #: changes what the adapter is applied to.
    student_base_model: str | None = None
    student_base_revision: str | None = None
    #: Stage-B checkpoint-75; the adapter directory actually loaded.
    #: Known location: vektori-trace-adapters / sft/qwen3-14b-stage-b-lora/
    #: checkpoint-75 (docs/SOL-HANDOFF.md). Its hashes are not in the repo and
    #: must be read off the box/volume before the first paid call.
    student_adapter_path: str | None = None
    #: sha256 of `adapter_model.safetensors` specifically — the weights, not the
    #: directory. Two checkpoints can share a config and differ only here.
    student_adapter_sha256: str | None = None
    #: sha256 of `adapter_config.json` — rank, alpha, target modules. A config
    #: change with identical weights still changes what gets loaded.
    student_adapter_config_sha256: str | None = None
    student_tokenizer: str = CROSS_STUDENT
    #: sha256 of the tokenizer + chat template as served. CLAUDE.md records that
    #: Qwen3's template decides where the think-wrapper lands, so this is a
    #: correctness pin, not bookkeeping.
    student_tokenizer_sha256: str | None = None
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
    #: Pinned at the reference implementation's default (verl `clip_ratio`).
    #: Recorded rather than left to the code default so a run report states it.
    #: Near-inert on a one-step smoke — behaviour and current policy start
    #: identical, so every ratio is 1 — and only bites once updates go stale or
    #: repeat over epochs. Do not tune before the probe/smoke produces evidence.
    clip_eps: float = DEFAULT_CLIP_EPS
    large_chunk_threshold: int = DEFAULT_LARGE_CHUNK_THRESHOLD
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
        "student_adapter_config_sha256",
        "student_tokenizer_sha256",
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
        d["reference_sha256"] = dict(REFERENCE_SHA256)
        d["paper_repo"] = PAPER_REPO
        return d


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_reference_pins(reference_dir: Path | None = None) -> dict[str, str]:
    """Re-hash the upstream reference sources against `REFERENCE_SHA256`.

    Returns the observed hashes on success. Raises `PinError` naming every file
    that is missing or altered — the port in `chunk_opd.py` is only meaningful
    relative to these exact bytes.
    """
    root = reference_dir or REFERENCE_DIR
    observed: dict[str, str] = {}
    problems: list[str] = []

    for name, expected in REFERENCE_SHA256.items():
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
            "reference paper code does not match its pin: "
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




def hash_adapter_dir(adapter_path: Path) -> dict[str, str | None]:
    """The two adapter pins §6.1 wants, read off disk.

    Weights and config are hashed separately and deliberately: two Stage-B
    checkpoints can share `adapter_config.json` byte-for-byte and differ only in
    `adapter_model.safetensors`, so a single directory-level hash would let a
    wrong checkpoint pass as the right one. Returns `None` for a file that is
    absent rather than raising — `missing_pins()` is what refuses, and it names
    the field.
    """
    root = Path(adapter_path)
    out: dict[str, str | None] = {
        "student_adapter_sha256": None,
        "student_adapter_config_sha256": None,
    }
    weights = root / "adapter_model.safetensors"
    if weights.is_file():
        out["student_adapter_sha256"] = sha256_file(weights)
    config = root / "adapter_config.json"
    if config.is_file():
        out["student_adapter_config_sha256"] = sha256_file(config)
    return out


def hash_tokenizer_dir(tokenizer_path: Path) -> str | None:
    """One hash over the tokenizer files that decide rendering.

    `tokenizer.json` alone is not enough: CLAUDE.md records that Qwen3's chat
    template governs where the empty-think wrapper lands, and the template
    ships in `tokenizer_config.json`. A tokenizer pin that ignored the template
    would call two materially different renderers identical. Files are hashed in
    a fixed order, each with its name, so the digest is stable and a missing
    file cannot collide with an empty one.
    """
    root = Path(tokenizer_path)
    if not root.is_dir():
        return None
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    )
    h = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        found = True
        h.update(name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest() if found else None


def build_run_manifest(
    *,
    base_model: str,
    adapter_path: str | None,
    task_corpus: str,
    fireworks_model_id: str,
    student_tokenizer: str | None = None,
    tokenizer_dir: str | Path | None = None,
    harbor_revision: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    teacher_serving_precision: str | None = None,
    extra: dict[str, Any] | None = None,
) -> OPDRunManifest:
    """Assemble a §6.1 manifest, deriving only what can be read from disk.

    The split matters. Hashes of files this process actually loads are *derived*
    — deriving them is strictly better than trusting a human to retype them, and
    they cannot drift from what ran. Everything else (Harbor revision, serving
    precision) is *recorded*: this process cannot observe it, so an inferred
    value would be a guess wearing a pin's clothing.

    `vektori_trace_commit` comes from `current_commit()`, which suffixes
    `-dirty` on an unclean tree, so a manifest cannot claim a clean commit for
    code that was edited after it.

    This does not call `require_complete()`. The caller decides when to refuse,
    because a dry run legitimately builds an incomplete manifest to *report*
    what is unpinned; only a paid run must refuse.
    """
    adapter_hashes: dict[str, str | None] = {
        "student_adapter_sha256": None,
        "student_adapter_config_sha256": None,
    }
    if adapter_path:
        adapter_hashes = hash_adapter_dir(Path(adapter_path))

    tok_dir = tokenizer_dir if tokenizer_dir is not None else adapter_path
    tok_sha = hash_tokenizer_dir(Path(tok_dir)) if tok_dir else None

    return OPDRunManifest(
        student_base_model=base_model,
        student_adapter_path=str(adapter_path) if adapter_path else None,
        student_adapter_sha256=adapter_hashes["student_adapter_sha256"],
        student_adapter_config_sha256=adapter_hashes["student_adapter_config_sha256"],
        student_tokenizer=student_tokenizer or CROSS_STUDENT,
        student_tokenizer_sha256=tok_sha,
        fireworks_model_id=fireworks_model_id,
        teacher_serving_precision=teacher_serving_precision,
        harbor_revision=harbor_revision,
        task_corpus=task_corpus,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        vektori_trace_commit=current_commit(),
        extra=dict(extra or {}),
    )

__all__ = [
    "build_run_manifest",
    "hash_adapter_dir",
    "hash_tokenizer_dir",
    "PAPER_ARXIV_ID",
    "PAPER_CODE_REVISION",
    "PAPER_REPO",
    "REFERENCE_DIR",
    "REFERENCE_SHA256",
    "OPDRunManifest",
    "PinError",
    "current_commit",
    "sha256_file",
    "verify_reference_pins",
]
