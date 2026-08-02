"""argparse `type=` validators and shared argument groups."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _min_gap_arg(value: str) -> float:
    """`--min-gap` as a threshold that can actually reject something.

    `float("nan")` parses fine, and every `gap < nan` comparison in
    `select_deficit` is then False — so a NaN threshold doesn't loosen the
    filter, it *removes* it, and inverted evidence (gap = -1.0) gets reported
    as a confident diagnosis again. That is the exact failure this flag was
    added to prevent, so it's rejected at the boundary rather than downstream.
    """
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            f"must be a finite number, got {value!r} (a non-finite threshold rejects nothing)"
        )
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be >= 0, got {parsed} (a negative gap is evidence against the hypothesis)"
        )
    return parsed


def _min_support_arg(value: str) -> int:
    """`--min-support` below 1 would admit capabilities with no evidence at all."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {parsed} (0 admits capabilities backed by no traces)"
        )
    return parsed


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _model_info_arg(value: str) -> dict[str, Any]:
    """`--model-info` as inline JSON or `@path.json`.

    Harbor refuses to start a `hosted_vllm/<name>` run without it, and the
    literal dict is long enough that pasting it into a shell is where the typo
    goes, so the file form exists too.
    """
    raw = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}") from None
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _add_endpoint_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Flags for pointing an arm at an OpenAI-compatible endpoint we host.

    `run_trial` has taken `api_base`/`model_info` since the Modal work, but no
    command exposed them, so every measurement was silently restricted to models
    a public provider already serves — which excludes the served candidate.
    """
    flag = f"--{prefix}api-base" if prefix else "--api-base"
    info = f"--{prefix}model-info" if prefix else "--model-info"
    dest = f"{prefix.replace('-', '_')}api_base"
    info_dest = f"{prefix.replace('-', '_')}model_info"
    parser.add_argument(
        flag,
        dest=dest,
        default=None,
        help="OpenAI-compatible base URL (e.g. a Modal vLLM /v1); reaches the agent via harbor --ak",
    )
    parser.add_argument(
        info,
        dest=info_dest,
        type=_model_info_arg,
        default=None,
        help="JSON (or @file.json) of litellm model_info; required by harbor for hosted_vllm/ models",
    )
