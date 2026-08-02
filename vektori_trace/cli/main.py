"""Process entry point: configure logging, parse, dispatch."""

from __future__ import annotations

import logging
import os
import sys

from .parser import build_parser


def main(argv: list[str] | None = None) -> int:
    # Nothing configured logging, so every logger.info/warning in the mining
    # path went nowhere. `validate_pr` logs the one line that distinguishes
    # "this PR genuinely has no failing test" from "validation never ran"
    # ("parsed pre=0 post=0 tests"), and it was invisible for the whole
    # prefect smoke run — the skip histogram looked like a measurement.
    # VEKTORI_LOG_LEVEL overrides; INFO is the useful default for long mines.
    # Timestamps are not decoration here: a sweep's console log has to line up
    # against passk_log.jsonl, gpu_log.jsonl and the per-call token captures,
    # all of which carry wall-clock UTC. Without them "it stalled around here"
    # cannot be turned into "the GPU was idle for 90s waiting on the verifier".
    logging.basicConfig(
        level=os.environ.get("VEKTORI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
