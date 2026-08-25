#!/usr/bin/env python3
"""Stage the recovered retail policy onto the Modal volume.

The system policy lives in the simulation files on the EC2 box, which a Modal
container cannot see. It is also not optional: the student's prompt ids were
rendered with it, so scoring without it grades the action under a strictly
smaller context than the student saw.

This recovers it on the box, hashes it, and writes it where the ReOPD container
can read it. Render parity inside the run is what proves the staged copy is the
one the corpus was built with -- staging alone proves nothing.

    .venv/bin/python scripts/tau2_stage_policy_modal.py \\
        --artifacts /data/tau2/artifacts_16384 \\
        --simulations-dir /data/tau2/data/simulations
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import modal  # noqa: E402

from vektori_trace.tau2.c30_loader import recover_system_policy  # noqa: E402

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
POLICY_IN_VOLUME = "tau2/reopd/retail_policy.txt"

app = modal.App("tau2-stage-policy")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(volumes={VOLUME_MOUNT: vol}, timeout=300)
def put(text: str, expect_sha: str) -> dict:
    import hashlib as h
    import os

    dst = os.path.join(VOLUME_MOUNT, POLICY_IN_VOLUME)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text)
    vol.commit()

    # Re-read from the volume rather than trusting the write: a truncated
    # policy renders a plausible prefix and would only surface as a parity
    # failure much later, after a GPU is already allocated.
    got = open(dst, encoding="utf-8").read()
    sha = h.sha256(got.encode()).hexdigest()
    if sha != expect_sha:
        raise SystemExit(
            f"staged policy hash {sha[:16]} != source {expect_sha[:16]}; "
            "the write did not round-trip"
        )
    return {"path": f"{VOLUME_MOUNT}/{POLICY_IN_VOLUME}",
            "sha256": sha, "chars": len(got)}


@app.local_entrypoint()
def main(artifacts: str = "/data/tau2/artifacts_16384",
         simulations_dir: str = "/data/tau2/data/simulations"):
    policy, rep = recover_system_policy(artifacts, simulations_dir=simulations_dir)
    sha = hashlib.sha256(policy.encode()).hexdigest()
    print(f"recovered policy {sha[:16]} ({len(policy):,} chars, "
          f"{rep['n_tasks_agreeing']} tasks agree)")

    out = put.remote(policy, sha)
    print(f"staged -> {out['path']}")
    print(f"  sha256 {out['sha256'][:16]}  chars {out['chars']:,}")
    print(f"\nPass to the ReOPD runner:\n  --policy-file {out['path']}")
