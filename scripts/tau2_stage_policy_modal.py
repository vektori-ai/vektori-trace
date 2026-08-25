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
SCHEDULE_IN_VOLUME = "tau2/reopd/schedule.json"

#: The frozen 32x16 stream both continuation arms must consume. Pinned by hash
#: so a regenerated schedule cannot be staged over it silently -- that would
#: give the two arms different streams while both still reported 32 updates.
EXPECT_SCHEDULE_HASH = "24c0aa5395d69772"

app = modal.App("tau2-stage-policy")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(volumes={VOLUME_MOUNT: vol}, timeout=300)
def put(text: str, expect_sha: str, schedule_json: str = "") -> dict:
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
    out = {"policy_path": f"{VOLUME_MOUNT}/{POLICY_IN_VOLUME}",
           "policy_sha256": sha, "policy_chars": len(got)}

    if schedule_json:
        import json as j

        sdst = os.path.join(VOLUME_MOUNT, SCHEDULE_IN_VOLUME)
        with open(sdst, "w", encoding="utf-8") as fh:
            fh.write(schedule_json)
        vol.commit()

        # Re-read off the volume, not the string we just wrote: the point is to
        # prove another container will see this exact schedule.
        back = j.load(open(sdst, encoding="utf-8"))
        got_hash = back.get("schedule_hash")
        if got_hash != EXPECT_SCHEDULE_HASH:
            raise SystemExit(
                f"staged schedule hash {got_hash} != expected "
                f"{EXPECT_SCHEDULE_HASH}. Both arms must consume the identical "
                "stream; staging a different one invalidates the comparison."
            )
        out.update({
            "schedule_path": f"{VOLUME_MOUNT}/{SCHEDULE_IN_VOLUME}",
            "schedule_hash": got_hash,
            "n_updates": back.get("n_updates"),
            "n_per_update": back.get("n_per_update"),
            "n_exposures": back.get("n_exposures"),
        })
    return out


@app.local_entrypoint()
def main(artifacts: str = "/data/tau2/artifacts_16384",
         simulations_dir: str = "/data/tau2/data/simulations",
         schedule: str = ""):
    policy, rep = recover_system_policy(artifacts, simulations_dir=simulations_dir)
    sha = hashlib.sha256(policy.encode()).hexdigest()
    print(f"recovered policy {sha[:16]} ({len(policy):,} chars, "
          f"{rep['n_tasks_agreeing']} tasks agree)")

    # Build the schedule from the frozen prefix order rather than reading a
    # file that may or may not exist: it is a pure function of the manifest,
    # and its hash is pinned, so deriving it here cannot drift.
    sched_json = ""
    if schedule:
        sched_json = Path(schedule).read_text()
    else:
        from vektori_trace.tau2.c30_loader import load_c30_prefixes
        from vektori_trace.tau2.reopd_schedule import build_schedule, describe
        import json as j

        prefixes, _ = load_c30_prefixes(artifacts, system_policy=policy)
        sched = build_schedule([p.prefix_id for p in prefixes])
        print(f"schedule  {describe(sched)}")
        sched_json = j.dumps(sched, indent=1)

    out = put.remote(policy, sha, sched_json)
    print(f"\nstaged policy   -> {out['policy_path']}")
    print(f"  sha256 {out['policy_sha256'][:16]}  chars {out['policy_chars']:,}")
    if out.get("schedule_path"):
        print(f"staged schedule -> {out['schedule_path']}")
        print(f"  hash {out['schedule_hash']}  "
              f"{out['n_updates']}x{out['n_per_update']} = "
              f"{out['n_exposures']} exposures")
    print("\nReOPD:\n"
          f"  --policy-file {out['policy_path']}\n"
          "continued SFT:\n"
          f"  --schedule {out.get('schedule_path')}")
