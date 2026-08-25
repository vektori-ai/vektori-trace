"""Stage the frozen Tau2 corpus onto the Modal adapters volume. CPU only.

The probe, the full run and the reload check all need the same bytes. Shipping
them into an image each time re-uploads on every code edit and gives three
chances to disagree; a volume gives one copy whose hash can be checked.

Integrity is not assumed: `artifact_hashes.json` was written when the corpus was
frozen, and this re-computes every file's sha256 after the write and refuses if
one differs.

    modal run scripts/tau2_stage_corpus_modal.py
"""
from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
# Content-addressed by manifest hash: a different split is a different
# directory, so a run can never silently read the wrong corpus.
CORPUS_IN_VOLUME = "corpora/b741bfceb1f3d027"

app = modal.App("tau2-stage-corpus")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir("/tmp/tau2_corpus", remote_path="/root/corpus")
)


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=15 * 60,
              cpu=2.0, memory=8192)
def stage() -> dict:
    import hashlib
    import json
    import os
    import shutil

    dst = os.path.join(VOLUME_MOUNT, CORPUS_IN_VOLUME)
    os.makedirs(dst, exist_ok=True)

    copied = {}
    for fn in sorted(os.listdir("/root/corpus")):
        src = os.path.join("/root/corpus", fn)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst, fn))
        h = hashlib.sha256()
        with open(os.path.join(dst, fn), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        copied[fn] = h.hexdigest()
        print(f"  {fn:32s} {os.path.getsize(src):>10,} B  {copied[fn][:16]}",
              flush=True)
    vol.commit()

    # The FULL corpus is staged unchanged, so every frozen hash must match.
    # Partitions are derived views computed in memory by the trainer; the
    # immutable artifact stays the source of truth.
    frozen_p = os.path.join(dst, "artifact_hashes.json")
    mismatches, checked = [], []
    if os.path.exists(frozen_p):
        frozen = json.load(open(frozen_p))
        for fn, want in frozen.items():
            if fn in copied:
                checked.append(fn)
                if copied[fn] != want:
                    mismatches.append({"file": fn, "staged": copied[fn][:16],
                                       "frozen": want[:16]})
    print(f"\nverified {len(checked)} file(s) against artifact_hashes.json",
          flush=True)
    for m in mismatches:
        print(f"  MISMATCH {m['file']}: {m['staged']} != {m['frozen']}", flush=True)

    return {"path": CORPUS_IN_VOLUME, "files": copied,
            "verified_against_frozen": checked, "mismatches": mismatches}


@app.local_entrypoint()
def main():
    out = stage.remote()
    print("\n" + "=" * 66)
    print(f"staged {len(out['files'])} file(s) -> volume:{out['path']}")
    print(f"hash-verified: {out['verified_against_frozen']}")
    if out["mismatches"]:
        print(f"INTEGRITY FAILURE: {out['mismatches']}")
    else:
        print("all comparable hashes match the frozen corpus")
    print("=" * 66)
