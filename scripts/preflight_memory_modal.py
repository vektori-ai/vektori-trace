"""Run the §14 memory preflight on a Modal GPU.

Mirrors `sft_train_modal.py`'s image and volume setup rather than importing it:
Modal re-imports this module inside the container where the local package is
not installed. Same pins, because a memory number measured on a different torch
is a different number.

    modal run scripts/preflight_memory_modal.py --gpu A100-80GB

A100-80GB first, deliberately. The projection lands near an L40S's 48 GiB and
the components that decide it — CUDA context, cuBLAS workspaces, attention
temporaries, allocator fragmentation, AdamW state — are exactly the ones a
projection cannot see. The run records the nvidia-smi peak so the L40S question
is answered by measurement rather than re-estimated.
"""
from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

ADAPTER_IN_VOLUME = "sft/qwen3-14b-stage-b-lora/checkpoint-75"

app = modal.App("vektori-trace-opd-preflight")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.5.3",
        "peft==0.19.1",
        "accelerate==1.14.0",
    )
    .env({"HF_HOME": HF_CACHE_MOUNT})
    .add_local_dir(
        "vektori_trace", remote_path="/root/vektori_trace", ignore=["__pycache__"]
    )
    .add_local_file("scripts/preflight_memory.py", "/root/scripts/preflight_memory.py")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60,
    max_containers=1,
)
def preflight(
    prefix_tokens: int = 31591,
    action_tokens: int = 9216,
    no_checkpointing: bool = False,
) -> dict:
    import json
    import subprocess
    import sys
    from pathlib import Path

    adapter = Path(VOLUME_MOUNT) / ADAPTER_IN_VOLUME
    if not adapter.is_dir():
        raise SystemExit(f"adapter not on the volume: {adapter}")

    cmd = [
        sys.executable, "/root/scripts/preflight_memory.py",
        "--adapter-path", str(adapter),
        "--device", "cuda",
        "--prefix-tokens", str(prefix_tokens),
        "--action-tokens", str(action_tokens),
        "--out", "/tmp/preflight_memory.json",
        "--gpu-log", "/tmp/preflight_gpu.jsonl",
    ]
    if no_checkpointing:
        cmd.append("--no-checkpointing")

    # Stream child output live rather than capturing it: an OOM that kills the
    # process would otherwise take the whole log with it.
    proc = subprocess.Popen(
        cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()

    out: dict = {"returncode": rc}
    for key, path in (
        ("report", "/tmp/preflight_memory.json"),
        ("gpu_log", "/tmp/preflight_gpu.jsonl"),
    ):
        p = Path(path)
        if not p.exists():
            continue
        if key == "report":
            out[key] = json.loads(p.read_text())
        else:
            out[key] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


@app.local_entrypoint()
def main(
    prefix_tokens: int = 31591,
    action_tokens: int = 9216,
    no_checkpointing: bool = False,
):
    import json
    from pathlib import Path

    res = preflight.remote(
        prefix_tokens=prefix_tokens,
        action_tokens=action_tokens,
        no_checkpointing=no_checkpointing,
    )
    Path("preflight_result.json").write_text(json.dumps(res, indent=2))
    rep = res.get("report") or {}
    print("\n=== PREFLIGHT ===")
    print(f"returncode        {res['returncode']}")
    print(f"gpu               {rep.get('gpu')} ({rep.get('gpu_total_gib')} GiB)")
    print(f"tokens            {rep.get('total_tokens')}")
    print(f"torch peak alloc  {rep.get('peak_gib')} GiB")
    print(f"torch peak resv   {rep.get('peak_reserved_gib')} GiB")
    smi = (rep.get("gpu_sampler") or {}).get("peak_mem_used_gib")
    print(f"nvidia-smi peak   {smi} GiB")
    print(f"verdict           {rep.get('l40s_verdict')}")
    print("written: preflight_result.json")
