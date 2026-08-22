"""Run the replay OPD optimizer step on a Modal A100 (plan §14, stage 3c).

The training host is not the sampling host, deliberately (§14): sampling holds
a serving GPU, scoring is paid HTTP, and the optimizer step needs 67 GiB that
neither of the others do. This wrapper is what makes that split runnable —
ck75's adapter lives on the `vektori-trace-adapters` volume, which only a Modal
container mounts, while the captures and teacher scores live on the EC2 box.

The batch is ~3.4 MB, so it ships into the image rather than through a volume.

    modal run scripts/replay_train_modal.py --max-trace-share 0.45

Per-example progress is streamed as it happens and also written to
`train_progress.jsonl`, which is returned with the result: a crash at example
30 leaves the 29 rows that preceded it rather than an empty report.
"""
from __future__ import annotations

import os

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

ADAPTER_IN_VOLUME = "sft/qwen3-14b-stage-b-lora/checkpoint-75"
#: Where the trained adapter lands. On the volume so it survives the container.
V_REPLAY_IN_VOLUME = os.environ.get("V_REPLAY_PATH", "opd/v_replay")

#: Where captures.jsonl / teacher_scores.jsonl / replay_run.json live on the
#: launching host. Defaults to the box's run directory so training launches
#: from the machine that already holds them; override with REPLAY_BATCH_DIR.
BATCH_DIR = os.environ.get("REPLAY_BATCH_DIR", "/data/replay-v1")

app = modal.App("vektori-trace-replay-train")
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
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        # The run at 07:55 logged "memory allocation failed with OOM ... while
        # trying to allocate 2.82 GB (free: 2.73 GB)" and then recovered: a
        # fragmentation stall, not exhaustion. Expandable segments let a
        # segment grow rather than stranding freed blocks at the wrong size.
        # A mitigation, not extra memory — the batch already fits.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("vektori_trace", remote_path="/root/vektori_trace",
                   ignore=["__pycache__"])
    .add_local_dir(BATCH_DIR, remote_path="/root/batch")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=90 * 60,
    max_containers=1,
)
def train(
    max_trace_share: float = 0.35,
    learning_rate: float = 1e-5,
    n_samples_per_prefix: int = 4,
    max_new_tokens: int = 9216,
) -> dict:
    import base64
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root")

    import transformers

    from vektori_trace.replay_opd import (
        SampledAction,
        run_replay_chunk_opd,
        validate_sample_set,
    )
    from vektori_trace.replay_sample import token_bytes_from_ids
    from vektori_trace.replay_select import ReplayPrefix
    from vektori_trace.replay_train import (
        ReplayTrainConfig,
        build_optimizer,
        load_v0_for_training,
        make_optimizer_step,
    )

    def log(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    batch_dir = Path("/root/batch")
    out = Path("/tmp/replay-train")
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "train_progress.jsonl"

    adapter = Path(VOLUME_MOUNT) / ADAPTER_IN_VOLUME
    if not adapter.is_dir():
        raise SystemExit(f"adapter not on the volume: {adapter}")

    tok = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-14B", trust_remote_code=True
    )

    # -- rebuild the batch from what sampling and scoring already produced ----
    actions = []
    for line in (batch_dir / "captures.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = r["action_token_ids"]
        actions.append(
            SampledAction(
                prefix_id=r["prefix_id"],
                sample_index=r["sample_index"],
                action_bytes=base64.b64decode(r["action_bytes_b64"]),
                action_token_ids=ids,
                action_token_bytes=token_bytes_from_ids(tok, ids),
                behavior_logprobs=r["behavior_logprobs"],
                policy_version=r["policy_version"],
                prompt_token_ids=r.get("prompt_token_ids") or None,
                termination_reason=r.get("termination_reason"),
            )
        )

    scored = {}
    for line in (batch_dir / "teacher_scores.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        scored[r["key"]] = (
            [base64.b64decode(b) for b in r["teacher_token_bytes_b64"]],
            r["teacher_logprobs"],
        )

    prior = json.loads((batch_dir / "replay_run.json").read_text())
    prefixes = [
        ReplayPrefix(
            task=p["task"],
            trace_id=p["prefix_id"].rsplit("@", 1)[0],
            step_index=p["step"],
            prefix_turns=[],
            post_compaction=p.get("post_compaction", False),
        )
        for p in prior["prefixes"]
    ]
    log(f"{len(prefixes)} prefixes, {len(actions)} actions, {len(scored)} scores")

    # Completeness before the GPU is touched: a partial set silently changes
    # the loss denominator.
    validate_sample_set(
        prefixes, actions, n_samples_per_prefix=n_samples_per_prefix
    )
    log("sample set validated")

    cfg = ReplayTrainConfig(
        base_model="Qwen/Qwen3-14B",
        adapter_path=str(adapter),
        output_dir=out / "v_replay",
        learning_rate=learning_rate,
        device="cuda",
    )

    log("loading ck75 (bf16 + LoRA, checkpointing on)...")
    t0 = time.time()
    model = load_v0_for_training(cfg)
    opt = build_optimizer(model, cfg)
    log(f"loaded in {time.time() - t0:.0f}s")

    def _echo(row: dict) -> None:
        log(
            f"  ex {row['example']:>2} {row.get('key','')} "
            f"tok={row.get('total_tokens')} sup={row.get('supervised')} "
            f"loss/tok={row.get('loss_per_token')} "
            f"peak={row.get('peak_gib')}/{row.get('reserved_gib')}GiB "
            f"gpu={row.get('gpu_util_pct')}% "
            f"{row.get('seconds')}s"
        )

    log("stage 4/4: one optimizer step")
    result = run_replay_chunk_opd(
        prefixes,
        actions,
        scored,
        make_optimizer_step(
            model, opt, cfg, progress_path=progress, on_example=_echo
        ),
        max_new_tokens=max_new_tokens,
        max_trace_share=max_trace_share,
        n_samples_per_prefix=n_samples_per_prefix,
        selection_policy="stratified-diagnostic",
    )
    log(f"loss {result['optimizer']['loss']:.6f}, "
        f"{result['optimizer']['lora_tensors_moved']} tensors moved")

    rows = []
    if progress.exists():
        rows = [json.loads(x) for x in progress.read_text().splitlines() if x.strip()]

    # Persist v_replay to the volume, not through the function result. A LoRA
    # over all six projection modules is ~490 MB — returning it base64-encoded
    # was both a size limit waiting to be hit and a guarantee that the weights
    # die with the container's /tmp. The volume outlives the run.
    dest = Path(VOLUME_MOUNT) / V_REPLAY_IN_VOLUME
    if dest.exists():
        raise SystemExit(
            f"{dest} already exists — refusing to overwrite a previous v_replay"
        )
    dest.mkdir(parents=True)
    saved = []
    for f in sorted((out / "v_replay").iterdir()):
        if f.is_file():
            (dest / f.name).write_bytes(f.read_bytes())
            saved.append((f.name, f.stat().st_size))
    vol.commit()
    log(f"v_replay -> {dest}: " + ", ".join(f"{n} ({b:,}B)" for n, b in saved))

    return {
        "result": result,
        "progress": rows,
        "v_replay_volume_path": str(dest),
        "v_replay_files": dict(saved),
    }


@app.local_entrypoint()
def main(
    max_trace_share: float = 0.35,
    learning_rate: float = 1e-5,
    out_dir: str = "./replay-train-out",
):
    import base64
    import json
    from pathlib import Path

    res = train.remote(
        max_trace_share=max_trace_share, learning_rate=learning_rate
    )
    out = Path(out_dir)
    (out / "v_replay").mkdir(parents=True, exist_ok=True)

    (out / "replay_train_result.json").write_text(
        json.dumps({k: v for k, v in res.items() if k != "adapter_files"},
                   indent=2, default=str)
    )
    print(f"v_replay on volume: {res.get('v_replay_volume_path')}")
    for n, b in (res.get("v_replay_files") or {}).items():
        print(f"  {n}  {b:,} bytes")

    r = res["result"]["optimizer"]
    print("\n=== REPLAY OPD TRAINING ===")
    print(f"loss                {r['loss']:.6f}")
    print(f"supervised tokens   {r['global_supervised_tokens']}")
    print(f"grad norm           {r['grad_norm']}")
    print(f"clip fraction       {r['clip_fraction']}")
    print(f"LoRA tensors moved  {r['lora_tensors_moved']}/{r['lora_tensors_total']}")
    print(f"max param delta     {r['max_param_delta']}")
    print(f"examples logged     {len(res.get('progress') or [])}")
    print(f"written             {out}")
