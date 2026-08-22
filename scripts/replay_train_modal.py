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
#: ck75's own adapter is 513,877,864 bytes and one Adam step does not shrink it.
#: A file materially smaller than this is a truncated write, not an adapter.
MIN_ADAPTER_BYTES = 400 * 1024 * 1024

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
    # §8.4: the supervised action must be ck75's, not the stored DeepSeek one.
    # Omitting this made assert_action_is_student_sampled a silent no-op.
    stored_actions = {}
    stored_path = batch_dir / "stored_teacher_actions.json"
    if stored_path.is_file():
        stored_actions = {
            k: base64.b64decode(v)
            for k, v in json.loads(stored_path.read_text()).items()
        }
        log(f"loaded {len(stored_actions)} stored DeepSeek actions for the §8.4 check")
    else:
        log("no stored_teacher_actions.json — §8.4 identity check will be skipped")
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

    # Write the adapter onto the volume directly, the way sft_train_modal does.
    # Saving to /tmp and copying afterwards is what lost the last run.
    dest_dir = Path(VOLUME_MOUNT) / V_REPLAY_IN_VOLUME
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise SystemExit(
            f"{dest_dir} already exists and is not empty — refusing to "
            "overwrite a previous v_replay. Move it aside or set V_REPLAY_PATH."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReplayTrainConfig(
        base_model="Qwen/Qwen3-14B",
        adapter_path=str(adapter),
        output_dir=dest_dir,
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
    try:
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
            stored_teacher_actions=stored_actions or None,
            selection_policy="stratified-diagnostic",
        )
    finally:
        # Commit whatever exists even on an abort: the per-example rows are the
        # only record of how far a failed step got, and they live on the volume
        # rather than the container's /tmp precisely so a crash cannot take
        # them. GPU time is already spent by the time anything here runs.
        try:
            if progress.exists():
                (dest_dir / "train_progress.jsonl").write_bytes(progress.read_bytes())
            vol.commit()
        except Exception as e:  # never mask the original failure
            print(f"[warn] progress commit failed: {type(e).__name__}: {e}", flush=True)

    log(f"loss {result['optimizer']['loss']:.6f}, "
        f"{result['optimizer']['lora_tensors_moved']} tensors moved")

    rows = []
    if progress.exists():
        rows = [json.loads(x) for x in progress.read_text().splitlines() if x.strip()]

    # `output_dir` is already on the volume, so save_pretrained wrote the
    # weights there directly — no /tmp copy to lose. What remains is proving it
    # before claiming success: the previous run reported a perfect optimizer
    # step while adapter_model.safetensors never left the container.
    vol.commit()
    weights = dest_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(
            f"v_replay has no adapter_model.safetensors at {dest_dir}. The "
            "optimizer step ran but produced no durable artifact (§8.4)."
        )
    size = weights.stat().st_size
    if size < MIN_ADAPTER_BYTES:
        raise SystemExit(
            f"v_replay weights are {size:,} bytes, below the {MIN_ADAPTER_BYTES:,} "
            "floor — a truncated or partial write, not a usable adapter."
        )

    # §8.4 wants the *archived* artifact reloadable, not a file about to be
    # garbage-collected. Verified on the volume path for that reason.
    from vektori_trace.replay_train import verify_adapter_reloadable

    reload_check = verify_adapter_reloadable(dest_dir)
    saved = {f.name: f.stat().st_size for f in sorted(dest_dir.iterdir()) if f.is_file()}
    log(f"v_replay verified on volume: {weights.name} {size:,}B, "
        f"{reload_check.get('n_tensors')} tensors")

    return {
        "result": result,
        "progress": rows,
        "v_replay_volume_path": str(dest_dir),
        "v_replay_files": saved,
        "v_replay_weights_bytes": size,
        "v_replay_reload_check": reload_check,
    }


@app.local_entrypoint()
def main(
    # 0.45, not the library default of 0.35: this batch's click-3482@37 prefix
    # holds 43.01% of the supervised tokens because ck75 wrote two ~4k-token
    # actions there. Accepted and recorded rather than excluded — see the run
    # report's post_compaction_coverage / spread block.
    max_trace_share: float = 0.45,
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
        json.dumps(res, indent=2, default=str)
    )
    files = res.get("v_replay_files") or {}
    weights = files.get("adapter_model.safetensors")
    print(f"v_replay on volume: {res.get('v_replay_volume_path')}")
    for n, b in sorted(files.items()):
        print(f"  {n}  {b:,} bytes")

    # The banner is not the artifact. The previous run printed a perfect
    # optimizer report while the weights stayed in a container that had already
    # exited, so success is asserted against the file, not the metrics.
    if not weights:
        raise SystemExit(
            "TRAINING REPORTED SUCCESS BUT NO adapter_model.safetensors IS ON "
            "THE VOLUME — the run produced no durable v_replay."
        )
    if weights < 400 * 1024 * 1024:
        raise SystemExit(
            f"v_replay weights are only {weights:,} bytes — truncated write."
        )
    print(f"\nv_replay VERIFIED on volume: {weights:,} bytes")

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
