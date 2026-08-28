#!/usr/bin/env python3
"""Run the live Tau2 OPD driver on a Modal GPU.

Mirrors `tau2_reopd_modal.py`, with one structural difference that drives every
other: a live update *rolls out complete Tau2 episodes*, so this container needs
the benchmark itself -- environment, tools, user simulator and grader -- not
just a frozen corpus. The replay wrapper stages `artifacts_16384` and never
imports tau2; this one must import it.

Two GPUs are alive during a run, doing different jobs:

    L40S  serving A_sft_new under vLLM      (scripts/serve_student.py)
    L40S  running this: the optimizer step  (this file)

They cannot be the same card: vLLM claims its GPU's memory wholesale and leaves
nothing to train beside it.

    # 1. endpoint, in its own tmux window
    .venv/bin/python scripts/serve_student.py --base-model Qwen/Qwen3-4B \\
        --adapter a-sft-new=/adapters/tau2/runs/<a_sft_new>/checkpoint-35 \\
        --gpu L40S --max-model-len 24576 --write-env /data/tau2/live_env.sh

    # 2. the two-update proof, then the canary run
    .venv/bin/modal run scripts/tau2_live_opd_modal.py --two-update-proof \\
        --api-base "$STUDENT_API_BASE" --reload-url "$STUDENT_RELOAD_URL"

Nothing allocates a GPU without `--canary`, `--two-update-proof` or `--yes`.

Why `set_commit_fn` is the whole point of this file
--------------------------------------------------
The driver calls `commit_scores()` after every durable write, but that is a
no-op until something installs the callback. Run the driver directly on Modal
without this wrapper and `_COMMIT` stays `None`: captures, paid DeepSeek scores
and the trained adapter live only on container-local disk and vanish when the
container exits. The run can still report success. That is the single most
expensive failure available here, which is why the install is not optional and
why the tail of this file re-verifies the adapter actually reached the volume.
"""

from __future__ import annotations

import modal

VOLUME_NAME = "vektori-trace-adapters"
VOLUME_MOUNT = "/adapters"
HF_CACHE_VOLUME_NAME = "hf-model-cache"
HF_CACHE_MOUNT = "/root/.cache/huggingface"

#: The parent A_sft_new adapter and the run output, both on the volume. A Modal
#: container cannot see the box's /data/tau2, and -- more importantly -- the
#: serving container reloads checkpoints from this same volume, so a checkpoint
#: written anywhere else can never be served to sample the next update.
#: Verified against the volume 2026-08-28: `modal volume ls` shows
#: `adapter_config.json`/`adapter_model.safetensors` at this run's ROOT, not
#: under a `checkpoint-N` subdirectory (there is a `checkpoint-32`, which is a
#: mid-training step, not the selected artifact). Naming a path that does not
#: exist would fail only after the GPU had been allocated.
PARENT_IN_VOLUME = "tau2/runs/a_sft_new_ck35_r2"
RUNS_IN_VOLUME = "tau2/live-opd"

GPU = "L40S"

#: The pinned benchmark. Changing either invalidates the tool-schema hash the
#: run records, which is the intended behaviour.
TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT = "f8de30c"

app = modal.App("tau2-live-opd")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.5.3",
        "peft==0.19.1",
        "accelerate==1.14.0",
        "numpy<3",
        "requests",
        # Live rollouts need the benchmark itself: Tau2's orchestrator,
        # environment, user simulator and grader all run in this container.
        # The replay wrapper deliberately omits this -- it replays frozen
        # prefixes and never steps an environment.
        "litellm",
    )
    # Pinned to the exact commit the box runs (verified 2026-08-28:
    # /data/tau2/src is sierra-research/tau2-bench at f8de30c). Tau2 is NOT on
    # the Modal volume -- `modal volume ls` shows only artifacts/runs/reopd --
    # so installing it here is what makes a live rollout possible at all. A
    # floating branch would let the grader and tool schemas change underneath a
    # run, which is exactly what `tools.load_domain_tools` exists to prevent.
    .pip_install(f"tau2-bench @ git+{TAU2_REPO}@{TAU2_COMMIT}")
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir("vektori_trace", remote_path="/root/vektori_trace",
                   ignore=["__pycache__"])
    .add_local_file("scripts/tau2_live_opd_train.py",
                    remote_path="/root/scripts/tau2_live_opd_train.py")
)


@app.function(
    gpu=GPU,
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60 * 6,
    max_containers=1,
    secrets=[
        modal.Secret.from_name("fireworks-api-key"),
        # The user simulator is a separate paid model. Its key is a distinct
        # secret from the teacher's, and a run that reaches the first user turn
        # without it fails mid-episode -- after the student generation for turn
        # 0 has already been paid for.
        modal.Secret.from_name("openai-api-key"),
    ],
)
def train(
    api_base: str,
    student_model: str,
    run_id: str,
    task_ids: str,
    seeds: str,
    n_updates: int,
    mode: str,
    learning_rate: float,
    temperature: float,
    tools_file: str,
    tau2_src: str,
    reload_url: str,
    user_model: str,
    adapter_hash: str,
    allow_missing_reasoning: bool,
    max_input_tokens: int,
) -> dict:
    import importlib.util
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    # Tau2 is pip-installed into the image at TAU2_COMMIT. `tau2_src` is an
    # override for a checkout staged on the volume, and takes precedence.
    if tau2_src:
        sys.path.insert(0, tau2_src)

    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)

    for name in ("adapter_config.json",):
        if not os.path.isfile(os.path.join(parent, name)):
            raise SystemExit(
                f"A_sft_new parent incomplete: missing {parent}/{name}. Stage "
                "the adapter on the volume first -- the container cannot see "
                "the box's /data."
            )
    try:
        import tau2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"tau2 is not importable in this container ({exc}). A live update "
            "rolls out real episodes and needs the benchmark's environment, "
            "user simulator and grader. The image pins "
            f"tau2-bench@{TAU2_COMMIT}; if that install failed, rebuild the "
            "image or stage a checkout on the volume and pass --tau2-src."
        ) from exc

    # The tool schemas are DERIVED from the installed domain, not read from a
    # staged file. `vektori_trace.tau2.tools.load_domain_tools` imports the
    # live registry precisely so the schema cannot drift from what serving
    # sends; a hand-staged JSON copy is a second source of truth that goes
    # stale silently, and the prompt-parity proof would then pass against the
    # wrong tools -- worse than not proving it. Written into the run directory
    # so the run records exactly what it used.
    if not tools_file:
        from vektori_trace.tau2.tools import load_domain_tools, tools_hash

        tools = load_domain_tools("retail")
        os.makedirs(out, exist_ok=True)
        tools_file = os.path.join(out, "retail_tools.json")
        with open(tools_file, "w") as fh:
            json.dump(tools, fh)
        print(f"derived {len(tools)} retail tool schemas "
              f"(hash {tools_hash(tools)}) -> {tools_file}")
    elif not os.path.isfile(tools_file):
        raise SystemExit(
            f"tools file not found at {tools_file}. Pass an empty --tools-file "
            "to derive the schemas from the installed tau2 domain instead."
        )

    argv = [
        "tau2_live_opd_train.py",
        "--run-dir", out,
        "--parent", parent,
        "--api-base", api_base,
        "--student-model", student_model,
        "--task-ids", task_ids,
        "--seeds", seeds,
        "--n-updates", str(n_updates),
        "--learning-rate", str(learning_rate),
        "--temperature", str(temperature),
        "--tools-file", tools_file,
        "--user-model", user_model,
        "--max-input-tokens", str(max_input_tokens),
    ]
    if reload_url:
        argv += ["--reload-url", reload_url]
    if adapter_hash:
        argv += ["--adapter-hash", adapter_hash]
    if allow_missing_reasoning:
        argv += ["--allow-missing-reasoning"]
    argv += [f"--{mode}"]
    sys.argv = argv

    spec = importlib.util.spec_from_file_location(
        "tau2_live_opd_train", "/root/scripts/tau2_live_opd_train.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # The critical wiring, and the reason this wrapper exists. fsync reaches
    # only the container's local disk; a Modal Volume publishes nothing to
    # shared storage until commit(). A container killed by OOM, timeout,
    # `app stop` or an infrastructure fault never runs a `finally`, so without
    # per-write commits every paid score, every captured behaviour logprob and
    # -- worse -- the trained adapter would be lost with the container.
    #
    # Live has a second reason replay does not: the next update's rollout is
    # served FROM the volume. An uncommitted checkpoint is not merely lost, it
    # cannot be served, so update 1 would silently resample update 0's policy.
    from vektori_trace.tau2.opd_stages import set_commit_fn

    set_commit_fn(vol.commit)

    rc = 1
    try:
        rc = mod.main()
    except BaseException:
        # Publish before re-raising: a partial run's markers are what a resume
        # reads, and its captures and scores were already billed.
        try:
            vol.commit()
        except Exception:
            pass
        raise
    finally:
        try:
            vol.commit()
        except Exception:
            pass

    # Last line of defence: prove the adapters this run exists to produce are
    # actually on the volume, rather than trusting that a commit worked.
    saved = []
    if os.path.isdir(out):
        for d in sorted(os.listdir(out)):
            cp = os.path.join(out, d, "checkpoint")
            if os.path.isdir(cp):
                ok = all(os.path.isfile(os.path.join(cp, f))
                         for f in ("adapter_config.json", "state.json"))
                w = any(os.path.isfile(os.path.join(cp, f)) for f in
                        ("adapter_model.safetensors", "adapter_model.bin"))
                saved.append({"update": d, "complete": bool(ok and w),
                              "path": os.path.join(RUNS_IN_VOLUME, run_id, d,
                                                   "checkpoint")})
    if rc == 0 and not saved:
        raise SystemExit(
            f"the run reported success but no checkpoint is on the volume "
            f"under {out}. Refusing to report a trained adapter that does not "
            "exist."
        )
    # The two-update proof's entire claim is that update 1 sampled from update
    # 0's adapter. Two complete checkpoints are the artifact of that claim.
    if rc == 0 and mode == "two-update-proof":
        complete = [s for s in saved if s["complete"]]
        if len(complete) < 2:
            raise SystemExit(
                f"two-update proof produced {len(complete)} complete "
                "checkpoint(s), not 2. The reload/next-policy transition is "
                "exactly what this mode exists to prove, so a single "
                "checkpoint is a failed proof, not a partial success."
            )

    result = {"returncode": rc, "run_id": run_id, "mode": mode,
              "out_dir": f"{RUNS_IN_VOLUME}/{run_id}",
              "checkpoints": saved}
    p = os.path.join(out, "manifest.json")
    if os.path.exists(p):
        result["manifest"] = json.load(open(p))
    tel = os.path.join(out, "telemetry.jsonl")
    if os.path.exists(tel):
        rows = [json.loads(line) for line in open(tel) if line.strip()]
        result["telemetry_rows"] = len(rows)
        result["stages"] = [r for r in rows if r.get("event") == "stage"]
        result["refreshes"] = [r for r in rows if r.get("event") == "refresh"]
    return result


@app.local_entrypoint()
def main(
    api_base: str = "",
    student_model: str = "a-sft-new",
    run_id: str = "",
    task_ids: str = "57,73,75,93",
    seeds: str = "0",
    n_updates: int = 5,
    canary: bool = False,
    two_update_proof: bool = False,
    yes: bool = False,
    learning_rate: float = 1e-5,
    temperature: float = 1.0,
    # Empty by default: derive from the installed tau2 domain rather than a
    # staged copy that can silently go stale. Pass a path only to pin one.
    tools_file: str = "",
    # Empty: tau2 is pip-installed into the image at TAU2_COMMIT. Pass a path
    # only to override with a checkout staged on the volume.
    tau2_src: str = "",
    reload_url: str = "",
    user_model: str = "gpt-4o-mini",
    adapter_hash: str = "",
    allow_missing_reasoning: bool = False,
    max_input_tokens: int = 16384,
):
    import json
    import time

    modes = [m for m, on in (("canary", canary),
                             ("two-update-proof", two_update_proof),
                             ("yes", yes)) if on]
    if not modes:
        raise SystemExit(
            "GPU run refused. Use --two-update-proof for the smallest run that "
            "proves checkpoint -> reload -> next-policy rollout, --canary for a "
            "single update, or --yes for the full arm."
        )
    if len(modes) > 1:
        raise SystemExit(f"pass exactly one of --canary/--two-update-proof/--yes, got {modes}")
    mode = modes[0]

    if not api_base:
        raise SystemExit(
            "--api-base is required: start the student endpoint first with "
            "scripts/serve_student.py and pass its URL."
        )
    if mode != "canary" and not reload_url:
        raise SystemExit(
            "--reload-url is required for any multi-update run. Without it the "
            "endpoint keeps serving update 0's policy while later updates claim "
            "to be on-policy, and every importance ratio silently compares two "
            "different distributions. Source the env file serve_student.py "
            "wrote."
        )

    n = 1 if mode == "canary" else (2 if mode == "two-update-proof" else n_updates)
    rid = run_id or f"{mode.replace('-', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    n_eps = len(task_ids.split(",")) * len(seeds.split(","))
    print(f"live OPD: A_sft_new -> {n} update(s) x {n_eps} episode(s)")
    print(f"  endpoint {api_base}")
    print(f"  output   {RUNS_IN_VOLUME}/{rid}")
    if allow_missing_reasoning:
        print("  WARNING --allow-missing-reasoning: actions without a <think> "
              "span will be admitted. Diagnostic only.")

    result = train.remote(
        api_base, student_model, rid, task_ids, seeds, n, mode,
        learning_rate, temperature, tools_file, tau2_src, reload_url,
        user_model, adapter_hash, allow_missing_reasoning, max_input_tokens,
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("stages", "refreshes")},
        indent=2))
    for s in result.get("stages", []):
        print(f"  u{s.get('update')} {str(s.get('stage')):8s} "
              f"{s.get('seconds')}s {s.get('loss', '')}")
    for r in result.get("refreshes", []):
        print(f"  refresh -> u{r.get('update')} "
              f"adapter={str(r.get('adapter_hash'))[:16]} "
              f"delta={r.get('max_logprob_delta')}")
