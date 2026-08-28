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
#: Verified against the volume 2026-08-28 via `checkpoints.jsonl`: the run root
#: and `checkpoint-32` carry the SAME adapter -- `adapter_model.safetensors`
#: hashes to 3869b147ab7ce5d2 in both, which is also the adapter_hash the
#: 2026-08-28 smoke run archived. Step 32 is the final step, not a mid-training
#: one (the root row is `final: true`).
#:
#: `checkpoint-32` is nonetheless the right parent: it is the only one carrying
#: `optimizer.pt`, `scheduler.pt` and `rng_state.pth`. The root holds the
#: inference files only. Both would load and train identically, so naming the
#: root would not error -- it would just quietly forgo the resume state, which
#: is the kind of difference that shows up as an unexplained learning-rate
#: change three updates later.
PARENT_IN_VOLUME = "tau2/runs/a_sft_new_ck35_r2/checkpoint-32"

#: The parent's adapter weights, from `checkpoints.jsonl`. Recorded so a run
#: cannot silently train from a different adapter than the one it claims.
PARENT_ADAPTER_HASH = "3869b147ab7ce5d2"
RUNS_IN_VOLUME = "tau2/live-opd"

GPU = "L40S"

#: The pinned benchmark. Changing either invalidates the tool-schema hash the
#: run records, which is the intended behaviour.
TAU2_REPO = "https://github.com/sierra-research/tau2-bench.git"
TAU2_COMMIT = "f8de30c"

#: The Tau2 user simulator, served through litellm's Fireworks provider. The
#: `fireworks_ai/` prefix is what routes it off OpenAI.
USER_MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731"

#: Tau2's domain data (db.json, policy.md, tasks.json), staged on the volume.
#: The pip install ships CODE ONLY -- with no TAU2_DATA_DIR, tau2 computes
#: `Path(tau2/utils/utils.py).parents[3]/data`, which under site-packages
#: resolves to the nonsensical /usr/local/lib/python3.12/data and every domain
#: load raises FileNotFoundError. Caught by the no-spend preflight 2026-08-28.
#: `TAU2_DATA_DIR` is tau2's own documented override (utils.py:17).
TAU2_DATA_IN_VOLUME = "tau2/tau2_data"

TEACHER_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
TEACHER_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash-0731"

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
    # `tau2`, NOT `tau2-bench`. The repo is named tau2-bench but its
    # pyproject declares `name = "tau2"`, and pip refuses the mismatch:
    #   "has inconsistent name: expected 'tau2-bench', but metadata has 'tau2'"
    # Caught by the no-spend preflight 2026-08-28 before any GPU was allocated.
    .pip_install(f"tau2 @ git+{TAU2_REPO}@{TAU2_COMMIT}")
    .env({
        "HF_HOME": HF_CACHE_MOUNT,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Points tau2 at the domain data staged on the volume. Without it the
        # package resolves a path that does not exist and every retail load
        # fails -- after the GPU would have been allocated.
        "TAU2_DATA_DIR": f"{VOLUME_MOUNT}/{TAU2_DATA_IN_VOLUME}",
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
    # Fireworks only. Verified 2026-08-28: `modal secret list` shows exactly
    # one secret, `fireworks-api-key`. Requiring an `openai-api-key` secret
    # would fail the container at start -- and the default user simulator is
    # now a Fireworks model, so nothing here needs an OpenAI key.
    secrets=[modal.Secret.from_name("fireworks-api-key")],
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
    # Deliberately NOT forwarded by default. The driver derives the parent
    # hash from --parent's weights, which cannot be forgotten the way a flag
    # can (update 0 of the 2026-08-28 proof archived "" for exactly that
    # reason). An explicit value is still honoured as a cross-check.
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
    # The two-update proof's entire claim is that update 1's episodes were
    # sampled from the adapter update 0 produced. Counting checkpoints does
    # NOT establish that: a run whose serving refresh silently no-opped
    # produces two checkpoints and a green log while update 1 resampled the
    # parent policy -- which is precisely the failure the proof exists to
    # catch. So compare the identities.
    if rc == 0 and mode == "two-update-proof":
        complete = [s for s in saved if s["complete"]]
        if len(complete) < 2:
            raise SystemExit(
                f"two-update proof produced {len(complete)} complete "
                "checkpoint(s), not 2. The reload/next-policy transition is "
                "exactly what this mode exists to prove, so a single "
                "checkpoint is a failed proof, not a partial success."
            )
        u0_state = os.path.join(out, "update-000", "checkpoint", "state.json")
        u0_hash = json.load(open(u0_state)).get("adapter_hash")

        # What update 1 actually sampled under, read off its own archived
        # episodes rather than off anything the driver reported.
        eps = os.path.join(out, "update-001", "live_archive", "episodes.jsonl")
        sampled_hashes = {
            json.loads(line).get("adapter_hash")
            for line in open(eps) if line.strip()
        }
        if sampled_hashes != {u0_hash}:
            raise SystemExit(
                "FAILED two-update proof: update 1 archived adapter_hash(es) "
                f"{sorted(str(h) for h in sampled_hashes)} but update 0 "
                f"produced {u0_hash!r}. Update 1 did not sample from the "
                "adapter update 0 trained, so this run does not demonstrate an "
                "on-policy loop -- it demonstrates two independent updates."
            )
        if u0_hash == PARENT_ADAPTER_HASH:
            raise SystemExit(
                "FAILED two-update proof: update 0's checkpoint hashes to the "
                f"PARENT adapter ({PARENT_ADAPTER_HASH}). The optimizer step "
                "did not change the weights, so update 1 resampled the SFT "
                "policy under a new name."
            )
        result_proof = {
            "update0_adapter_hash": u0_hash,
            "update1_sampled_under": sorted(str(h) for h in sampled_hashes),
            "parent_adapter_hash": PARENT_ADAPTER_HASH,
            "on_policy_transition_verified": True,
        }
        print(json.dumps({"two_update_proof": result_proof}, indent=2))

    result = {"returncode": rc, "run_id": run_id, "mode": mode,
              "two_update_proof": locals().get("result_proof"),
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


@app.function(
    # NO gpu= argument: this allocates CPU only and cannot cost GPU time. It
    # exists so the expensive run's prerequisites fail here, in seconds, rather
    # than after a GPU has been allocated and turn 0 has been paid for.
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def preflight(tau2_src: str = "") -> dict:
    """Prove every prerequisite exists. No GPU, no teacher call, no rollout."""
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    if tau2_src:
        sys.path.insert(0, tau2_src)

    out: dict = {"ok": True, "checks": {}}

    def check(name, fn):
        try:
            out["checks"][name] = {"ok": True, "value": fn()}
        except Exception as exc:
            out["ok"] = False
            out["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)

    def _parent():
        missing = [
            f for f in ("adapter_config.json", "adapter_model.safetensors",
                        "optimizer.pt", "scheduler.pt", "rng_state.pth")
            if not os.path.isfile(os.path.join(parent, f))
        ]
        if missing:
            raise FileNotFoundError(f"{parent} missing {missing}")
        return parent

    def _tau2():
        import tau2
        return getattr(tau2, "__version__", "installed")

    def _tools():
        from vektori_trace.tau2.tools import load_domain_tools, tools_hash
        tools = load_domain_tools("retail")
        return {"n": len(tools), "hash": tools_hash(tools)}

    def _driver():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "t", "/root/scripts/tau2_live_opd_train.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return {"max_action_tokens": m.MAX_ACTION_TOKENS}

    def _bridge():
        from vektori_trace.tau2.live_train import live_max_trace_share
        from vektori_trace.tau2.opd_stages import set_commit_fn
        set_commit_fn(vol.commit)
        return {"share_4_episodes": round(live_max_trace_share(4), 4),
                "commit_fn_installed": True}

    def _teacher_key():
        if not os.environ.get("FIREWORKS_API_KEY"):
            raise KeyError("FIREWORKS_API_KEY absent from the container env")
        return "present"

    def _torch():
        import torch
        return {"cuda_available": torch.cuda.is_available()}

    def _domain_data():
        root = os.environ.get("TAU2_DATA_DIR")
        if not root:
            raise KeyError(
                "TAU2_DATA_DIR unset; tau2 would resolve its data dir relative "
                "to site-packages and every retail load would fail"
            )
        need = os.path.join(root, "tau2", "domains", "retail")
        missing = [f for f in ("db.json", "policy.md", "tasks.json")
                   if not os.path.isfile(os.path.join(need, f))]
        if missing:
            raise FileNotFoundError(f"{need} missing {missing}")
        # The USER SIMULATOR reads its own prompts from the same tree, and a
        # rollout dies at episode start without them -- after the endpoint is
        # up and paid for. Staging only the domain is not enough (cost one
        # discarded episode 2026-08-28).
        usim = os.path.join(root, "tau2", "user_simulator")
        umissing = [f for f in ("simulation_guidelines.md",
                                "simulation_guidelines_tools.md")
                    if not os.path.isfile(os.path.join(usim, f))]
        if umissing:
            raise FileNotFoundError(f"{usim} missing {umissing}")
        return {"data_dir": root, "retail": "complete",
                "user_simulator": "complete"}

    def _parent_hash():
        from vektori_trace.tau2.reopd_checkpoint import adapter_hash as _ah
        got = _ah(parent)
        if got != PARENT_ADAPTER_HASH:
            raise ValueError(
                f"parent adapter at {parent} hashes to {got}, but this file "
                f"pins {PARENT_ADAPTER_HASH}. The run would train from a "
                "different adapter than it claims."
            )
        return got

    check("tau2_domain_data", _domain_data)
    check("parent_adapter_hash", _parent_hash)
    check("parent_adapter", _parent)
    check("tau2_importable", _tau2)
    check("retail_tools", _tools)
    check("driver_loads", _driver)
    check("bridge_and_commit_fn", _bridge)
    check("fireworks_key", _teacher_key)
    check("torch", _torch)

    print(json.dumps(out, indent=2))
    return out


@app.function(
    # NO gpu= argument. Steps 2-4 of the ladder: roll out ONE episode, then
    # score exactly that episode with the real DeepSeek teacher. No optimizer,
    # no checkpoint, no weights. This is deliberately cheaper than the
    # two-update proof and comes first, because `capture_live_update` catches a
    # per-episode failure and continues to the next plan -- so a four-episode
    # update whose reasoning capture fails on episode 1 still pays the endpoint
    # and user simulator for episodes 2-4 before `batch_report` rejects the
    # batch. One episode cannot waste three.
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def diagnose(
    api_base: str,
    student_model: str,
    task_id: str,
    seed: int,
    run_id: str,
    max_input_tokens: int,
    temperature: float,
    user_model: str,
    score: bool,
    tau2_src: str = "",
) -> dict:
    """One reasoning-required episode, then real teacher scoring of it."""
    import json
    import os
    import sys

    sys.path.insert(0, "/root")
    if tau2_src:
        sys.path.insert(0, tau2_src)

    from transformers import AutoTokenizer

    from vektori_trace.tau2.live_rollout import (
        EpisodePlan, RolloutSettings, Tau2EpisodeRunner, capture_live_update,
    )
    from vektori_trace.tau2.opd_stages import set_commit_fn
    from vektori_trace.tau2.reopd_state import RunState

    set_commit_fn(vol.commit)
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)

    from vektori_trace.tau2.tools import load_domain_tools, tools_hash
    tools = load_domain_tools("retail")

    settings = RolloutSettings(
        domain="retail",
        student_model=student_model,
        api_base=api_base,
        policy_version="diagnose-u000",
        adapter_hash=PARENT_ADAPTER_HASH,
        gen_config_hash="diagnose",
        max_tokens=4096,
        max_input_tokens=max_input_tokens,
        temperature=temperature,
        user_model=user_model,
        # NOT relaxed. The old 13/13 `reasoning: None` result came from the
        # obsolete pre-closed prompt; the corrected boundary deserves a real
        # test, and admitting reasoning-less actions here would prove nothing.
        require_reasoning=True,
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    run = RunState(out, n_updates=1)
    run.freeze_manifest({
        "kind": "tau2_live_opd_diagnose", "task_id": task_id, "seed": seed,
        "student_model": student_model, "tools_hash": tools_hash(tools),
        "require_reasoning": True,
    })
    plan = EpisodePlan(episode_id=f"diag-task{task_id}-seed{seed}",
                       task_id=task_id, seed=seed)

    result: dict = {"run_id": run_id, "out_dir": f"{RUNS_IN_VOLUME}/{run_id}"}
    try:
        report = capture_live_update(
            run.update(0), [plan], settings=settings,
            teacher_context={"model": TEACHER_MODEL,
                             "tokenizer": TEACHER_TOKENIZER,
                             "renderer": "deepseek-v4-native"},
            runner=Tau2EpisodeRunner(settings, tok),
        )
        result["rollout"] = report
    except BaseException as exc:
        vol.commit()
        result["rollout_error"] = f"{type(exc).__name__}: {exc}"[:600]
        # The archive is the point: even a failed episode recorded WHY.
        from vektori_trace.tau2.live_episode import EpisodeArchive
        arch = EpisodeArchive(os.path.join(out, "update-000", "live_archive"))
        eps = arch.load_episodes()
        result["episodes"] = {k: {"status": v.get("status"),
                                  "discard_reason": v.get("discard_reason"),
                                  "num_turns": v.get("num_turns"),
                                  "num_failed_turns": v.get("num_failed_turns")}
                              for k, v in eps.items()}
        result["failures"] = [
            {"turn": f.get("turn_index"), "kind": f.get("failure_kind"),
             "error": str(f.get("error"))[:300]}
            for f in arch.load_failures(plan.episode_id)
        ][:8]
        vol.commit()
        return result
    vol.commit()

    # What the rollout actually captured -- the reasoning question, answered.
    from vektori_trace.tau2.live_episode import EpisodeArchive
    arch = EpisodeArchive(os.path.join(out, "update-000", "live_archive"))
    turns = arch.load_turns(plan.episode_id)
    result["turns"] = [
        {"turn": t["capture"]["turn_index"],
         "finish": t["capture"].get("finish_reason"),
         "n_tok": len(t["capture"]["sampled_token_ids"]),
         "n_logprobs": len(t["capture"]["behavior_logprobs"]),
         "lengths_agree": (len(t["capture"]["sampled_token_ids"])
                           == len(t["capture"]["behavior_logprobs"])),
         "has_reasoning": bool((t["capture"].get("reasoning") or "").strip()),
         "reasoning_chars": len((t["capture"].get("reasoning") or "")),
         "n_reasoning_tokens": len(t["capture"].get("reasoning_token_indices") or []),
         # Diagnostic only -- False is normal (trailing <|im_end|>). The
         # real byte gate is verify_ids_reconstruct_text, which must have
         # passed for the turn to exist at all.
         "raw_eq_token_bytes": t["capture"].get(
             "raw_equals_token_bytes_exactly")}
        for t in turns
    ]
    n = len(result["turns"])
    result["reasoning_summary"] = {
        "turns": n,
        "with_reasoning": sum(1 for r in result["turns"] if r["has_reasoning"]),
        "all_lengths_agree": all(r["lengths_agree"] for r in result["turns"]),
        # Every archived turn passed verify_ids_reconstruct_text by
        # construction, so this is the meaningful statement about bytes.
        "all_ids_reconstruct_text": True,
        "raw_eq_token_bytes_all": all(
            r["raw_eq_token_bytes"] for r in result["turns"]),
    }

    if not score:
        return result

    # Step 4: prove REAL cross-tokenizer byte alignment on these exact bytes.
    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.tau2.live_train import load_live_update_inputs
    from vektori_trace.vocab_bridge import load_tokenizer
    from vektori_trace.replay_score import score_replay_batch

    inputs = load_live_update_inputs(run.update(0),
                                     policy_version="diagnose-u000")
    teacher_tok = load_tokenizer(TEACHER_TOKENIZER)
    pool = FireworksTeacherPool(model=TEACHER_MODEL)
    scored, ledger = score_replay_batch(
        inputs.actions, inputs.rendered, teacher_tok, pool)
    vol.commit()
    tin = ledger.get("teacher_input_tokens", 0)
    result["scoring"] = {
        "n_scored": len(scored),
        "teacher_input_tokens": tin,
        "est_cost_usd_uncached": round(tin * 0.22 / 1e6, 4),
        "all_finite": all(
            all(x == x and x not in (float("inf"), float("-inf")) for x in lps)
            for _, lps in scored.values()),
    }
    return result


@app.function(
    # CPU only. Reads the archive off the volume and prints what Qwen sampled.
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=60 * 10,
)
def show_actions(run_id: str = "", update: int = 0, full: bool = False) -> str:
    """Print every sampled action, its reasoning and its observation."""
    import glob
    import json
    import os

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    if run_id:
        runs = [os.path.join(base, run_id)]
    else:
        runs = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime)
    if not runs:
        return "no runs on the volume yet"
    run = runs[-1]
    arch = os.path.join(run, f"update-{update:03d}", "live_archive")
    if not os.path.isdir(arch):
        return f"no live_archive at {arch}"

    lim = 100000 if full else 500
    out = [f"run: {os.path.basename(run)}", f"archive: {arch}", ""]

    eps = os.path.join(arch, "episodes.jsonl")
    if os.path.isfile(eps):
        rows = [json.loads(x) for x in open(eps) if x.strip()]
        last = {}
        for r in rows:
            last[r["episode_id"]] = r
        for e in last.values():
            out.append(
                f"EPISODE {e['episode_id']}: status={e.get('status')} "
                f"turns={e.get('num_turns')} failed={e.get('num_failed_turns')} "
                f"reward={e.get('reward')} term={e.get('termination_reason')}"
            )
            if e.get("discard_reason"):
                out.append(f"  discard_reason: {e['discard_reason']}")
        out.append("")

    for f in sorted(glob.glob(os.path.join(arch, "turns", "*.jsonl"))):
        out.append("=" * 76)
        out.append(f"TURNS: {os.path.basename(f)}")
        out.append("=" * 76)
        for line in open(f):
            r = json.loads(line)
            kind = r.get("kind")
            if kind == "failed_turn":
                out.append(f"\n--- turn {r.get('turn_index')} FAILED "
                           f"[{r.get('failure_kind')}] ---")
                out.append(f"    error: {str(r.get('error'))[:400]}")
                out.append(f"    salvaged raw: {str(r.get('raw_text'))[:lim]!r}")
                continue
            if kind == "turn_observed":
                obs = r.get("observation") or {}
                txt = obs.get("content") or json.dumps(obs)
                out.append(f"    ENV/USER -> {str(txt)[:300]}")
                continue
            c = r.get("capture") or {}
            ids = c.get("sampled_token_ids") or []
            lps = c.get("behavior_logprobs") or []
            reasoning = c.get("reasoning") or ""
            out.append(
                f"\n--- turn {c.get('turn_index')} [{c.get('finish_reason')}] "
                f"{len(ids)} tok / {len(lps)} logprobs "
                f"{'OK' if len(ids) == len(lps) else 'MISMATCH!'} ---"
            )
            out.append(f"  REASONING ({len(reasoning)} chars): {reasoning[:lim]!r}")
            out.append(f"  CONTENT  : {str(c.get('content'))[:lim]!r}")
            for tc in (c.get("tool_calls") or []):
                out.append(f"  TOOL_CALL: {json.dumps(tc)[:400]}")
            if full:
                out.append(f"  RAW      : {str(c.get('raw_text'))[:lim]!r}")
    text = "\n".join(out)
    print(text)
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def show_telemetry(run_id: str = "") -> str:
    """Print telemetry.jsonl and report.json for a run. CPU only."""
    import glob
    import json
    import os

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    if not runs:
        return "no runs found"
    run = runs[-1]
    out = [f"run: {os.path.basename(run)}", "", "=== files ==="]
    for dp, dn, fn_ in os.walk(run):
        if "turns" in dn:
            dn.remove("turns")
            out.append(f"  {os.path.relpath(os.path.join(dp,'turns'), run)}/ (dir)")
        for f in fn_:
            fp = os.path.join(dp, f)
            out.append(f"  {os.path.relpath(fp, run)}  ({os.path.getsize(fp)}b)")

    tel = os.path.join(run, "telemetry.jsonl")
    out += ["", "=== telemetry.jsonl ==="]
    if not os.path.isfile(tel):
        out.append("  MISSING")
    else:
        for line in open(tel):
            if not line.strip():
                continue
            r = json.loads(line)
            r.pop("t", None); r.pop("iso", None); r.pop("probe_logprobs", None)
            ev, st = r.get("event"), r.get("stage")
            if ev == "stage" and st == "TRAINED":
                out.append("[TRAINED] " + json.dumps(r, indent=1))
            elif ev in ("live_batch", "refresh") or ev == "stage":
                out.append(f"[{ev}/{st}] " + json.dumps(r)[:420])

    rep = os.path.join(run, "update-000", "report.json")
    out += ["", "=== report.json (update 0) ==="]
    if not os.path.isfile(rep):
        out.append("  MISSING")
    else:
        d = json.load(open(rep))
        out.append("keys: " + str(sorted(d.keys())))
        out.append("optimizer: " + json.dumps(d.get("optimizer"))[:600])
        for k in ("global_supervised_tokens", "spread", "selection_policy",
                  "realized_step_histogram"):
            if k in d:
                out.append(f"{k}: {json.dumps(d[k])[:300]}")
        pas = d.get("per_action_stats") or []
        advs = [s.get("mean_advantage") for s in pas
                if isinstance(s, dict) and s.get("mean_advantage") is not None]
        out.append(f"per_action_stats: {len(pas)}")
        if advs:
            sv = sorted(advs)
            out.append(f"  mean_advantage min={min(advs):.6f} max={max(advs):.6f} "
                       f"median={sv[len(sv)//2]:.6f} "
                       f"n_distinct={len({round(a,6) for a in advs})}")
    text = "\n".join(out)
    print(text)
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 20)
def classify_advantages(run_id: str = "", update: int = 0) -> str:
    """Recompute per-token advantages and break them down BY TOKEN CLASS.

    CPU only, no teacher call: it replays the exact alignment and chunk-
    advantage math over the archived actions and paid scores, then attributes
    every supervised token to markup / reasoning / tool-json / content.

    This is what a global histogram cannot do. The 2026-08-28 proof reported a
    single `-55.289` advantage and `</think>` negative in 27/31 turns; without
    attribution there is no way to tell a structural-syntax penalty from a
    genuine semantic one, and masking a class we have not measured throws away
    real supervision alongside the spurious kind.
    """
    import base64
    import glob
    import json
    import os

    from vektori_trace.align import align_by_bytes
    from vektori_trace.chunk_opd import assign_chunk_advantages
    from vektori_trace.tau2.live_token_classes import (
        class_report, classify_action,
    )

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    if not runs:
        return "no runs found"
    run = runs[-1]
    u = os.path.join(run, f"update-{update:03d}")

    actions = {}
    for line in open(os.path.join(u, "actions.jsonl")):
        if line.strip():
            r = json.loads(line)
            actions[r["key"]] = r
    scores = {}
    for line in open(os.path.join(u, "scores.jsonl")):
        if line.strip():
            r = json.loads(line)
            scores[r["key"]] = r

    per_token = []
    n_actions = 0
    skipped = []
    for key, a in sorted(actions.items()):
        sc = scores.get(key)
        if sc is None:
            skipped.append((key, "no score"))
            continue
        stu = [base64.b64decode(x) for x in a["action_token_bytes_b64"]]
        tea = [base64.b64decode(x) for x in sc["teacher_token_bytes_b64"]]
        try:
            alignment = align_by_bytes(stu, tea)
            advs, supervised, _stats = assign_chunk_advantages(
                alignment,
                [float(x) for x in a["behavior_logprobs"]],
                [float(x) for x in sc["teacher_logprobs"]],
            )
        except Exception as exc:
            skipped.append((key, f"{type(exc).__name__}: {exc}"[:120]))
            continue
        classes = classify_action(stu)
        for i, (adv, sup) in enumerate(zip(advs, supervised)):
            if not sup:
                continue
            tok = stu[i].decode("utf-8", "replace")
            per_token.append((classes[i], float(adv), tok))
        n_actions += 1

    rep = class_report(per_token)
    rep["n_actions"] = n_actions
    rep["skipped"] = skipped[:10]

    out = [f"run: {os.path.basename(run)}  update {update}",
           f"actions analysed: {n_actions}",
           f"supervised tokens: {rep['n_supervised_tokens']}",
           f"markup share: {rep['markup_share']}", ""]
    for cls, st in rep["by_class"].items():
        out.append(
            f"{cls:10s} n={st['n']:6d}  +{st['n_positive']:5d} "
            f"-{st['n_negative']:5d}  mean={st['mean']:+.4f}  "
            f"min={st['min']}  max={st['max']}"
        )
    out.append("")
    for cls, st in rep["by_class"].items():
        out.append(f"--- {cls} extremes ---")
        for e in st["extremes"]:
            out.append(f"   {e['advantage']:+10.4f}  {e['token']!r}")
    if skipped:
        out.append("")
        out.append(f"skipped: {skipped[:5]}")
    text = "\n".join(out)
    print(text)
    return text


@app.local_entrypoint()
def main(
    api_base: str = "",
    student_model: str = "a-sft-new",
    run_id: str = "",
    task_ids: str = "57,73,75,93",
    seeds: str = "0",
    n_updates: int = 5,
    preflight_only: bool = False,
    show: bool = False,
    telemetry_only: bool = False,
    classify_only: bool = False,
    classify_update: int = 0,
    show_full: bool = False,
    show_update: int = 0,
    diagnose_one: bool = False,
    diagnose_task: str = "57",
    diagnose_seed: int = 0,
    diagnose_score: bool = True,
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
    # The user simulator runs on Fireworks, like the teacher. `gpt-4o-mini`
    # would need an OpenAI key this environment does not have as a Modal
    # secret, and would fail mid-episode -- after turn 0's student generation
    # had already been paid for. Same provider the 2026-08-28 smoke run used.
    user_model: str = USER_MODEL,
    adapter_hash: str = "",
    allow_missing_reasoning: bool = False,
    max_input_tokens: int = 16384,
):
    import json
    import time

    if preflight_only:
        print("preflight: CPU only, no GPU, no teacher call, no rollout")
        result = preflight.remote(tau2_src)
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            raise SystemExit("preflight FAILED -- do not launch a paid run")
        print("preflight OK")
        return

    if classify_only:
        classify_advantages.remote(run_id, classify_update)
        return

    if telemetry_only:
        show_telemetry.remote(run_id)
        return

    if show or show_full:
        show_actions.remote(run_id, show_update, show_full)
        return

    if diagnose_one:
        if not api_base:
            raise SystemExit("--api-base is required for the diagnostic episode")
        rid = run_id or f"diagnose_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"diagnostic: ONE reasoning-required episode, task "
              f"{diagnose_task} seed {diagnose_seed}")
        print(f"  scoring:  {'yes (real DeepSeek)' if diagnose_score else 'no'}")
        print(f"  no GPU, no optimizer, no checkpoint")
        r = diagnose.remote(api_base, student_model, diagnose_task,
                            diagnose_seed, rid, max_input_tokens, temperature,
                            user_model, diagnose_score, tau2_src)
        print(json.dumps(r, indent=2)[:6000])
        if r.get("rollout_error"):
            raise SystemExit("diagnostic FAILED -- see episodes/failures above")
        return

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
