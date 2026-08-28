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
    parent_override: str = "",
    resume_optimizer_from: str = "",
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

    # Default: the frozen A_sft_new parent. `parent_override` exists for an
    # ITERATED update -- update 2 must train from update 1's child, not from
    # the SFT adapter again, or it is a second first-step wearing the name of
    # an iteration: fresh on-policy data trained onto the wrong base, with a
    # loss curve and a checkpoint that both look entirely normal.
    parent = (parent_override if parent_override
              else os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME))
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    print(f"parent adapter: {parent}"
          f"{' (OVERRIDE)' if parent_override else ' (pinned A_sft_new)'}")

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

    # Step 4: prove cross-tokenizer alignment through the SEMANTIC PROJECTION.
    # Deliberately not `score_replay_batch`: that scores raw Qwen bytes, the
    # path the 2026-08-28 update took, and no live code path may reach it.
    import base64 as _b64

    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.tau2.live_score import score_live_action
    from vektori_trace.tau2.live_train import load_live_update_inputs
    from vektori_trace.vocab_bridge import load_tokenizer

    inputs = load_live_update_inputs(run.update(0),
                                     policy_version="diagnose-u000")
    teacher_tok = load_tokenizer(TEACHER_TOKENIZER)
    pool = FireworksTeacherPool(model=TEACHER_MODEL)

    n_sup = n_exc = tin = 0
    reasons: dict = {}
    for row in inputs.capture_rows:
        raw_a = _b64.b64decode(row["action_bytes_b64"]).decode("utf-8", "replace")
        for sp in ("<|im_end|>", "<|endoftext|>"):
            if raw_a.endswith(sp):
                raw_a = raw_a[: -len(sp)]
        sc = score_live_action(
            key=row["key"], raw_text=raw_a,
            student_token_bytes=[
                _b64.b64decode(b) for b in row["action_token_bytes_b64"]
            ],
            semantic_history=inputs.rendered[row["prefix_id"]],
            teacher_tokenizer=teacher_tok, pool=pool,
        )
        n_sup += sc.n_supervised
        n_exc += len(sc.excluded)
        tin += sc.n_prefix_tokens + sc.n_teacher_tokens
        for r in sc.excluded.values():
            reasons[r] = reasons.get(r, 0) + 1
    vol.commit()
    result["scoring"] = {
        "projection": "semantic",
        "n_actions": len(inputs.capture_rows),
        "n_supervised_tokens": n_sup,
        "n_excluded_tokens": n_exc,
        "excluded_by_reason": reasons,
        "teacher_input_tokens": tin,
        "est_cost_usd_uncached": round(tin * 0.22 / 1e6, 4),
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


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 20)
def projection_report(run_id: str = "", update: int = 0) -> str:
    """Offline acceptance report for the semantic projection. No API calls.

    Runs `project_action` over every archived action and reports what would be
    supervised, what would be excluded and why. This is the gate that must pass
    before a single teacher token is re-purchased.
    """
    import base64
    import glob
    import json
    import os

    from vektori_trace.tau2.live_projection import ProjectionError, project_action

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    if not runs:
        return "no runs found"
    run = runs[-1]
    apath = os.path.join(run, f"update-{update:03d}", "actions.jsonl")
    if not os.path.isfile(apath):
        return f"no actions.jsonl at {apath}"

    rows = [json.loads(l) for l in open(apath) if l.strip()]
    n_ok = 0
    failures = []
    tot_tokens = tot_sup = 0
    by_kind: dict = {}
    by_reason: dict = {}
    per_action = []

    for r in rows:
        stu = [base64.b64decode(x) for x in r["action_token_bytes_b64"]]
        raw = b"".join(stu).decode("utf-8", "replace")
        # Trailing specials are allowed by the capture path; strip for parsing.
        for special in ("<|im_end|>", "<|endoftext|>"):
            if raw.endswith(special):
                raw = raw[: -len(special)]
        try:
            proj = project_action(raw, stu)
        except ProjectionError as exc:
            failures.append({"key": r["key"], "error": str(exc)[:200]})
            continue
        rep = proj.report()
        # Invariant: nothing silently dropped.
        assert rep["n_supervised"] + rep["n_excluded"] == len(stu), r["key"]
        n_ok += 1
        tot_tokens += len(stu)
        tot_sup += rep["n_supervised"]
        for k, v in rep["supervised_by_kind"].items():
            by_kind[k] = by_kind.get(k, 0) + v
        for k, v in rep["excluded_by_reason"].items():
            by_reason[k] = by_reason.get(k, 0) + v
        per_action.append((r["key"], rep["retained_fraction"],
                           rep["has_reasoning"], rep["n_tool_calls"]))

    out = [
        f"run: {os.path.basename(run)}  update {update}",
        f"actions: {len(rows)}   projected OK: {n_ok}   failed: {len(failures)}",
        f"tokens: {tot_tokens}   supervised: {tot_sup}   "
        f"retained: {round(tot_sup / tot_tokens, 4) if tot_tokens else 0}",
        "",
        "supervised by kind:  " + json.dumps(by_kind),
        "excluded by reason:  " + json.dumps(by_reason),
        "",
        "ACCEPTANCE:",
        f"  all actions project        : {n_ok == len(rows)}",
        f"  zero markup supervised     : {'markup' not in by_kind}",
        f"  zero tool-json supervised  : {'tool_json' not in by_kind}",
        "",
    ]
    for key, frac, has_r, n_tc in per_action[:40]:
        out.append(f"  {key:34s} retained={frac:.3f} reasoning={has_r} tools={n_tc}")
    for f in failures[:10]:
        out.append(f"  FAIL {f['key']}: {f['error']}")
    text = "\n".join(out)
    print(text)
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 20)
def reconcile_counts(run_id: str = "", update: int = 0) -> str:
    """Account for EVERY token, three ways. No API calls.

    Three numbers were reported for the same batch and did not agree:

        14,178  optimizer `supervised_tokens`   (chunk_opd, post-alignment)
        14,206  projection `tokens`             (raw archived token count)
           711  classifier markup+tool_json     vs 789 projection exclusions

    A 28-token and a 78-token gap, unexplained, is exactly the kind of drift
    that makes a later result unfalsifiable. This reconciles them token by
    token: sentinel (over-long chunk / unaligned tail), trailing special, and
    class-by-class agreement between the two accountings.
    """
    import base64
    import glob
    import json
    import os

    from vektori_trace.align import align_by_bytes
    from vektori_trace.chunk_opd import assign_chunk_advantages
    from vektori_trace.tau2.live_projection import project_action
    from vektori_trace.tau2.live_token_classes import classify_action

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    run = runs[-1]
    u = os.path.join(run, f"update-{update:03d}")
    actions = {json.loads(l)["key"]: json.loads(l)
               for l in open(os.path.join(u, "actions.jsonl")) if l.strip()}
    scores = {json.loads(l)["key"]: json.loads(l)
              for l in open(os.path.join(u, "scores.jsonl")) if l.strip()}

    tot_raw = tot_chunk_sup = tot_sentinel = 0
    tot_trailing = 0
    proj_sup = proj_exc = 0
    # cross-tab: chunk-supervised x projection-decision
    cross: dict = {}
    cls_counts: dict = {}

    for key, a in sorted(actions.items()):
        stu = [base64.b64decode(x) for x in a["action_token_bytes_b64"]]
        sc = scores[key]
        tea = [base64.b64decode(x) for x in sc["teacher_token_bytes_b64"]]
        tot_raw += len(stu)

        alignment = align_by_bytes(stu, tea)
        _advs, supervised, stats = assign_chunk_advantages(
            alignment,
            [float(x) for x in a["behavior_logprobs"]],
            [float(x) for x in sc["teacher_logprobs"]],
        )
        tot_chunk_sup += sum(1 for x in supervised if x)
        tot_sentinel += stats.n_sentinel_tokens

        raw = b"".join(stu).decode("utf-8", "replace")
        trimmed = raw
        for special in ("<|im_end|>", "<|endoftext|>"):
            if trimmed.endswith(special):
                trimmed = trimmed[: -len(special)]
                tot_trailing += 1
        proj = project_action(trimmed, stu)
        proj_sup += proj.n_supervised
        proj_exc += proj.n_excluded

        classes = classify_action(stu)
        for i in range(len(stu)):
            in_chunk = bool(supervised[i])
            in_proj = i in proj.supervised
            cross[(in_chunk, in_proj)] = cross.get((in_chunk, in_proj), 0) + 1
            if in_chunk and not in_proj:
                cls_counts[classes[i]] = cls_counts.get(classes[i], 0) + 1

    out = [
        f"run {os.path.basename(run)} update {update}",
        "",
        "TOKEN ACCOUNTING",
        f"  raw archived student tokens        : {tot_raw}",
        f"  chunk_opd supervised (optimizer)   : {tot_chunk_sup}",
        f"  chunk_opd sentinel (unsupervised)  : {tot_sentinel}",
        f"  raw - supervised - sentinel        : "
        f"{tot_raw - tot_chunk_sup - tot_sentinel}",
        f"  actions with a trailing special    : {tot_trailing}",
        "",
        "PROJECTION",
        f"  supervised : {proj_sup}",
        f"  excluded   : {proj_exc}",
        f"  sum        : {proj_sup + proj_exc}  (must equal raw {tot_raw})",
        "",
        "CROSS-TAB  (chunk_supervised, projection_supervised) -> n",
    ]
    for k in sorted(cross, key=lambda t: (not t[0], not t[1])):
        out.append(f"  chunk={k[0]!s:5s} proj={k[1]!s:5s} : {cross[k]}")
    out += ["", "TOKENS THE PROJECTION REMOVES FROM SUPERVISION, by class:"]
    for k, v in sorted(cls_counts.items(), key=lambda kv: -kv[1]):
        out.append(f"  {k:12s} {v}")
    text = "\n".join(out)
    print(text)
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 30)
def scoring_dryrun(run_id: str = "", update: int = 0) -> str:
    """Run the FULL projected scoring adapter offline. No teacher calls.

    Step 6 of the repair: prove the scoring->training path works end to end on
    every archived action before a single teacher token is re-purchased. The
    teacher is a stub returning a constant, so what is exercised is the render,
    the id extension, the per-payload alignment and the index mapping -- every
    place a silent mismatch could hide.
    """
    import base64
    import glob
    import json
    import os

    from vektori_trace.tau2.live_score import LiveScoreError, score_live_action
    from vektori_trace.vocab_bridge import load_tokenizer

    class _StubPool:
        """Constant logprobs: this run must never bill the teacher."""

        def __init__(self):
            self.n_calls = 0
            self.n_tokens = 0

        def score_ids(self, prompt_ids, tokens):
            self.n_calls += 1
            self.n_tokens += len(tokens)
            return [-0.5] * len(tokens)

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    run = runs[-1]
    u = os.path.join(run, f"update-{update:03d}")
    actions = [json.loads(l) for l in open(os.path.join(u, "actions.jsonl"))
               if l.strip()]
    rendered = json.loads(open(os.path.join(u, "rendered.json")).read())

    tok = load_tokenizer(TEACHER_TOKENIZER)
    pool = _StubPool()

    n_ok = n_fail = 0
    tot_tokens = tot_sup = 0
    reasons: dict = {}
    payload_skips: dict = {}
    failures = []
    per_action = []

    for a in actions:
        stu = [base64.b64decode(x) for x in a["action_token_bytes_b64"]]
        raw = b"".join(stu).decode("utf-8", "replace")
        for sp in ("<|im_end|>", "<|endoftext|>"):
            if raw.endswith(sp):
                raw = raw[: -len(sp)]
        history = rendered.get(a["prefix_id"])
        if history is None:
            failures.append((a["key"], "no rendered history"))
            n_fail += 1
            continue
        try:
            sc = score_live_action(
                key=a["key"], raw_text=raw, student_token_bytes=stu,
                semantic_history=history, teacher_tokenizer=tok, pool=pool,
            )
        except LiveScoreError as exc:
            failures.append((a["key"], f"{exc}"[:160]))
            n_fail += 1
            continue
        n_ok += 1
        tot_tokens += len(stu)
        tot_sup += sc.n_supervised
        # Invariant: nothing silently dropped.
        covered = set(sc.teacher_logprob_by_index) | set(sc.excluded)
        if covered != set(range(len(stu))):
            failures.append((a["key"], "token accounting incomplete"))
        for r in sc.excluded.values():
            reasons[r] = reasons.get(r, 0) + 1
        for kind, info in sc.payload_report.items():
            if isinstance(info, dict) and "skipped" in info:
                k = f"{kind}: {info['skipped']}"
                payload_skips[k] = payload_skips.get(k, 0) + 1
        per_action.append((a["key"], sc.n_supervised, len(stu)))

    out = [
        f"run {os.path.basename(run)} update {update}",
        f"actions: {len(actions)}   scored OK: {n_ok}   failed: {n_fail}",
        f"teacher STUB calls: {pool.n_calls}  (tokens {pool.n_tokens}) -- $0",
        f"student tokens: {tot_tokens}   supervised: {tot_sup}   "
        f"retained: {round(tot_sup/tot_tokens, 4) if tot_tokens else 0}",
        "",
        "exclusion reasons: " + json.dumps(reasons),
        "payload skips    : " + json.dumps(payload_skips),
        "",
        "ACCEPTANCE:",
        f"  all actions scored          : {n_fail == 0}",
        f"  token accounting complete   : {not any('accounting' in f[1] for f in failures)}",
        f"  some supervision retained   : {tot_sup > 0}",
        "",
    ]
    for key, sup, tot in per_action[:40]:
        out.append(f"  {key:34s} {sup:4d}/{tot:4d} = {sup/tot:.3f}")
    for f in failures[:12]:
        out.append(f"  FAIL {f[0]}: {f[1]}")
    text = "\n".join(out)
    print(text)
    return text


@app.function(
    # NO gpu= argument, and no trainer is ever constructed. This is the paid
    # re-score and NOTHING else: the repaired scorer has never made a real
    # teacher call, so the first valid advantage distribution must be readable
    # before any weight moves. A clamp decision, or the discovery of a second
    # scoring defect, has to happen while the parent adapter is still untouched.
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def rescore(run_id: str = "", update: int = 0,
                 clamp: float = 0.0) -> str:
    """SCORED only. The real teacher, the production scorer, no optimizer step.

    Calls `run_projected_score_stage` -- the same function `train_live_update`
    calls -- rather than a re-implementation, so what is measured here is what
    training would consume. Then builds the projected batch **for analysis
    only** and reports the distribution.

    Deliberately absent: `run_projected_train_stage`, any trainer, any
    checkpoint, any reload, any rollout, any GPU. `--clamp` only annotates the
    report with what a given threshold *would* do; it changes nothing on disk.
    """
    import base64
    import json
    import os

    from vektori_trace.providers.teacher.fireworks import FireworksTeacherPool
    from vektori_trace.tau2.live_batch import build_projected_batch
    from vektori_trace.tau2.live_token_classes import classify_action
    from vektori_trace.tau2.live_train import (
        load_live_update_inputs,
        run_projected_score_stage,
    )
    from vektori_trace.tau2.opd_stages import set_commit_fn
    from vektori_trace.tau2.reopd_state import RunState, read_jsonl
    from vektori_trace.vocab_bridge import load_tokenizer

    # Without this the paid scores live on container-local disk and vanish when
    # the container exits, having been billed for.
    set_commit_fn(vol.commit)

    if not run_id:
        raise ValueError("--run-id is required; a paid re-score must name its "
                         "target rather than guess the newest run")
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    if not os.path.isdir(out):
        raise FileNotFoundError(f"no run at {out}")

    with open(os.path.join(out, "manifest.json")) as _fh:
        _m = json.load(_fh)
    run = RunState(out, n_updates=int(_m.get("n_updates", 1)))
    u = run.update(update)

    # Refuse to re-buy scores, and refuse to touch a trained update.
    if u.reached("TRAINED"):
        raise RuntimeError(
            f"{run_id} update {update} is already TRAINED; re-scoring it would "
            "spend money to overwrite the record of a run that already "
            "consumed its scores"
        )

    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = json.load(fh)
    # Identity comes from THIS update's .SAMPLED marker, never the run
    # manifest -- the manifest names the run's initial parent, which for
    # update k>0 is the wrong adapter and the wrong policy version.
    from vektori_trace.tau2.live_train import sampled_identity

    ident = sampled_identity(
        u, expect_adapter_hash=PARENT_ADAPTER_HASH if update == 0 else None)
    sampled_hash = ident["adapter_hash"]
    print(f"scoring update {update}, sampled from {sampled_hash} "
          f"(policy {ident['policy_version']})")

    policy_version = ident["policy_version"]
    inputs = load_live_update_inputs(u, policy_version=policy_version)
    teacher_tok = load_tokenizer(TEACHER_TOKENIZER)
    pool = FireworksTeacherPool(model=TEACHER_MODEL)

    paid = run.paid_scores(update)
    by_key = {r["key"]: r for r in inputs.capture_rows}

    result = run_projected_score_stage(
        u, inputs,
        teacher_tok=teacher_tok, pool=pool, run_dir=out,
        paid=paid, drop_keys=set(), by_key=by_key,
    )
    vol.commit()

    # --- analysis ONLY: build the batch, never hand it to a trainer --------
    # Same arguments `run_projected_train_stage` would pass, so the reported
    # distribution is the one training would consume -- including the derived
    # task share (the replay default of 0.5 rejects a legitimate live batch).
    n_tasks = len({p.task for p in inputs.prefixes})
    max_task_share = min(1.0, (1.0 / n_tasks) * 1.5) if n_tasks else 1.0
    batch = build_projected_batch(
        inputs.prefixes, inputs.actions, result.projected,
        policy_version=policy_version,
        max_task_share=max_task_share,
        max_trace_share=inputs.max_trace_share,
    )

    def stats(xs):
        if not xs:
            return {}
        s = sorted(xs)
        n = len(s)
        return {
            "n": n,
            "mean": round(sum(s) / n, 4),
            "min": round(s[0], 4),
            "p1": round(s[max(0, int(0.01 * n) - 1)], 4),
            "p50": round(s[n // 2], 4),
            "p99": round(s[min(n - 1, int(0.99 * n))], 4),
            "max": round(s[-1], 4),
            "n_pos": sum(1 for x in s if x > 0),
            "n_neg": sum(1 for x in s if x < 0),
            "n_zero": sum(1 for x in s if x == 0),
        }

    all_adv = []
    by_class: dict = {}
    per_action = []
    extremes = []
    for ta, row in zip(batch.advantages, inputs.capture_rows):
        stu = [base64.b64decode(b) for b in row["action_token_bytes_b64"]]
        classes = classify_action(stu)
        vals = []
        for i, a in enumerate(ta.advantages):
            if not ta.supervised_mask[i]:
                continue
            vals.append(a)
            all_adv.append(a)
            by_class.setdefault(classes[i], []).append(a)
            extremes.append((a, row["key"], i, classes[i],
                             stu[i].decode("utf-8", "replace")))
        per_action.append({
            "key": row["key"],
            "task": row.get("task_id"),
            "turn": row.get("turn_index"),
            "n_tokens": len(stu),
            "n_supervised": ta.n_supervised,
            **({"mean_adv": round(sum(vals) / len(vals), 4)} if vals else {}),
        })

    # structural / tool weight must be exactly zero
    struct = {k: v for k, v in by_class.items()
              if k in ("markup", "tool_json", "special")}

    n_finite = sum(
        1 for sc in result.projected.values()
        for v in sc.teacher_logprob_by_index.values()
        if v == v and abs(v) != float("inf")
    )
    n_total_credit = sum(len(sc.teacher_logprob_by_index)
                         for sc in result.projected.values())

    tot_tok = sum(len(r["action_token_bytes_b64"])
                  for r in inputs.capture_rows)
    tot_sup = sum(sc.n_supervised for sc in result.projected.values())
    tot_exc = sum(len(sc.excluded) for sc in result.projected.values())

    skips = []
    for sc in result.projected.values():
        for name, rep in (sc.payload_report or {}).items():
            if isinstance(rep, dict) and rep.get("skipped"):
                skips.append((sc.key, name, rep.get("reason")))

    extremes.sort(key=lambda t: t[0])
    lines = [
        f"POST-FIX ADVANTAGE REPORT -- {run_id} update {update}",
        "  SCORING ONLY. No optimizer step, no checkpoint, no GPU.",
        "",
        f"  scored           : {result.n_newly_scored} new / "
        f"{result.n_reused} reused  ({result.seconds}s)",
        f"  teacher tokens   : {result.teacher_input_tokens:,}",
        f"  est cost         : "
        f"${round(result.teacher_input_tokens * 0.22 / 1e6, 4)}",
        "",
        f"  actions scored   : {len(result.projected)}/"
        f"{len(inputs.capture_rows)}",
        f"  credits finite   : {n_finite}/{n_total_credit}",
        f"  payload skips    : {len(skips) or 'none'}",
        f"  accounting       : {tot_sup:,} supervised + {tot_exc:,} excluded "
        f"= {tot_sup + tot_exc:,} vs {tot_tok:,} tokens "
        f"({'OK' if tot_sup + tot_exc == tot_tok else 'MISMATCH'})",
        f"  structural weight: "
        f"{ {k: len(v) for k, v in struct.items()} or 'zero (correct)'}",
        "",
        "  ADVANTAGE DISTRIBUTION (supervised tokens only)",
        f"    all      : {json.dumps(stats(all_adv))}",
    ]
    for cls in sorted(by_class):
        lines.append(f"    {cls:9s}: {json.dumps(stats(by_class[cls]))}")
    lines += ["", "  10 MOST NEGATIVE SUPERVISED TOKENS"]
    for a, k, i, c, t in extremes[:10]:
        lines.append(f"    {a:10.3f}  {c:9s} {k}[{i}] {t!r}")
    lines += ["", "  10 MOST POSITIVE SUPERVISED TOKENS"]
    for a, k, i, c, t in reversed(extremes[-10:]):
        lines.append(f"    {a:10.3f}  {c:9s} {k}[{i}] {t!r}")

    if clamp:
        would = [x for x in all_adv if abs(x) > clamp]
        lines += [
            "",
            f"  CLAMP PREVIEW at +/-{clamp} (nothing written)",
            f"    would clip {len(would)} of {len(all_adv)} "
            f"({100.0 * len(would) / max(1, len(all_adv)):.3f}%)",
        ]

    lines += ["", "  PER-ACTION (supervision, mean advantage)"]
    for pa in per_action:
        lines.append(
            f"    {pa['key']:34s} task {pa.get('task'):>3s} turn "
            f"{pa.get('turn'):>2} {pa['n_supervised']:>5}/{pa['n_tokens']:<5} "
            f"mean={pa.get('mean_adv', 'n/a')}"
        )

    # fingerprints + stage, read back off disk
    rows = read_jsonl(u.scores_path)
    fp_ok = all(
        r.get("fingerprint") == by_key[r["key"]].get("score_fingerprint")
        for r in rows if r.get("key") in by_key
    )
    try:
        u.validate()
        val = "accepted"
    except Exception as exc:  # noqa: BLE001
        val = f"REFUSED: {exc}"[:120]
    lines += [
        "",
        f"  score rows       : {len(rows)}",
        f"  fingerprints     : {'all bound' if fp_ok else 'MISMATCH'}",
        f"  UpdateDir.validate: {val}",
        f"  stage            : {u.stage()}",
        f"  TRAINED marker   : "
        f"{'absent (correct)' if not u.reached('TRAINED') else 'PRESENT -- BUG'}",
    ]

    text = "\n".join(lines)
    print(text)

    if skips:
        raise RuntimeError(f"payload skips: {skips[:4]}")
    if tot_sup + tot_exc != tot_tok:
        raise RuntimeError("token accounting incomplete")
    if struct:
        raise RuntimeError(f"structural tokens carry weight: "
                           f"{ {k: len(v) for k, v in struct.items()} }")
    if n_finite != n_total_credit:
        raise RuntimeError("non-finite teacher credit present")
    return text


@app.function(
    gpu=GPU,
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 60,
    max_containers=1,
)
def one_step(run_id: str = "", update: int = 0,
             learning_rate: float = 1e-5, parent_override: str = "",
             resume_optimizer_from: str = "") -> str:
    """Exactly one optimizer step from cached scores. No rollout, no teacher.

    The `train` driver cannot do this: its loop samples fresh episodes before
    scoring, so pointing it at an already-SCORED canary would resample over the
    evidence rather than consume it. This is the narrow path -- load the paid
    scores, build the projected batch, take one step, checkpoint, stop.

    No teacher pool is constructed at all, so "zero new teacher calls" is a
    property of the code rather than a number to check afterwards. No endpoint
    is contacted: reload verification and the Tau2 episode are a separate,
    later container, so the training GPU is never held while an endpoint runs.
    """
    import json
    import os

    from vektori_trace.tau2.live_score import ProjectedScore
    from vektori_trace.tau2.live_train import (
        load_live_update_inputs,
        run_projected_train_stage,
    )
    from vektori_trace.tau2.opd_stages import set_commit_fn
    from vektori_trace.tau2.reopd_checkpoint import adapter_hash
    from vektori_trace.tau2.reopd_state import RunState, read_jsonl
    from vektori_trace.tau2.reopd_trainer import ReOPDTrainer

    set_commit_fn(vol.commit)

    if not run_id:
        raise ValueError("--run-id is required")
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    with open(os.path.join(out, "manifest.json")) as _fh:
        _m = json.load(_fh)
    run = RunState(out, n_updates=int(_m.get("n_updates", 1)))
    u = run.update(update)

    if not u.reached("SCORED"):
        raise RuntimeError(
            f"{run_id} update {update} is not SCORED; this path trains from "
            "cached scores and never buys new ones"
        )
    if u.reached("TRAINED"):
        raise RuntimeError(
            f"{run_id} update {update} is already TRAINED; refusing to train "
            "twice from one parent"
        )

    # Identity from THIS update's .SAMPLED marker, never the run manifest:
    # the manifest names the run's INITIAL parent, so for update k>0 it holds
    # both the wrong adapter hash and the wrong policy version. Reading it
    # there would reject update k-1's correct checkpoint for "disagreeing"
    # with the SFT hash, or load the actions under a version they were never
    # sampled with -- and both read as a provenance check passing.
    from vektori_trace.tau2.live_train import sampled_identity

    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = json.load(fh)

    # Lineage: update 0 must come from the untouched parent; update k>0 must
    # come from exactly update k-1's checkpoint.
    expect = PARENT_ADAPTER_HASH if update == 0 else adapter_hash(
        run.update(update - 1).checkpoint_path)
    ident = sampled_identity(u, expect_adapter_hash=expect)
    claimed = ident["adapter_hash"]

    parent = parent_override or (
        os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME) if update == 0
        else str(run.update(update - 1).checkpoint_path))
    parent_hash = adapter_hash(parent)
    if parent_hash != claimed:
        raise ValueError(
            f"the adapter at {parent} hashes to {parent_hash}, but "
            f"update {update} was sampled from {claimed}. Training a batch "
            "onto an adapter that did not generate it breaks the on-policy "
            "assumption: every importance ratio would compare two "
            "distributions that never met."
        )
    print(f"parent: {parent}\n  hash {parent_hash} == .SAMPLED adapter_hash "
          f"(policy {ident['policy_version']})")

    policy_version = ident["policy_version"]
    inputs = load_live_update_inputs(u, policy_version=policy_version)

    # --- rehydrate the PAID scores. No pool exists in this container. ------
    rows = {r["key"]: r for r in read_jsonl(u.scores_path)
            if r.get("projection") == "semantic"}
    missing = [r["key"] for r in inputs.capture_rows if r["key"] not in rows]
    if missing:
        raise RuntimeError(
            f"{len(missing)} actions have no cached semantic score "
            f"({missing[:3]}); this path will not buy them"
        )
    projected = {
        k: ProjectedScore(
            key=k,
            teacher_logprob_by_index={
                int(i): float(v)
                for i, v in r["teacher_logprob_by_index"].items()
            },
            excluded={int(i): v for i, v in r["excluded"].items()},
            n_prefix_tokens=int(r.get("n_prefix_tokens", 0)),
            n_teacher_tokens=int(r.get("n_teacher_tokens", 0)),
            payload_report=r.get("payloads", {}),
        )
        for k, r in rows.items()
    }
    print(f"rehydrated {len(projected)} cached semantic scores; "
          "no teacher pool constructed")

    trainer = ReOPDTrainer(
        base_model=manifest.get("base_model", "Qwen/Qwen3-4B"),
        parent_adapter=parent,
        learning_rate=learning_rate,
        run_dir=out,
        device="cuda",
    )

    # Adam's moments, the scheduler and the RNG -- NOT just the weights.
    #
    # `resume_from=None` builds a FRESH optimizer. Chaining updates by pointing
    # `--parent-override` at the previous adapter therefore gives correct
    # weights with zero moment estimates every time: with bias correction, the
    # first step after a reset is near-full-magnitude regardless of history, so
    # ten "iterative" updates are ten independent first steps. Nothing in the
    # logs shows it -- `max_param_delta` reads 1.0e-05 either way, which is
    # exactly what both engineering updates reported.
    #
    # So the optimizer state comes from the update that produced this parent.
    # `ReOPDTrainer.load` reloads that checkpoint's adapter weights too, which
    # is what makes the pair consistent: its docstring warns that resuming
    # CK35's weights with update 19's optimizer "would be neither run".
    resume_from = None
    if update > 0:
        prev = run.update(update - 1).checkpoint_path
        if not (prev / "optimizer.pt").exists():
            raise RuntimeError(
                f"update {update} must resume optimizer state from update "
                f"{update - 1}, but {prev}/optimizer.pt is missing. Training "
                "with a fresh optimizer would silently change the recipe at "
                "every update while every reported metric looked normal."
            )
        resume_from = prev
    elif resume_optimizer_from:
        resume_from = Path(resume_optimizer_from)
        if not (resume_from / "optimizer.pt").exists():
            raise RuntimeError(
                f"--resume-optimizer-from {resume_from} has no optimizer.pt"
            )

    loaded = trainer.load(resume_from=resume_from)
    print(f"trainer: parent_hash={loaded.get('parent_hash')} "
          f"resumed={loaded.get('resumed')}"
          + (f" from update {loaded.get('resumed_from_update')}"
             if loaded.get("resumed") else " (fresh optimizer -- update 0)"))

    before = adapter_hash(parent)
    state = run_projected_train_stage(
        u, inputs, projected, trainer,
        run_dir=out, policy_version=policy_version,
        max_trace_share=inputs.max_trace_share,
    )
    vol.commit()

    cp = u.checkpoint_path
    child = adapter_hash(cp)
    moved = child != before

    lines = [
        f"ONE-STEP OPD -- {run_id} update {update}",
        "",
        f"  cached scores    : {len(projected)} reused, 0 new teacher calls",
        f"  parent           : {parent}",
        f"  parent hash      : {before}",
        f"  child hash       : {child}",
        f"  weights moved    : {moved}",
        "",
        f"  loss             : {state.get('loss')}",
        f"  grad_norm        : {state.get('grad_norm')}",
        f"  policy_version   : {state.get('policy_version')}",
        f"  parent_policy    : {state.get('parent_policy_hash')}",
        f"  reload_verified  : {state.get('reload_verified')}",
        f"  checkpoint       : {cp}",
        f"  stage            : {u.stage()}",
    ]
    for k in ("supervised_tokens", "clip_fraction", "max_param_delta",
              "n_examples"):
        if k in state:
            lines.append(f"  {k:17s}: {state[k]}")
    text = "\n".join(lines)
    print(text)

    if not moved:
        raise RuntimeError(
            "child adapter hash equals the parent: the step changed nothing"
        )
    return text



@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def token_shares(run_id: str = "", update: int = 0) -> str:
    """True per-task gradient share: supervised TOKENS, not turn counts.

    A turn share answers "how many turns did this task contribute"; the loss is
    normalised by supervised tokens, so that is the number that decides
    influence. They are not the same when turns differ in length, which for
    heterogeneous Tau2 conversations is always.
    """
    import json
    import os

    u = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id,
                     f"update-{update:03d}")
    acts = [json.loads(l) for l in open(os.path.join(u, "actions.jsonl"))
            if l.strip()]
    scores = {}
    for l in open(os.path.join(u, "scores.jsonl")):
        if l.strip():
            r = json.loads(l)
            scores[r["key"]] = r

    turns, raw, sup, adv = {}, {}, {}, {}
    t_raw = t_sup = 0
    t_adv = 0.0
    for a in acts:
        t = a["task_id"]
        sc = scores.get(a["key"], {})
        tl = sc.get("teacher_logprob_by_index", {}) or {}
        ntok = len(a["action_token_ids"])
        beh = list(a["behavior_logprobs"])
        mag = 0.0
        for idx, tv in tl.items():
            i = int(idx)
            if i >= len(beh):
                continue
            ls = float(beh[i])
            if abs(ls) < 1e-8:
                continue
            mag += abs((float(tv) / ls - 1.0) * ls)
        turns[t] = turns.get(t, 0) + 1
        raw[t] = raw.get(t, 0) + ntok
        sup[t] = sup.get(t, 0) + len(tl)
        adv[t] = adv.get(t, 0.0) + mag
        t_raw += ntok
        t_sup += len(tl)
        t_adv += mag

    n = len(acts)
    out = [f"TRUE SHARES -- {run_id} update {update}", "",
           f"  turns {n}   raw tokens {t_raw:,}   supervised {t_sup:,}", "",
           f"  {'task':<6}{'turns':>7}{'turn%':>9}{'sup_tok':>10}{'SUP%':>9}"
           f"{'|adv|%':>9}"]
    for t in sorted(turns):
        out.append(
            f"  {t:<6}{turns[t]:>7}{100 * turns[t] / n:>8.1f}%"
            f"{sup[t]:>10,}{100 * sup[t] / max(1, t_sup):>8.1f}%"
            f"{100 * adv[t] / max(1e-9, t_adv):>8.1f}%")
    out += ["",
            f"  worst by TURNS      : {max(turns.values()) / n:.4f}",
            f"  worst by SUP TOKENS : "
            f"{max(sup.values()) / max(1, t_sup):.4f}   <- gradient share",
            f"  worst by |advantage|: "
            f"{max(adv.values()) / max(1e-9, t_adv):.4f}   <- influence"]
    text = "\n".join(out)
    print(text)
    return text



@app.function(
    # NO gpu=. Rollout drives the SERVING endpoint over HTTP -- the generation
    # happens on that card, not in this container. The monolithic `train`
    # allocated a training GPU here and then held it through scoring too,
    # which over a 10-update run is hours of an idle card.
    image=image,
    volumes={VOLUME_MOUNT: vol, HF_CACHE_MOUNT: hf_cache},
    timeout=60 * 90,
    secrets=[modal.Secret.from_name("fireworks-api-key")],
)
def rollout_only(run_id: str = "", update: int = 0, api_base: str = "",
                 student_model: str = "", adapter_hash_expect: str = "",
                 tau2_src: str = "",
                 allow_missing_reasoning: bool = False) -> dict:
    """Stage 1: sample one update's episodes, stop at SAMPLED.

    No teacher pool, no trainer, no GPU. Every turn is archived the moment it
    lands, so a crash costs at most the episode in flight -- and nothing has
    been paid to DeepSeek yet, which is why this stage runs before scoring.

    The roster, seeds and generation settings come from the FROZEN manifest,
    not from CLI arguments, so a preregistered run cannot drift by a typo.
    """
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    if tau2_src:
        sys.path.insert(0, tau2_src)

    from vektori_trace.tau2.live_rollout import (
        RolloutSettings, Tau2EpisodeRunner, capture_live_update,
    )
    from vektori_trace.tau2.opd_stages import log, set_commit_fn, telemetry
    from vektori_trace.tau2.reopd_refresh import served_models
    from vektori_trace.tau2.reopd_state import RunState

    set_commit_fn(vol.commit)

    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = json.load(fh)
    run = RunState(out, n_updates=int(manifest.get("n_updates", 1)))
    u = run.update(update)
    u.validate()

    # Idempotent: a resumed pilot must never resample an update it already has.
    # The archived turns ARE the recovery mechanism.
    if u.reached("SAMPLED"):
        rep = json.loads(u.report_path.read_text())
        log(f"update {update} already SAMPLED "
            f"({rep.get('trainable_turns')} turns); nothing resampled")
        return rep

    if not api_base or not student_model:
        raise ValueError("--api-base and --student-model are required")

    # The endpoint must already serve the policy this update samples from.
    # vLLM resolves an unknown adapter name against the base model, so without
    # this the run would sample the BASE weights and report success.
    advertised = served_models(api_base, timeout=30.0)
    if student_model not in advertised:
        raise ValueError(
            f"endpoint does not advertise {student_model!r}; it serves "
            f"{advertised}. An unknown name silently resolves to the base "
            "model, so the adapter would do nothing."
        )

    # Provenance comes from the caller, which read it off the checkpoint that
    # is actually being served. A blank here is what made update 0 of the
    # 2026-08-28 proof unverifiable.
    if not adapter_hash_expect:
        raise ValueError(
            "--adapter-hash-expect is required: every archived episode is "
            "stamped with it, and batch_report checks the batch against it"
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        manifest.get("student_tokenizer", "Qwen/Qwen3-4B"),
        trust_remote_code=True)

    # Episodes come from the FROZEN plan, not a task x seed cross recomputed
    # here. `plan_update` crosses the same tasks and seeds at every update --
    # only the episode id carries the update index -- so a 10x8 run would
    # sample the same 8 scenarios ten times and call it 80 episodes.
    # `plans_by_update` names all 80 distinct (task, seed) pairs before update
    # 0 runs, which is what makes the schedule auditable afterwards.
    from vektori_trace.tau2.live_rollout import EpisodePlan

    by_update = manifest.get("plans_by_update")
    if not by_update:
        raise ValueError(
            "manifest has no plans_by_update: this run's episode schedule was "
            "never frozen, so what it sampled cannot be checked against what "
            "it intended. Build it with scripts/tau2_pilot_manifest.py."
        )
    if update >= len(by_update):
        raise ValueError(
            f"update {update} is outside the frozen plan "
            f"({len(by_update)} updates)"
        )
    # The plan must be the one that was frozen. A manifest edited after
    # generation -- a task swapped, a seed nudged -- would silently change the
    # experiment while every downstream check still passed.
    import hashlib as _hashlib

    recorded = manifest.get("plan_hash")
    actual = _hashlib.sha256(
        json.dumps(by_update, sort_keys=True).encode()).hexdigest()[:16]
    if recorded and recorded != actual:
        raise ValueError(
            f"plans_by_update hashes to {actual} but the manifest records "
            f"{recorded}: the schedule was modified after it was frozen. "
            "Refusing to sample a plan that is not the preregistered one."
        )

    block = by_update[update]
    expected_n = int(manifest.get("episodes_per_update", len(block)))
    if len(block) != expected_n:
        raise ValueError(
            f"update {update} has {len(block)} planned episodes, expected "
            f"{expected_n}"
        )
    ids = [p["episode_id"] for p in block]
    pairs = [(str(p["task_id"]), int(p["seed"])) for p in block]
    if len(set(ids)) != len(ids):
        raise ValueError(f"update {update} has duplicate episode ids")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"update {update} has duplicate (task, seed) pairs")
    pool = {str(t) for t in manifest.get("task_ids", [])}
    stray = sorted({t for t, _ in pairs} - pool) if pool else []
    if stray:
        raise ValueError(
            f"update {update} plans tasks outside the frozen pool: {stray}"
        )

    plans = [
        EpisodePlan(episode_id=p["episode_id"], task_id=str(p["task_id"]),
                    seed=int(p["seed"]))
        for p in block
    ]
    log(f"frozen plan for update {update} (plan_hash {actual}): "
        + ", ".join(f"{p.task_id}/s{p.seed}" for p in plans))

    settings = RolloutSettings(
        domain=manifest.get("domain", "retail"),
        student_model=student_model,
        api_base=api_base,
        policy_version=f"live-u{update:03d}",
        adapter_hash=adapter_hash_expect,
        gen_config_hash=manifest.get("gen_config_hash", ""),
        max_tokens=int(manifest.get("max_action_tokens", 4096)),
        max_input_tokens=int(manifest.get("max_input_tokens", 16384)),
        temperature=float(manifest.get("temperature", 1.0)),
        timeout=600.0,
        max_steps=int(manifest.get("max_turns", 100)),
        max_errors=10,
        user="user_simulator",
        user_model=manifest.get("user_model", USER_MODEL),
        user_model_args={},
        require_reasoning=not allow_missing_reasoning,
    )

    t0 = time.time()
    log(f"rolling out {len(plans)} episode(s) on {student_model} "
        f"(adapter {adapter_hash_expect}, policy live-u{update:03d})")
    report = capture_live_update(
        u, plans,
        settings=settings,
        teacher_context={
            "model": manifest.get("teacher_model", TEACHER_MODEL),
            "tokenizer": manifest.get("teacher_tokenizer", TEACHER_TOKENIZER),
            "renderer": manifest.get("teacher_renderer", "deepseek-v4-native"),
        },
        runner=Tau2EpisodeRunner(settings, tokenizer),
    )
    secs = round(time.time() - t0, 1)
    vol.commit()

    log(f"rollout took {secs}s: {report.get('trainable')} episodes, "
        f"{report.get('trainable_turns')} turns, {report.get('failed')} "
        f"failed, {report.get('discarded')} discarded")
    telemetry(out, {
        "event": "stage", "stage": "SAMPLED", "update": update,
        "seconds": secs, "n_episodes": report.get("trainable"),
        "n_turns": report.get("trainable_turns"),
        "failed": report.get("failed"), "discarded": report.get("discarded"),
        "adapter_hash": adapter_hash_expect,
    })
    return report


@app.function(
    # NO gpu=. The swap happens on the ALREADY-RUNNING serving card; this
    # container only issues the load/verify calls.
    image=image,
    volumes={VOLUME_MOUNT: vol},
    timeout=60 * 20,
)
def refresh_only(run_id: str = "", update: int = 0, api_base: str = "",
                 reload_url: str = "", base_served_name: str = "",
                 previous_served_name: str = "") -> dict:
    """Stage 4: serve update `update-1`'s checkpoint before sampling `update`.

    This is the step that makes the loop on-policy, and the one whose absence
    is invisible: skip it and update k samples from update k-2's weights while
    every marker, hash and loss still looks right. OPD is on-policy in the
    action -- `log pi_old` must come from the policy that sampled it -- so a
    stale endpoint makes every importance ratio compare two distributions that
    never met, with a finite loss and nothing in the logs to show for it.

    Verifies the swap by fingerprint rather than trusting the load call: the
    probe logprobs must actually move.
    """
    import json
    import os

    from transformers import AutoTokenizer

    from vektori_trace.tau2.live_train import refresh_live_policy
    from vektori_trace.tau2.opd_stages import set_commit_fn
    from vektori_trace.tau2.reopd_refresh import probe_logprobs
    from vektori_trace.tau2.reopd_state import RunState

    set_commit_fn(vol.commit)

    if update <= 0:
        raise ValueError(
            "update 0 needs no refresh: the endpoint already serves the SFT "
            "checkpoint it samples from"
        )
    if not api_base or not base_served_name:
        raise ValueError("--api-base and --base-served-name are required")

    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = json.load(fh)
    run = RunState(out, n_updates=int(manifest.get("n_updates", 1)))

    prev = run.update(update - 1)
    # The checkpoint must be complete BEFORE the endpoint is asked to serve it.
    prev.validate_checkpoint()

    tok = AutoTokenizer.from_pretrained(
        manifest.get("student_tokenizer", "Qwen/Qwen3-4B"),
        trust_remote_code=True)
    probe_ids = tok("You are a retail agent.",
                    add_special_tokens=False)["input_ids"][:256]

    # Fingerprint the CURRENT policy first, so the swap is verified against the
    # policy it replaces rather than accepted because a name appeared.
    served_now = previous_served_name or base_served_name
    before = probe_logprobs(api_base, served_now, probe_ids, timeout=300.0)

    class _Args:
        pass

    args = _Args()
    args.api_base = api_base
    args.reload_url = reload_url
    args.initial_served_name = base_served_name
    args.served_name = served_now
    args.student_model = served_now
    args.probe_prompt_ids = probe_ids
    args.probe_logprobs = before
    args.adapter_hash = ""

    rep = refresh_live_policy(args, update, prev.checkpoint_path, run_dir=out)
    vol.commit()

    return {
        "update": update,
        "served_name": args.student_model,
        "adapter_hash": args.adapter_hash,
        "max_logprob_delta": rep.get("max_logprob_delta"),
        "checkpoint": str(prev.checkpoint_path),
    }


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def pilot_status(run_id: str = "") -> dict:
    """Which stages are done, per update, read from the volume itself.

    The orchestrator runs on the box, which cannot see `/adapters` -- that path
    exists inside Modal containers. Without this it would test local paths that
    are always absent, conclude every stage is unfinished, and resample an
    update it already has.

    Also reports whether each checkpoint carries `optimizer.pt`, so the caller
    can refuse to dispatch a training GPU that is going to fail on arrival:
    update k>0 resumes update k-1's Adam state, and a fresh optimizer would be
    silently WRONG rather than an error.
    """
    import json
    import os

    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    if not os.path.isdir(out):
        return {"run_id": run_id, "exists": False, "updates": []}

    with open(os.path.join(out, "manifest.json")) as fh:
        manifest = json.load(fh)
    n = int(manifest.get("n_updates", 1))

    updates = []
    for i in range(n):
        u = os.path.join(out, f"update-{i:03d}")
        cp = os.path.join(u, "checkpoint")

        def _marker(name):
            p = os.path.join(u, name)
            if not os.path.isfile(p):
                return None
            try:
                with open(p) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return {}

        sampled = _marker(".SAMPLED")
        scored = _marker(".SCORED")
        trained = _marker(".TRAINED")
        n_scores = 0
        sp = os.path.join(u, "scores.jsonl")
        if os.path.isfile(sp):
            with open(sp) as fh:
                n_scores = sum(1 for line in fh if line.strip())

        updates.append({
            "update": i,
            "planned": os.path.isfile(os.path.join(u, ".PLANNED")),
            "sampled": sampled is not None,
            "scored": scored is not None,
            # What the ledger values DeepSeek spend from. Without it the
            # teacher contributes $0 to the running estimate.
            "teacher_input_tokens": (scored or {}).get("teacher_input_tokens"),
            "trained": trained is not None,
            "n_scores": n_scores,
            "sampled_adapter_hash": (sampled or {}).get("adapter_hash"),
            "sampled_policy_version": (sampled or {}).get("policy_version"),
            "n_actions": (sampled or {}).get("actions"),
            "trained_adapter_hash": (trained or {}).get("adapter_hash"),
            # What `preflight_checkpoint` needs, without a GPU to ask.
            "checkpoint_complete": all(
                os.path.isfile(os.path.join(cp, f))
                for f in ("optimizer.pt", "state.json", "adapter_config.json")
            ),
        })

    done = [u["update"] for u in updates if u["trained"]]
    nxt = next((u["update"] for u in updates if not u["trained"]), n)
    result = {
        "run_id": run_id,
        "exists": True,
        "n_updates": n,
        "plan_hash": manifest.get("plan_hash"),
        "parent_adapter_hash": manifest.get("adapter_hash"),
        "trained_updates": done,
        "next_update": nxt,
        "complete": len(done) == n,
        "updates": updates,
    }
    # A sentinel line, not pretty-printed JSON: the orchestrator parses this
    # exact prefix. Scanning for the first "{" through the last "}" breaks the
    # moment Modal or a library prints any other object.
    print("PILOT_STATUS_JSON=" + json.dumps(result, separators=(",", ":")))
    return result


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def stage_manifest(run_id: str = "", manifest_json: str = "") -> dict:
    """Create the run directory and freeze its manifest. No GPU, no teacher.

    Refuses to overwrite an existing manifest: the preregistration is the
    record of what the run intended, and silently replacing it would make the
    schedule unfalsifiable after the fact.
    """
    import hashlib
    import json
    import os

    if not run_id or not manifest_json:
        raise ValueError("--run-id and --manifest-json are required")

    manifest = json.loads(manifest_json)
    out = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id)
    path = os.path.join(out, "manifest.json")
    if os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)
        if existing.get("plan_hash") != manifest.get("plan_hash"):
            raise FileExistsError(
                f"{path} already exists with plan_hash "
                f"{existing.get('plan_hash')}, not {manifest.get('plan_hash')}. "
                "Refusing to overwrite a frozen preregistration; use a new "
                "run id."
            )
        print(f"manifest already staged (plan {existing.get('plan_hash')})")
        return {"run_id": run_id, "staged": False,
                "plan_hash": existing.get("plan_hash")}

    # The plan must hash to what it claims before anything samples from it.
    actual = hashlib.sha256(
        json.dumps(manifest["plans_by_update"], sort_keys=True).encode()
    ).hexdigest()[:16]
    if actual != manifest.get("plan_hash"):
        raise ValueError(
            f"plan_hash mismatch: manifest says {manifest.get('plan_hash')}, "
            f"content hashes to {actual}"
        )

    os.makedirs(out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    vol.commit()
    print(f"staged {path} (plan {actual}, "
          f"{manifest.get('n_planned_episodes')} episodes)")
    return {"run_id": run_id, "staged": True, "plan_hash": actual,
            "path": path}


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def inspect_failed_turns(run_id: str = "", update: int = 0) -> str:
    """Why did a capture fail? Read the RAW bytes, do not infer.

    "No non-empty <think> span" is a statement about the PARSER, not about the
    model. It is consistent with at least three different things: the model
    really answered directly, it reasoned in a form `split_generation` does not
    recognise (malformed or absent tags, a different marker), or something
    stripped the span server-side. Those have different fixes, and guessing
    between them costs a rollout.

    So this prints the first 600 raw characters of every failed generation.
    No GPU, no teacher, no rollout.
    """
    import base64
    import glob
    import json
    import os

    root = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME, run_id,
                        f"update-{update:03d}", "live_archive", "turns")
    if not os.path.isdir(root):
        return f"no turns archive at {root}"

    out = [f"RAW FAILED GENERATIONS -- {run_id} update {update}", ""]
    n_fail = 0
    for path in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        ep = os.path.basename(path)[:-6]
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            # A FailedTurn row carries an error; a capture row does not.
            err = r.get("error") or r.get("failure") or r.get("reason")
            if not err:
                continue
            n_fail += 1
            raw = r.get("raw_text") or r.get("text") or ""
            if not raw:
                b64 = r.get("raw_sampled_bytes_b64") or r.get("action_bytes_b64")
                if b64:
                    raw = base64.b64decode(b64).decode("utf-8", "replace")
            out += [
                f"── {ep} turn {r.get('turn_index')} ──",
                f"   error       : {str(err)[:150]}",
                f"   finish      : {r.get('finish_reason')}",
                f"   n_tokens    : {len(r.get('sampled_token_ids') or r.get('action_token_ids') or [])}",
                f"   has '<think>' : {'<think>' in raw}",
                f"   has '</think>': {'</think>' in raw}",
                f"   raw[:600]   : {raw[:600]!r}",
                "",
            ]
    if not n_fail:
        out.append("no failed-turn rows found (they may not carry an error key)")
    text = "\n".join(out)
    print(text)
    return text

@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 10)
def show_markers(run_id: str = "", update: int = 0) -> str:
    """Dump a run's manifest and stage markers verbatim. Read-only.

    Exists so provenance can be audited from what is *on the volume* rather
    than from the report of the function that wrote it.
    """
    import json
    import os

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    run = os.path.join(base, run_id)
    u = os.path.join(run, f"update-{update:03d}")
    out = [f"run {run_id} update {update}", ""]
    mf = os.path.join(run, "manifest.json")
    if os.path.isfile(mf):
        with open(mf) as fh:
            out += ["=== manifest.json ===", json.dumps(json.load(fh),
                                                        indent=1,
                                                        sort_keys=True)]
    for marker in (".PLANNED", ".SAMPLED", ".SCORED", ".TRAINED"):
        p = os.path.join(u, marker)
        out += ["", f"=== {marker} ==="]
        if os.path.isfile(p):
            with open(p) as fh:
                out.append(json.dumps(json.load(fh), indent=1, sort_keys=True))
        else:
            out.append("(absent)")
    text = "\n".join(out)
    print(text)
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 20)
def build_canary(source_run_id: str = "", canary_run_id: str = "",
                 update: int = 0) -> str:
    """Gate 4 -- a fresh SAMPLED-only run directory from immutable evidence.

    The repaired scorer has never made a real teacher call, so the ~$0.04
    re-score needs somewhere to run that is *not* the failed run. Resuming that
    one would short-circuit the repair: its `.SCORED`/`.TRAINED` markers and its
    31 contaminated score rows would be reused verbatim, the projected scorer
    would never execute, and the run would report success having changed
    nothing.

    So this copies only what sampling produced and the teacher cannot influence:

        actions.jsonl      the 31 archived generations (ids, bytes, logprobs)
        rendered.json      the semantic histories the teacher will condition on
        live_archive/      episodes, turns, events, Tau2 SimulationRuns
        .PLANNED/.SAMPLED  the stage markers sampling legitimately reached

    and refuses to copy scores, checkpoints, optimizer state or the
    `SCORED`/`TRAINED` markers. The parent hash is read **fresh from the
    adapter weights on the volume**, never from the source manifest -- update 0
    archived `""` there, so trusting it would propagate the very blank this
    gate exists to replace.

    The stage markers are **reconstructed, not copied**, for the same reason.
    The source `.PLANNED` and `.SAMPLED` both carry `"adapter_hash": ""`, so
    copying them verbatim would leave the canary's lifecycle provenance
    blank while its manifest claimed the real hash -- a contradiction between
    two records of the same fact, and exactly the defect this gate exists to
    remove. Everything the markers assert about *sampling* (plans, episode
    ids, counts, policy version, generation-config and teacher-context hashes)
    is preserved verbatim: that evidence is real and the teacher cannot
    influence it. Only the empty provenance field is replaced.
    """
    import glob
    import json
    import os
    import shutil

    from vektori_trace.tau2.reopd_checkpoint import adapter_hash

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, source_run_id)] if source_run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    if not runs or not os.path.isdir(runs[-1]):
        raise FileNotFoundError(f"no source run under {base}")
    src_run = runs[-1]
    src = os.path.join(src_run, f"update-{update:03d}")

    # --- the parent, hashed fresh from its weights ------------------------
    parent = os.path.join(VOLUME_MOUNT, PARENT_IN_VOLUME)
    got = adapter_hash(parent)
    if got != PARENT_ADAPTER_HASH:
        raise ValueError(
            f"parent adapter at {parent} hashes to {got}, but this file pins "
            f"{PARENT_ADAPTER_HASH}. The canary would train from a different "
            "adapter than it claims."
        )

    name = canary_run_id or f"canary_rescore_{os.path.basename(src_run)}"
    dst_run = os.path.join(base, name)
    if os.path.exists(dst_run):
        raise FileExistsError(
            f"{dst_run} already exists; refusing to overwrite. Pass a new "
            "--canary-run-id rather than resuming a directory whose history "
            "is unknown."
        )
    dst = os.path.join(dst_run, f"update-{update:03d}")
    os.makedirs(dst, exist_ok=True)

    # --- copy ONLY the immutable sampling evidence ------------------------
    copied = []
    for fname in ("actions.jsonl", "rendered.json"):
        s = os.path.join(src, fname)
        if not os.path.isfile(s):
            raise FileNotFoundError(f"source update is missing {fname}")
        shutil.copy2(s, os.path.join(dst, fname))
        copied.append(fname)
    if os.path.isdir(os.path.join(src, "live_archive")):
        shutil.copytree(os.path.join(src, "live_archive"),
                        os.path.join(dst, "live_archive"))
        copied.append("live_archive/")
    # --- reconstruct the lifecycle markers with REAL provenance -----------
    # Copying these verbatim is what the first build did wrong: both carry
    # `adapter_hash: ""`. Preserve every sampling fact, replace only that.
    source_markers = {}
    rebuilt = {}
    for marker in (".PLANNED", ".SAMPLED"):
        s = os.path.join(src, marker)
        if not os.path.isfile(s):
            raise FileNotFoundError(
                f"source update is missing {marker}; a canary cannot claim a "
                "stage the evidence does not support"
            )
        with open(s) as fh:
            payload = json.load(fh)
        source_markers[marker] = {
            "adapter_hash": payload.get("adapter_hash", None),
            "policy_version": payload.get("policy_version"),
            "gen_config_hash": payload.get("gen_config_hash"),
        }
        payload["adapter_hash"] = got
        payload["parent_adapter_hash"] = got
        payload["provenance"] = (
            f"rebuilt for canary from {os.path.basename(src_run)} "
            f"update {update}; adapter_hash restored from parent weights"
        )
        with open(os.path.join(dst, marker), "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        rebuilt[marker] = payload
        copied.append(f"{marker} (rebuilt)")

    # --- the manifest, with a REAL parent hash and full provenance --------
    with open(os.path.join(src_run, "manifest.json")) as fh:
        manifest = json.load(fh)
    manifest["adapter_hash"] = got
    manifest["parent"] = "/" + PARENT_IN_VOLUME.lstrip("/")
    manifest["parent_adapter_hash"] = got
    manifest["mode"] = "canary-rescore"
    manifest["n_updates"] = 1
    manifest["source_run"] = os.path.basename(src_run)
    manifest["source_update"] = update
    manifest["derived_from"] = (
        "SAMPLED-only evidence; scores, checkpoints and SCORED/TRAINED "
        "markers deliberately not copied"
    )
    manifest["scoring"] = "semantic projection"
    # What the source markers actually said, kept so the substitution is
    # auditable rather than invisible. The blank is the defect being repaired.
    manifest["source_marker_provenance"] = source_markers
    manifest["marker_rebuild"] = (
        "adapter_hash in .PLANNED/.SAMPLED was empty in the source and has "
        f"been set to the freshly hashed parent weights ({got}); all sampling "
        "fields preserved verbatim"
    )
    with open(os.path.join(dst_run, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    for fname in ("retail_tools.json",):
        s = os.path.join(src_run, fname)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst_run, fname))

    vol.commit()

    # --- prove the thing that would silently ruin the re-score ------------
    forbidden = [f for f in (".SCORED", ".TRAINED", "scores.jsonl",
                             "checkpoint")
                 if os.path.exists(os.path.join(dst, f))]
    n_actions = sum(1 for l in open(os.path.join(dst, "actions.jsonl"))
                    if l.strip())
    with open(os.path.join(dst, "rendered.json")) as fh:
        n_rendered = len(json.load(fh))

    # --- the four records of the parent must agree, read back off disk ----
    # Read from the written files rather than the in-memory values: the point
    # is that what a later resume will *load* is consistent, not what this
    # function believed it wrote.
    with open(os.path.join(dst_run, "manifest.json")) as fh:
        m_disk = json.load(fh)
    with open(os.path.join(dst, ".PLANNED")) as fh:
        p_disk = json.load(fh)
    with open(os.path.join(dst, ".SAMPLED")) as fh:
        s_disk = json.load(fh)
    hashes = {
        "parent weights (fresh)": got,
        "manifest.adapter_hash": m_disk.get("adapter_hash"),
        ".PLANNED.adapter_hash": p_disk.get("adapter_hash"),
        ".SAMPLED.adapter_hash": s_disk.get("adapter_hash"),
        "pinned constant": PARENT_ADAPTER_HASH,
    }
    disagree = {k: v for k, v in hashes.items() if v != PARENT_ADAPTER_HASH}

    out = [
        "GATE 4 -- fresh canary from immutable SAMPLED evidence",
        "",
        f"  source run      : {os.path.basename(src_run)} update {update}",
        f"  canary run      : {name}",
        f"  path            : {dst_run}",
        "",
        f"  parent          : {parent}",
        f"  copied          : {', '.join(copied)}",
        f"  actions         : {n_actions}",
        f"  rendered        : {n_rendered} histories",
        f"  forbidden found : {forbidden or 'none'}",
        "",
        "  PARENT PROVENANCE (all five must agree)",
    ]
    for k, v in hashes.items():
        flag = "ok " if v == PARENT_ADAPTER_HASH else "BAD"
        out.append(f"    [{flag}] {k:24s} {v!r}")
    out += [
        "",
        "  source markers said (the defect being repaired):",
    ]
    for k, v in source_markers.items():
        out.append(f"    {k:10s} adapter_hash={v['adapter_hash']!r}")
    out += [
        "",
        f"  stage           : "
        f"{'SAMPLED (ready to score)' if not forbidden and not disagree else 'CONTAMINATED'}",
    ]
    text = "\n".join(out)
    print(text)
    if forbidden:
        raise RuntimeError(
            f"canary carries {forbidden}; it would short-circuit the re-score"
        )
    if n_actions != n_rendered:
        raise RuntimeError(
            f"{n_actions} actions but {n_rendered} rendered histories"
        )
    if disagree:
        raise RuntimeError(
            f"parent provenance disagrees across records: {disagree}; the "
            "canary would train from an adapter it cannot name consistently"
        )
    return text


@app.function(image=image, volumes={VOLUME_MOUNT: vol}, timeout=60 * 20)
def predict_advantages(run_id: str = "", update: int = 0) -> str:
    """What the OLD advantages become once the projection's mask is applied.

    Free preview, no teacher call: it takes the advantages already paid for in
    the archived run and reports them under the NEW eligibility rules -- which
    tokens keep credit, which lose it, and what happens to the tails.

    This is a *lower bound* on the improvement, not the post-fix distribution:
    the surviving numbers were still computed against a chat-mode prefix and
    raw-byte alignment. What it does show exactly is the damage the mask
    removes -- the -55 tool markup and the -27 EOS.
    """
    import base64
    import glob
    import json
    import os

    from vektori_trace.align import align_by_bytes
    from vektori_trace.chunk_opd import assign_chunk_advantages
    from vektori_trace.tau2.live_projection import project_action
    from vektori_trace.tau2.live_token_classes import classify_action

    base = os.path.join(VOLUME_MOUNT, RUNS_IN_VOLUME)
    runs = ([os.path.join(base, run_id)] if run_id
            else sorted(glob.glob(os.path.join(base, "two_update_proof_*")),
                        key=os.path.getmtime))
    run = runs[-1]
    u = os.path.join(run, f"update-{update:03d}")
    actions = {json.loads(l)["key"]: json.loads(l)
               for l in open(os.path.join(u, "actions.jsonl")) if l.strip()}
    scores = {json.loads(l)["key"]: json.loads(l)
              for l in open(os.path.join(u, "scores.jsonl")) if l.strip()}

    kept: list = []
    dropped: list = []
    kept_by_cls: dict = {}
    drop_by_cls: dict = {}
    drop_extremes: list = []

    for key, a in sorted(actions.items()):
        sc = scores.get(key)
        if sc is None:
            continue
        stu = [base64.b64decode(x) for x in a["action_token_bytes_b64"]]
        tea = [base64.b64decode(x) for x in sc["teacher_token_bytes_b64"]]
        advs, sup, _st = assign_chunk_advantages(
            align_by_bytes(stu, tea),
            [float(x) for x in a["behavior_logprobs"]],
            [float(x) for x in sc["teacher_logprobs"]],
        )
        raw = b"".join(stu).decode("utf-8", "replace")
        trimmed = raw
        for spc in ("<|im_end|>", "<|endoftext|>"):
            if trimmed.endswith(spc):
                trimmed = trimmed[: -len(spc)]
        proj = project_action(trimmed, stu)
        cls = classify_action(stu)
        for i, (adv, s) in enumerate(zip(advs, sup)):
            if not s:
                continue
            if i in proj.supervised:
                kept.append(float(adv))
                kept_by_cls.setdefault(cls[i], []).append(float(adv))
            else:
                dropped.append(float(adv))
                drop_by_cls.setdefault(cls[i], []).append(float(adv))
                drop_extremes.append((float(adv),
                                      stu[i].decode("utf-8", "replace")))

    def stats(xs):
        if not xs:
            return "none"
        xs2 = sorted(xs)
        pos = sum(1 for x in xs if x > 0)
        return (f"n={len(xs):6d} +{pos:5d} -{len(xs)-pos:5d} "
                f"mean={sum(xs)/len(xs):+.4f} min={xs2[0]:+.3f} "
                f"max={xs2[-1]:+.3f} p1={xs2[len(xs2)//100]:+.3f}")

    drop_extremes.sort()
    out = [
        f"run {os.path.basename(run)} update {update}",
        "",
        "BEFORE (everything the old update trained on):",
        f"  {stats(kept + dropped)}",
        "",
        "AFTER the projection mask:",
        f"  KEPT    {stats(kept)}",
        f"  DROPPED {stats(dropped)}",
        "",
        "kept, by class:",
    ]
    for c, xs in sorted(kept_by_cls.items()):
        out.append(f"  {c:10s} {stats(xs)}")
    out.append("")
    out.append("dropped, by class:")
    for c, xs in sorted(drop_by_cls.items()):
        out.append(f"  {c:10s} {stats(xs)}")
    out.append("")
    out.append("worst advantages the mask REMOVES:")
    for adv, t in drop_extremes[:12]:
        out.append(f"  {adv:+10.3f}  {t!r}")
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
    projection_only: bool = False,
    reconcile_only: bool = False,
    scoring_dryrun_only: bool = False,
    predict_only: bool = False,
    build_canary_only: bool = False,
    show_markers_only: bool = False,
    token_shares_only: bool = False,
    rescore_only: bool = False,
    one_step_only: bool = False,
    parent_override: str = "",
    clamp_preview: float = 0.0,
    canary_run_id: str = "",
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

    if token_shares_only:
        token_shares.remote(run_id, classify_update)
        return

    if show_markers_only:
        show_markers.remote(run_id, classify_update)
        return

    if one_step_only:
        print(f"one-step: TRAINING GPU on {run_id!r} update {classify_update}")
        print("  cached scores only, no teacher pool, no rollout, no endpoint")
        one_step.remote(run_id, classify_update, learning_rate,
                        parent_override, resume_optimizer_from)
        return

    if rescore_only:
        # The paid step. Scoring only: no trainer is constructed, no GPU is
        # requested, and the update stops at SCORED so the first valid
        # advantage distribution can be read before any weight moves.
        print(f"rescore: SCORING ONLY on {run_id!r} update {classify_update}")
        print("  no optimizer step, no checkpoint, no reload, no GPU")
        rescore.remote(run_id, classify_update, clamp_preview)
        return

    if build_canary_only:
        print("build-canary: CPU only, no GPU, no teacher call")
        build_canary.remote(run_id, canary_run_id, classify_update)
        return

    if predict_only:
        predict_advantages.remote(run_id, classify_update)
        return

    if scoring_dryrun_only:
        scoring_dryrun.remote(run_id, classify_update)
        return

    if reconcile_only:
        reconcile_counts.remote(run_id, classify_update)
        return

    if projection_only:
        projection_report.remote(run_id, classify_update)
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
        parent_override,
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
