#!/usr/bin/env python3
"""Live Tau2 evaluation of `A_warm` checkpoints and the frozen baseline.

This is the missing piece named in `docs/TAU2-SFT-HANDOFF.md` §7.1: three
`A_warm` checkpoints exist and nothing can *select* among them, because V2 §6.4
forbids selecting on training loss. Selection needs live Tau2 reward on S16.

**Correction to the handoff.** It says `serve.py` has "no `--tool-call-parser`
flag" and calls that the first thing to fix. `serve_model` does not need one:
it forwards `extra_vllm_args` verbatim (serve.py), and
`scripts/run_tau2_smoke.py` has passed `--enable-auto-tool-choice`,
`--reasoning-parser qwen3` and a per-model `--tool-call-parser` through that
door for a while. Nothing was structurally blocked. What was actually missing
is what this file adds: adapter serving wired to a *partition* of the frozen
split, with the arms evaluated under one base load.

What this does NOT do is decide anything. It runs arms and writes tau2's own
result files. Reading them against the V2 §10 gates is a separate step, on
purpose -- a script that both produces and grades a number is a script that can
quietly grade it wrong.

Arms
----
One vLLM boot hosts the frozen base *and* every checkpoint, because a LoRA
adapter is ~132 MB on top of base weights that cost minutes and most of the
VRAM to load. `ServedModel.adapter_models` exists for exactly this; Phase 7
graded seven checkpoints on one load. tau2 is then invoked once per arm, with
the `model` name selecting the adapter server-side:

    A0        Qwen3-4B                  frozen base, the §7.1 baseline
    ck35      a_warm .../checkpoint-35   epoch 1
    ck70      a_warm .../checkpoint-70   epoch 2
    ck105     a_warm .../checkpoint-105  epoch 3

Guardrails that exist because each one has cost a run somewhere in this repo:

* **Partition is refused by name.** S16 selects, C30 is a pre-adaptation
  baseline (§7.1, measurable only before continuation training starts). F38 is
  the frozen final test and is rejected outright -- it is run once, after every
  recipe is locked, and never from a selection script.
* **Context is pinned, not defaulted.** See `_model_info` below; the default
  arithmetic silently starves the prompt budget.
* **`--max-loras` is set.** vLLM defaults it to 1 and refuses the second
  adapter *after* the GPU is allocated.
* **Every arm's served name is verified against /v1/models before any rollout.**
  A name vLLM does not advertise resolves to base weights or 404s, and both
  read as "the adapter changed nothing" with a full GPU bill behind them.

Nothing here allocates a GPU without `--yes`; `--dry-run` is CPU-only and
prints every command. Per CLAUDE.md, a GPU launch needs explicit per-run
approval, and this plan is not that approval.

    python scripts/tau2_eval_modal.py --dry-run
    python scripts/tau2_eval_modal.py --partition S16 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vektori_trace.runtime.modal_env import VOLUME_MOUNT  # noqa: E402
from vektori_trace.runtime.serve import serve_model  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from run_replay_opd import require_endpoint_model  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-4B"

# The trained artifact (handoff §2). Checkpoints are subdirectories of the run.
#
# The prefix is `VOLUME_MOUNT` ("/adapters"), not "/vol": `_resolve_volume_adapter`
# accepts a path only if it starts with the real mount, and otherwise tries to
# stage it as a *local* directory -- which fails with a message about the path
# being "neither a Volume path nor a local adapter dir". Derived from the
# constant rather than written out, so a remount cannot leave this stale.
DEFAULT_RUN = f"{VOLUME_MOUNT}/tau2/runs/a_warm_20260825_003343"

# Serving pins (handoff §4.3, §5). 16,384 is the measured cap: at 8,192 only
# 20/73 retail traces survive, and 32,768 buys nothing while doubling KV cache.
MAX_MODEL_LEN = 16384
# Reserve for generation. Measured generation p99 <= 344 across all domains, so
# 2,048 is ~6x headroom, and it leaves 14,336 for the prompt against a measured
# retail final-turn p90 of ~11,942.
GENERATION_RESERVE = 2048

# S16 selects; C30 is the §7.1 pre-adaptation baseline. F38 is deliberately
# absent -- see `_resolve_tasks`.
EVALUABLE_PARTITIONS = ("S16", "C30", "W30")

# Qwen3 emits <tool_call> blocks; `hermes` is what parses them. The handoff
# recorded qwen3_coder extracting *nothing* from a Qwen3 dense model -- 22 raw
# blocks left in content, 0 parsed, no tool ever executed. Verified for this
# corpus by `tau2_parser_parity_modal.py`: 25/25 targets round-trip through
# Hermes2ProToolParser.
VLLM_ARGS = [
    "--enforce-eager",
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
]


def _model_info() -> dict:
    """Token budget litellm advertises. Pinned, because the default starves it.

    `serve_model` clamps an unspecified budget to `out_cap = min(default,
    max_model_len // 2)`, which at a 16,384 window gives 8,192 in and 8,192
    out. Retail final-turn prompts reach p90 ~11,942 (handoff §4.3), so most
    S16 episodes would be refused for context -- and a context refusal that
    reaches tau2 is graded as a model failure, not as infrastructure. Passing
    the split explicitly is what keeps the prompt budget at 14,336.
    """
    return {
        "max_input_tokens": MAX_MODEL_LEN - GENERATION_RESERVE,
        "max_output_tokens": GENERATION_RESERVE,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }


def _resolve_tasks(artifacts: str, partition: str) -> tuple[list[str], str]:
    """Task ids for `partition`, from the frozen manifest. Returns (ids, hash).

    F38 is refused here rather than in the caller so that no flag combination
    reaches it. It is the single frozen final comparison (V2 §4, §11): run once
    after every recipe is locked. Evaluating it during checkpoint selection is
    how a blind test stops being blind, and unlike most mistakes in this repo
    it cannot be undone by rerunning something.
    """
    if partition == "F38":
        raise SystemExit(
            "F38 is the frozen final test and is never run from a selection "
            "script. It is evaluated once, after every recipe is locked "
            "(V2 §4, §11). Use S16 to select."
        )
    if partition not in EVALUABLE_PARTITIONS:
        raise SystemExit(
            f"unknown partition {partition!r}; "
            f"expected one of {', '.join(EVALUABLE_PARTITIONS)}"
        )
    man_path = os.path.join(artifacts, "task_split_manifest.json")
    if not os.path.exists(man_path):
        raise SystemExit(
            f"missing {man_path}. The split manifest is built on the box by "
            f"scripts/tau2_build_split.py; this script never regenerates it, "
            f"because a manifest rebuilt under an evaluation is a different "
            f"experiment."
        )
    manifest = json.load(open(man_path))
    ids = list(manifest["partitions"][partition])
    return ids, str(manifest.get("manifest_hash", "?"))


def _canonical_base_name() -> str:
    """The name vLLM serves the frozen base under.

    `serve_model` derives it from the base model with `_canonical_name`, which
    strips the org prefix: "Qwen/Qwen3-4B" is served as "Qwen3-4B". Asking for
    the full HF path 404s every call. Imported rather than reimplemented so the
    two cannot drift.
    """
    from vektori_trace.runtime.serve import _canonical_name

    return _canonical_name(BASE_MODEL)


def _subset(ids: list[str], partition: str,
            tasks: list[str] | None, max_tasks: int | None) -> list[str]:
    """Narrow a partition to a probe-sized subset, without leaving it.

    A checkpoint smoke test does not need all 16 selection tasks. What it does
    need is that the tasks it runs are *still selection tasks* -- a typo that
    silently pulls in a test task is the one failure here that rerunning cannot
    undo, so an explicit id is checked against the partition rather than
    trusted.

    Prefer naming ids over `--max-tasks`: the baseline 4B/8B trajectories from
    2026-08-24 cover tasks 57, 73, 75 and 93, so choosing from those gives the
    probe free context that would otherwise cost a rerun. `--max-tasks` takes a
    manifest-order prefix, which carries no such guarantee.
    """
    if tasks:
        stray = [t for t in tasks if t not in set(ids)]
        if stray:
            raise SystemExit(
                f"tasks {stray} are not in {partition}. Supplying an id from "
                f"another partition is how a frozen split stops being frozen; "
                f"{partition} holds: {' '.join(ids)}"
            )
        # De-duplicate while keeping the order the caller asked for.
        seen: set[str] = set()
        return [t for t in tasks if not (t in seen or seen.add(t))]
    if max_tasks is not None:
        if max_tasks < 1:
            raise SystemExit(f"--max-tasks must be >= 1, got {max_tasks}")
        return ids[:max_tasks]
    return ids


def _arms(run_dir: str, checkpoints: list[str]) -> dict[str, str]:
    """Served suffix -> adapter path. `A0` is the base and has no entry."""
    table: dict[str, str] = {}
    for ck in checkpoints:
        table[f"ck{ck}"] = f"{run_dir.rstrip('/')}/checkpoint-{ck}"
    return table


def _spend_so_far(results: Path) -> float | None:
    """Agent + user cost across every simulation written so far, or None.

    Partial reads are expected: tau2 rewrites the file as it goes, so a poll can
    land mid-write. The next poll sees a complete file.
    """
    try:
        d = json.loads(results.read_text())
    except Exception:
        return None
    return sum((s.get("agent_cost") or 0) + (s.get("user_cost") or 0)
               for s in d.get("simulations", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="S16",
                    help="S16 selects; C30 is the §7.1 pre-adaptation baseline")
    ap.add_argument("--artifacts", default="/data/tau2/artifacts_16384")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="run only these task ids; each is asserted to belong "
                         "to --partition. Baseline trajectories exist for "
                         "57/73/75/93, so a probe over those has free context.")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="run the first N tasks of the partition (manifest "
                         "order). Prefer --tasks when the ids matter.")
    ap.add_argument("--run-dir", default=DEFAULT_RUN,
                    help="Volume path of the A_warm run holding the checkpoints")
    ap.add_argument("--checkpoints", nargs="*", default=["35", "70", "105"],
                    help="checkpoint numbers to grade; empty grades base only")
    ap.add_argument("--no-base", action="store_true",
                    help="skip the frozen A0 arm (it is the §7.1 baseline)")
    ap.add_argument("--gpu", default="L40S")
    ap.add_argument("--num-trials", type=int, default=1,
                    help="1 is a plumbing probe. §4.4 wants multiple fixed "
                         "trials before any number is read as a rate.")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--user-llm",
                    default="fireworks_ai/accounts/fireworks/models/deepseek-v4-flash-0731")
    ap.add_argument("--tau2-dir", default="/data/tau2")
    ap.add_argument("--save-prefix", default=None,
                    help="default: tau2_eval_<partition>_<timestamp>")
    # Measured ~$0.0075/retail task with deepseek-v4-flash on both sides. Four
    # arms x 16 tasks is well under $1; the cap stops a runaway retry loop.
    ap.add_argument("--max-cost", type=float, default=2.0,
                    help="stop an arm once agent+user spend exceeds this (USD)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command and exit; allocates nothing")
    ap.add_argument("--yes", action="store_true",
                    help="required to allocate a GPU (CLAUDE.md: per-run approval)")
    a = ap.parse_args()

    task_ids, man_hash = _resolve_tasks(a.artifacts, a.partition)
    all_n = len(task_ids)
    task_ids = _subset(task_ids, a.partition, a.tasks, a.max_tasks)
    adapters = _arms(a.run_dir, a.checkpoints)
    if not adapters and a.no_base:
        raise SystemExit("nothing to evaluate: no checkpoints and --no-base")

    prefix = a.save_prefix or f"tau2_eval_{a.partition.lower()}_{time.strftime('%Y%m%d_%H%M%S')}"

    # Checked before the environment checks below, not after. An unapproved run
    # must be refused *for that reason*, not incidentally because some unrelated
    # dependency happened to be missing first -- otherwise the approval gate is
    # only as reliable as the box's install state.
    if not a.dry_run and not a.yes:
        raise SystemExit(
            "this allocates a GPU and needs explicit per-run approval "
            "(CLAUDE.md). Re-run with --yes, or --dry-run to see the commands."
        )

    tau2_bin = shutil.which("tau2") or str(Path(sys.executable).parent / "tau2")
    # Checked only on a real run. A dry-run exists to be runnable anywhere --
    # including a laptop that has never installed tau2 -- so that the task ids,
    # the arm table and the context budget can be reviewed before anyone is
    # near a GPU. Requiring the binary here would defeat that.
    if not a.dry_run and not Path(tau2_bin).exists():
        raise SystemExit(
            f"tau2 entry point not found at {tau2_bin}; "
            f"uv pip install --python {sys.executable} -e {a.tau2_dir}"
        )

    if not a.dry_run and not os.environ.get("FIREWORKS_API_KEY"):
        # tau2 also reads <tau2-dir>/.env, so a missing env var is not fatal --
        # but failing here beats discovering it after the model has loaded.
        if not (Path(a.tau2_dir) / ".env").exists():
            raise SystemExit(
                "FIREWORKS_API_KEY unset and no .env; the user simulator cannot run"
            )

    def tau2_cmd(api_base: str, agent_llm: str, save_to: str) -> list[str]:
        args = f'{{"api_base": "{api_base}", "temperature": 0.0}}'
        return [
            tau2_bin, "run",
            "--domain", a.domain,
            "--task-ids", *task_ids,
            "--num-trials", str(a.num_trials),
            "--agent-llm", agent_llm,
            "--agent-llm-args", args,
            "--user-llm", a.user_llm,
            "--user-llm-args", '{"temperature": 0.0}',
            "--max-concurrency", str(a.max_concurrency),
            "--save-to", save_to,
            # tau2 v0.2.0 defaults to ERROR, which hides the request/response
            # detail that tells an empty tool_calls apart from a routing 404.
            "--log-level", "DEBUG",
        ]

    arm_names = ([] if a.no_base else ["A0"]) + sorted(adapters)
    subset = "" if len(task_ids) == all_n else f" (subset of {all_n})"
    print(f"partition   {a.partition}: {len(task_ids)} tasks{subset}, "
          f"manifest_hash={man_hash}")
    print(f"tasks       {' '.join(task_ids)}")
    print(f"arms        {', '.join(arm_names)}")
    print(f"context     max_model_len={a.max_model_len}, "
          f"prompt budget {a.max_model_len - GENERATION_RESERVE}, "
          f"generation reserve {GENERATION_RESERVE}")
    print(f"episodes    {len(arm_names)} arms x {len(task_ids)} tasks x "
          f"{a.num_trials} trials = {len(arm_names) * len(task_ids) * a.num_trials}")

    if a.dry_run:
        print("\nwould serve:", BASE_MODEL, "on", a.gpu,
              "with", " ".join(VLLM_ARGS))
        for suffix, path in sorted(adapters.items()):
            print(f"  lora {suffix:8s} {path}")
        for arm in arm_names:
            # The served name is only knowable once the endpoint is up --
            # `serve_model` derives it from the base model and the suffix. A
            # guess printed here is what let a wrong model name survive a
            # dry-run and 404 minutes into a paid run.
            print(f"\n[{arm}] would run:")
            print("  " + " ".join(
                tau2_cmd("<api_base>", f"hosted_vllm/<{arm} @ boot>",
                         f"{prefix}_{arm.lower()}")))
        return 0

    results_dir = Path(a.tau2_dir) / "data" / "simulations"
    rc_total = 0
    t0 = time.time()
    with serve_model(
        BASE_MODEL,
        adapter_paths=adapters or None,
        # vLLM defaults --max-loras to 1 and refuses the second adapter after
        # the GPU is already allocated. len(adapters) is the exact need.
        max_loras=max(1, len(adapters)),
        gpu=a.gpu,
        max_model_len=a.max_model_len,
        model_info=_model_info(),
        gpu_memory_utilization=0.90,
        extra_vllm_args=list(VLLM_ARGS),
    ) as served:
        print(f"[serve] up in {time.time()-t0:.0f}s at {served.api_base}", flush=True)

        # Map arm -> served name. `serve_model` registers each adapter as
        # "<canonical base>-<suffix>"; the base keeps the canonical name. Ask
        # the endpoint rather than reconstructing the string, then verify every
        # one before spending anything: an unadvertised name silently resolves
        # to base weights, which reads as "the adapter did nothing".
        served_for: dict[str, str] = {}
        if not a.no_base:
            # Derived from the base model, never inferred from what is
            # advertised. Picking "whatever is not an adapter" out of the served
            # list is how A0 silently becomes one of the checkpoints -- and a
            # baseline that is actually a trained arm makes every delta wrong
            # in a direction nothing in the logs would show.
            served_for["A0"] = _canonical_base_name()
        for suffix in sorted(adapters):
            match = [n for n in served.adapter_models if n.endswith(f"-{suffix}")]
            if not match:
                raise SystemExit(
                    f"adapter {suffix!r} is not registered on the endpoint; "
                    f"registered={sorted(served.adapter_models)}"
                )
            served_for[suffix] = match[0]

        for arm, name in served_for.items():
            require_endpoint_model(served.api_base, name)
            print(f"[serve] {arm:8s} -> {name!r} advertised", flush=True)

        for arm, name in served_for.items():
            save_to = f"{prefix}_{arm.lower()}"
            cmd = tau2_cmd(served.api_base, f"hosted_vllm/{name}", save_to)
            print(f"\n[{arm}] {' '.join(cmd)}", flush=True)
            ta = time.time()
            # Inherit stdout/stderr so a `tail -f` shows tau2's own progress.
            # stdin is /dev/null on purpose: a resume prompt nobody answers is
            # how an endpoint idles on a paid GPU.
            proc = subprocess.Popen(cmd, cwd=a.tau2_dir, stdin=subprocess.DEVNULL)
            results = results_dir / f"{save_to}.json"
            while proc.poll() is None:
                time.sleep(20)
                spent = _spend_so_far(results)
                if spent is not None and spent > a.max_cost:
                    print(f"[budget] ${spent:.2f} > --max-cost ${a.max_cost:.2f}; "
                          f"stopping {arm}. Rerun with --save-prefix {prefix} "
                          f"to resume.", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            rc = proc.returncode if proc.returncode is not None else 1
            rc_total |= rc
            print(f"[{arm}] exit {rc} after {time.time()-ta:.0f}s, "
                  f"~${_spend_so_far(results) or 0:.2f} -> {results}", flush=True)

    print(f"\n[done] {len(served_for)} arms in {time.time()-t0:.0f}s")
    print(f"       results: {results_dir}/{prefix}_*.json")
    print("       Grade these against the V2 §10 gates separately; this script "
          "deliberately does not.")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
