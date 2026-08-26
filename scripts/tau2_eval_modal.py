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

#: Default --checkpoints. Named so --adapter can detect that the user left it
#: alone; comparing against a literal in two places drifts.
_DEFAULT_CHECKPOINTS = ("35", "70", "105")

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


def _parse_adapter_args(specs: list[str]) -> dict[str, str]:
    """`name=path` pairs -> {served_suffix: adapter_path}, order preserved.

    `_arms` can only express `{run_dir}/checkpoint-{N}`, which reaches neither a
    second run directory nor ReOPD's `update-031/checkpoint` (no `-N` suffix at
    all). Forcing those through --run-dir/--checkpoints would need one
    invocation per arm: three model loads instead of one, and three different
    servers, so the arms would no longer be compared under identical conditions.

    Names are served suffixes, so they must be distinct and URL-safe; vLLM
    resolves an unknown model name against the *base* model, which is how a
    typo becomes three silent A0 runs that still report numbers.
    """
    table: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--adapter expects name=path, got {spec!r}"
            )
        name, _, path = spec.partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--adapter expects name=path, got {spec!r}")
        if not all(c.isalnum() or c in "-_" for c in name):
            raise SystemExit(
                f"--adapter name {name!r} must be alphanumeric/-/_ : it becomes "
                "a served model name"
            )
        if name in table:
            raise SystemExit(
                f"--adapter name {name!r} given twice; served names must be "
                "distinct or one arm silently shadows the other"
            )
        if name.upper() == "A0":
            raise SystemExit(
                "--adapter name 'A0' is reserved for the unadapted base model"
            )
        table[name] = path
    return table


def _ck_order(suffixes) -> list[str]:
    """Checkpoint suffixes in numeric order: ck35, ck70, ck105.

    `sorted()` puts ck105 first, because "1" < "3" as strings. Arms run
    sequentially, so on 2026-08-25 a hung ck105 arm consumed the whole session
    and ck35/ck70 never started -- the two checkpoints the selection actually
    cared about. Numeric order runs the earliest epoch first, which is both the
    likeliest pick and the least likely to misbehave.
    """
    def key(s: str):
        digits = "".join(c for c in s if c.isdigit())
        return (0, int(digits)) if digits else (1, 0)

    return sorted(suffixes, key=key)


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


# The Tau2 user simulator sometimes opens a conversation playing the *agent*
# rather than the customer. Measured 2026-08-25 over stored retail simulations:
# task 57 shows this in 2 of 8 runs, task 93 in 0 of 10. One role-confused
# task-57 run still scored reward 1.0 (Qwen3-14B), which is exactly why the
# official reward cannot be trusted to filter these -- the simulator can drift
# into the wrong role and still walk the agent to the expected DB state.
_AGENT_TELLS = (
    "happy to help",
    "how can i assist",
    "how may i assist",
    "how can i help you with your order",
    "please provide your order",
    "please provide your account",
    "could you please provide me with your",
    "i'd be glad to",
    "welcome to",
)


def _role_confused(text: str | None) -> str | None:
    """Return the phrase that marks a simulator opening as agent-like, or None."""
    t = (text or "").lower()
    for tell in _AGENT_TELLS:
        if tell in t:
            return tell
    return None


def _audit_simulator_roles(results: Path) -> list[dict]:
    """Flag simulations whose first user turn reads as the agent.

    Post-hoc on the result file rather than inline: tau2 owns the conversation
    loop, so checking mid-episode would mean patching tau2 itself. A flagged
    trajectory is simulator failure -- it must not be graded as a checkpoint
    result in either direction, whatever reward it carries.
    """
    try:
        d = json.loads(results.read_text())
    except Exception:
        return []
    bad = []
    for sim in d.get("simulations", []):
        first_user = next((m for m in (sim.get("messages") or [])
                           if m.get("role") == "user"), None)
        if not first_user:
            continue
        tell = _role_confused(first_user.get("content"))
        if tell:
            bad.append({
                "task_id": str(sim.get("task_id")),
                "trial": sim.get("trial"),
                "reward": (sim.get("reward_info") or {}).get("reward"),
                "tell": tell,
                "opening": (first_user.get("content") or "")[:90],
            })
    return bad


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
    ap.add_argument("--checkpoints", nargs="*", default=list(_DEFAULT_CHECKPOINTS),
                    help="checkpoint numbers to grade; empty grades base only")
    ap.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                    help="explicit arm as name=path; repeatable. Use this for "
                         "adapters that are not {run-dir}/checkpoint-N -- "
                         "different runs, or ReOPD's update-NNN/checkpoint. "
                         "Mutually exclusive with --checkpoints.")
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
    ap.add_argument("--user-timeout", type=float, default=60.0,
                    help="per-request timeout (s) for the user simulator. "
                         "Without it litellm defaults to 600 s and tau2 retries "
                         "3x, so one dropped connection stalls the whole run "
                         "for ~40 min with no log output.")
    ap.add_argument("--user-retries", type=int, default=2,
                    help="user-simulator retries. tau2 forces 3 when unset; "
                         "setting it explicitly bounds the worst case.")
    ap.add_argument("--continue-after-infra", action="store_true",
                    help="keep running later arms after one fails for "
                         "infrastructure reasons. Off by default: the arms "
                         "share an endpoint and a provider, so the next arm "
                         "usually hits the same wall and spends GPU minutes to "
                         "produce another ungradeable result.")
    ap.add_argument("--stall-timeout", type=float, default=900.0,
                    help="kill an arm after this many seconds with no new "
                         "results. Guards the hung-provider-call failure: the "
                         "socket stays open, tau2 blocks forever, and the GPU "
                         "bills. Recorded as infrastructure, never as a model "
                         "result.")
    ap.add_argument("--arm-timeout", type=float, default=1800.0,
                    help="hard wall-clock cap per arm, in seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command and exit; allocates nothing")
    ap.add_argument("--yes", action="store_true",
                    help="required to allocate a GPU (CLAUDE.md: per-run approval)")
    a = ap.parse_args()

    task_ids, man_hash = _resolve_tasks(a.artifacts, a.partition)
    all_n = len(task_ids)
    task_ids = _subset(task_ids, a.partition, a.tasks, a.max_tasks)
    if a.adapter:
        # Explicit paths win outright rather than merging with the template
        # arms: --checkpoints has a non-empty default, so a silent merge would
        # add three unrequested ck* arms to every explicit run and pay for them.
        if tuple(a.checkpoints) != _DEFAULT_CHECKPOINTS:
            raise SystemExit(
                "--adapter and --checkpoints are mutually exclusive; --adapter "
                "already names every arm explicitly"
            )
        adapters = _parse_adapter_args(a.adapter)
        arm_order = list(adapters)
    else:
        adapters = _arms(a.run_dir, a.checkpoints)
        arm_order = _ck_order(adapters)
    if not adapters and a.no_base:
        raise SystemExit("nothing to evaluate: no checkpoints and --no-base")

    # Fail before the GPU, not after. `_arms` never checked this: a wrong path
    # reaches vLLM, which resolves an unknown adapter against the BASE model, so
    # every arm silently becomes A0 and still reports plausible numbers. A
    # dry-run cannot see the volume, so this is a real-run check only.
    if not a.dry_run:
        for name, path in adapters.items():
            weights = Path(path) / "adapter_model.safetensors"
            if not weights.exists():
                raise SystemExit(
                    f"arm {name!r}: no adapter weights at {weights}. Refusing "
                    "to start: vLLM resolves a missing adapter against the base "
                    "model, so this arm would silently be A0."
                )

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
            "--user-llm-args", json.dumps({
                "temperature": 0.0,
                # tau2 passes no `timeout` to litellm for the user simulator
                # (utils/llm_utils.py), so litellm falls back to 600 s -- and
                # `litellm.request_timeout`'s 6000 is a sentinel that is never
                # applied. tau2 then forces num_retries=3
                # (llm_utils.py, config.DEFAULT_MAX_RETRIES), so ONE stalled
                # Fireworks connection burns 600 s four times over: ~40 minutes
                # of total silence, because litellm retries internally and tau2
                # logs only on the final exception.
                #
                # It freezes the whole run rather than one episode because
                # concurrency is `list(executor.map(...))` (run.py): results
                # drain in submission order, so a single blocked worker parks
                # the main thread on a futex and every other thread with it.
                # Observed three times on 2026-08-25 -- twice mid-conversation,
                # once before the first turn.
                #
                # `--user-llm-args` is json.loads'd by tau2's CLI and splatted
                # straight into litellm.completion, so these land as real
                # kwargs. It REPLACES tau2's default args, which is why
                # temperature is restated here -- dropping it would silently
                # change user-simulator sampling.
                "timeout": a.user_timeout,
                "num_retries": a.user_retries,
            }),
            "--max-concurrency", str(a.max_concurrency),
            "--save-to", save_to,
            # tau2 v0.2.0 defaults to ERROR, which hides the request/response
            # detail that tells an empty tool_calls apart from a routing 404.
            "--log-level", "DEBUG",
        ]

    arm_names = ([] if a.no_base else ["A0"]) + arm_order
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
        for suffix in arm_order:
            print(f"  lora {suffix:8s} {adapters[suffix]}")
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
    infra_failures: list[dict] = []
    confused: list[dict] = []
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
        for suffix in arm_order:
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
            results = results_dir / f"{save_to}.json"
            # Progress is measured from tau2's own output, captured to a file
            # this process owns.
            #
            # It used to watch the RESULTS file, which was wrong in a way only a
            # slow task exposed: tau2 writes results when a *simulation
            # completes*, and a task-93 episode runs 400+ s on a 4B student
            # (DeepSeek does the same task in 43 s). So the file sat untouched
            # through a perfectly healthy conversation and the watchdog killed
            # two good arms at 240 s (2026-08-25, run 93c -- ck70 was at
            # "Step 12" and still progressing when it was shot). A per-turn
            # signal is the only thing that separates "slow episode" from "hung
            # provider call"; per-episode granularity cannot, because one
            # episode outlasts any useful timeout.
            #
            # Defined BEFORE the Popen that opens it: an earlier edit left the
            # open() above this line and the arm died with UnboundLocalError
            # after paying the full 106 s boot.
            progress_log = results_dir / f"{save_to}.progress.log"
            # stdin is /dev/null on purpose: a resume prompt nobody answers is
            # how an endpoint idles on a paid GPU.
            progress_fh = open(progress_log, "wb")
            proc = subprocess.Popen(cmd, cwd=a.tau2_dir, stdin=subprocess.DEVNULL,
                                    stdout=progress_fh, stderr=subprocess.STDOUT)
            last_size, last_change = -1, time.time()
            killed_for = None
            while proc.poll() is None:
                time.sleep(20)
                now = time.time()
                try:
                    size = progress_log.stat().st_size
                except OSError:
                    size = -1
                if size != last_size:
                    last_size, last_change = size, now

                # Heartbeat. tau2's own output now goes to progress_log, so the
                # caller's `tee` log would otherwise sit silent for the whole
                # arm -- which is precisely how a healthy 400 s episode and a
                # hung one looked identical from the outside.
                idle = now - last_change
                print(f"[{arm}] {now - ta:6.0f}s elapsed, last progress "
                      f"{idle:4.0f}s ago (kill at {a.stall_timeout:.0f}s)",
                      flush=True)

                spent = _spend_so_far(results)
                if spent is not None and spent > a.max_cost:
                    killed_for = (f"[budget] ${spent:.2f} > --max-cost "
                                  f"${a.max_cost:.2f}")
                # A provider call with no read timeout hangs forever, and the
                # GPU bills the whole time. Observed 2026-08-25: a task-57 arm
                # sat on an idle socket for minutes and blocked every later
                # arm, since arms run sequentially.
                elif now - last_change > a.stall_timeout:
                    killed_for = (f"[stall] no result activity for "
                                  f"{now - last_change:.0f}s "
                                  f"(--stall-timeout {a.stall_timeout})")
                elif now - ta > a.arm_timeout:
                    killed_for = (f"[timeout] arm exceeded "
                                  f"{a.arm_timeout}s wall clock")

                if killed_for:
                    print(f"{killed_for}; stopping {arm}. Rerun with "
                          f"--save-prefix {prefix} to resume.", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
            progress_fh.close()
            # Replay what tau2 printed, so the caller's `tee` log keeps the full
            # DEBUG trace it had before stdout was redirected.
            try:
                sys.stdout.write(progress_log.read_text(errors="replace"))
                sys.stdout.flush()
            except OSError:
                pass
            rc = proc.returncode if proc.returncode is not None else 1
            # An arm killed by a watchdog is INFRASTRUCTURE, not a model
            # result. Grading a stalled arm as a checkpoint failure is how a
            # provider outage turns into a wrong experimental conclusion.
            if killed_for:
                infra_failures.append({"arm": arm, "reason": killed_for})
            rc_total |= rc
            print(f"[{arm}] exit {rc} after {time.time()-ta:.0f}s, "
                  f"~${_spend_so_far(results) or 0:.2f} -> {results}"
                  + ("  INFRASTRUCTURE FAILURE" if killed_for else ""), flush=True)

            for bad in _audit_simulator_roles(results):
                confused.append({"arm": arm, **bad})
                print(f"[simulator] {arm} task {bad['task_id']} trial "
                      f"{bad['trial']}: user opened in the AGENT role "
                      f"({bad['tell']!r}, reward={bad['reward']}). Not a "
                      f"checkpoint result.", flush=True)

            # Stop the session on an infrastructure failure rather than running
            # the remaining arms into the same wall.
            #
            # The arms share one endpoint and one user-simulator provider, so a
            # stall or timeout is almost always a property of the *session*, not
            # of the checkpoint that happened to be running. Continuing spends
            # GPU minutes to collect more ungradeable arms -- run 93c produced
            # two INFRASTRUCTURE FAILUREs back to back for exactly this reason.
            # Teardown is the context manager's job and happens on the way out.
            if killed_for and not a.continue_after_infra:
                print(f"[abort] {arm} failed for infrastructure reasons; "
                      f"skipping the remaining arms because they share this "
                      f"endpoint and provider. Re-run with "
                      f"--continue-after-infra to override.", flush=True)
                break

    if confused:
        print(f"\n[SIMULATOR FAILURE] {len(confused)} simulation(s) opened with "
              f"the user playing the agent:", flush=True)
        for c in confused:
            print(f"    {c['arm']} task {c['task_id']} trial {c['trial']} "
                  f"(reward={c['reward']}): {c['opening']!r}", flush=True)
        print("    Discard these and retry under a recorded seed. A reward of "
              "1.0 here is not evidence the checkpoint solved the task.",
              flush=True)

    if infra_failures:
        print(f"\n[INFRASTRUCTURE] {len(infra_failures)} arm(s) did not produce "
              f"a gradeable result:", flush=True)
        for f in infra_failures:
            print(f"    {f['arm']}: {f['reason']}", flush=True)
        print("    These are NOT model failures. Do not grade them.", flush=True)

    print(f"\n[done] {len(served_for)} arms in {time.time()-t0:.0f}s")
    print(f"       results: {results_dir}/{prefix}_*.json")
    print("       Grade these against the V2 §10 gates separately; this script "
          "deliberately does not.")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
