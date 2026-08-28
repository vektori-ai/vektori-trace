# Stage B train — handoff, 2026-08-19

Read `docs/SFT-SCRATCH-PLAN.md` first — it's still authoritative. This file is
the state of the *current attempt* to run step 8's GPU train, for a fresh
session to pick up without re-deriving it.

## Where things stand right now

- Branch `sft-scratch`, commit `9251648`, pushed to `origin`, box
  (`/data/vektori-trace` on `i-0a348ff3d7be9769a`, via SSM only, no SSH) is
  pulled to the same commit. Working trees clean on both sides.
- Modal: **zero running apps, confirmed.** No GPU is currently billing.
- Dataset: `/data/sft-stage-b` (both locally-referenced and on the
  `vektori-trace-adapters` volume) — 742 rows, `dataset_sha256`
  `c371033d1aa6bfbee9ac3041a0898e83580e0b6bf447bbff2dc9d5d601c805e4`, cold
  share 30.0% weighted / 14.5% uniform, `mix_problems: []`. The old stale
  copy (`segments: 60` in its mix_report) was deleted; this is the only copy
  now, both locally and on the volume.
- Plan doc amendments landed this session (all in commit `9251648`):
  - **Amendment 4**: cold replay (Stage A's 165 rows replayed as Stage B's
    cold-start component) signed off. Mechanism was always in the builder
    code; the plan text only had the floor/target numbers, not the
    mechanism, until now.
  - **ck84 selected**, explicitly, as Stage A's checkpoint. "Earliest
    checkpoint clearing all 45" is unmeasured (ck20/ck40 untested) and
    **stays unmeasured by decision** — not worth GPU to re-measure
    (`docs/STAGE-A-RUN-LOG.md:191`).
  - Row-count correction: 577 later+recovery rows (not the ≈660 estimated
    pre-build), + 165 cold_replay = 742.

## The sampler-check bug (found and fixed this session)

`scripts/sft_stage_b_train_modal.py`'s pre-train safety check used to read
`loader.sampler` after `Accelerate.prepare()`, falling back to
`loader.batch_sampler.sampler` only `if seen is None`. That fallback never
fired: Accelerate's `DataLoaderShard` always sets `.sampler` to a
`SequentialSampler` placeholder for its own epoch bookkeeping (never
`None`), so the check was reading a decoy and aborting with a false
"Accelerate replaced the sampler" error every time, before any training step.

**Verified off-GPU**, no cost: reproduced with the exact pinned versions
(`torch==2.13.0 transformers==5.5.3 trl==1.10.0 accelerate==1.14.0`,
throwaway venv at `/tmp/repro_venv`, script at `/tmp/repro_sampler.py`).
Confirmed the real sampler — `loader.batch_sampler.sampler` — was
`WeightedRandomSampler` the whole time; direct index draws showed repeats
and non-sequential order, proof it was actually sampling weighted-with-
replacement, not sequential.

Fixed in `scripts/sft_stage_b_train_modal.py` around line 592: now reads
`batch_sampler.sampler` directly when a `batch_sampler` is present, falling
back to `loader.sampler` only when it isn't. Committed as `9251648`.

**This fix is real and confirmed working on GPU**: the third probe run
printed `dataloader sampler: WeightedRandomSampler (checked on the prepared
loader, not just the factory)` and passed cleanly. The full train run also
showed `sampler: WeightedRandomSampler over 742 rows, 742 draws/epoch, cold
draws 304 (plan floor 253, target 325)` — 304 is between the floor and
target, so the realized draw looks right in practice, not just in theory.

**Two non-blocking nits flagged by the user's own review, both confirmed
accurate, neither fixed (not needed for a 1-GPU run):**
1. The cold-draw *print* (line ~605, `drawn = list(iter(trainer._get_train_sampler(ds)))`)
   re-derives a fresh sampler rather than reading the prepared loader — harmless
   because `_get_train_sampler` reseeds deterministically from `self.args.seed`
   each call, so the print is representative, just not literally the same object.
2. On multi-process, Accelerate wraps in `BatchSamplerShard`, whose `.sampler`
   is not the weighted one — doesn't apply here, this run is `max_containers=1`,
   single A100.

## Probe: passed (3rd attempt)

1st probe: hit the sampler bug (unfixed code). 2nd probe: **also** hit it —
turned out my local fix never reached the box (separate git checkouts,
`/data/vektori-trace` on the box vs local repo; I'd only edited locally).
Fixed by syncing the file (base64-over-SSM at first, then properly via
`git commit` → `push` → `git pull` on the box once the user asked to commit).
3rd probe, with the real fix on the box: **passed clean**.

Probe numbers (NF4, 3 steps, 24 longest rows, 34250–36993 tokens):
peak 61.2 GiB allocated / 63.1 GiB reserved (well under the A100's 80 GiB —
the "ceiling 60 GiB" text in the peak-print line is a mislabeled leftover
from the bf16 arm's check, `BF16_PEAK_CEILING_GIB`, which only gates when
`not nf4`; not a real violation for nf4). Loss 0.655→0.609→0.769, grad_norm
0.049–0.074 finite/nonzero, 560/560 LoRA tensors moved, 280/280 continued
`lora_B` tensors carry trained weight (correctly continuing ck84, not fresh).
258.4 s/step on the longest rows (worst case, not representative of the real
run average).

## Full train: crashed at step 42/93 — infra mistake, not a training bug

Launched via `aws ssm send-command ... "modal run scripts/sft_stage_b_train_modal.py"`
with `--timeout-seconds 28800` (8h), expecting that to cover the run.
**Wrong assumption**: that flag only bounds how long AWS waits for the
command to *start*, not how long it's allowed to *run*. The actual runtime
cap is `AWS-RunShellScript`'s own internal step timeout, which defaults to
**3600s (1 hour)** and is set separately (inside the document's own
`Parameters`, not the top-level `send-command` flag). Nobody set that one.

At ~58:30 elapsed (step 42/93), the SSM command was killed by that default.
Training itself was healthy the entire time it ran — loss oscillating
0.44–0.77 with no divergence, no NaN, grad_norm always finite/nonzero,
checkpoint-25 had already saved cleanly at the expected step. This was not
a training failure.

Killing the SSM command killed the local `modal run` client on the box, but
left the remote GPU container running detached (`ap-AJ0vS0sYjfFSUFBa6mIwmg`,
state `ephemeral`, `1` task, no local process left to supervise it — checked
`ps aux | grep modal`, nothing running). Per the standing rule ("tear down
the moment the next step is uncertain," not only on outright failures), I
tore it down: `modal app stop ap-AJ0vS0sYjfFSUFBa6mIwmg --yes`, then
re-listed and confirmed `stopped`, `0` tasks. **Confirmed clean — nothing
is running or billing right now.**

Only `checkpoint-25` survived (steps 26–42 lost; next save point,
checkpoint-50, hadn't been reached yet).

## What checkpoint-25 actually has

Listed via `modal volume ls vektori-trace-adapters sft/qwen3-14b-stage-b-lora/checkpoint-25`:
`optimizer.pt`, `scheduler.pt`, `trainer_state.json`, `rng_state.pth`,
`adapter_model.safetensors`, `adapter_config.json`, plus tokenizer/template
files. **Everything a real HF Trainer resume needs is there.**

**But the script doesn't support resuming.** `trainer.train()` is called
bare at `scripts/sft_stage_b_train_modal.py:620` — no
`resume_from_checkpoint` argument, no CLI flag for it.
`BASE_ADAPTER_IN_VOLUME` (line 65) is a hardcoded constant pointing at
Stage A's `checkpoint-84` — that's the *starting LoRA weights* path, a
different thing from Trainer-state resume (optimizer/scheduler/step count).
Just repointing that constant at `checkpoint-25` would reload the right
LoRA weights but restart the LR schedule and step counter from zero, not a
true resume.

**I was explicitly told not to change anything further this session** —
last user message was "NO DONT CHANGE ANYTHING" — so the script is
untouched beyond the sampler-check fix already committed. Wiring up a real
`resume_from_checkpoint` is unstarted work.

## The two open decisions for whoever picks this up

1. **Resume from checkpoint-25, or restart from ck84 clean?** Resuming saves
   ~25 steps of already-paid GPU time but needs a code change
   (`trainer.train(resume_from_checkpoint=...)`) that hasn't been written
   or tested. Restarting is simpler and the code is already known-good
   (3rd probe passed), but throws away real progress.
2. **Fix the launch mechanism before relaunching either way.** The SSM
   1-hour default timeout is what actually caused this. Options: set the
   document's own execution-timeout parameter explicitly to cover the full
   run, or (cleaner) detach the job entirely — `nohup ... modal run ... > logfile 2>&1 &
   disown` on the box, then poll `logfile` with short SSM commands instead
   of keeping one long-running SSM command open for the whole 1-2h run.
   Given the memory note about heredocs being unreliable over SSM, keep the
   detach command simple/single-line.

## Standing rules that apply to whatever comes next

- **Always ask before relaunching a GPU step** — even a retry of something
  already approved once, even "just a rerun after a fix." (User feedback
  this session, saved to memory as `gpu_relaunch_always_ask`.)
- **Teardown needs no approval, ever, and comes before diagnosis** — the
  moment anything fails *or the next step is uncertain*, tear down first.
  Exact commands: `sudo -u ubuntu /data/vektori-trace/.venv/bin/modal app
  list`, then `... modal app stop <id> --yes` (bare `modal` isn't on PATH
  under `sudo -iu ubuntu`; `app stop` silently no-ops without `--yes` in a
  non-interactive shell — see memory `modal_teardown_required`).
- User wants **real-time visibility** on any GPU run: grab and share the
  Modal dashboard URL (`https://modal.com/apps/...`) the moment a run
  starts — it's printed near the top of the run's stdout.
- User wants **eval runs to log raw data**, not just summary numbers — full
  rollout transcripts, parser output, per-checkpoint gate results on disk.
  Not yet implemented since eval hasn't run yet.
- Once Stage B has a selected checkpoint: eval should score checkpoints
  **earliest-first and stop at the first that clears** (45-gate + beats
  Stage A on `first_edit`/`test_exec`), not score every checkpoint upfront.
  And **after selection, run it on an actual task** (mirroring step 7's
  guarded rollout, but against the Stage B checkpoint) — user was explicit
  this is required, doesn't matter if the task was in Stage B's training
  data or not.
- No PR exists for `sft-scratch` yet; user said commit+push there directly,
  didn't ask for a PR to be opened.
