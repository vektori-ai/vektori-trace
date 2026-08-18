# SFT probe monitor — GPT-5.6 Sol

This is a read-only operational log for the corrective SFT probe. It records
observations and validation only; it does not authorize starting GPU work.

## Operating constraints

- Never start a Modal endpoint, GPU, probe, rollout, or training run without
  explicit per-run approval.
- Monitor active approved work through completion, failure, or timeout.
- Confirm Modal teardown as soon as the run exits; stop an orphaned app
  immediately if needed.
- Validate new commits and working-tree changes before relying on them.
- Do not commit this log or other changes unless explicitly requested.

## 2026-08-18 TRL chunked-NLL probe

- Modal app: `ap-lR3MEkSyB16K6oUVey7psk`
- Created: 2026-08-18 01:08:45 IST
- Commit observed and validated: `551dce0` (`fix(sft): two half-wired options
  in the trainer`)
  - Explicitly skips TRL dataset preparation for fingerprinted tokenized rows.
  - Dumps optional CUDA memory history on both success and OOM.
  - Python compilation and `git diff --check` passed after the run.
- Mode observed in logs:
  - NF4 base: 9.1 GiB footprint
  - 280 bitsandbytes four-bit modules
  - Existing v1 adapter: 560 LoRA tensors, 128,450,560 trainable parameters,
    rank 32
  - TRL `SFTTrainer`
  - `loss_type=chunked_nll`
  - `use_liger_kernel=False`
  - `assistant_only_loss=False`; precomputed labels remain authoritative
  - Probe set: 24 longest rows, 34,250–36,993 tokens
- Memory before training:
  - Base loaded: 9.25 GiB allocated / 9.54 GiB reserved
  - Adapter loaded: 12.62 GiB allocated / 15.58 GiB reserved
  - Pre-train: 12.86 GiB allocated / 15.70 GiB reserved
- Step 1 completed at 01:18:02 IST:
  - Loss: 0.7133
  - Gradient norm: 0.1167
  - Learning rate: 1e-5
  - Mean token accuracy: 0.81
  - Duration: 268.42 seconds
- Step 2 completed at 01:22:29 IST:
  - Loss: 0.6835
  - Gradient norm: 0.1572
  - Mean token accuracy: 0.817
- Step 3 completed at 01:26:57 IST:
  - Loss: 0.6165
  - Gradient norm: 0.08447
  - Mean token accuracy: 0.8316
- Final outcome: success
  - Aggregate training loss: 0.6711238424
  - Runtime: 803.6 seconds
  - 267.9 seconds per optimizer step
  - Peak allocated VRAM: 63.7 GiB
  - Peak reserved VRAM: 65.9 GiB
  - All losses finite
  - All gradient norms finite and nonzero
  - All 560/560 LoRA tensors changed
  - No probe checkpoint was saved, by design
  - Estimated configured full run: 63 optimizer steps at the probe's
    conservative worst-row rate
- Teardown:
  - Local entrypoint completed at 01:26:57 IST
  - Modal app confirmed `stopped`, zero tasks, at 01:27:06 IST
  - No active probe GPU remains

## 2026-08-18 full corrective SFT

- Modal app: `ap-zNhKaJKc33RslzK56guD6Y`
- Created by the user at 01:28:48 IST
- Monitoring attached at 01:29:45 IST
- Loaded all 165 repaired segments at 01:29:27 IST
- Local log monitoring stopped at the user's request at approximately
  01:33 IST. This did not stop or modify the Modal training app.
- Final configuration, training progress, checkpoints, outcome, and teardown
  confirmation are pending.
