# Training VRAM on a 24GB card — measured

`FINAL-PLAN.md` §2 asserts "an 8B LoRA fits one A10G." It does not, in the
configuration the plan describes. This is the measurement that replaces the
assumption. Reproduce any row with `scripts/vram_probe.py`, which drives the
real `train_lora` rather than a reimplementation.

Model `Qwen/Qwen3-8B`, LoRA r=16 on q/k/v/o, batch 1, gradient checkpointing on,
A10G (22,588 MiB usable).

| # | dtype | ctx | peak alloc | peak reserved | headroom | result |
|---|---|---|---|---|---|---|
| 1 | bf16 | 8192 | 20,595 | 21,562 | 1,026 | OOM (4.64 GiB) |
| 2 | bf16 | 4096 | 20,533 | 21,054 | 1,534 | OOM (2.32 GiB) |
| 3 | 4-bit | 8192 | 21,458 | 22,076 | 512 | OOM (4.64 GiB) |
| 4 | 4-bit + embed fix | 8192 | 17,829 | 19,568 | 3,020 | OOM (4.64 GiB) |
| 5 | **4-bit + embed fix** | **4096** | **15,459** | **17,514** | **5,074** | **trains** |

Row 5 is the only configuration that completes a step. `docs/vram-probe*.json`
holds the raw records.

## Why shortening the context was not the first answer

Rows 1 and 2 differ by 4096 tokens and 62 MiB of peak. The failed allocation
halves with sequence length, but the ~20.5 GiB preceding it does not move — that
is the bf16 base (15.6 GiB) plus LoRA and optimizer state. bf16 does not fit at
*any* usable context; there is no sequence length that rescues row 1.

## Why 4-bit alone made it worse

Quantization worked: `from_pretrained` went 15,623 → 5,806 MiB, with 3,312 MiB
of `uint8` in the layers (`docs/vram-stages-*.json`). But
`prepare_model_for_kbit_training` upcasts every non-quantized parameter to fp32,
and on a 151,936-token vocab the two it catches are `model.embed_tokens.weight`
and `lm_head.weight` at 1,187 MiB each — the +2,375 MiB visible between stages.

4-bit freed 7.4 GiB of weights and the training step consumed 8.3 GiB more.
Peak-only numbers cannot explain that; the `--stages` mode exists because of it.

`train_lora` now casts those two tensors back to the compute dtype after
`prepare_model_for_kbit_training`. The layer norms the upcast exists to
stabilise are kilobytes and stay fp32. That is row 3 → row 4, worth 3.6 GiB.

## What actually caps the context at 4096

The failed allocation is **4.64 GiB in every 8192 row, at both dtypes**. It is
not the weights and not the activations:

    151,936 vocab x 8,192 tokens x 4 bytes = 4.64 GiB

Transformers' causal-LM loss calls `logits.float()` before cross-entropy
regardless of the head's dtype, so the fp32 logit tensor — and its gradient —
are produced no matter how the base is loaded. ~9.3 GiB on one tensor. Neither
quantization nor gradient checkpointing touches it: checkpointing recomputes
transformer-layer activations, and this allocation happens after the last layer.

Halving the context halves that tensor, which is why row 5 fits.

## Consequence

Train at 4096. `reopd.py` already tokenizes teacher continuations at
`max_length=4096`, so the OPD path was built for this window — the 8192 figure
in `FINAL-PLAN.md` appears never to have been checked against the code, and
should be corrected rather than reconciled.

To recover 8192 later, the lever is a fused linear cross-entropy (Liger kernel),
which chunks the loss and never materialises full logits — worth ~9 GiB. That is
a new dependency inside a billed training path and is unrun, so it is a
deliberate follow-up, not a default.

Row 5's 5,074 MiB of headroom is what gradient accumulation, a longer eval
batch, or a larger LoRA rank must come out of. Re-measure before spending it.
