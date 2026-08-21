# ck75 pinned identity (plan §6.1)

Recorded 2026-08-21. Source: Modal volume `vektori-trace-adapters`, path
`sft/qwen3-14b-stage-b-lora/checkpoint-75`. Hashes computed by reading the
files off the volume; Modal exposes no server-side digest, so the bytes have to
be fetched to be hashed. CPU only — no GPU was allocated, and the local copy
was deleted after hashing.

## Hashes

| file | sha256 | bytes |
| --- | --- | --- |
| `adapter_model.safetensors` | `a41e2347e5ab0f0b863471d36a2528a0b06e4b9c62ca2a9c9752fea734745329` | 513,877,864 |
| `adapter_config.json` | `31d26e99d14486e124da971622f8cb9a45758a93531f7eb2b4d368e5070bfdbc` | 1,093 |
| `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` | 11,422,650 |
| `tokenizer_config.json` | `47bfa3e7727312946b29ac10d6dd0672d63cf7815b2a160b9523872040d2e536` | 665 |
| `chat_template.jinja` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` | 4,168 |
| `trainer_state.json` | `1b5cec11502349a5cde153f6d5f07519360dc0c9cf4250ffa0852b965352d767` | 23,102 |

`chat_template.jinja` is hashed separately from `tokenizer_config.json` because
it is a separate file in this checkpoint, and it is the file that decides where
Qwen3's think-wrapper lands — a correctness pin, not bookkeeping (CLAUDE.md).

## Configuration

- base model: `Qwen/Qwen3-14B`
- LoRA rank 32, alpha 64
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- `global_step` 75, epoch 0.809

Rank 32 matters operationally: a vLLM server must be launched with
`--max-lora-rank 32` or higher, and `--enable-lora` cannot be turned on after
launch.

## Not yet pinned

- `student_base_revision` — the `Qwen/Qwen3-14B` HF commit the adapter was
  trained against is not recorded in `adapter_config.json`, which stores only
  the repo name. Resolve it from the box's HF cache before the first paid run;
  a re-tagged base silently changes what the adapter applies to.
