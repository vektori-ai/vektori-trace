# Step 4 handoff — gate manifest

> **RESOLVED 2026-08-18.** The decision below was taken (**A**, with reuse as a
> fallback and not a default) and recorded as amendment 3 in
> `docs/SFT-SCRATCH-PLAN.md`. The manifest is frozen — see "Frozen" at the
> bottom. The rest of this file is the record of how it got there.

State as of 2026-08-18, branch `sft-scratch` @ `cb986c0` (pushed; box is on it).
Plan of record is `docs/SFT-SCRATCH-PLAN.md`. **No GPU. No Modal.** Step 4 is CPU only.

## Done

- **Plan amended and signed off**: Stage A is **165 rows**, not 183. The 18
  parse-error recoveries move to Stage B (`max_length` 40960) and Stage B now
  *requires* them. Fail-closed no longer demands 18 recoveries on Stage A.
  Amendment 2 in "Decisions taken here".
- **Step 3 verified on the box**: `/data/sft-stage-a/`, `dataset_sha256`
  `c2d4d2e7…`, source pin `7ecfee31…`, 165 rows, tokens 952–6494, supervised
  8.6%, `dropped_targets`/`mix_problems`/`tokenization_problems` all empty.
  All three reports on disk.
- **Step 4 code** (`243d488`):
  - `has_read` / `has_edit` / `has_test` in `vektori_trace.evaluate.phase7` are
    the single home. `EDIT_RE`/`TEST_RE` are `^`-anchored without MULTILINE, so
    the old `search(joined)` only ever saw command 0. Gates, manifest `_ops`
    and the dataset audit now all classify per command. Duplicate regexes
    deleted from `scripts/sft_repair_dataset.py`.
  - Selection is `SELECTION_CATEGORIES` (orientation / first_inspection /
    post_compaction) × `SELECTION_SUITES` (acquisition / control /
    generalization) = **45**. `SELECTION_SUITE = "generalization"` is gone;
    `selection_prefix_ids()` reads the set off the frozen manifest. Tripwires
    (`TRIPWIRE_CATEGORIES`, n=1) are reported, never gating.
  - `pick()` takes per-category counts, never reuses a task, returns a
    shortfall; `main` refuses to freeze on a short set (`--allow-short` exists,
    **must stay off**).
  - 1087 tests pass, ruff clean. New: `tests/test_phase7_manifest.py`, plus
    later-command edit/test cases in `tests/test_phase7.py`.

## Blocked — the freeze refused, correctly

    .venv/bin/python scripts/phase7_manifest.py \
      --acquisition /data/sft-repaired \
      --control /data/phase7-corpora/trained-failing \
      --generalization /data/phase7-corpora/heldout-failing \
      --selection-per-category 5 --tripwire-per-category 1 \
      --tokenizer Qwen/Qwen3-14B --out /data/phase7-stage-a/manifest.json

    selection set: 40 prefixes (35 distinct tasks, 5 repos); expected 45
    SHORT: acquisition{long_context 1}  control{post_compaction 5, first_edit 1,
           test_exec 1, parse_error_recovery 1, long_context 1}
    refusing to freeze

Nothing was written. `/data/phase7-stage-a/` does not exist yet.

**Note the output path.** `/data/phase7/manifest.json` is the *v1* frozen
manifest (24 prefixes, `per_category` 1) that `/data/phase7/results.json` was
graded against. Do not overwrite it. Stage A's manifest goes to
`/data/phase7-stage-a/`.

### Why 40, measured

Candidates per suite (`/tmp/probe.py` on the box, or rerun the same walk):

| suite | tasks | orientation | first_inspection | post_compaction |
|---|---|---|---|---|
| acquisition | 34 | 117 / 34 tasks | 145 / 34 | 48 / **17** |
| control | **11** | 18 / 11 | 34 / 11 | 20 / **5 tasks, 3 repos** |
| generalization | 26 | 103 / 26 | 190 / 26 | 117 / 17 |

The control corpus holds 38 segments over **11 tasks**. Selection wants 15
prefixes from it, and the current rule is one prefix per task *across the whole
suite*, so 15 distinct tasks cannot exist. Control also has no `anyio` at all,
and its `post_compaction` lives in exactly 5 tasks across 3 repos.

## The decision needed (user's call, do not pick it silently)

- **A — relax distinctness to one prefix per task *per category*.** Yields
  exactly 45: control spends its 5 post_compaction tasks, all of them. Cost: a
  control task may appear once in orientation and once in post_compaction
  (different turns), and control's post_compaction covers 3 repos, not 5.
  One-line change in `pick()` (`used_tasks` becomes per-category).
- **B — enlarge the control corpus.** More trained-failing rollouts → more
  tasks. Real work, and depends on rollouts existing on disk.
- **C — accept 40 with `--allow-short`.** The user explicitly ruled this out.

Recommendation: **A**, and record it in the plan as amendment 3 before freezing.

## After the freeze

Show sha + 45-count by suite and category + distinct tasks + repos. Never
regenerate. Then stop: step 5 is GPU and needs its own approval.

## Frozen

    /data/phase7-stage-a/manifest.json
    sha256 804771ca9243f983250a8125512669605374f155c8c2a0192dd35f32b65c6f92
    60 prefixes, 45 selecting, 35 distinct tasks, 5 repos
    prefix tokens: min 825  median 3,418  max 35,609  (budget 40,448)

5 x 3 in every cell:

| suite | orientation | first_inspection | post_compaction | repos |
|---|---|---|---|---|
| acquisition | 5 | 5 | 5 | all 5 |
| control | 5 | 5 | 5 | 4 — **no anyio** |
| generalization | 5 | 5 | 5 | all 5 |

Cross-category reuse stamped **10** entries, of which **5 select**. The
selecting 5 are all control `post_compaction`, exactly where predicted:
`pypa__hatch-2127`, `pallets__click-3363`, `prefecthq__prefect-22591`,
`pallets__click-3704`, `pallets__click-3534`. The other 5 are tripwires
(`acquisition/long_context`; `control/first_edit`, `test_exec`,
`parse_error_recovery`, `long_context`), which never gate. No acquisition or
generalization *selection* prefix reused a task.

**Per-cell repo coverage — `post_compaction` is 3 repos in every suite**, not
just control: acquisition click/hatch/jinja, control click/hatch/prefect,
generalization click/hatch/prefect. Control `first_inspection` is also 3
(click/jinja/prefect). Do not read repo coverage off any `post_compaction`
cell at eval time; orientation and first_inspection carry it.

Never regenerate. Step 5 is GPU and needs its own approval.
