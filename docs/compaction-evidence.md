# Compaction in the DeepSeek corpus — measured, not assumed

Probe: `pallets__click-3704__sCRM7w4` (dsv4-corpus60), 2026-08-21.
Script: `scratchpad/probe_compaction.py`. Free; no paid call.

Written to answer only: marker step -> subsequent main-trajectory messages ->
referenced sidecar contents -> what `prefix_turns_through_step` produces.
It does **not** authorize a parser design.

## 1. Markers exist and are machine-readable

`trajectory.json` has **72 raw steps**, two of which carry compaction:

| raw list pos | `step_id` | `extra.context_management` | sidecars |
| --- | --- | --- | --- |
| 25 | 26 | `{"type":"compaction","boundary":"replace"}` | `summarization-1-{summary,questions,answers}` |
| 56 | 57 | `{"type":"compaction","boundary":"replace"}` | `summarization-2-{summary,questions,answers}` |

Two boundaries in one trace, so multi-boundary is real, not hypothetical.
`raw_step_id` is 1-based and equals `raw_list_pos + 1` here; that relationship
is **not** verified across the corpus.

## 2. The parser splices sidecars in ADDITIVELY

`parse_job_trajectory` yields **379 turns** from 72 raw steps, because
`atif._subagent_turns` follows `subagent_trajectory_ref` and inlines each
sidecar at `subagent_depth=1`. Depth counts: 145 at depth 0, 234 at depth 1.

`boundary: "replace"` is **not** honored. The sidecar content is appended after
the pre-compaction history, which remains present in full.

## 3. What the prefix actually contains

First depth-0 marker is turn 49; the first replay step after it is **T=81**.
`prefix_turns_through_step(turns, 81)`:

| | |
| --- | --- |
| prefix turns | 158 |
| turns before the marker | 49 |
| turns after the marker | 108 |
| depth-0 / depth-1 turns | 56 / 102 |
| total chars | 380,540 |
| chars before the marker | 114,048 |
| chars from depth-1 (sidecar) turns | 256,291 |

So the prefix at a post-compaction step contains **both** the pre-compaction
history (114k chars) **and** the sidecar content (256k chars). `turns[:end]` is
a flat slice from index 0 ([reopd.py:211](../vektori_trace/reopd.py#L211)) and
filters neither.

## 4. Step indices are NOT inflated

`assistant_tool_steps` skips `subagent_depth > 0`
([resume.py:91](../vektori_trace/evaluate/resume.py#L91)), so the replay step
pool counts only parent-agent actions. An earlier claim in conversation that
step counts were inflated by spliced turns was **wrong**; §8's length table is
not impeached by this probe.

Note the two depth-1 marker turns (265, 333) map to the same
`first_replay_step_after` as the depth-0 marker at 214 — a depth filter is
needed when deriving boundaries, or one boundary is counted three times.

## 5. Sidecar inventory (hashes for pinning, §6.1/§10)

| file | sha256[:16] | steps | summarization_index |
| --- | --- | --- | --- |
| summarization-1-summary | `855d0b1716ff258e` | 26 | 1 |
| summarization-1-questions | `dfa897a04092bf62` | 2 | 1 |
| summarization-1-answers | `cef5829b44d6b442` | 28 | 1 |
| summarization-2-summary | `0cff34fcf6ea7da8` | 33 | 2 |
| summarization-2-questions | `a1f7e6b55e17986e` | 2 | 2 |
| summarization-2-answers | `c9c919fab65a7061` | 35 | 2 |

## 6. What this establishes, and what it does not

**Established (this trace):** compaction is observable and machine-readable;
there can be several boundaries; the parser inlines sidecars additively rather
than replacing; a post-compaction prefix therefore carries pre-compaction
history plus sidecar text; step indices are correctly depth-filtered.

**Not established:** whether the retained state DeepSeek actually conditioned on
equals `[system, task] + sidecar summary`, or something else — the sidecars hold
a summary/questions/answers *handoff conversation*, not a drop-in message list,
and which of the three (if any) constitutes replacement content is unread;
whether every compacted trace in the corpus has this shape; the raw
`step_id` -> replay `step_index` mapping beyond this one trace; whether the
observed prefix is actually wrong for OPD, which depends on the unanswered
retained-state question above.

Until the retained-state question is answered by reading sidecar bodies, no
parser design (mark-only / replace-from-sidecar / both) is justified.
