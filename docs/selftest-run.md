# Planted-deficit recovery

**Planted capability:** Reads a failed tool call's error and adjusts the arguments

> When a tool call fails with an informative error, the agent reads the error text and issues a corrected call, instead of repeating the identical call.


Model `default`, min_gap=0.2, min_support=3, seed=0, 3 repeat(s) per cell.


Losses that do *not* carry the planted deficit fail for unrelated reasons (stops_to_ask, wrong_output_shape, drops_second_requirement), so the ranker has to choose between real competing explanations.


## Recovery rate

| corpus | prevalence | ceiling | recovered | proposed | label acc. | verdicts |
|---|---|---|---|---|---|---|
| 3w/3l | 1 | 0% | 0% | 100% | 83% | outranked_by_distractor×1, top_ranked_but_below_threshold×2 |
| 3w/3l | 0.6 | 0% | 0% | 100% | 94% | top_ranked_but_below_threshold×3 |
| 3w/3l | 0.3 | 0% | 0% | 100% | 89% | top_ranked_but_below_threshold×3 |
| 6w/6l | 1 | 100% | 67% | 100% | 81% | outranked_by_distractor×1, recovered×2 |
| 6w/6l | 0.6 | 100% | 67% | 100% | 86% | recovered×2, top_ranked_but_below_threshold×1 |
| 6w/6l | 0.3 | 0% | 0% | 100% | 92% | outranked_by_distractor×2, top_ranked_but_below_threshold×1 |
| 12w/12l | 1 | 100% | 67% | 100% | 82% | outranked_by_distractor×1, recovered×2 |
| 12w/12l | 0.6 | 100% | 67% | 100% | 83% | recovered×2, top_ranked_but_below_threshold×1 |
| 12w/12l | 0.3 | 100% | 100% | 100% | 100% | recovered×3 |

## Reading this

- **ceiling** — recovery under a perfect proposer and a perfect labeller on the same corpora, computed without an LLM. The live rate cannot exceed it. A 0% ceiling means the config is unrecoverable by construction at these thresholds, and a live run there measures the thresholds, not the ranker.
- **recovered** — the planted capability was selected as *the* deficit.
- **proposed** — it was named at all. A gap between the two columns is a labelling or threshold problem, not a proposer problem.
- **label acc.** — how often the labeller reproduced the label we know to be correct, per trace. This is the labeller blur that shrinks every gap downstream toward zero.
- **verdicts** — `not_proposed` is a proposer problem, `outranked_by_distractor` a labeller problem, `top_ranked_but_below_threshold` a calibration problem.


Every corpus, its ground truth, and its per-run recovery detail are under `corpora/` — nothing above is derived from anything not on disk.
