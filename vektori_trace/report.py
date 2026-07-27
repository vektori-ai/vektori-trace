"""Assemble the final diagnose+prove report: JSON for machines, Markdown for humans."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .diagnose import MIN_DISCORDANT_PAIRS, DeficitScore, ReplayDiagnosis


def score_dict(s: DeficitScore) -> dict:
    return {
        "capability": asdict(s.capability),
        "baseline_rate": s.baseline_rate,
        "incident_rate": s.incident_rate,
        "gap": s.gap,
        "prevalence": s.prevalence,
        "priority": s.priority,
        "n_relevant_wins": s.n_relevant_wins,
        "n_relevant_losses": s.n_relevant_losses,
        "lacking_loss_run_ids": [t.run_id for t in s.lacking_loss_traces],
    }


def build_report(
    deficit: DeficitScore | None,
    all_scores: list[DeficitScore],
    task_dir: Path | None,
    validity: dict | None,
    thresholds: dict | None = None,
) -> dict:
    report = {
        "chosen_deficit": score_dict(deficit) if deficit else None,
        "all_deficits_ranked": [score_dict(s) for s in all_scores],
        "task_dir": str(task_dir) if task_dir else None,
        "thresholds": thresholds or {},
    }
    if validity:
        report["validity"] = {
            "valid": validity["valid"],
            "oracle": {
                "passed": validity["oracle"].passed,
                "reward": validity["oracle"].reward,
            },
            "base": (
                {"agent": validity["base"].agent, "passed": validity["base"].passed, "reward": validity["base"].reward}
                if validity.get("base")
                else None
            ),
        }
    return report


def build_replay_report(
    diagnosis: ReplayDiagnosis,
    task_dir: Path | None,
    validity: dict | None,
    thresholds: dict | None = None,
) -> dict:
    """The plain report, plus the second contrast and the same-task test.

    Everything `build_report` already emits keeps its shape and meaning — the
    ranked list and `chosen_deficit` are the *cross-model* contrast, because
    that's what `diagnose_replay` scored. The `replay` block adds what the
    cross-model numbers alone can't say: whether the candidate can be trained on
    this, and whether the difference survives comparing the models task by task.
    """
    report = build_report(
        diagnosis.chosen, diagnosis.cross_model_scores, task_dir, validity, thresholds
    )
    m = diagnosis.mcnemar
    report["replay"] = {
        "frontier_model": diagnosis.frontier_model,
        "candidate_model": diagnosis.candidate_model,
        "within_model_deficit": (
            score_dict(diagnosis.within_model_score) if diagnosis.within_model_score else None
        ),
        "mcnemar": (
            {
                "capability_id": m.capability_id,
                "frontier_only": m.frontier_only,
                "candidate_only": m.candidate_only,
                "concordant": m.concordant,
                "discordant_n": m.discordant_n,
                "p_value": m.p_value,
                "underpowered": m.underpowered,
                "min_discordant_pairs": MIN_DISCORDANT_PAIRS,
            }
            if m
            else None
        ),
        "trainable": diagnosis.trainable,
    }
    return report


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnosis.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_lines = ["# Vektori-trace diagnosis\n"]
    r = report.get("replay")
    if r:
        md_lines.append(
            f"Cross-model contrast: frontier `{r['frontier_model']}` wins vs "
            f"candidate `{r['candidate_model']}` losses. Every rate below is that "
            "contrast unless it says otherwise.\n"
        )
    d = report["chosen_deficit"]
    if d is None:
        t = report.get("thresholds") or {}
        md_lines.append("## No deficit found\n")
        md_lines.append(
            "No candidate capability cleared the thresholds "
            f"(min_gap={t.get('min_gap')}, min_support={t.get('min_support')}). "
            "That is a result, not a failure: on this evidence nothing separates "
            "the wins from the losses well enough to be worth training against. "
            "The ranked list below is reported for inspection only — its top entry "
            "was rejected.\n"
        )
    else:
        md_lines.append(f"## Diagnosed deficit: {d['capability']['name']}\n")
        md_lines.append(d["capability"]["description"] + "\n")
        md_lines.append(
            f"- baseline rate (in wins): {_fmt(d['baseline_rate'])} "
            f"(N={d['n_relevant_wins']})\n"
            f"- incident rate (in losses): {_fmt(d['incident_rate'])} "
            f"(N={d['n_relevant_losses']})\n"
            f"- gap: {_fmt(d['gap'])}\n"
            f"- prevalence (share of failures explained): {_fmt(d['prevalence'])}\n"
        )
        if report["task_dir"]:
            md_lines.append(f"\nGenerated task: `{report['task_dir']}`\n")

    if r and d is not None:
        md_lines.extend(_replay_sections(r))

    if "validity" in report:
        v = report["validity"]
        md_lines.append("\n## Validity proof\n")
        md_lines.append(f"- Oracle solution: {'PASS' if v['oracle']['passed'] else 'FAIL'}")
        if v["base"]:
            md_lines.append(
                f"- Base agent ({v['base']['agent']}): {'PASS' if v['base']['passed'] else 'FAIL'}"
            )
        md_lines.append(f"- **Valid: {v['valid']}**\n")

    md_lines.append("\n## All diagnosed deficits (ranked)\n")
    for s in report["all_deficits_ranked"]:
        md_lines.append(
            f"- {s['capability']['name']}: priority={_fmt(s['priority'])}, "
            f"gap={_fmt(s['gap'])}, prevalence={_fmt(s['prevalence'])}, "
            f"N={s['n_relevant_wins']}w/{s['n_relevant_losses']}l"
        )

    md_path = out_dir / "diagnosis.md"
    md_path.write_text("\n".join(md_lines))
    return md_path


def _replay_sections(r: dict) -> list[str]:
    """The within-model contrast, the same-task test, and the trainability call."""
    lines: list[str] = ["\n## Within-model contrast (is there anything to train from?)\n"]
    w = r["within_model_deficit"]
    if w is None:
        lines.append("Not assessed.\n")
    else:
        lines.append(
            f"Same capability, `{r['candidate_model']}`'s own wins vs its own losses:\n\n"
            f"- lacking rate in its wins: {_fmt(w['baseline_rate'])} (N={w['n_relevant_wins']})\n"
            f"- lacking rate in its losses: {_fmt(w['incident_rate'])} (N={w['n_relevant_losses']})\n"
            f"- gap: {_fmt(w['gap'])}\n"
        )

    m = r["mcnemar"]
    lines.append("\n## Same-task comparison (exact McNemar)\n")
    if m is None:
        lines.append("Not assessed.\n")
    else:
        lines.append(
            "Difficulty cancels because each pair is one task both models attempted:\n\n"
            f"- frontier had it, candidate didn't (b): {m['frontier_only']}\n"
            f"- candidate had it, frontier didn't (c): {m['candidate_only']}\n"
            f"- concordant pairs: {m['concordant']}\n"
            f"- discordant pairs (b+c): {m['discordant_n']}\n"
            f"- two-sided exact p: {_fmt_p(m['p_value'])}\n"
        )
        if m["underpowered"]:
            lines.append(
                f"\n**Underpowered**: {m['discordant_n']} discordant pair(s) is below the "
                f"{m['min_discordant_pairs']}-pair floor. Under 6 pairs no split can reach "
                "p<0.05 at all, and under 9 only a perfectly one-sided one can — so a "
                "significant p here means the split was maximally lopsided, and a "
                "non-significant one says the test had no power, not that the models "
                "are alike. Either way, more paired tasks before believing it.\n"
            )

    if r["trainable"] is False:
        lines.append("\n## Identified, not trainable\n")
        lines.append(
            "The cross-model contrast cleared the thresholds, but "
            f"`{r['candidate_model']}` never demonstrated this capability often enough "
            "in its own wins. Rejection sampling keeps only rollouts that pass, so "
            "there is nothing here for it to keep: a task built against this deficit "
            "yields an empty training set, not a hard one. No task was generated.\n\n"
            "Next: widen the replay (more tasks), or pick a candidate that succeeds "
            "here at least sometimes.\n"
        )
    return lines


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _fmt_p(x) -> str:
    return "n/a (no discordant pairs)" if x is None else f"{x:.4f}"
