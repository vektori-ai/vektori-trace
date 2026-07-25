"""Assemble the final diagnose+prove report: JSON for machines, Markdown for humans."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .diagnose import DeficitScore


def build_report(
    deficit: DeficitScore | None,
    all_scores: list[DeficitScore],
    task_dir: Path | None,
    validity: dict | None,
    thresholds: dict | None = None,
) -> dict:
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


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnosis.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_lines = ["# Vektori-trace diagnosis\n"]
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
        md_lines.append(f"\nGenerated task: `{report['task_dir']}`\n")

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


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"
