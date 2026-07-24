"""Assemble the final diagnose+prove report: JSON for machines, Markdown for humans."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .diagnose import DeficitScore


def build_report(
    deficit: DeficitScore,
    all_scores: list[DeficitScore],
    task_dir: Path,
    validity: dict | None,
) -> dict:
    def score_dict(s: DeficitScore) -> dict:
        return {
            "capability": asdict(s.capability),
            "baseline_rate": s.baseline_rate,
            "incident_rate": s.incident_rate,
            "gap": s.gap,
            "prevalence": s.prevalence,
            "priority": s.priority,
            "lacking_loss_run_ids": [t.run_id for t in s.lacking_loss_traces],
        }

    report = {
        "chosen_deficit": score_dict(deficit),
        "all_deficits_ranked": [score_dict(s) for s in all_scores],
        "task_dir": str(task_dir),
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
    md_lines.append(f"## Diagnosed deficit: {d['capability']['name']}\n")
    md_lines.append(d["capability"]["description"] + "\n")
    md_lines.append(
        f"- baseline rate (in wins): {_fmt(d['baseline_rate'])}\n"
        f"- incident rate (in losses): {_fmt(d['incident_rate'])}\n"
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
            f"gap={_fmt(s['gap'])}, prevalence={_fmt(s['prevalence'])}"
        )

    md_path = out_dir / "diagnosis.md"
    md_path.write_text("\n".join(md_lines))
    return md_path


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"
