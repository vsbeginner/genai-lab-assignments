"""Capstone Evaluation Aggregator: Statistical Scoring and Summary Generation.

Course: ETMMGA304 - Generative AI and Its Applications
Institution: K.R. Mangalam University (Master of Computer Applications)

This script merges raw model generations (lab_outputs/outputs.json) with multi-criteria
human evaluation ratings (data/scores.json). It generates:
  1. A comprehensive markdown evaluation record table.
  2. Aggregate mean metrics (Relevance, Readability, Usefulness, Total out of 6).
  3. Win / Tie / Loss statistical comparative metrics.
  4. Formatted summary exported to: lab_outputs/eval_summary.md

Run:
  python score_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def generate_evaluation_summary() -> str:
    """Aggregates raw outputs and scores into a markdown report."""
    outputs_path = Path("lab_outputs/outputs.json")
    scores_path = Path("data/scores.json")
    output_summary_path = Path("lab_outputs/eval_summary.md")

    with open(outputs_path, "r", encoding="utf-8") as f:
        outputs_raw = json.load(f)
    outputs: Dict[str, Dict[str, str]] = {r["topic"]: r for r in outputs_raw}

    with open(scores_path, "r", encoding="utf-8") as f:
        scores: Dict[str, Dict[str, List]] = json.load(f)

    assert set(outputs.keys()) == set(scores.keys()), "Mismatch between outputs and evaluation scores!"

    markdown_lines: List[str] = [
        "| Topic | Model | Relevance | Readability | Usefulness | Total /6 | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    total_scores = {"base": [], "adapted": []}
    criteria_scores = {
        "base": [[], [], []],     # Relevance, Readability, Usefulness
        "adapted": [[], [], []],
    }

    for topic in outputs:
        for model_key in ("base", "adapted"):
            rel, read, use, note = scores[topic][model_key]
            topic_total = rel + read + use
            total_scores[model_key].append(topic_total)

            criteria_scores[model_key][0].append(rel)
            criteria_scores[model_key][1].append(read)
            criteria_scores[model_key][2].append(use)

            markdown_lines.append(
                f"| {topic} | {model_key.capitalize()} | {rel} | {read} | {use} | {topic_total} | {note} |"
            )

    markdown_lines += [
        "",
        "| Model | Mean relevance | Mean readability | Mean usefulness | Mean total /6 |",
        "|---|---|---|---|---|",
    ]

    for model_key in ("base", "adapted"):
        means = [sum(c) / len(c) for c in criteria_scores[model_key]]
        overall_mean = sum(total_scores[model_key]) / len(total_scores[model_key])
        markdown_lines.append(
            f"| {model_key.capitalize()} | {means[0]:.2f} | {means[1]:.2f} | {means[2]:.2f} | {overall_mean:.2f} |"
        )

    # Statistical Win / Tie / Loss breakdown
    wins = sum(a > b for a, b in zip(total_scores["adapted"], total_scores["base"]))
    ties = sum(a == b for a, b in zip(total_scores["adapted"], total_scores["base"]))
    losses = len(outputs) - wins - ties

    markdown_lines.append(
        f"\nAdapted scored higher on {wins} topics, tied on {ties}, lower on {losses}."
    )

    summary_text = "\n".join(markdown_lines)
    with open(output_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text


if __name__ == "__main__":
    report = generate_evaluation_summary()
    print("Generated evaluation summary successfully:\n")
    print(report)
