"""Analyze private industry outputs into aggregate-only results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_judge_bias.config import SEED

REPS = 10_000


def paired_metrics(wide: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Compute Group B paired accuracy, anchor, and induced-error estimates."""
    truth = wide["human_label"]
    anchor = wide["anchor"]
    c0_correct = wide["C0"] == truth
    c2_correct = wide["C2"] == truth
    c0_anchor = wide["C0"] == anchor
    c2_anchor = wide["C2"] == anchor
    accuracy_change = c2_correct.astype(int) - c0_correct.astype(int)
    anchor_change = c2_anchor.astype(int) - c0_anchor.astype(int)
    induced = c0_correct & c2_anchor
    indices = rng.integers(0, len(wide), size=(REPS, len(wide)))
    accuracy_boot = accuracy_change.to_numpy()[indices].mean(axis=1)
    anchor_boot = anchor_change.to_numpy()[indices].mean(axis=1)
    induced_boot = []
    for sample in indices:
        denominator = c0_correct.to_numpy()[sample].sum()
        if denominator:
            induced_boot.append(induced.to_numpy()[sample].sum() / denominator)
    return {
        "n": len(wide),
        "c0_accuracy": c0_correct.mean(),
        "c2_accuracy": c2_correct.mean(),
        "accuracy_change": accuracy_change.mean(),
        "accuracy_ci_low": np.quantile(accuracy_boot, 0.025),
        "accuracy_ci_high": np.quantile(accuracy_boot, 0.975),
        "c0_anchor_agreement": c0_anchor.mean(),
        "c2_anchor_agreement": c2_anchor.mean(),
        "anchor_change": anchor_change.mean(),
        "anchor_ci_low": np.quantile(anchor_boot, 0.025),
        "anchor_ci_high": np.quantile(anchor_boot, 0.975),
        "c0_correct_n": int(c0_correct.sum()),
        "induced_n": int(induced.sum()),
        "induced_among_c0_correct": induced.sum() / c0_correct.sum(),
        "induced_ci_low": np.quantile(induced_boot, 0.025),
        "induced_ci_high": np.quantile(induced_boot, 0.975),
    }


def group_a_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute correction and original-label persistence for Group A."""
    rows = []
    for (strategy, condition), group in data.groupby(["strategy", "condition"]):
        rows.append(
            {
                "strategy": strategy,
                "condition": condition,
                "n": len(group),
                "correction_rate": (
                    group["output_label"] == group["human_label"]
                ).mean(),
                "mislabel_persistence": (
                    group["output_label"] == group["original_label"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def run(input_path: Path, output_dir: Path) -> None:
    """Analyze private results and write aggregate-only CSV files."""
    data = pd.read_parquet(input_path)
    valid = data[data["error"].isna()].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    group_a_metrics(valid[valid["group"] == "A"]).to_csv(
        output_dir / "industry_group_a.csv", index=False
    )

    rng = np.random.default_rng(SEED)
    rows = []
    group_b = valid[valid["group"] == "B"]
    for strategy, group in group_b.groupby("strategy"):
        wide = (
            group.pivot(
                index=["sample_id", "human_label", "anchor"],
                columns="condition",
                values="output_label",
            )
            .dropna()
            .reset_index()
        )
        rows.append({"strategy": strategy, **paired_metrics(wide, rng)})
    pd.DataFrame(rows).to_csv(output_dir / "industry_group_b.csv", index=False)


def main() -> None:
    """Parse paths and analyze private outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
