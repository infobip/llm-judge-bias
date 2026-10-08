"""Threshold-derived acceptance analysis and paper figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_judge_bias.analysis import build_cells, build_contrasts, load_main_data
from llm_judge_bias.config import FIGURES_DIR, RESULTS_DIR, SEED

COLORS = {"C1": "#55a868", "C2": "#c44e52"}
REPS = 10_000


def bootstrap_by_temperature(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap task-level acceptance effects for each model and temperature."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, temperature), group in contrasts.groupby(
        ["model_short", "temperature"], sort=True
    ):
        boot = np.zeros((REPS, 2))
        for category in sorted(group["category"].unique()):
            values = (
                group[group["category"] == category]
                .sort_values("task_id")[["accept_c1_c0", "accept_c2_c0"]]
                .to_numpy()
            )
            indices = rng.integers(0, 5, size=(REPS, 5))
            boot += values[indices].mean(axis=1) / 4
        estimates = group[["accept_c1_c0", "accept_c2_c0"]].mean()
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                **{name: estimates[name] for name in ("accept_c1_c0", "accept_c2_c0")},
                "c1_low": np.quantile(boot[:, 0], 0.025),
                "c1_high": np.quantile(boot[:, 0], 0.975),
                "c2_low": np.quantile(boot[:, 1], 0.025),
                "c2_high": np.quantile(boot[:, 1], 0.975),
            }
        )
    return pd.DataFrame(rows)


def plot(results: pd.DataFrame) -> None:
    """Write the threshold-derived acceptance figure used in the paper."""
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 7, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 4, figsize=(7.0, 3.46), sharey=True)
    order = results.groupby("model")["accept_c2_c0"].mean().sort_values().index
    for axis, temperature in zip(
        axes, sorted(results["temperature"].unique()), strict=True
    ):
        subset = (
            results[results["temperature"] == temperature].set_index("model").loc[order]
        )
        positions = np.arange(len(order))
        for condition, marker, offset, low, high in (
            ("C1", "s", 0.125, "c1_low", "c1_high"),
            ("C2", "o", -0.125, "c2_low", "c2_high"),
        ):
            column = f"accept_{condition.lower()}_c0"
            axis.errorbar(
                subset[column],
                positions + offset,
                xerr=[subset[column] - subset[low], subset[high] - subset[column]],
                fmt=marker,
                color=COLORS[condition],
                label=f"{condition}-C0" if axis is axes[0] else None,
            )
        axis.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        axis.set_yticks(positions, order)
        axis.set_title(f"T = {temperature}")
        axis.grid(axis="y", alpha=0.3)
    figure.supxlabel("Acceptance-rate difference")
    axes[0].set_ylabel("Model")
    figure.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "acceptance_rate_shift.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURES_DIR / "acceptance_rate_shift.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def run() -> None:
    """Write acceptance aggregates and figure."""
    RESULTS_DIR.mkdir(exist_ok=True)
    contrasts = build_contrasts(build_cells(load_main_data()))
    results = bootstrap_by_temperature(contrasts)
    results.to_csv(RESULTS_DIR / "acceptance_by_temperature.csv", index=False)
    contrasts.groupby("model_short", observed=True)[
        ["accept_c1_c0", "accept_c2_c0"]
    ].mean().to_csv(RESULTS_DIR / "acceptance_by_model.csv")
    plot(results)


if __name__ == "__main__":
    run()
