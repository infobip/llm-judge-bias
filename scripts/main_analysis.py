"""Task-aware analysis of the main experiment."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from llm_judge_bias.analysis import (
    build_cells,
    build_contrasts,
    load_main_data,
    stratified_task_bootstrap,
    write_json,
)
from llm_judge_bias.config import FIGURES_DIR, RESULTS_DIR, SEED

BOOTSTRAP_REPS = 10_000
COLORS = {"C1": "#55a868", "C2": "#c44e52"}


def historical_cohens_d(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce descriptive Cohen's d on complete condition triplets."""
    key = ["model", "model_short", "task_id", "temperature", "trial"]
    wide = (
        data.pivot(index=key, columns="condition", values="score")
        .dropna()
        .reset_index()
    )
    rows = []
    for model, group in wide.groupby("model_short", sort=True):
        pooled_sd = np.sqrt((group["C0"].var() + group["C2"].var()) / 2)
        rows.append(
            {
                "model_short": model,
                "complete_triplets": len(group),
                "cohens_d": (group["C2"].mean() - group["C0"].mean()) / pooled_sd,
            }
        )
    return pd.DataFrame(rows)


def joint_model(
    contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Fit the paper's fixed-model-set OLS with task-bootstrap intervals."""
    formula = (
        "c2_c0 ~ C(category, Treatment(reference='code_review')) + "
        "C(temperature, Treatment(reference=0.0)) + C(model_short, Sum)"
    )
    fit = smf.ols(formula, data=contrasts).fit()
    design, outcome = fit.model.exog, fit.model.endog
    category_tasks = [
        [
            np.flatnonzero(contrasts["task_id"].eq(task).to_numpy())
            for task in sorted(
                contrasts.loc[contrasts["category"] == category, "task_id"].unique()
            )
        ]
        for category in sorted(contrasts["category"].unique())
    ]
    rng = np.random.default_rng(SEED)
    coefficients = np.empty((BOOTSTRAP_REPS, design.shape[1]))
    for repetition in range(BOOTSTRAP_REPS):
        sampled: list[int] = []
        for tasks in category_tasks:
            for index in rng.integers(0, len(tasks), size=len(tasks)):
                sampled.extend(tasks[index])
        coefficients[repetition] = np.linalg.lstsq(
            design[sampled], outcome[sampled], rcond=None
        )[0]
    low, high = np.quantile(coefficients, [0.025, 0.975], axis=0)
    table = pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.to_numpy(),
            "ci_low": low,
            "ci_high": high,
        }
    )
    diagnostics = {
        "formula": formula,
        "observations": int(fit.nobs),
        "tasks": int(contrasts["task_id"].nunique()),
        "r_squared": float(fit.rsquared),
        "condition_number": float(fit.condition_number),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": SEED,
    }
    return table, diagnostics


def figure_bootstrap(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap score effects for each model and temperature."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, temperature), group in contrasts.groupby(
        ["model_short", "temperature"], sort=True
    ):
        boot = np.zeros((BOOTSTRAP_REPS, 2))
        for category in sorted(group["category"].unique()):
            values = (
                group[group["category"] == category]
                .sort_values("task_id")[["c1_c0", "c2_c0"]]
                .to_numpy()
            )
            indices = rng.integers(0, 5, size=(BOOTSTRAP_REPS, 5))
            boot += values[indices].mean(axis=1) / 4
        estimates = group[["c1_c0", "c2_c0"]].mean()
        rows.append(
            {
                "model": model,
                "temperature": temperature,
                "c1_c0": estimates["c1_c0"],
                "c2_c0": estimates["c2_c0"],
                "c1_low": np.quantile(boot[:, 0], 0.025),
                "c1_high": np.quantile(boot[:, 0], 0.975),
                "c2_low": np.quantile(boot[:, 1], 0.025),
                "c2_high": np.quantile(boot[:, 1], 0.975),
            }
        )
    return pd.DataFrame(rows)


def plot_effects(contrasts: pd.DataFrame) -> None:
    """Regenerate the paper's score-effect and category figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 7, "axes.spines.top": False, "axes.spines.right": False}
    )
    values = figure_bootstrap(contrasts)
    figure, axes = plt.subplots(1, 4, figsize=(7.0, 3.46), sharey=True)
    order = values.groupby("model")["c2_c0"].mean().sort_values().index
    for axis, temperature in zip(
        axes, sorted(values["temperature"].unique()), strict=True
    ):
        subset = (
            values[values["temperature"] == temperature].set_index("model").loc[order]
        )
        positions = np.arange(len(order))
        for condition, marker, offset, low, high in (
            ("C1", "s", 0.125, "c1_low", "c1_high"),
            ("C2", "o", -0.125, "c2_low", "c2_high"),
        ):
            column = f"{condition.lower()}_c0"
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
    figure.supxlabel("Score difference")
    axes[0].set_ylabel("Model")
    figure.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "score_effects.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "score_effects.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    heatmap = contrasts.groupby(["model_short", "category"])["c2_c0"].mean().unstack()
    limit = max(abs(heatmap.min().min()), abs(heatmap.max().max()))
    figure, axis = plt.subplots(figsize=(3.5, 2.9))
    sns.heatmap(
        heatmap,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=axis,
        cbar_kws={"label": "Mean C2-C0"},
    )
    axis.set_xlabel("Category")
    axis.set_ylabel("Model")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "category_heatmap.pdf", bbox_inches="tight")
    figure.savefig(FIGURES_DIR / "category_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run() -> None:
    """Write count, model, category, temperature, joint-model, and figure results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_main_data()
    cells = build_cells(data)
    contrasts = build_contrasts(cells)
    pairs = data.pivot(
        index=["model", "task_id", "temperature", "trial"],
        columns="condition",
        values="score",
    )
    counts = {
        "attempted": 192_000,
        "valid": len(data),
        "complete_triplets": len(pairs.dropna(subset=["C0", "C1", "C2"])),
        "pair_c0_c2": len(pairs.dropna(subset=["C0", "C2"])),
        "out_of_bounds_clipped": int((data["score_raw"] != data["score"]).sum()),
        "complete_cells": len(cells),
        "stored_accept_disagreements": int(
            (data["accept"].astype(bool) != data["accepted"]).sum()
        ),
    }
    write_json(RESULTS_DIR / "counts.json", counts)
    bootstrap = stratified_task_bootstrap(
        contrasts, ["c2_c0", "c1_c0", "c2_c1"], reps=BOOTSTRAP_REPS
    )
    model_results = (
        bootstrap[bootstrap["contrast"] == "c2_c0"]
        .drop(columns="contrast")
        .merge(historical_cohens_d(data), on="model_short")
        .sort_values("estimate")
    )
    model_results.to_csv(RESULTS_DIR / "model_effects.csv", index=False)
    contrasts.to_csv(RESULTS_DIR / "cell_contrasts.csv", index=False)
    contrasts.groupby("category", observed=True)["c2_c0"].mean().to_csv(
        RESULTS_DIR / "category_effects.csv"
    )
    contrasts.groupby("temperature", observed=True)["c2_c0"].mean().to_csv(
        RESULTS_DIR / "temperature_effects.csv"
    )
    coefficients, diagnostics = joint_model(contrasts)
    coefficients.to_csv(RESULTS_DIR / "joint_model.csv", index=False)
    write_json(RESULTS_DIR / "joint_model_diagnostics.json", diagnostics)
    plot_effects(contrasts)


if __name__ == "__main__":
    run()
