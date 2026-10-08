"""Task-aware robustness analysis for a score-ceiling explanation."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from llm_judge_bias.analysis import (
    build_cells,
    build_contrasts,
    load_main_data,
    write_json,
)
from llm_judge_bias.config import RESULTS_DIR, SEED

REPS = 10_000


def clustered_fit(formula: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Fit OLS with task-clustered uncertainty."""
    fit = smf.ols(formula, data=frame).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["task_id"], "use_correction": True},
    )
    interval = fit.conf_int()
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.values,
            "ci_low": interval[0].values,
            "ci_high": interval[1].values,
            "p_value": fit.pvalues.values,
        }
    )


def run() -> None:
    """Write ceiling regressions and non-ceiling sensitivity results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    contrasts = build_contrasts(build_cells(load_main_data()))
    units = (
        contrasts.groupby(["model_short", "task_id", "category"], observed=True)
        .agg(mean_c0=("c0_mean", "mean"), c2_c0=("c2_c0", "mean"))
        .reset_index()
    )
    units["distance_to_ceiling"] = 5 - units["mean_c0"]
    units["absolute_c2_c0"] = units["c2_c0"].abs()
    formulas = {
        "signed_unadjusted": "c2_c0 ~ distance_to_ceiling",
        "signed_adjusted": "c2_c0 ~ distance_to_ceiling + C(model_short) + C(category)",
        "absolute_unadjusted": "absolute_c2_c0 ~ distance_to_ceiling",
        "absolute_adjusted": (
            "absolute_c2_c0 ~ distance_to_ceiling + C(model_short) + C(category)"
        ),
    }
    tables = []
    for name, formula in formulas.items():
        table = clustered_fit(formula, units)
        table.insert(0, "specification", name)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(
        RESULTS_DIR / "ceiling_regressions.csv", index=False
    )

    subset = units[(units["mean_c0"] >= 3.0) & (units["mean_c0"] < 4.5)]
    task_ids = np.array(sorted(subset["task_id"].unique()))
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(REPS)
    for index in range(REPS):
        selected = rng.choice(task_ids, len(task_ids), replace=True)
        sample = pd.concat([subset[subset["task_id"] == task] for task in selected])
        bootstrap[index] = sample["c2_c0"].mean()
    write_json(
        RESULTS_DIR / "non_ceiling_sensitivity.json",
        {
            "model_task_units": len(subset),
            "represented_tasks": len(task_ids),
            "negative_units": int((subset["c2_c0"] < 0).sum()),
            "mean": subset["c2_c0"].mean(),
            "ci_low": np.quantile(bootstrap, 0.025),
            "ci_high": np.quantile(bootstrap, 0.975),
            "bootstrap_reps": REPS,
            "seed": SEED,
        },
    )


if __name__ == "__main__":
    run()
