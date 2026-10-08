"""Analyze the public trial-level numerical mitigation data."""

import numpy as np
import pandas as pd
from scipy import stats

from llm_judge_bias.config import DATA_DIR, RESULTS_DIR, SEED

REPS = 10_000


def analyze(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute total effects, dose-response slopes, and token costs."""
    valid = frame[frame["error"].isna() & frame["score"].notna()].copy()
    rng = np.random.default_rng(SEED)
    rows = []
    for strategy in ("baseline", "cot", "warning"):
        group = valid[valid["strategy"] == strategy]
        c0 = group[group["condition"] == "C0"]["score"].to_numpy(float)
        c2 = group[group["condition"] == "C2"]["score"].to_numpy(float)
        bootstrap = c2[rng.integers(0, len(c2), size=(REPS, len(c2)))].mean(axis=1)
        bootstrap -= c0[rng.integers(0, len(c0), size=(REPS, len(c0)))].mean(axis=1)
        dose = group[group["condition"] == "C2"]
        regression = stats.linregress(dose["prior_score"], dose["score"])
        margin = stats.t.ppf(0.975, len(dose) - 2) * regression.stderr
        rows.append(
            {
                "strategy": strategy,
                "valid_calls": len(group),
                "c0_mean": c0.mean(),
                "c2_mean": c2.mean(),
                "total_effect_c2_c0": c2.mean() - c0.mean(),
                "effect_ci_low": np.quantile(bootstrap, 0.025),
                "effect_ci_high": np.quantile(bootstrap, 0.975),
                "within_c2_slope": regression.slope,
                "slope_ci_low": regression.slope - margin,
                "slope_ci_high": regression.slope + margin,
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_total_tokens": group["total_tokens"].mean(),
            }
        )
    output = pd.DataFrame(rows)
    baseline_effect = abs(
        output.loc[output["strategy"] == "baseline", "total_effect_c2_c0"].iloc[0]
    )
    baseline_tokens = output.loc[
        output["strategy"] == "baseline", "mean_total_tokens"
    ].iloc[0]
    output["effect_reduction_vs_baseline"] = (
        1 - output["total_effect_c2_c0"].abs() / baseline_effect
    )
    output["total_token_increase_pct"] = (
        output["mean_total_tokens"] / baseline_tokens - 1
    ) * 100
    return output


def run() -> None:
    """Write numerical mitigation results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    data = pd.read_parquet(DATA_DIR / "mitigation_trials.parquet")
    if len(data) != 1_500:
        raise ValueError(f"Expected 1,500 mitigation calls, found {len(data)}")
    analyze(data).to_csv(RESULTS_DIR / "mitigation.csv", index=False)


if __name__ == "__main__":
    run()
