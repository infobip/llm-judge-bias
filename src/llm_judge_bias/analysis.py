"""Shared loading, aggregation, bootstrap, and output helpers."""

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, MODEL_SHORT_NAMES, SEED

CATEGORIES = ("code_review", "creative_writing", "factual_qa", "summarization")


def load_main_data() -> pd.DataFrame:
    """Load, validate, and clean the public main-experiment data."""
    data = pd.read_parquet(DATA_DIR / "main_experiment.parquet")
    key = ["model", "task_id", "temperature", "condition", "trial"]
    if len(data) != 192_000 or data[key].duplicated().any():
        raise ValueError("Unexpected main-experiment dimensions or duplicate keys")
    score = pd.to_numeric(data["score"], errors="coerce")
    valid = data["error"].isna() & score.notna()
    data = data.loc[valid].copy()
    data["score_raw"] = score.loc[valid]
    data["score"] = data["score_raw"].clip(0.0, 5.0)
    data["accepted"] = data["score"] >= 4.0
    data["model_short"] = data["model"].map(MODEL_SHORT_NAMES)
    if len(data) != 185_271 or data["model_short"].isna().any():
        raise ValueError("Unexpected valid-response count or model name")
    return data


def build_cells(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate decodings into model-task-temperature-condition cells."""
    keys = ["model", "model_short", "task_id", "category", "temperature", "condition"]
    return (
        data.groupby(keys, observed=True)
        .agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            score_sd=("score", "std"),
            accept_rate=("accepted", "mean"),
            mean_prior_score=("prior_score_used", "mean"),
        )
        .reset_index()
    )


def build_contrasts(cells: pd.DataFrame) -> pd.DataFrame:
    """Create score and acceptance contrasts from cell means."""
    index = ["model", "model_short", "task_id", "category", "temperature"]
    scores = cells.pivot(
        index=index, columns="condition", values="mean_score"
    ).reset_index()
    accepts = cells.pivot(
        index=index, columns="condition", values="accept_rate"
    ).reset_index()
    output = scores[index].copy()
    for condition in ("C0", "C1", "C2"):
        output[f"{condition.lower()}_mean"] = scores[condition]
        output[f"accept_{condition.lower()}"] = accepts[condition]
    output["c1_c0"] = scores["C1"] - scores["C0"]
    output["c2_c0"] = scores["C2"] - scores["C0"]
    output["c2_c1"] = scores["C2"] - scores["C1"]
    output["accept_c1_c0"] = accepts["C1"] - accepts["C0"]
    output["accept_c2_c0"] = accepts["C2"] - accepts["C0"]
    return output


def stratified_task_bootstrap(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    *,
    group_column: str = "model_short",
    reps: int = 10_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Resample five tasks within each category while retaining temperatures."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    values = list(value_columns)
    for group, group_data in frame.groupby(group_column, sort=True):
        bootstrap = {value: np.zeros(reps) for value in values}
        for category in CATEGORIES:
            category_data = group_data[group_data["category"] == category]
            indices = rng.integers(0, 5, size=(reps, 5))
            for value in values:
                matrix = (
                    category_data.pivot(
                        index="task_id", columns="temperature", values=value
                    )
                    .sort_index()
                    .to_numpy()
                )
                if matrix.shape != (5, 4):
                    raise ValueError(f"Unexpected task matrix: {matrix.shape}")
                bootstrap[value] += matrix[indices].mean(axis=(1, 2)) / len(CATEGORIES)
        for value, estimates in bootstrap.items():
            rows.append(
                {
                    group_column: group,
                    "contrast": value,
                    "estimate": group_data[value].mean(),
                    "ci_low": np.quantile(estimates, 0.025),
                    "ci_high": np.quantile(estimates, 0.975),
                    "reps": reps,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def write_json(path: Any, value: Any) -> None:
    """Write deterministic JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
