"""Validate committed inputs and key paper results."""

import json

import pandas as pd

from llm_judge_bias.analysis import load_main_data
from llm_judge_bias.config import DATA_DIR, RESULTS_DIR


def close(actual: float, expected: float, tolerance: float = 0.001) -> None:
    """Raise when a reproduced value differs from the paper value."""
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, found {actual}")


def main() -> None:
    """Check privacy, row counts, and representative paper values."""
    raw = pd.read_parquet(DATA_DIR / "main_experiment.parquet")
    forbidden = {"system_prompt", "user_prompt", "response_raw", "rationale", "flags"}
    if forbidden & set(raw.columns):
        raise AssertionError("Main data contain unnecessary model text")
    data = load_main_data()
    if len(raw) != 192_000 or len(data) != 185_271:
        raise AssertionError("Main data counts do not match the paper")

    mitigation = pd.read_parquet(DATA_DIR / "mitigation_trials.parquet")
    if forbidden & set(mitigation.columns):
        raise AssertionError("Mitigation data contain prompt or response text")
    if len(mitigation) != 1_500:
        raise AssertionError("Mitigation call count does not match the paper")

    model = pd.read_csv(RESULTS_DIR / "model_effects.csv").set_index("model_short")
    close(model.loc["GPT-4.1", "estimate"], -0.4106)
    close(model.loc["Llama-3.2-3B", "cohens_d"], -0.71, 0.01)
    if int(((model["ci_low"] > 0) | (model["ci_high"] < 0)).sum()) != 7:
        raise AssertionError("Expected seven model intervals to exclude zero")

    counts = json.loads((RESULTS_DIR / "counts.json").read_text())
    expected = {
        "attempted": 192_000,
        "valid": 185_271,
        "complete_triplets": 59_136,
        "pair_c0_c2": 60_133,
        "out_of_bounds_clipped": 18,
        "complete_cells": 1_920,
        "stored_accept_disagreements": 1_453,
    }
    if {key: counts[key] for key in expected} != expected:
        raise AssertionError("Count audit does not match the paper")

    industry_files = list(RESULTS_DIR.glob("industry*"))
    if any(path.suffix != ".csv" for path in industry_files):
        raise AssertionError(
            "Public repository must contain aggregate industry CSVs only"
        )
    print("Validated public inputs and reproduced paper values.")


if __name__ == "__main__":
    main()
