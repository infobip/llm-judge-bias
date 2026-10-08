"""Summarize the paper's targeted token-probability probes."""

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from llm_judge_bias.config import DATA_DIR, MODEL_SHORT_NAMES, RESULTS_DIR

DIGIT = re.compile(r'["\s]*([0-5])(?:[.\s"\'\,]|$)')
SCORE_KEY = re.compile(r'score["\s]*:', re.IGNORECASE)


def score_token_probabilities(
    tokens: list[dict[str, Any]] | None,
) -> dict[str, float] | None:
    """Extract score-digit probabilities from one token sequence."""
    if not tokens:
        return None
    position = None
    for index, token in enumerate(tokens):
        if SCORE_KEY.search(token["token"]):
            position = index
            break
        if "score" in token["token"].lower():
            for lookahead in range(index + 1, min(index + 3, len(tokens))):
                if ":" in tokens[lookahead]["token"]:
                    position = lookahead
                    break
        if position is not None:
            break
    if position is None:
        return None
    for token in tokens[position + 1 :]:
        actual = next(
            (character for character in token["token"] if character in "012345"), None
        )
        if actual is None:
            continue
        probabilities: dict[str, float] = {}
        for alternative in token.get("top_alternatives", []):
            match = DIGIT.search(alternative["token"])
            digit = (
                match.group(1) if match else alternative["token"].strip().strip("\"'")
            )
            if digit in "012345" and len(digit) == 1:
                probabilities.setdefault(digit, float(np.exp(alternative["logprob"])))
        probabilities[actual] = float(np.exp(token["logprob"]))
        return probabilities
    return None


def run() -> None:
    """Write GPT-4.1 and Llama probe summaries."""
    RESULTS_DIR.mkdir(exist_ok=True)
    gpt = json.loads((DATA_DIR / "mechanism_gpt41_summary.json").read_text())
    rows = []
    for digit, values in gpt["token_probabilities"].items():
        rows.append(
            {
                "model": "GPT-4.1",
                "digit": digit,
                "p_c0": values["C0"],
                "p_c2": values["C2"],
                "change": values["C2"] - values["C0"],
                "within_c2_slope": values["within_c2_slope"],
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mechanism_gpt41.csv", index=False)

    raw = pd.read_parquet(DATA_DIR / "mechanism_llama.parquet")
    probability_rows = []
    for row in raw.itertuples(index=False):
        tokens = (
            json.loads(row.token_logprobs)
            if isinstance(row.token_logprobs, str)
            else row.token_logprobs
        )
        probabilities = score_token_probabilities(tokens)
        if probabilities:
            for digit, probability in probabilities.items():
                probability_rows.append(
                    {
                        "model": row.model,
                        "condition": row.condition,
                        "prior_score": row.prior_score,
                        "digit": digit,
                        "probability": probability,
                    }
                )
    probabilities = pd.DataFrame(probability_rows)
    behavioral = []
    token_results = []
    for model, group in raw.groupby("model", sort=True):
        c0 = group[group["condition"] == "C0"]["output_score"]
        c2 = group[group["condition"] == "C2"]["output_score"]
        short = MODEL_SHORT_NAMES[model]
        behavioral.append(
            {
                "model": short,
                "c0_mean": c0.mean(),
                "c2_mean": c2.mean(),
                "c2_c0": c2.mean() - c0.mean(),
            }
        )
        for digit in ("3", "4", "5"):
            values = probabilities[
                (probabilities["model"] == model) & (probabilities["digit"] == digit)
            ]
            c0_probability = values[values["condition"] == "C0"]["probability"].mean()
            c2_values = values[values["condition"] == "C2"]
            if not np.isfinite(c0_probability) or c2_values.empty:
                continue
            grouped = (
                c2_values.groupby("prior_score")["probability"].mean().reset_index()
            )
            token_results.append(
                {
                    "model": short,
                    "digit": digit,
                    "p_c0": c0_probability,
                    "p_c2": c2_values["probability"].mean(),
                    "change": c2_values["probability"].mean() - c0_probability,
                    "within_c2_slope": stats.linregress(
                        grouped["prior_score"], grouped["probability"]
                    ).slope,
                }
            )
    pd.DataFrame(behavioral).to_csv(
        RESULTS_DIR / "mechanism_llama_behavior.csv", index=False
    )
    pd.DataFrame(token_results).to_csv(
        RESULTS_DIR / "mechanism_llama_tokens.csv", index=False
    )


if __name__ == "__main__":
    run()
