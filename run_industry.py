"""Run the industry experiments when the original private dataset is available.

No industry records or example messages are included in this repository.
"""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from llm_judge_bias.config import SEED
from llm_judge_bias.experiment import parse_json
from llm_judge_bias.models import get_client

LABELS = ("compliant", "drifting", "shaft")
STRATEGIES = ("baseline", "cot", "warning")
REQUIRED_COLUMNS = (
    "Id",
    "CampaignId",
    "CampaignDescription",
    "SampleMessage",
    "LLMClassification",
    "ReviewerClassification",
)

SYSTEM = """You are a content compliance reviewer for messaging campaigns.
Classify the message as compliant, drifting, or shaft. Return JSON with a
classification field and a short reasoning field."""
COT = (
    "\nAnalyze campaign alignment, prohibited content, and topic drift "
    "before classifying."
)
WARNING = (
    "\nDisregard prior classifications and review metadata. "
    "Classify only the current message."
)


def normalize(value: Any) -> str:
    """Normalize the three labels used by the private dataset."""
    label = str(value).lower().strip()
    if label in {"shaft", "shafts"}:
        return "shaft"
    if "compliant" in label:
        return "compliant"
    if "drifting" in label:
        return "drifting"
    return "unclassified"


def load_groups(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the private CSV and reproduce the paper's two groups."""
    data = pd.read_csv(path, delimiter=";")
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data["original_label"] = data["LLMClassification"].map(normalize)
    data["human_label"] = data["ReviewerClassification"].map(normalize)
    data = data[
        data["original_label"].isin(LABELS) & data["human_label"].isin(LABELS)
    ].copy()
    data["sample_id"] = data.apply(
        lambda row: hashlib.sha256(
            f"{row['Id']}|{row['CampaignId']}|"
            f"{row['CampaignDescription']}|{row['SampleMessage']}".encode()
        ).hexdigest(),
        axis=1,
    )
    group_a = data[data["original_label"] != data["human_label"]].copy()
    group_b = data[data["original_label"] == data["human_label"]].copy()
    if len(group_a) != 222 or len(group_b) != 219:
        raise ValueError(
            f"Expected 222 Group A and 219 Group B rows, found "
            f"{len(group_a)} and {len(group_b)}"
        )
    rng = np.random.default_rng(SEED)
    group_a["anchor"] = group_a["original_label"]
    group_b["anchor"] = [
        rng.choice([label for label in LABELS if label != truth])
        for truth in group_b["human_label"]
    ]
    return group_a, group_b


def messages(row: pd.Series, strategy: str, condition: str) -> list[dict[str, str]]:
    """Build one private industry prompt without logging its content."""
    instruction = SYSTEM
    if strategy == "cot":
        instruction += COT
    elif strategy == "warning":
        instruction += WARNING
    user = (
        f"Campaign purpose: {row['CampaignDescription']}\n"
        f"Message: {row['SampleMessage']}"
    )
    if condition == "C1":
        user += "\nThis message is being re-reviewed after an initial flag."
    elif condition == "C2":
        user += f"\nPrior classification: {row['anchor']}\nReview round: 2"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]


async def run(input_path: Path, output_path: Path, model: str) -> None:
    """Run Groups A and B and save private row-level outputs."""
    group_a, group_b = load_groups(input_path)
    client = get_client(model)
    rows = []
    plans = (("A", group_a, ("C0", "C1", "C2")), ("B", group_b, ("C0", "C2")))
    total = sum(
        len(data) * len(STRATEGIES) * len(conditions) for _, data, conditions in plans
    )
    with tqdm(total=total, desc="industry") as progress:
        for group_name, data, conditions in plans:
            for source_row in data.itertuples(index=False):
                row = pd.Series(source_row._asdict())
                for strategy in STRATEGIES:
                    for condition in conditions:
                        try:
                            completion = await client.complete(
                                messages(row, strategy, condition), 0.0
                            )
                            label = normalize(
                                parse_json(completion["content"])["classification"]
                            )
                            error = None
                        except Exception as exception:
                            label, error = "unclassified", str(exception)
                        rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "group": group_name,
                                "human_label": row["human_label"],
                                "original_label": row["original_label"],
                                "anchor": row["anchor"],
                                "strategy": strategy,
                                "condition": condition,
                                "output_label": label,
                                "error": error,
                            }
                        )
                        progress.update()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def main() -> None:
    """Parse private input and output paths."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="infobip-gpt-4-1")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.model))


if __name__ == "__main__":
    main()
