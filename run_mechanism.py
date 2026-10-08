"""Run the targeted score-token mechanism probe."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from llm_judge_bias.experiment import evaluate
from llm_judge_bias.models import get_client
from llm_judge_bias.tasks import get_task

PRIORS = (1.0, 2.0, 3.0, 3.9)


async def run(model: str, trials: int, output_dir: Path) -> Path:
    """Run C0, C1, and a C2 anchor sweep with token log probabilities."""
    client = get_client(model)
    task = get_task("write_005_forgotten_language")
    specifications = [
        (condition, trial, prior)
        for condition, priors in (("C0", (None,)), ("C1", (None,)), ("C2", PRIORS))
        for prior in priors
        for trial in range(trials)
    ]
    rows: list[dict[str, Any]] = []
    for condition, trial, prior in tqdm(specifications, desc=model):
        rows.append(
            await evaluate(
                client,
                task,
                condition,
                trial,
                0.0,
                logprobs=True,
                prior_score=prior,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"mechanism_{timestamp}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def main() -> None:
    """Parse options and run the probe."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="infobip-gpt-4-1")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    print(asyncio.run(run(args.model, args.trials, args.output_dir)))


if __name__ == "__main__":
    main()
