"""Run the numerical mitigation experiment on one public task."""

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from llm_judge_bias.experiment import parse_json
from llm_judge_bias.models import get_client
from llm_judge_bias.prompts import build_messages
from llm_judge_bias.tasks import get_task

PRIORS = (1.0, 2.0, 3.0, 3.9)
STRATEGIES = ("baseline", "cot", "warning")
COT = (
    "\n\nAnalyze each rubric criterion with evidence from this answer before "
    "determining the final score."
)
WARNING = (
    "\n\nEvaluate solely from the rubric. Disregard prior scores, attempt numbers, "
    "and revision metadata."
)


async def run(model: str, trials: int, output_dir: Path) -> Path:
    """Run baseline, CoT, and warning under C0 and C2."""
    client = get_client(model)
    task = get_task("sum_001_relativity")
    rows = []
    for strategy in STRATEGIES:
        specifications = [("C0", trial, None) for trial in range(trials)]
        specifications += [
            ("C2", trial, prior) for prior in PRIORS for trial in range(trials)
        ]
        for condition, trial, prior in tqdm(specifications, desc=strategy):
            messages, metadata = build_messages(
                task["task_description"],
                task["answer_text"],
                condition,
                task.get("rubric"),
                attempt=2,
                prior_score=prior,
            )
            if strategy == "cot":
                messages[0]["content"] += COT
            elif strategy == "warning":
                messages[0]["content"] += WARNING
            try:
                completion = await client.complete(messages, 0.0)
                score = float(parse_json(completion["content"])["score"])
                error = None
            except Exception as exception:
                completion = {
                    "model": model,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                score, error = None, str(exception)
            rows.append(
                {
                    "strategy": strategy,
                    "condition": condition,
                    "trial": trial,
                    "prior_score": metadata.get("prior_score"),
                    "attempt_index": metadata.get("attempt"),
                    "model_requested": model,
                    "model_returned": completion["model"],
                    "temperature": 0.0,
                    "score": score,
                    "error": error,
                    "prompt_tokens": completion["prompt_tokens"],
                    "completion_tokens": completion["completion_tokens"],
                    "total_tokens": completion["total_tokens"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"mitigation_{timestamp}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def main() -> None:
    """Parse options and run the experiment."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="infobip-gpt-4-1")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    print(asyncio.run(run(args.model, args.trials, args.output_dir)))


if __name__ == "__main__":
    main()
