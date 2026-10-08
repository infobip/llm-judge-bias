"""CLI for the main text experiment."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from llm_judge_bias.config import API_MODELS, HUGGINGFACE_MODELS
from llm_judge_bias.experiment import run_experiment


def main() -> None:
    """Run selected tasks and models."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=("summarization", "code_review", "creative_writing", "factual_qa"),
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-concurrent", type=int, default=9)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        print("API models:", *API_MODELS, sep="\n  ")
        print("Local models:", *HUGGINGFACE_MODELS, sep="\n  ")
        return
    if not args.models:
        parser.error("--models is required unless --list-models is used")
    path = asyncio.run(
        run_experiment(
            args.models,
            task_ids=args.tasks,
            categories=args.categories,
            trials=args.trials,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
