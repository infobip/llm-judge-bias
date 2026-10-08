"""Run the text-only experiments used in the paper."""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .config import CONDITIONS, MAX_CONCURRENT, MAX_RETRIES, TEMPERATURES, THRESHOLD
from .models import ModelClient, get_client
from .prompts import build_messages
from .tasks import TaskCategory, get_tasks


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fences and surrounding text."""
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```").strip("` \n")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            return json.loads(value[start : end + 1])
        raise


async def evaluate(
    client: ModelClient,
    task: dict[str, Any],
    condition: str,
    trial: int,
    temperature: float,
    *,
    logprobs: bool = False,
    attempt: int | None = None,
    prior_score: float | None = None,
) -> dict[str, Any]:
    """Run one evaluation with retries and return a tabular row."""
    messages, metadata = build_messages(
        task["task_description"],
        task["answer_text"],
        condition,
        task.get("rubric"),
        attempt=attempt,
        prior_score=prior_score,
    )
    error = None
    for retry in range(MAX_RETRIES):
        try:
            completion = await client.complete(messages, temperature, logprobs=logprobs)
            parsed = parse_json(completion["content"])
            score = float(parsed["score"])
            return {
                "task_id": task["task_id"],
                "category": task["category"],
                "trial": trial,
                "condition": condition,
                "model": client.model_name,
                "temperature": temperature,
                "threshold": THRESHOLD,
                "score": score,
                "accept": score >= THRESHOLD,
                "prior_score_used": metadata.get("prior_score"),
                "attempt": metadata.get("attempt"),
                "error": None,
                "retries": retry,
                "timestamp": time.time(),
                "response_raw": completion["content"],
                "logprobs": json.dumps(completion["logprobs"]),
                "prompt_tokens": completion["prompt_tokens"],
                "completion_tokens": completion["completion_tokens"],
                "total_tokens": completion["total_tokens"],
                "model_returned": completion["model"],
            }
        except Exception as exception:
            error = str(exception)
            if retry < MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * 2**retry)
    return {
        "task_id": task["task_id"],
        "category": task["category"],
        "trial": trial,
        "condition": condition,
        "model": client.model_name,
        "temperature": temperature,
        "threshold": THRESHOLD,
        "score": None,
        "accept": None,
        "prior_score_used": metadata.get("prior_score"),
        "attempt": metadata.get("attempt"),
        "error": error,
        "retries": MAX_RETRIES,
        "timestamp": time.time(),
        "response_raw": "",
        "logprobs": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "model_returned": None,
    }


async def run_experiment(
    model_names: list[str],
    *,
    task_ids: list[str] | None = None,
    categories: list[TaskCategory] | None = None,
    trials: int = 100,
    temperatures: tuple[float, ...] = TEMPERATURES,
    output_dir: Path = Path("runs"),
    max_concurrent: int = MAX_CONCURRENT,
) -> Path:
    """Run the main C0/C1/C2 experiment and write one Parquet file."""
    tasks = get_tasks(task_ids, categories)
    if not tasks:
        raise ValueError("No tasks matched the requested filters")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        client = get_client(model_name)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited(
            *args: Any,
            selected_client: ModelClient = client,
            selected_semaphore: asyncio.Semaphore = semaphore,
        ) -> dict[str, Any]:
            async with selected_semaphore:
                return await evaluate(selected_client, *args)

        specifications = [
            (task, condition, trial, temperature)
            for task in tasks
            for condition in CONDITIONS
            for temperature in temperatures
            for trial in range(trials)
        ]
        for start in tqdm(
            range(0, len(specifications), max_concurrent),
            desc=model_name,
        ):
            rows.extend(
                await asyncio.gather(
                    *(
                        limited(*spec)
                        for spec in specifications[start : start + max_concurrent]
                    )
                )
            )
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"main_experiment_{timestamp}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
