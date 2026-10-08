"""Load the 20 public text tasks used in the paper."""

import json
from typing import Any, Literal

from .config import CREATIVE_WRITING_RUBRIC, FACTUAL_QA_RUBRIC, TASKS_DIR

TaskCategory = Literal["summarization", "code_review", "creative_writing", "factual_qa"]
RUBRICS = {
    "CREATIVE_WRITING_RUBRIC": CREATIVE_WRITING_RUBRIC,
    "FACTUAL_QA_RUBRIC": FACTUAL_QA_RUBRIC,
}


def _load_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(TASKS_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                task = json.loads(line)
                rubric = task.get("rubric")
                if isinstance(rubric, str):
                    task["rubric"] = RUBRICS[rubric]
                tasks.append(task)
    if len(tasks) != 20:
        raise ValueError(f"Expected 20 tasks, found {len(tasks)}")
    return tasks


TASKS = _load_tasks()


def get_tasks(
    task_ids: list[str] | None = None,
    categories: list[TaskCategory] | None = None,
) -> list[dict[str, Any]]:
    """Return tasks matching optional ID and category filters."""
    tasks = TASKS
    if task_ids:
        tasks = [task for task in tasks if task["task_id"] in task_ids]
    if categories:
        tasks = [task for task in tasks if task["category"] in categories]
    return tasks


def get_task(task_id: str) -> dict[str, Any]:
    """Return one task or raise when its ID is unknown."""
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown task: {task_id}")
