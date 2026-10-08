"""Prompt construction for the three experimental conditions."""

from typing import Any

import numpy as np

from .config import DEFAULT_RUBRIC, SEED, THRESHOLD

_rng = np.random.default_rng(SEED)

SYSTEM_PROMPT = """You are a meticulous evaluator.

Judge the answer strictly against the rubric and output JSON with fields:
  - score (a floating-point number between 0 and 5 with 2 decimal places),
  - accept (boolean true or false),
  - rationale (short text),
  - flags (array of strings).

Do not include any other text."""


def _rubric_block(rubric: dict[str, Any]) -> str:
    criteria = "\n".join(
        f"  - {item['name']}: {item['description']}"
        for item in rubric.get("criteria", [])
    )
    return f"""Rubric:
{criteria}

Scoring guidance:
  - [0-1): Unacceptable
  - [1-2): Poor
  - [2-3): Fair
  - [3-4): Good
  - [4-5): Very Good
  - [5]: Excellent

Threshold: accept if score >= {THRESHOLD}

Output format:
{{"score": ..., "accept": ..., "rationale": "...", "flags": ["..."]}}"""


def build_messages(
    task: str,
    answer: str,
    condition: str,
    rubric: dict[str, Any] | None = None,
    *,
    attempt: int | None = None,
    prior_score: float | None = None,
) -> tuple[list[dict[str, str]], dict[str, float | int]]:
    """Build judge messages and return the condition metadata used."""
    metadata: dict[str, float | int] = {}
    if condition == "C0":
        condition_text = ""
    elif condition == "C1":
        condition_text = "Attempt metadata:\n  - This is a revision\n"
    elif condition == "C2":
        attempt = attempt if attempt is not None else int(_rng.integers(1, 5))
        prior_score = (
            prior_score
            if prior_score is not None
            else round(float(_rng.uniform(0, 3.99)), 2)
        )
        metadata = {"attempt": attempt, "prior_score": prior_score}
        condition_text = (
            "Attempt metadata:\n"
            f"  - Attempt: {attempt} (This is a revision)\n"
            f"  - Prior score: {prior_score}\n"
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    parts = [f"Task:\n{task}\n"]
    if condition_text:
        parts.append(condition_text)
    parts.extend([f"Answer:\n{answer}\n", _rubric_block(rubric or DEFAULT_RUBRIC)])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ], metadata
