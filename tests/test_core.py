"""Focused tests for experiment and analysis code."""

import numpy as np
import pandas as pd
import pytest

from llm_judge_bias.analysis import (
    build_cells,
    build_contrasts,
    stratified_task_bootstrap,
)
from llm_judge_bias.experiment import parse_json
from llm_judge_bias.prompts import build_messages
from llm_judge_bias.tasks import get_tasks
from scripts.mechanism import score_token_probabilities
from scripts.mitigation import analyze


def test_public_tasks() -> None:
    """The repository contains the exact 20-task design."""
    tasks = get_tasks()
    assert len(tasks) == 20
    assert len({task["task_id"] for task in tasks}) == 20
    assert pd.Series([task["category"] for task in tasks]).value_counts().eq(5).all()


def test_conditions() -> None:
    """C0, C1, and C2 contain the intended metadata."""
    c0, _ = build_messages("task", "answer", "C0")
    c1, _ = build_messages("task", "answer", "C1")
    c2, metadata = build_messages("task", "answer", "C2", attempt=2, prior_score=3.0)
    assert "revision" not in c0[1]["content"].lower()
    assert "revision" in c1[1]["content"].lower()
    assert "Prior score: 3.0" in c2[1]["content"]
    assert metadata == {"attempt": 2, "prior_score": 3.0}


def test_json_parser() -> None:
    """The parser accepts JSON, fences, and surrounding prose."""
    assert parse_json('{"score": 4}') == {"score": 4}
    assert parse_json('```json\n{"score": 4}\n```') == {"score": 4}
    assert parse_json('Result: {"score": 4}.') == {"score": 4}


def test_cell_contrasts_do_not_pair_trial_indices() -> None:
    """Contrasts use all valid responses in each cell."""
    frame = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "model_short": ["M"] * 5,
            "task_id": ["t"] * 5,
            "category": ["code_review"] * 5,
            "temperature": [0.0] * 5,
            "condition": ["C0", "C0", "C1", "C2", "C2"],
            "score": [4.0, 5.0, 4.0, 3.0, 4.0],
            "accepted": [True, True, True, False, True],
            "prior_score_used": [None, None, None, 1.0, 2.0],
        }
    )
    contrast = build_contrasts(build_cells(frame)).iloc[0]
    assert contrast["c1_c0"] == -0.5
    assert contrast["c2_c0"] == -1.0


def test_bootstrap_reproducibility() -> None:
    """Task bootstrap results repeat for a fixed seed."""
    rows = []
    for category_index, category in enumerate(
        ("code_review", "creative_writing", "factual_qa", "summarization")
    ):
        for task in range(5):
            for temperature in (0.0, 0.5, 1.0, 2.0):
                rows.append(
                    {
                        "model_short": "M",
                        "category": category,
                        "task_id": f"{category_index}-{task}",
                        "temperature": temperature,
                        "c2_c0": float(category_index - task),
                    }
                )
    frame = pd.DataFrame(rows)
    first = stratified_task_bootstrap(frame, ["c2_c0"], reps=100)
    second = stratified_task_bootstrap(frame, ["c2_c0"], reps=100)
    pd.testing.assert_frame_equal(first, second)


def test_score_token_parser() -> None:
    """Score-token probabilities include generated and alternative digits."""
    probabilities = score_token_probabilities(
        [
            {"token": '"score":', "logprob": -0.1, "top_alternatives": []},
            {
                "token": " 4",
                "logprob": 0.0,
                "top_alternatives": [{"token": " 5", "logprob": -1.0}],
            },
        ]
    )
    assert probabilities == pytest.approx({"4": 1.0, "5": np.exp(-1)})


def test_mitigation_analysis_uses_no_text_columns() -> None:
    """Mitigation aggregates need scores and token counts only."""
    rows = []
    for strategy in ("baseline", "cot", "warning"):
        for condition, prior, values in (
            ("C0", None, (4.0, 4.0)),
            ("C2", 1.0, (3.0, 3.0)),
            ("C2", 2.0, (3.5, 3.5)),
        ):
            for score in values:
                rows.append(
                    {
                        "strategy": strategy,
                        "condition": condition,
                        "prior_score": prior,
                        "score": score,
                        "error": None,
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                )
    result = analyze(pd.DataFrame(rows))
    assert set(result["strategy"]) == {"baseline", "cot", "warning"}
    assert not {"prompt", "response_raw", "rationale"} & set(result.columns)
