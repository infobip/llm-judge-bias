"""Tests for the industry runner that do not require private data."""

import numpy as np
import pandas as pd

from run_industry import LABELS, normalize
from scripts.industry import paired_metrics


def test_label_normalization() -> None:
    """Industry labels follow the experiment's normalization rules."""
    assert normalize("SHAFTS") == "shaft"
    assert normalize("non-compliant") == "compliant"
    assert normalize(" drifting ") == "drifting"
    assert normalize(None) == "unclassified"
    assert set(LABELS) == {"compliant", "drifting", "shaft"}


def test_paired_metrics_use_c0_correct_denominator() -> None:
    """Induced errors are conditional on each strategy's C0-correct cases."""
    frame = pd.DataFrame(
        {
            "human_label": ["drifting", "shaft", "drifting"],
            "anchor": ["shaft", "drifting", "shaft"],
            "C0": ["drifting", "shaft", "compliant"],
            "C2": ["shaft", "shaft", "shaft"],
        }
    )
    metrics = paired_metrics(frame, np.random.default_rng(42))
    assert metrics["c0_correct_n"] == 2
    assert metrics["induced_n"] == 1
    assert metrics["induced_among_c0_correct"] == 0.5
