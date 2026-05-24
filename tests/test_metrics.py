from __future__ import annotations

import numpy as np

from evaluation.metrics import binary_metrics, multitask_metrics


def test_binary_metrics() -> None:
    metrics = binary_metrics(np.array([0.1, 0.4, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["minpse"] == 1.0


def test_multitask_metrics_prefixes() -> None:
    metrics = multitask_metrics(
        np.array([0.1, 0.9]),
        np.array([0.8, 0.2]),
        np.array([0, 1]),
        np.array([1, 0]),
    )
    assert "outcome_auroc" in metrics
    assert "readmission_auprc" in metrics
