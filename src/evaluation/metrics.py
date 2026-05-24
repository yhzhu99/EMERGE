from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics


def minpse(preds: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, _ = sk_metrics.precision_recall_curve(labels, preds)
    return float(max(min(p, r) for p, r in zip(precision, recall, strict=False)))


def binary_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=int)
    out: dict[str, float] = {}
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(sk_metrics.roc_auc_score(labels, preds))
        out["auprc"] = float(sk_metrics.average_precision_score(labels, preds))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    out["minpse"] = minpse(preds, labels) if len(labels) else float("nan")
    hard = (preds >= 0.5).astype(int)
    out["accuracy"] = float(sk_metrics.accuracy_score(labels, hard))
    out["f1"] = float(sk_metrics.f1_score(labels, hard, zero_division=0))
    return out


def multitask_metrics(
    outcome_pred: np.ndarray,
    readmission_pred: np.ndarray,
    outcome_true: np.ndarray,
    readmission_true: np.ndarray,
) -> dict[str, float]:
    outcome = binary_metrics(outcome_pred, outcome_true)
    readmission = binary_metrics(readmission_pred, readmission_true)
    return {
        **{f"outcome_{key}": value for key, value in outcome.items()},
        **{f"readmission_{key}": value for key, value in readmission.items()},
    }


def bootstrap_multitask_metrics(
    outcome_pred: np.ndarray,
    readmission_pred: np.ndarray,
    outcome_true: np.ndarray,
    readmission_true: np.ndarray,
    *,
    rounds: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    size = len(outcome_true)
    samples: dict[str, list[float]] = {}
    for _ in range(rounds):
        idx = rng.choice(size, size=size, replace=True)
        metrics = multitask_metrics(
            outcome_pred[idx],
            readmission_pred[idx],
            outcome_true[idx],
            readmission_true[idx],
        )
        for key, value in metrics.items():
            samples.setdefault(key, []).append(value)
    return {
        key: {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
        for key, values in samples.items()
    }
