"""Metrics: Accuracy, Precision, Recall, F1 (FAKE = positive class)."""
from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import RESULTS_DIR

METRICS_PATH = RESULTS_DIR / "metrics.json"


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def save_metrics(model_name: str, metrics: dict, **extra) -> None:
    """Merge this model's metrics into results/metrics.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    all_metrics = load_metrics()
    all_metrics[model_name] = {**metrics, **extra}
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))
