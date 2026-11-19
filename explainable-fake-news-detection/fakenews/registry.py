"""Find and load trained models by short name: nb, lr, lstm, bert."""
from __future__ import annotations

from .config import MODELS_DIR

PREFERENCE = ("bert", "lstm", "lr", "nb")  # best-first


def _path(name: str):
    return {
        "nb": MODELS_DIR / "nb.joblib",
        "lr": MODELS_DIR / "lr.joblib",
        "lstm": MODELS_DIR / "lstm.pt",
        "bert": MODELS_DIR / "bert" / "detector.json",
    }[name]


def available_models() -> list[str]:
    return [m for m in PREFERENCE if _path(m).exists()]


def load_model(name: str):
    """Return an object with .predict_proba(list_of_texts) -> array of shape (n, 2) [REAL, FAKE]."""
    if name not in PREFERENCE:
        raise ValueError(f"Unknown model '{name}'. Choose from {PREFERENCE}.")
    if not _path(name).exists():
        raise FileNotFoundError(f"Model '{name}' is not trained yet. Run:  python train.py --models {name}")
    if name in ("nb", "lr"):
        import joblib

        return joblib.load(_path(name))
    if name == "lstm":
        from .lstm import LSTMDetector

        return LSTMDetector.load(MODELS_DIR / "lstm.pt")
    from .bert import BertDetector

    return BertDetector.load(MODELS_DIR / "bert")
