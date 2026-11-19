"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_PATH = DATA_DIR / "sample" / "sample_news.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Label convention used everywhere: 0 = REAL, 1 = FAKE (FAKE is the positive class)
REAL, FAKE = 0, 1
CLASS_NAMES = ["REAL", "FAKE"]

# Pretty names for charts / UI (also defines the display order)
MODEL_LABELS = {
    "nb": "Naive Bayes",
    "lr": "Logistic Regression",
    "lstm": "LSTM",
    "bert": "BERT (Ours)",
}
