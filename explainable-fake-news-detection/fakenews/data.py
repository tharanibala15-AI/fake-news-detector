"""Dataset loading. Every loader returns a DataFrame with columns: text, label (0=REAL, 1=FAKE)."""
from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_DIR, SAMPLE_PATH, SEED

DATASETS = ("sample", "kaggle", "liar", "both")

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",
]
LIAR_FAKE_LABELS = {"pants-fire", "false", "barely-true"}
LIAR_REAL_LABELS = {"half-true", "mostly-true", "true"}

# "WASHINGTON (Reuters) - " at the start of real articles is a well-known giveaway in the
# Fake-and-Real-News dataset (models just learn the word "Reuters"). We strip it.
_REUTERS = re.compile(r"^.{0,60}?\(Reuters\)\s*-?\s*", flags=re.S)


def _first_existing(candidates):
    return next((p for p in candidates if p.exists()), None)


def load_sample() -> pd.DataFrame:
    """Tiny built-in dataset (60 headlines) - only for checking that the pipeline runs."""
    return pd.read_csv(SAMPLE_PATH)[["text", "label"]]


def load_kaggle() -> pd.DataFrame:
    """Fake and Real News Dataset (Fake.csv + True.csv) from Kaggle, ~44,898 articles."""
    fake = _first_existing([RAW_DIR / "Fake.csv", RAW_DIR / "kaggle" / "Fake.csv"])
    true = _first_existing([RAW_DIR / "True.csv", RAW_DIR / "kaggle" / "True.csv"])
    if fake is None or true is None:
        raise FileNotFoundError(
            "Fake.csv / True.csv not found. Download the 'Fake and Real News Dataset' from Kaggle "
            f"and put both files in {RAW_DIR}  (see data/README.md)."
        )
    parts = []
    for path, label in ((fake, 1), (true, 0)):
        df = pd.read_csv(path)
        body = df["text"].fillna("").str.replace(_REUTERS, "", regex=True)
        text = (df["title"].fillna("") + ". " + body).str.strip()
        parts.append(pd.DataFrame({"text": text, "label": label}))
    out = pd.concat(parts, ignore_index=True)
    out = out[out["text"].str.len() >= 20].drop_duplicates("text")
    return out.reset_index(drop=True)


def load_liar() -> pd.DataFrame:
    """LIAR dataset (train/valid/test .tsv), 12,836 statements. Six labels -> binary."""
    folder = _first_existing(
        [d for d in (RAW_DIR / "liar", RAW_DIR / "liar_dataset", RAW_DIR) if (d / "train.tsv").exists()]
    )
    if folder is None:
        raise FileNotFoundError(
            f"LIAR files not found. Put train.tsv, valid.tsv and test.tsv in {RAW_DIR / 'liar'} "
            "(see data/README.md)."
        )
    frames = []
    for name in ("train", "valid", "test"):
        path = folder / f"{name}.tsv"
        if path.exists():
            frames.append(
                pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, quoting=3, on_bad_lines="skip")
            )
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(LIAR_FAKE_LABELS | LIAR_REAL_LABELS)]  # also drops any header row
    out = pd.DataFrame(
        {"text": df["statement"].astype(str), "label": df["label"].isin(LIAR_FAKE_LABELS).astype(int)}
    )
    return out.drop_duplicates("text").reset_index(drop=True)


def load_dataset(name: str = "sample") -> pd.DataFrame:
    if name == "sample":
        return load_sample()
    if name == "kaggle":
        return load_kaggle()
    if name == "liar":
        return load_liar()
    if name == "both":
        both = pd.concat([load_kaggle(), load_liar()], ignore_index=True)
        return both.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")


def split_dataset(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = SEED):
    """Stratified train / validation / test split (default 80 / 10 / 10)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
