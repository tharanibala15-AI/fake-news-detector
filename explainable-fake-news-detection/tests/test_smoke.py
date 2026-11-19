"""Quick checks: python -m pytest -q"""
import numpy as np

from fakenews.classic import build_pipeline
from fakenews.data import load_dataset, split_dataset
from fakenews.evaluate import compute_metrics
from fakenews.explain import explain
from fakenews.preprocess import tokenize, tokenize_and_stem


def test_preprocess():
    assert tokenize("The GOVERNMENT has announced https://x.com a new tax cut!") == [
        "government", "announced", "new", "tax", "cut"]
    assert tokenize_and_stem("Announcing announced") == ["announc", "announc"]


def test_sample_dataset_and_split():
    df = load_dataset("sample")
    assert set(df["label"]) == {0, 1}
    train, val, test = split_dataset(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_train_predict_explain():
    df = load_dataset("sample")
    train, _, test = split_dataset(df)
    model = build_pipeline("lr").fit(train["text"], train["label"])
    proba = model.predict_proba(test["text"].tolist())
    assert proba.shape == (len(test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    m = compute_metrics(test["label"], proba.argmax(1))
    assert 0.0 <= m["accuracy"] <= 1.0
    pairs = explain("occlusion", model.predict_proba, "Shocking secret miracle cure exposed", top_k=5)
    assert pairs and all(isinstance(w, str) for w, _ in pairs)
