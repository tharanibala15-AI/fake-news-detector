"""Classical baselines: TF-IDF + Naive Bayes / Logistic Regression.

The whole preprocessing chain lives inside the sklearn Pipeline, so the saved model accepts
RAW text. That is what lets LIME / SHAP / occlusion perturb the original words.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocess import tokenize_and_stem


def build_pipeline(kind: str, max_features: int = 50_000) -> Pipeline:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "lr":
        clf = LogisticRegression(max_iter=1000, C=5.0)
    else:
        raise ValueError("kind must be 'nb' or 'lr'")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])
