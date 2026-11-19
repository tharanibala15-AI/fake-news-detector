"""Explainability module: LIME, SHAP and a dependency-free occlusion explainer.

All explainers take `predict_proba(list_of_texts) -> (n, 2) array [REAL, FAKE]` and return a list of
(word, weight) pairs.  weight > 0  -> the word pushes the prediction toward FAKE
                        weight < 0  -> the word pushes the prediction toward REAL
"""
from __future__ import annotations

import re
import warnings

import numpy as np

from .config import CLASS_NAMES, SEED

METHODS = ("lime", "shap", "occlusion")
_PARTS = re.compile(r"[\w']+|[^\w']+")


def _fake_prob(predict_proba):
    return lambda texts: np.asarray(predict_proba(list(texts)))[:, 1]


# --------------------------------------------------------------------------- LIME
def explain_lime(predict_proba, text: str, num_features: int = 8, num_samples: int = 500):
    from lime.lime_text import LimeTextExplainer  # pip install lime

    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=SEED)
    exp = explainer.explain_instance(
        text, lambda t: np.asarray(predict_proba(list(t))),
        labels=(1,), num_features=num_features, num_samples=num_samples,
    )
    return [(w, float(v)) for w, v in exp.as_list(label=1)]


# --------------------------------------------------------------------------- SHAP
def explain_shap(predict_proba, text: str, max_words: int = 60, max_evals: int = 400):
    import shap  # pip install shap

    text = " ".join(text.split()[:max_words])  # SHAP cost grows fast with length
    n_tokens = len(re.split(r"\W+", text))
    explainer = shap.Explainer(_fake_prob(predict_proba), shap.maskers.Text(r"\W+"))
    sv = explainer([text], max_evals=max(max_evals, 2 * n_tokens + 2))
    pairs = {}
    for tok, val in zip(sv.data[0], sv.values[0]):
        tok = str(tok).strip()
        if tok:
            pairs[tok] = pairs.get(tok, 0.0) + float(val)
    return list(pairs.items())


# --------------------------------------------------------------------------- Occlusion
def explain_occlusion(predict_proba, text: str, max_words: int = 200):
    """Remove each distinct word in turn and measure how much P(FAKE) changes. No extra libraries."""
    parts = _PARTS.findall(text)
    is_word = [bool(re.fullmatch(r"[\w']+", p)) for p in parts]
    vocab = list(dict.fromkeys(p.lower() for p, w in zip(parts, is_word) if w and len(p) > 1))[:max_words]
    variants = [text] + [
        "".join(p for p, w in zip(parts, is_word) if not (w and p.lower() == word)) for word in vocab
    ]
    probs = _fake_prob(predict_proba)(variants)
    return [(word, float(probs[0] - probs[i + 1])) for i, word in enumerate(vocab)]


# --------------------------------------------------------------------------- Public API
def explain(method: str, predict_proba, text: str, top_k: int = 8):
    """Return the top_k most influential words (by absolute weight), strongest first."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    try:
        if method == "lime":
            pairs = explain_lime(predict_proba, text, num_features=top_k)
        elif method == "shap":
            pairs = explain_shap(predict_proba, text)
        else:
            pairs = explain_occlusion(predict_proba, text)
    except ImportError:
        warnings.warn(f"'{method}' is not installed - falling back to the built-in occlusion explainer.")
        pairs = explain_occlusion(predict_proba, text)
    pairs = [p for p in pairs if abs(p[1]) > 1e-6]  # drop words that had no influence at all
    return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:top_k]


def plot_contributions(pairs, title: str = "Important words (contribution to FAKE)", save_path=None):
    """Horizontal bar chart like the SHAP example on the poster. Returns a matplotlib Figure."""
    from matplotlib.figure import Figure

    pairs = sorted(pairs, key=lambda p: p[1])  # largest FAKE-contribution ends up on top
    words = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    fig = Figure(figsize=(6, 0.45 * max(len(pairs), 3) + 1.2))
    ax = fig.subplots()
    ax.barh(words, vals, color=["#d62728" if v > 0 else "#2ca02c" for v in vals])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.margins(x=0.25)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("<- pushes toward REAL      pushes toward FAKE ->")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
