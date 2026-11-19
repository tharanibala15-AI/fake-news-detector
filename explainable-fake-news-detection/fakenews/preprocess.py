"""Text preprocessing: cleaning, tokenization, stop-word removal, stemming.

Uses scikit-learn's built-in English stop-word list and NLTK's PorterStemmer
(pure Python - no extra NLTK data download needed).
"""
from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:  # NLTK is listed in requirements.txt, but fall back gracefully if missing
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    def _stem(word: str) -> str:
        return _stemmer.stem(word)

except ImportError:  # pragma: no cover

    def _stem(word: str) -> str:
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z]{2,}")


def clean_text(text: str) -> str:
    """Lower-case and strip URLs / HTML tags / extra whitespace."""
    text = str(text).lower().replace("'", "")
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Clean -> tokenize -> remove stop-words (no stemming)."""
    return [w for w in _WORD.findall(clean_text(text)) if w not in STOPWORDS]


@lru_cache(maxsize=300_000)
def stem_word(word: str) -> str:
    return _stem(word)


def tokenize_and_stem(text: str) -> list[str]:
    """Full pipeline used by the TF-IDF models: tokenization, stop-word removal, stemming."""
    return [stem_word(w) for w in tokenize(text)]
