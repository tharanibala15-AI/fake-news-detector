"""Bidirectional LSTM text classifier (PyTorch). Requires: pip install -r requirements-deep.txt"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .config import SEED

PAD, UNK = 0, 1
_TOKEN = re.compile(r"[a-z0-9]+")


def simple_tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


class _Net(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        out, _ = self.lstm(self.emb(x))
        out = out.masked_fill(~mask, -1e9)  # ignore padding, then max-pool over time
        return self.fc(self.drop(out.max(dim=1).values))


class LSTMDetector:
    def __init__(self, max_len: int = 256, max_vocab: int = 30_000, device: str | None = None):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.vocab: dict[str, int] = {}
        self.net: _Net | None = None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---------- helpers ----------
    def _build_vocab(self, texts):
        counts = Counter(w for t in texts for w in simple_tokens(t))
        words = [w for w, c in counts.most_common(self.max_vocab) if c >= 2]
        self.vocab = {w: i + 2 for i, w in enumerate(words)}

    def _encode(self, texts) -> torch.Tensor:
        rows = []
        for t in texts:
            ids = [self.vocab.get(w, UNK) for w in simple_tokens(t)[: self.max_len]] or [UNK]
            rows.append(ids + [PAD] * (self.max_len - len(ids)))
        return torch.tensor(rows, dtype=torch.long)

    # ---------- training ----------
    def fit(self, texts, labels, epochs: int = 4, batch_size: int = 64, lr: float = 1e-3, val=None):
        torch.manual_seed(SEED)
        texts, labels = list(texts), np.asarray(labels)
        self._build_vocab(texts)
        self.net = _Net(len(self.vocab) + 2).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.default_rng(SEED)

        for epoch in range(1, epochs + 1):
            self.net.train()
            order = rng.permutation(len(texts))
            total = 0.0
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                x = self._encode([texts[i] for i in idx]).to(self.device)
                y = torch.tensor(labels[idx], dtype=torch.long, device=self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            msg = f"[LSTM] epoch {epoch}/{epochs}  train_loss={total / len(texts):.4f}"
            if val is not None:
                acc = (self.predict_proba(val[0]).argmax(1) == np.asarray(val[1])).mean()
                msg += f"  val_acc={acc:.4f}"
            print(msg)
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 128) -> np.ndarray:
        self.net.eval()
        texts = list(texts)
        out = []
        for start in range(0, len(texts), batch_size):
            x = self._encode(texts[start : start + batch_size]).to(self.device)
            out.append(torch.softmax(self.net(x), dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 2))

    # ---------- persistence ----------
    def save(self, path):
        torch.save(
            {"state": self.net.state_dict(), "vocab": self.vocab, "max_len": self.max_len},
            path,
        )

    @classmethod
    def load(cls, path, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        det = cls(max_len=ckpt["max_len"], device=device)
        det.vocab = ckpt["vocab"]
        det.net = _Net(len(det.vocab) + 2)
        det.net.load_state_dict(ckpt["state"])
        det.net.to(det.device).eval()
        return det
