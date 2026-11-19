"""BERT fine-tuning (Hugging Face Transformers + PyTorch). Requires: pip install -r requirements-deep.txt

Tip: BERT is slow on CPU. Use a GPU (e.g. free Google Colab) or run with
     --bert-model distilbert-base-uncased --max-len 128 --max-train-samples 5000
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import SEED


class BertDetector:
    def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 256, device: str | None = None):
        self.model_name = model_name
        self.max_len = max_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = None
        self.model = None

    def _tok(self, texts):
        return self.tokenizer(
            list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt"
        ).to(self.device)

    # ---------- training ----------
    def fit(self, texts, labels, epochs: int = 2, batch_size: int = 16, lr: float = 2e-5, val=None):
        torch.manual_seed(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model.to(self.device)

        texts, labels = list(texts), np.asarray(labels)
        steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        rng = np.random.default_rng(SEED)

        for epoch in range(1, epochs + 1):
            self.model.train()
            order = rng.permutation(len(texts))
            total = 0.0
            for n, start in enumerate(range(0, len(order), batch_size), 1):
                idx = order[start : start + batch_size]
                batch = self._tok([texts[i] for i in idx])
                y = torch.tensor(labels[idx], dtype=torch.long, device=self.device)
                opt.zero_grad()
                loss = self.model(**batch, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(idx)
                if n % 50 == 0:
                    print(f"[BERT] epoch {epoch} step {n}  loss={loss.item():.4f}", flush=True)
            msg = f"[BERT] epoch {epoch}/{epochs}  train_loss={total / len(texts):.4f}"
            if val is not None:
                acc = (self.predict_proba(val[0]).argmax(1) == np.asarray(val[1])).mean()
                msg += f"  val_acc={acc:.4f}"
            print(msg)
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        texts = list(texts)
        out = []
        for start in range(0, len(texts), batch_size):
            logits = self.model(**self._tok(texts[start : start + batch_size])).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 2))

    # ---------- persistence ----------
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "detector.json").write_text(
            json.dumps({"model_name": self.model_name, "max_len": self.max_len})
        )

    @classmethod
    def load(cls, directory, device: str | None = None):
        directory = Path(directory)
        meta = json.loads((directory / "detector.json").read_text())
        det = cls(meta["model_name"], meta["max_len"], device)
        det.tokenizer = AutoTokenizer.from_pretrained(directory)
        det.model = AutoModelForSequenceClassification.from_pretrained(directory).to(det.device).eval()
        return det
