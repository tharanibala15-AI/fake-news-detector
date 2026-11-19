"""Train and evaluate models.

Examples
--------
python train.py --dataset sample --models nb lr                  # 5-second smoke test
python train.py --dataset kaggle --models nb lr                  # real data, classical baselines
python train.py --dataset kaggle --models lstm --epochs 4        # needs requirements-deep.txt
python train.py --dataset kaggle --models bert --max-len 128 --max-train-samples 5000 --epochs 1
"""
from __future__ import annotations

import argparse
import time

import joblib

from fakenews.classic import build_pipeline
from fakenews.config import MODEL_LABELS, MODELS_DIR, SEED
from fakenews.data import DATASETS, load_dataset, split_dataset
from fakenews.evaluate import compute_metrics, save_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, default="sample")
    p.add_argument("--models", nargs="+", choices=list(MODEL_LABELS), default=["nb", "lr"])
    p.add_argument("--epochs", type=int, default=None, help="default: 4 for LSTM, 2 for BERT")
    p.add_argument("--batch-size", type=int, default=None, help="default: 64 for LSTM, 16 for BERT")
    p.add_argument("--max-len", type=int, default=256, help="max tokens per article (LSTM/BERT)")
    p.add_argument("--bert-model", default="bert-base-uncased", help="e.g. distilbert-base-uncased for CPU")
    p.add_argument("--max-train-samples", type=int, default=None, help="subsample training set (speed)")
    return p.parse_args()


def train_one(name, train_df, val_df, args):
    texts, labels = train_df["text"].tolist(), train_df["label"].to_numpy()
    val = (val_df["text"].tolist(), val_df["label"].to_numpy())
    MODELS_DIR.mkdir(exist_ok=True)

    if name in ("nb", "lr"):
        model = build_pipeline(name).fit(texts, labels)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    elif name == "lstm":
        from fakenews.lstm import LSTMDetector

        model = LSTMDetector(max_len=args.max_len).fit(
            texts, labels, epochs=args.epochs or 4, batch_size=args.batch_size or 64, val=val
        )
        model.save(MODELS_DIR / "lstm.pt")
    else:
        from fakenews.bert import BertDetector

        model = BertDetector(args.bert_model, args.max_len).fit(
            texts, labels, epochs=args.epochs or 2, batch_size=args.batch_size or 16, val=val
        )
        model.save(MODELS_DIR / "bert")
    return model


def main():
    args = parse_args()
    df = load_dataset(args.dataset)
    train_df, val_df, test_df = split_dataset(df)
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = train_df.sample(args.max_train_samples, random_state=SEED).reset_index(drop=True)
    print(f"Dataset '{args.dataset}': {len(df)} articles  ->  train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
    print(f"Fake share in train: {train_df['label'].mean():.1%}\n")

    rows = []
    for name in args.models:
        start = time.time()
        print(f"=== Training {MODEL_LABELS[name]} ===")
        model = train_one(name, train_df, val_df, args)
        preds = model.predict_proba(test_df["text"].tolist()).argmax(axis=1)
        m = compute_metrics(test_df["label"].to_numpy(), preds)
        save_metrics(name, m, dataset=args.dataset, train_size=len(train_df), test_size=len(test_df))
        rows.append((MODEL_LABELS[name], m, time.time() - start))
        print(f"Saved model to models/ ({time.time() - start:.1f}s)\n")

    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Time(s)':>9}")
    for label, m, secs in rows:
        print(f"{label:<22}{m['accuracy']:>10.2%}{m['precision']:>11.2%}{m['recall']:>9.2%}{m['f1']:>9.2%}{secs:>9.1f}")
    print("\nMetrics saved to results/metrics.json  (run: python plot_results.py)")


if __name__ == "__main__":
    main()
