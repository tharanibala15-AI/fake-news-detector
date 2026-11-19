"""Draw the grouped bar chart (Accuracy / Precision / Recall / F1 per model) from results/metrics.json."""
from matplotlib.figure import Figure

from fakenews.config import MODEL_LABELS, RESULTS_DIR
from fakenews.evaluate import load_metrics

COLORS = {"accuracy": "#1f77b4", "precision": "#ff7f0e", "recall": "#2ca02c", "f1": "#9467bd"}


def main():
    metrics = load_metrics()
    names = [m for m in MODEL_LABELS if m in metrics]
    if not names:
        raise SystemExit("No results yet. Run:  python train.py --dataset sample --models nb lr")

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    width = 0.2
    for j, key in enumerate(COLORS):
        xs = [i + (j - 1.5) * width for i in range(len(names))]
        ax.bar(xs, [metrics[n][key] for n in names], width, label=key.capitalize() if key != "f1" else "F1-Score", color=COLORS[key])
    ax.set_xticks(range(len(names)), [MODEL_LABELS[n] for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance (test set)")
    ax.legend(ncol=4, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    out = RESULTS_DIR / "model_comparison.png"
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    for n in names:
        m = metrics[n]
        print(f"{MODEL_LABELS[n]:<22} acc={m['accuracy']:.2%} prec={m['precision']:.2%} rec={m['recall']:.2%} f1={m['f1']:.2%}  [{m.get('dataset','?')}]")


if __name__ == "__main__":
    main()
