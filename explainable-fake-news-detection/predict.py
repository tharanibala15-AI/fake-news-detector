"""Classify a news text and explain the decision.

Examples
--------
python predict.py --text "Shocking! Miracle pill cures everything overnight"
python predict.py --model lr --explainer lime --text "The government has announced a new tax cut for all citizens."
"""
import argparse

from fakenews.config import CLASS_NAMES, RESULTS_DIR
from fakenews.explain import METHODS, explain, plot_contributions
from fakenews.registry import PREFERENCE, available_models, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="news headline / article text")
    ap.add_argument("--model", choices=PREFERENCE, default=None, help="default: best trained model found")
    ap.add_argument("--explainer", choices=METHODS, default="occlusion",
                    help="occlusion needs no extra libs; lime/shap need requirements installed")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--save-plot", default=str(RESULTS_DIR / "explanation.png"))
    args = ap.parse_args()

    name = args.model or next(iter(available_models()), None)
    if name is None:
        raise SystemExit("No trained model found. Run first:  python train.py --dataset sample --models nb lr")
    model = load_model(name)

    proba = model.predict_proba([args.text])[0]
    label = CLASS_NAMES[int(proba.argmax())]
    print(f"\nModel      : {name}")
    print(f"Prediction : {label}   (REAL {proba[0]:.1%} | FAKE {proba[1]:.1%})")

    pairs = explain(args.explainer, model.predict_proba, args.text, top_k=args.top_k)
    print(f"\nImportant words ({args.explainer}; + = toward FAKE, - = toward REAL):")
    for word, weight in pairs:
        print(f"  {word:<18}{weight:+.3f}")

    if args.save_plot:
        RESULTS_DIR.mkdir(exist_ok=True)
        plot_contributions(pairs, save_path=args.save_plot)
        print(f"\nChart saved to {args.save_plot}")


if __name__ == "__main__":
    main()
