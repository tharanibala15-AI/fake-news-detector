# Explainable Fake-News Detection System

> **Truth with Reasons** - detects fake news *and* explains why, using word-level explanations (LIME / SHAP).

**Team**
| # | Name | Register No. |
|---|------|--------------|
| 1 | Nithika KP | RA2411047050017 |
| 2 | Rupavathi M | RA2411047050018 |
| 3 | Tharanibala B | RA2411047050043 |

## What it does
```
Input news text -> Preprocessing -> Feature extraction -> Classifier -> Explainability module
                   (tokenize,       (TF-IDF /             (NB, LR,      (LIME / SHAP / occlusion)
                    stop-words,      embeddings)           LSTM, BERT)
                    stemming)
```
* Four models for comparison: **Naive Bayes**, **Logistic Regression** (TF-IDF), **BiLSTM**, **BERT**
* Metrics: Accuracy, Precision, Recall, F1 (FAKE = positive class)
* Explanations: LIME, SHAP, plus a built-in dependency-free *occlusion* explainer
* Command-line tools, a Streamlit web app and a Jupyter notebook

## Project structure
```
.
├── fakenews/            # the library
│   ├── data.py          # loaders: sample, Kaggle Fake&Real, LIAR
│   ├── preprocess.py    # cleaning, stop-words, stemming
│   ├── classic.py       # TF-IDF + Naive Bayes / Logistic Regression
│   ├── lstm.py          # BiLSTM (PyTorch)
│   ├── bert.py          # BERT fine-tuning (Transformers)
│   ├── explain.py       # LIME / SHAP / occlusion + plotting
│   ├── evaluate.py      # metrics
│   └── registry.py      # load trained models by name
├── train.py             # train + evaluate
├── predict.py           # classify + explain one text
├── app.py               # Streamlit web app
├── plot_results.py      # model-comparison bar chart
├── notebooks/demo.ipynb
├── tests/test_smoke.py
├── data/                # see data/README.md
├── models/              # trained models are saved here (git-ignored)
└── results/             # metrics.json and charts
```

## 1. Setup in VS Code
Requires **Python 3.10+**. Open the unzipped folder in VS Code (`File > Open Folder`), open the terminal (`Ctrl+``) and run:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then press `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose the `.venv` one.

> PowerShell blocks the activate script? Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 2. Quick check (5 seconds, no downloads)
```bash
python train.py --dataset sample --models nb lr
python predict.py --text "Scientists discover cure for cancer in just 5 days!"
streamlit run app.py
python -m pytest -q
```
The bundled sample has only 60 headlines - it just proves the pipeline works. **Its scores are meaningless.**

## 3. Train on the real datasets
Download the data as described in [`data/README.md`](data/README.md), then:
```bash
python train.py --dataset kaggle --models nb lr      # minutes on a laptop
python plot_results.py                               # -> results/model_comparison.png
```

### Deep models (LSTM / BERT)
```bash
pip install -r requirements-deep.txt
python train.py --dataset kaggle --models lstm
python train.py --dataset kaggle --models bert --epochs 2          # GPU recommended
```
BERT on a CPU is very slow. Use a GPU (free Google Colab works) or a lighter setup:
```bash
python train.py --dataset kaggle --models bert --bert-model distilbert-base-uncased --max-len 128 --max-train-samples 5000 --epochs 1
```
Train everything in one go: `--models nb lr lstm bert`.

## 4. Explanations
```bash
python predict.py --text "The government has announced a new tax cut for all citizens." --explainer lime
python predict.py --text "..." --explainer shap          # needs requirements-deep.txt
python predict.py --text "..." --explainer occlusion     # default, no extra library
```
Output: predicted label, confidence, the most influential words (`+` pushes toward FAKE, `-` toward REAL)
and a chart saved to `results/explanation.png`.

## 5. Your results
All numbers come from **your own** training runs (`results/metrics.json`). The bar chart on the project
poster is labelled *Example*; only quote accuracy figures in this README or your report if your runs
actually produce them. Add your numbers here once you have them:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Naive Bayes | | | | |
| Logistic Regression | | | | |
| LSTM | | | | |
| BERT | | | | |

## 6. Push to GitHub
1. Create an **empty** repository on github.com (no README/.gitignore/license - they already exist here).
2. In the VS Code terminal, inside the project folder:
```bash
git init
git add .
git commit -m "Initial commit: Explainable Fake-News Detection System"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
Large files (`data/raw/*`, `models/*`) are git-ignored on purpose. If asked to log in, use a
[personal access token](https://github.com/settings/tokens) as the password, or sign in through VS Code's GitHub prompt.

## Future scope
Multilingual detection - real-time social-media streams - fact-checking API integration -
stronger explainability - mobile/web deployment.

## References
1. Shu et al. (2017). *Fake News Detection on Social Media* (Kaggle dataset).
2. Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.*
3. Ribeiro et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier* (LIME).
4. Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP).
5. Wang (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection* (LIAR).

License: MIT
