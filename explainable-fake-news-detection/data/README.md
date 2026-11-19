# Datasets

`data/sample/sample_news.csv` is a tiny 60-headline file bundled only so you can check that the
code runs. **Do not report its scores** - download the real datasets below for your results.

## 1. Fake and Real News Dataset (Kaggle) - ~44,898 articles
1. Search Kaggle for **"Fake and Real News Dataset"** (Clément Bisaillon) and download it.
2. Put `Fake.csv` and `True.csv` in `data/raw/`.
3. Train with `python train.py --dataset kaggle --models nb lr`.

Note: the loader removes the `(Reuters)` dateline from real articles. Without that, models just
learn the word "Reuters" and score ~99% for the wrong reason.

## 2. LIAR dataset - 12,836 short statements
1. Download LIAR (Wang, 2017) - it is available on Kaggle and from the UCSB NLP group.
2. Put `train.tsv`, `valid.tsv`, `test.tsv` in `data/raw/liar/`.
3. Train with `python train.py --dataset liar --models nb lr`.

The six LIAR labels are merged into two classes:
FAKE = pants-fire, false, barely-true | REAL = half-true, mostly-true, true.

## 3. Both combined
`python train.py --dataset both --models nb lr`
