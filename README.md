# Huawei Tech Arena 2026: activity detection from physiology

Task description: https://docs.google.com/document/d/1i5zuvzHuiWaLst7EOqUP7m0WFmjIWhOos5pSyYq6his/edit?usp=sharing

Predict `Rest` / `Activity` / `Stress` for every row of wristband data (BVP, EDA, HR, IBI, skin
temperature). Score = 80 x macro-F1 over 10-second windows + 20 for speed (single CPU core).

**Current best: `submissions/submissionV3.zip`, leaderboard 72.** History and lessons: [submissions/README.md](submissions/README.md).

## Layout

| Path | What |
|---|---|
| `src/your_model.py` | The submitted model (self-contained: cleaning, features, ONNX inference, post-processing). |
| `src/your_model.onnx` | Trained gradient-boosting model; its metadata holds the feature list, smoothing width and class bias. |
| `src/train_robust.py` | Trains the V3 model (recordings cut into 1-30 minute pieces + whole). |
| `src/train_v2.py` | Feature dataset, person-grouped cross-validation, tuning and ONNX export (used by `train_robust.py`). |
| `src/train_baseline.py` | The original V1 Random Forest baseline. |
| `src/eval_harness.py` | Local scoring on the 6 holdout people (S16-S18, f11-f13). Passes whole recordings only. |
| `src/package_submission.py` | Edge-case tests + builds `submission.zip` in the project root. |
| `signal_processing/` | Exploration: cleaning and feature research, figures, feature evaluation. |
| `submissions/` | Every uploaded zip with its leaderboard score. |
| `data/` | Training recordings (`data/_cache/` holds generated feature caches and is not in git). |

## Setup

```
pip install -r requirements.txt -r requirements-train.txt
```

`requirements.txt` is the official evaluation environment; `requirements-train.txt` adds what training needs.

## Train, evaluate, package (from `src/`)

```
python train_robust.py rebuild         # build the chunked feature cache (data/_cache/robust_features.parquet)
python train_robust.py export holdout  # train without the 6 holdout people -> honest local score
python eval_harness.py                 # local score
python train_robust.py export all      # final model trained on everyone
python package_submission.py           # tests + ../submission.zip
```

Notes:
- After `export all`, `eval_harness.py` scores people the model was trained on, so its quality number is inflated.
- `train_baseline.py` overwrites `src/your_model.onnx` with the V1 model, which the current `your_model.py` cannot read.
- Agorize allows 3 successful submissions per day.
