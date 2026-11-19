# Huawei Tech Arena 2026: activity detection from physiology

Predict `Rest` / `Activity` / `Stress` for every row of wristband data (BVP, EDA, HR, IBI, skin
temperature). Score = 80 × macro-F1 over 10-second windows + 20 for speed (single CPU core).
Task description: https://docs.google.com/document/d/1i5zuvzHuiWaLst7EOqUP7m0WFmjIWhOos5pSyYq6his/edit?usp=sharing

**Leaderboard: best 74 (TULUI).** V3 72, V4 71. **Next candidate: `submissions/candidates/V5.zip`.** Every upload and its score: [submissions/README.md](submissions/README.md).
**New here? Read [DATASET_AND_MODELS.md](DATASET_AND_MODELS.md) first** (dataset, models, how to gain 1 %, ~20 min).
Strategy, findings and team split: [PLAN.md](PLAN.md).

## How to: train, score, submit

All commands run from `src/` with the project's environment active:

```
.venv\Scripts\activate          (first time: python -m venv .venv, then
cd src                           pip install -r ..\requirements.txt -r ..\requirements-train.txt)
```

| Step | Command | Result |
|---|---|---|
| 1. Train a test copy | `python train_v5.py holdout` | `models/v5_holdout/`: model trained **without** the 6 holdout people (~2 min) |
| 2. Get its score | `python score.py v5_holdout` | Quality on shuffled recordings (like the official test), speed, total |
| 3. Train the real one | `python train_v5.py all` | `src/your_model.onnx`: trained on everyone (~2 min) |
| 4. Package | `python package_submission.py` | Runs the safety tests, writes `submission.zip` in the project root |
| 5. Upload and record | copy `submission.zip` to `submissions/submissionVn.zip`, upload, add a row to `submissions/README.md` | |

Scoring other models:

```
python score.py ../submissions/candidates/TULUI.zip       # any zip or unzipped folder
python score.py A.zip B.zip ../models/holdout              # several side by side
python score.py ../models/holdout --quick                  # whole recordings only (faster)
```

Only score models trained with `holdout`. A model trained on everyone has already seen the holdout
people, so its local score is inflated (`score.py` warns; V4 as uploaded "scores" 91 that way).

## Where things are

| Path | What |
|---|---|
| `src/your_model.py` + `src/your_model.onnx` | **The submission** (what gets zipped). Features, both models, post-processing. |
| `src/adm_base_class.py` | Huawei's base class (official). |
| `src/train_v5.py` | Trains the current model (V5: one model, no position features, trained with shuffled copies). |
| `src/train_v4.py` | V4 (two stages, position features). Scored 71: do not use; the official test punishes timetable features. |
| `src/score.py` | Local score of any model. Main number: holdout recordings with phases shuffled, like the official test. |
| `src/package_submission.py` | Safety tests + `submission.zip`. |
| `src/experiment.py` | For trying ideas: feature cache, cross-validation, ablation; results in `experiments/log.csv`. See [PLAN.md](PLAN.md) §7. |
| `src/eval_harness.py` | The team's original local bench (whole recordings only); `score.py` uses its scoring rules. |
| `src/old/` | V1-V3 trainers, kept for reference. They overwrite `src/your_model.onnx`: restore with `git checkout src/your_model.onnx`. |
| `models/holdout/` | Test copy from step 1 (not in git). |
| `submissions/` | Every uploaded zip + leaderboard scores. `candidates/` = zips not uploaded yet (e.g. TULUI). |
| `experiments/` | Experiment log, feature rankings, the V3 feature list. |
| `signal_processing/` | Early exploration: cleaning, figures, feature evaluation. |
| `data/` | Training recordings (`data/_cache/` = generated feature caches, safe to delete). |
| `archive/` | Not used by the pipeline: the WESAD soft-label notebook, `ml_ready_10s_dataset_v1.parquet` (unusable for training, see PLAN.md §3). |

`requirements.txt` = the official evaluation environment; `requirements-train.txt` adds what training
needs (LightGBM, onnxmltools, ...). Agorize allows 3 successful submissions per day.
