# Huawei Tech Arena 2026: activity detection from physiology

Predict `Rest` / `Activity` / `Stress` for every row of wristband data (BVP, EDA, HR, IBI, skin temperature).
Score = **80 × macro-F1** over clock-aligned 10-second windows (pooled over all test recordings) **+ up to 20 for speed**
(single CPU core). Agorize allows 3 successful submissions per day. Deadline: 1 December 2026.
Task description: https://docs.google.com/document/d/1i5zuvzHuiWaLst7EOqUP7m0WFmjIWhOos5pSyYq6his/edit?usp=sharing ·
[Public leaderboard](https://huawei.agorize.com/en/challenges/2026-finland-tech-arena/pages/leaderboard?lang=en) (team "Witty name (Placeholder)").

**Leaderboard: best 75 (`submissions/TULUI35.zip`, 20 Nov).** Target: 77, ideally 78.
Every upload and what it taught us: [submissions/README.md](submissions/README.md).

## Read in this order

| Document | What | Time |
|---|---|---|
| this README | where we are, how to work, the route to 77-78 | 10 min |
| [DATASET_AND_MODELS.md](DATASET_AND_MODELS.md) | the dataset, every model, glossary (for newcomers) | 20 min |
| [submissions/README.md](submissions/README.md) | all 12 uploads, scores, lessons | 5 min |
| [HUAWEI_EVALUATION.md](HUAWEI_EVALUATION.md) | how Huawei evaluates, from our uploads; review of V5 | 10 min |
| [NEXT_MODEL_TO_77.md](NEXT_MODEL_TO_77.md), [experiments.md](experiments.md) | detailed experiment notes (20 Nov) | reference |
| [PLAN.md](PLAN.md) | the 18 Nov plan (V4 era, partly outdated) | reference |

## Official rules we know (topic page + Huawei's forum answers, 20 Nov; verbatim in [HUAWEI_EVALUATION.md](HUAWEI_EVALUATION.md))

- Validation data: **entirely new people**, one whole recording of **15-35 minutes** per `run()` call, **not** following
  the training protocols ("random free-living data"). Tuning to the protocol is no longer possible.
- **No protocol information** (phase durations, structure, order, time since start): significant jury penalty.
- **Look-ahead** = time needed past a prediction point. **Recommended: at most 30 seconds. Whole-recording analysis =
  infinite look-ahead = "practically zero for feasibility"** at the final judging. The penalty grows with the look-ahead.
- Macro-F1 is **pooled** over all scored 10 s windows of the validation set (ties: first occurrence wins).
- Speed: score = (1 − MET / cap) × 20, MET (mean execution time) measured on the 15-35 min validation recordings.
  **The cap is still unclear** (topic page 0.3 s per [experiments.md](experiments.md), a later prompt said 1.0 s): asked on the forum.
- Finalists: top 10 by leaderboard **and** jury review together.

## What we learned (18-20 Nov)

1. **Our first local test was wrong.** It used the training recordings in their protocol order. That rewarded models that
   memorised the timetable: ranked by it, our uploads come out almost in **reverse** website order (−0.21).
2. **Timetable features fail** (V2: 82 locally → 61; V4: best locally → 71).
3. **A better local test** (`src/score.py`, `src/truth_eval.py`): the 6 holdout people's phases shuffled and mixed
   across sessions, random lengths. It ranks the uploads like the website (+0.77), but with ±4 points of error.
4. **It cannot judge models trained on shuffled copies.** V5.1 trained on copies spliced like the test: predicted ~85, got 73.
5. **Big models lose:** SHEH (3 ensembles) 54, the GRU network 69. One LightGBM with good features wins.
6. **More training people help:** TULUI's recipe on 29 → 35 people: 74 → 75.
7. **Speed alone did not show:** TULUI_fast (identical predictions, ~24 % faster on the laptop) stayed at 74.
8. **Only score a model trained without the 6 holdout people** (S16-S18, f11-f13). Otherwise the score is fake
   (V4 trained on everyone "scores" 91).
9. **Our labels are right:** alternative phase→class mappings fit the website worse.

## How to work (from `src/`, environment active)

```
.venv\Scripts\activate          (first time: python -m venv .venv, then
cd src                           pip install -r ..\requirements.txt -r ..\requirements-train.txt)
```

Current best recipe (TULUI35):

| Step | Command | Result |
|---|---|---|
| 1. Test copy | `python train_tului.py holdout` | `models/tului_re_holdout/`, trained without the holdout people (~5 min) |
| 2. Its score | `python score.py tului_re_holdout tului` | Quality (MIXED / shuffled / training order), speed, vs the original TULUI |
| 3. Real model | `python train_tului.py all` | `models/tului35/` + `submission.zip` in the project root |
| 4. Upload | upload `submission.zip`, copy it to `submissions/<name>.zip`, add a row to `submissions/README.md` | |

Other tools: `python truth_eval.py` (all uploads vs website, ~20 min), `python score.py NAME` (any zip or folder by name),
`python experiment.py ...` (feature experiments), Terminal → Run Task in VS Code. Rule: **one change per experiment,
one question per upload**, record every result.

## Route to 77-78

We need **+2 to +3 points** over 75. In order of certainty:

| # | Step | Expected | How we check it locally | Risk |
|---|---|---|---|---|
| 1 | **Speed pass on TULUI35:** fewer trees (1,650 now), drop features the model barely uses, faster rolling medians and spectra, compute only used features | If the cap is 0.3 s and Huawei is ~2× slower than the laptop: halving our ~45 ms ≈ **+2 to +3** | Predictions identical, or quality unchanged in person-held-out CV; side-by-side timing | Speed-only TULUI_fast stayed at 74; first read the runtime the leaderboard shows for TULUI35 |
| 2 | **Re-tune class bias + smoothing** with leave-one-person-out CV on all 35 people (Tului tuned them on 29) | +0.5 to +1 | Pooled out-of-fold macro-F1 | Low |
| 3 | **More training windows:** 5 s stride (2× overlap) in training, test unchanged | +0 to +1 | Person-held-out CV | Low |
| 4 | **One controlled quality probe:** TULUI recipe + shuffle augmentation, trained on 35 people (`candidates/TULUI_shuffle_aug.zip` is the 29-person version) | unknown: +3.6 on our shuffled test, but V5.1 warns this test flatters shuffle training | Only the website can tell | Medium |
| 5 | Average 3-5 seeds of the final model | +0 to +0.5 | CV | Low, costs speed |

Before every upload: zero fallbacks on real recordings (`package_submission.py` tests), files at the zip root, and a
gain on held-out people that holds for a second seed.

## Jury readiness: the 30-second look-ahead rule

**Our 75 model (TULUI35) uses the whole recording**, which Huawei calls infinite look-ahead ("practically zero for
feasibility"). It is fine for the leaderboard, but a finalist needs a version that looks at most 30 s ahead. What has to change in
`src/tului_model.py` / the feature code:

| Uses the future | Feature(s) | Past-only (≤ 30 s) replacement |
|---|---|---|
| whole recording | `rec_*` quantiles, `*_rel`, `*_z`, TULUI's "floor" features (`hr_above_floor`, ... use recording quantiles) | expanding or trailing (e.g. last 5-10 min) quantiles/median up to the current window |
| up to 2.5 min | `*_local` (centred 5-min median), `*_2m`, `*_5m` (centred means) | trailing windows |
| 30 s - 3 min | lag features `*_lag3` ... `*_lag18` | keep only past lags and `lag3` (= 30 s) |
| long (filter) | EDA `sosfiltfilt` (zero-phase, runs backwards over the whole segment) | causal `sosfilt` |
| ≤ 30 s (allowed) | centred 60 s context, smoothing k = 7 (±30 s) | keep, or make trailing |

Expected cost: a teammate's test with only window + past-trend features lost about 0.10 macro-F1 (0.748 → 0.648 in
person-held-out CV, [NEXT_MODEL_TO_77.md](NEXT_MODEL_TO_77.md)). Plan: keep TULUI35 as the leaderboard model, build a
≤ 30 s version in parallel, measure its cost with `score.py`, and ask which submission the jury reviews.

## Documentation still to do

- [ ] After every upload: zip into `submissions/`, row in `submissions/README.md` (score, runtime shown, what changed).
- [ ] Ask Agorize: the speed cap (0.3 s or 1.0 s) and the reference CPU; and which submission the jury reviews.
- [ ] Jury version (≤ 30 s look-ahead, table above): build, measure the cost, document it.
- [ ] Final write-up for the jury: architecture (one LightGBM on physiological features, ONNX), validation method, look-ahead audit, speed.
- [ ] Tidy: mark PLAN.md as outdated, keep one experiment log (experiments.md + experiments/log.csv).

## Where things are

| Path | What |
|---|---|
| `submissions/` | All 12 uploads + `candidates/` (not uploaded). [README](submissions/README.md) = the history. |
| `src/train_tului.py` | **Current best recipe** (TULUI35, 75): trains with `src/tului_model.py`. |
| `src/score.py`, `src/truth_eval.py`, `truth_eval.py` | Local scoring of any model; calibration of all uploads against the website (two versions: `src/` = holdout quality + speed, root = speed audit incl. V27/TULUI35). |
| `src/package_submission.py` | Safety tests (also: no silent "Rest" fallback). |
| `src/tului_model.py` | The TULUI35 inference code (shipped as `your_model.py` in the zip). |
| `src/your_model.py` + `.onnx` | The older V4/V5 code line (V5.1 model). |
| `src/train_v5.py`, `src/train_v4.py`, `src/old/` | Earlier recipes (V5/V5.1, V4, V1-V3). |
| `src/experiment.py`, `experiments/` | Feature experiments, logs, other team members' experiment scripts. |
| `models/` | Trained test copies (not in git). |
| `data/` | The 85 training recordings. |
| `archive/` | Not used by the pipeline (WESAD notebook, unusable `ml_ready` dataset). |

`requirements.txt` = the official evaluation environment; `requirements-train.txt` adds what training needs.
