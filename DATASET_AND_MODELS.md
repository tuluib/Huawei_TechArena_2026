# Dataset and models: a guide for the team

**Reading time:** about 20 minutes if you are new to machine learning (sections 1-2 ≈ 10 min, section 3 ≈ 5 min, sections 4-5 ≈ 5 min).
Words in *italics* are explained in the glossary at the end.
All numbers were measured on our data on 18 November 2026.

**Current submission status (20 Nov):** the best public score is 74 (`TULUI_fast.zip`). The next quality candidate is `submissions/candidates/TULUI_shuffle_aug.zip`; its holdout results are in [NEXT_MODEL_TO_77.md](NEXT_MODEL_TO_77.md). The historical instructions below that say to upload V5 are no longer current.

---

## 1. The task in one paragraph

A wristband (an Empatica E4-style sensor) records a person's pulse, sweat, heart rate and skin temperature.
For every moment, our model must say whether the person is at **Rest**, doing **Activity** (exercise) or under **Stress**.
Huawei scores us on two things:

- **Quality (80 points)** = 80 × *macro-F1* over 10-second windows, pooled over all their test recordings.
- **Speed (20 points)** = how fast `run()` processes one recording on a single CPU core (preprocessing and output included).

The test uses **new people** we never see. Each test recording is **one whole recording of 15-35 minutes**, and its
phases are **not in the training order or durations** (Huawei: it "mimics free-living data").

---

## 2. The dataset

### What we have

| | |
|---|---|
| People | **35**: 18 named `S01`-`S18` and 17 named `f01`-`f18` (no `f14`). S and f did slightly different versions of the protocols. |
| Recordings | **85** files, one per person per protocol (not everyone did all three) |
| Total | **44 hours**, 10.1 million rows |
| Where | `data/AEROBIC/`, `data/ANAEROBIC/`, `data/STRESS/`, one `.parquet` file per person (e.g. `data/STRESS/S05.parquet`) |

### The three protocols (why the data exists)

Each recording is one lab session that follows a fixed script, so that every phase is labelled:

| Protocol | Recordings | Length (min / avg / max) | What the person does |
|---|---|---|---|
| **STRESS** | 31 | 22 / 34 / 52 min | Sits. Baseline rest, then mental stress tasks (*Stroop* colour test, *TMCT* mental arithmetic, giving a real and an opposite opinion, subtraction) with long rests in between. |
| **AEROBIC** | 27 | 24 / 34 / 41 min | Cycles: warm-up, then steady pedalling at 60 → 110 rpm in steps, cool-down, rest. |
| **ANAEROBIC** | 27 | 14 / 25 / 37 min | Cycles: warm-up, then 3-4 short all-out sprints (~30 s) each followed by easy cool-down pedalling, then rest. |

All timestamps are from 2013. Recordings follow the script in order, e.g. STRESS for S05:
`Baseline 3 min → Stroop 2 min → Rest 6 min → TMCT 3 min → Rest 4 min → opinions → Subtraction → Rest`.

### What is in each file (columns)

Each row is one moment, every 15.6 ms (64 rows per second). The sensors tick at different speeds, so most cells are empty (NaN) **on purpose**:

| Column | What it is | Rows with a value | Sensor rate |
|---|---|---|---|
| `BVP` | Blood volume pulse: the raw light signal of each heartbeat | 100 % | 64 per second |
| `EDA` | Electrodermal activity: skin sweat, in microsiemens (rises with stress) | 6.25 % | 4 per second |
| `Temperature` | Skin temperature, °C | 6.25 % | 4 per second |
| `HR` | Heart rate, beats per minute (computed by the device) | 1.6 % | 1 per second |
| `IBI_duration`, `IBI_dt` | Time between two heartbeats, only when the device was sure of a beat | 0.8 % | per beat |
| `Phase` | **The label** (training only): Baseline, Stroop, 70 rpm, Rest, ... | 100 % | |
| `user_id` | Person (training only) | 100 % | |

The test data has only the six sensor columns. Our model must return one label per row.

### From phases to the three classes

| Class | Phases | Share of rows |
|---|---|---|
| **Activity** | Warm-up, every "rpm" step, Sprint, Cooldown | 48 % |
| **Rest** | Baseline, Rest | 38 % |
| **Stress** | Stroop, TMCT, Real opinion, Opposite opinion, Subtraction | **7 %** |
| not scored | Undefined (instructions, transitions) | 7 % |

This mapping is in `src/eval_harness.py`. It was written by us, not by Huawei.

In 10-second windows of whole recordings this gives 6,035 Rest, 7,658 Activity and only **1,197 Stress** windows.
*Macro-F1* gives each class the same weight, so the rare Stress class counts as much as the other two together. **Stress is where most points are lost.**

### How we split it

- **Training people:** 29.
- **Holdout people:** `S16, S17, S18, f11, f12, f13` (17 recordings). A *test copy* of a model is trained without them, and then scored on them.
- **Shuffled test** (`src/score.py`): the holdout recordings with their phases cut to random lengths and joined in random order, inside each recording. This copies what Huawei says about the real test.

> **Common mistake:** jumbling the training people's data and testing the model on it. The model has already seen
> those people, so the score is fake (V4 "scored" 91 that way). Always jumble **holdout** people, never
> training people. `score.py` does this, and warns when a model was trained on everyone.

### Known problems in the data

- Stress is rare (7 %), short (tasks last 0.5-4 min) and weak: heart rate rises only 2-4 bpm during a task.
- The body stays "stressed" for 1-3 minutes after a task, but those minutes are labelled Rest.
- There is no motion sensor. Movement shows up only as noise in `BVP`.
- 16 recordings have short gaps (up to 270 s) where the device lost contact.

---

## 3. The models

All our models share the same five steps. Only steps 3-5 really differ:

```
raw rows → 1. clean each sensor → 2. cut into 10-second windows (on the clock: 12:00:00, 12:00:10 ...)
        → 3. compute features per window → 4. tree model gives Rest/Activity/Stress probabilities
        → 5. smooth over neighbouring windows → one label per window, copied to every row
```

The model itself is a set of *decision trees* (*gradient boosting* or a *random forest*), saved as an `.onnx` file.
The submission is always `your_model.py` (the code for all five steps) + `your_model.onnx` (the trained trees).

| Model | Leaderboard | Features per window | Tree model | Trained on | Key idea / problem |
|---|---|---|---|---|---|
| **V1** | 68 | 12: mean and spread of each raw column over 640 rows | Random forest, 60 trees | 29 people | No cleaning, never predicts Stress. Simple and robust. |
| **V2** | 61 | 138, incl. **time since start** | Gradient boosting, 1,200 trees | everyone | Cleaned signals and context. Learned the timetable, so it failed on the real test. |
| **V3** | 72 | 134 (V2 without the time features) | Gradient boosting, 1,200 trees | everyone, whole + cut pieces | Robust to how `run()` is called. |
| **TULUI** | **74** | 145 (V3 + 11 "relative to the person's resting level" features) | Gradient boosting, 1,650 trees (exported with OnnxMLTools) | 29 people | Best upload so far. |
| **TULUI + shuffle augmentation** | not uploaded | Same 145 TULUI features | LightGBM, 400 trees | all people + reordered, full-length reordered, and same-person session-mixed copies | Holdout comparison: +3.62 T3 quality points averaged over five seeds, plus faster inference; candidate `submissions/candidates/TULUI_shuffle_aug.zip`. |
| **V4** | 71 | 190, incl. time features | **Two** LightGBM models (2,100 trees): the 2nd reads the 1st's answers over ±3 min | everyone | Learned the protocol's order, so it failed on the real test. |
| **V5** | not uploaded | 186, **no time features** | LightGBM, 1,200 trees | everyone + **3 shuffled copies** of every recording | Cannot learn the order. Shuffled test: 60.7 quality, against TULUI 55.6 and V3 55.9. |
| **V5.1** | **73** | 186, no position features | LightGBM, 400 trees | all people + reordered, full-length reordered, and same-person session-mixed copies | Better generated T4 robustness than V5; its local T4 fit predicted ~85 but the leaderboard result was 73. |

### What the features measure (V3 to V5)

- **Per window:** heart rate (mean, spread, slope), pulse rate and pulse strength from `BVP`, how noisy the pulse is (movement), sweat level and sweat bursts (*SCR*), skin temperature, heart-rate variability (*HRV*).
- **Context:** the same values over the last and next 1, 2 and 5 minutes, and the values 30 s to 3 min before and after.
- **Relative to the same person:** each value minus that recording's median (people differ a lot: resting heart rate ranges from 60 to 100).
- **Recording summaries:** e.g. the highest heart rate in the recording. These tell the model what kind of session it is looking at.

### Rules the models must respect (from Huawei)

1. No features that use the protocol's structure: time since start, phase order, phase durations. They fail on the real test, and the **jury penalises them**.
2. *Look-ahead* is penalised at the final judging. All models so far use up to 3 minutes of future data. This must be reduced before the finals.
3. `run()` must not keep anything between recordings.

---

## 4. How to run it

From the `src` folder in the VS Code terminal (or *Terminal → Run Task…*):

```
python train_v5.py holdout              # test copy, without the holdout people (~2 min)
python score.py v5_holdout TULUI        # compare models by name; main number = shuffled test
python train_v5.py all                  # the real model, trained on everyone
python package_submission.py            # safety tests + submission.zip
```

A teammate's model: put its zip (trained **without** the 6 holdout people) in `submissions/candidates/`, then run `python score.py NAME`.

---

## 5. How to gain about 1 % (≈ 1 point) next

In order of certainty. Each one is checked with `train_v5.py holdout` + `score.py`, and kept only if it gains **≥ 1 point on the shuffled test**. Smaller changes are noise with 6 people.

| Idea | Expected | Why |
|---|---|---|
| **Speed:** compute only the features the model uses; faster rolling statistics | +1 point per 15 ms saved (now ~43 ms) | Certain: the speed formula is fixed. Accuracy does not change. |
| **More shuffled copies** in training (3 → 6), or shuffle with different block lengths | unknown, cheap to test | Shuffled training gave the biggest gain so far (+10 quality on the shuffled test). |
| **TULUI's 11 features inside V5** | unknown | They did not help V4, but V4 was judged with the old test. |
| **Average 3-5 models** trained with different random seeds | +0.3-0.5 typical | Smooths out chance effects of one training run. |
| **Less look-ahead** (use past data only) | probably −1 to −2 on the leaderboard | Not for points: it is for the jury. Measure the cost before deciding. |

First, upload **V5** (`submissions/candidates/V5.zip`). If it beats 74, the shuffled test is trustworthy and we keep using it. If not, we tell the team before building on it.

---

## Glossary

- **Macro-F1:** the F1 score (balance of "found most of it" and "rarely wrong") computed for each class, then averaged. A rare class counts as much as a common one.
- **Feature:** a number computed from the raw signal, e.g. "mean heart rate in this 10 s window".
- **Decision tree / gradient boosting / random forest:** models made of many yes/no questions on features ("heart rate above 120?"). Gradient boosting (LightGBM, sklearn's HistGradientBoosting) adds trees one by one to fix earlier mistakes; a random forest averages independent trees.
- **ONNX:** a file format for trained models that Huawei's machine can run without our training tools.
- **Holdout / test copy:** people left out of training so we can test on them honestly.
- **Look-ahead:** using data from after the moment being predicted.
- **SCR:** skin conductance response, a short burst of sweat that often comes with stress.
- **HRV:** heart-rate variability, how much the time between beats changes. It usually drops under stress.
- **Stroop:** naming the ink colour of a written colour word ("RED" printed in blue). **TMCT:** a timed mental arithmetic test.
