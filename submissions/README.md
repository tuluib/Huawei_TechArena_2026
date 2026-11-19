# Submissions

Each zip is exactly what was uploaded to Agorize (`your_model.py` + `your_model.onnx` at the root).

| File | Leaderboard | Local harness (holdout people) | What it is |
|---|---|---|---|
| `submissionV1.zip` | **68** | 53.9 (quality 38.8, speed 15-17) | Baseline: Random Forest on mean/std of the raw columns per 640 rows. |
| `submissionV2.zip` | **61** | 82.3 (quality 65.1, speed 17.2) | Cleaned-signal features + gradient boosting, with time-position features. Broke on the hidden test (see below). |
| `submissionV3.zip` | **72** | 57.0 quality on whole recordings, 45-57 on any input format | V2 made robust to how `run()` is called. Current best. |
| `V1.2(2nd trial65score).zip` | **65** | T4 36.3 / T3 37.2 (holdout people unknown) | V1-style random forest, 36 features; assumes the first 60 s is rest (fails on shuffled recordings). |
| `SHEH.zip` | **54** | T4 28.1 / training order 67.4 (very likely saw the holdout people) | 3 tree models (2,710 trees), 163 features; files inside a folder in the zip. Memorises sessions. |
| `submissionV4.zip` | **71** | 62.3 quality on whole recordings, 47.5-62.3 on any input format; ~1.2× V3's run time | Two-stage LightGBM (stage 2 reads the stage-1 probabilities of the surrounding minutes), position-in-input features, new pulse/skin features. See [../PLAN.md](../PLAN.md). |

Speed scores on the local machine vary by 1-3 points between runs.

## What we learned (V2 -> V3)

The local harness passes one whole recording per `run()` call. The hidden test clearly does not,
because V2 (82 locally) scored 61 while the length-independent baseline (54 locally) scored 68.
V2 failed when:

- the input was shorter than 5 minutes: `np.convolve(mode="same")` returns a longer array when the
  input has fewer windows than the kernel, the feature code crashed, and `run()` silently answered "Rest";
- several recordings were glued into one input: time going backwards crashed it, and a gap of months
  created over a million empty windows;
- the input was a piece of a recording: the "time since start" features no longer meant anything.

V3 fixes the crash, splits the input wherever time goes backwards or jumps more than 5 minutes,
drops the time-position features and is trained on recordings cut into 1, 2, 5, 10 and 30-minute
pieces as well as whole ones. Quality points (of 80) on the 6 holdout people, model trained without them:

| How `run()` is called | V1 | V2 | V3 |
|---|---|---|---|
| Whole recordings | 38.7 | 65.1 | 57.0 |
| 10-minute pieces | 39.2 | 37.2 | 51.1 |
| 2-minute pieces | 39.9 | 15.6 | 49.0 |
| One person's recordings joined | 38.8 | 41.5 | 45.2 |
| Everything glued into one input | 38.5 | crash / "Rest" | 57.0 |

## Candidates (not uploaded)

| File | Shuffled test (like official) | What it is |
|---|---|---|
| `candidates/TULUI.zip` | 55.6 quality, 72.6 total | **Uploaded: 74.** V3 + 11 person-normalised features, trained on 29 people. |
| `candidates/V5.zip` | T4 41.5 / T3 60.7 | One LightGBM, no position features, trained with within-session shuffled copies. Superseded by V5.1. |
| `candidates/V5.1.zip` | T4 52.6 / T3 60.8 | **Uploaded 19 Nov: 73.** V5 + full-length and session-mixed shuffled copies. Local T4 overestimated it by ~10 (trained on copies spliced like the test). |
| `candidates/TULUI_fast.zip` | identical predictions to TULUI (checked on 107 inputs); 24 % faster (58.2 → 44.0 ms, ≈ +1 speed point on a laptop) | TULUI (74) with faster code only: time conversion, output table, pulse spectrum placement, artifact counting. Same model file. **Recommended next upload.** |
