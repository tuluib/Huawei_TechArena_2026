# Submissions

Each zip is exactly what was uploaded to Agorize (`your_model.py` + `your_model.onnx` at the root).

| File | Leaderboard | Local harness (holdout people) | What it is |
|---|---|---|---|
| `submissionV1.zip` | **68** | 53.9 (quality 38.8, speed 15-17) | Baseline: Random Forest on mean/std of the raw columns per 640 rows. |
| `submissionV2.zip` | **61** | 82.3 (quality 65.1, speed 17.2) | Cleaned-signal features + gradient boosting, with time-position features. Broke on the hidden test (see below). |
| `submissionV3.zip` | **72** | 57.0 quality on whole recordings, 45-57 on any input format | V2 made robust to how `run()` is called. Current best. |

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
