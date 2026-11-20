# Submission history

Every zip in this folder was uploaded to Agorize. `candidates/` holds packages that were **not** uploaded.
All zips have `your_model.py` + `your_model.onnx` at the root (except SHEH, which has them in a subfolder).
Agorize allows 3 successful submissions per day; the leaderboard shows each team's best score.

**Best: 75, `TULUI35.zip`** (20 Nov 2026).

## All uploads (approximate upload order)

| # | File | Website | What it is | What it taught us |
|---|---|---|---|---|
| 1 | `submissionV1.zip` | 68 | Random forest (60 trees) on mean/std of the 6 raw columns per 640 rows. | A very simple model that only looks at the current 10 s holds up well on Huawei's data. |
| 2 | `V1.2(2nd trial65score).zip` | 65 | V1-style forest, 36 features; normalises by the first 60 s of the recording. | "The first minute is rest" is false in Huawei's recordings. |
| 3 | `submissionV2.zip` | 61 | Cleaned signals + gradient boosting, **with time-since-start features**. | Timetable features fail: 82 locally, 61 on the website. |
| 4 | `submissionV3.zip` | 72 | V2 without time features, trained on whole recordings + 1-30 min pieces. | Robust to how `run()` is called. |
| 5 | `submissionV4.zip` | 71 | Two-stage LightGBM (stage 2 reads stage 1 over ±3 min) + position features. | Best ever locally (62.3) but not on the website: learned the protocol order. |
| 6 | `TULUI.zip` | 74 | V3 + 11 "relative to the person's resting level" features, median fill, trained on 29 people. | Best until TULUI35. |
| 7 | `SHEH.zip` | 54 | 3 tree models (2,710 trees), 163 features; files in a subfolder. | Huge capacity memorises sessions: best on training order, worst on mixed sessions and on the website. |
| 8 | `V5.zip` | 73 | One LightGBM, 186 features, no time features, trained with within-session shuffled copies. | Shuffled-copy training looks great on our shuffled tests, but those tests are spliced the same way. |
| 9 | `V5.1.zip` | 73 | V5 + full-length and session-mixed shuffled copies. | Local mixed test predicted ~85; website 73. Our spliced tests cannot judge splice-trained models. |
| 10 | `TULUI_fast.zip` | 74 | TULUI with faster code, **identical predictions** (checked on 107 inputs). | Public runtime 87.81 shown for it. Speed gain did not change the total. |
| 11 | `V27_dualrate_GRU.zip` (was "Last update result 69 Tului promised") | 69 | Neural network (GRU), 2 s + 10 s feature streams, trained on everyone. | The sequence network did worse than the tree models. |
| 12 | `TULUI35.zip` | **75** | TULUI's recipe rebuilt with LightGBM (`src/train_tului.py`), trained on **all 35 people**, TULUI_fast code. | 6 more training people: 74 → 75. More data helps. |

## Local scores of the uploads

From `src/truth_eval.py` (20 Nov). Quality is out of 80, on the 6 holdout people (S16-S18, f11-f13), using
copies trained *without* them where available. Speed: 50 runs on the longest recording, cap 1.0 s.
MIXED = holdout phases from all sessions, random order and lengths: the best (but imperfect) local
match to the website. Raw table: `experiments/truth_eval_stage0.csv`.

| Upload | Website | Fair copy? | MIXED | Shuffled in session | Training order |
|---|---|---|---|---|---|
| TULUI / TULUI_fast | 74 | yes | 39.5 | 55.6 | 58.9 |
| TULUI35 recipe (trained without holdout) | 75 | yes | 40.4 | 54.7 | 56.9 |
| V5 | 73 | yes | 41.9 | 60.5 | 55.4 |
| V5.1 | 73 | yes | 52.6* | 60.8* | 53.2 |
| V3 | 72 | yes | 40.5 | 55.9 | 57.0 |
| V4 | 71 | yes | 32.6 | 49.2 | 62.3 |
| V1 | 68 | yes | 39.1 | 39.2 | 38.7 |
| V1.2 | 65 | unknown | 36.3 | 37.2 | 41.5 |
| V2 | 61 | yes | 29.3 | 43.8 | 65.1 |
| SHEH | 54 | no (saw holdout) | 28.1 | 52.8 | 67.4 |

\* Inflated: trained on copies spliced like these tests. Rank correlation with the website: MIXED +0.77,
shuffled +0.65, training order **−0.21** (all 10). The TULUI35 *upload* itself was trained on the holdout
people, so only its holdout-trained copy is scored here.

## Candidates (not uploaded)

| File | Evidence | What it is |
|---|---|---|
| `candidates/TULUI_shuffle_aug.zip` | Holdout comparison: TULUI baseline T3 mean 54.03 vs augmented 57.65 across five seeds (+3.62/80, 5/5 wins); quick T2 57.5 vs 56.4, T4 48.7 vs 36.0. Paired timing: TULUI-fast 55.1 ms, candidate 52.8 ms. Contract passed on a 130,609-row recording. | TULUI's 145 features and TULUI-fast code with V5.1/V7 shuffle augmentation and a 400-tree LightGBM, trained on 29 people. **Caution:** V5/V5.1 show that shuffle-trained models look better on our shuffled tests than on the website. |
| `candidates/V7_interactions.zip` | Holdout-trained variant: T3 61.5 vs V5.1 60.8; mean +0.52 over five shuffle seeds; speed 13.8 vs 13.6. | V5.1's augmentation recipe + 11 TULUI-style cross-sensor interactions. Controlled quality probe. |

## Rules for this folder

1. Before uploading, score the **holdout-trained** copy with `python score.py NAME`; never trust the local score of a model trained on everyone.
2. After uploading, move the zip from `candidates/` to here, add a row above (website score, runtime if shown, what changed) and commit.
3. Change one thing per upload, so each score answers one question.
