# How Huawei evaluates: what our 7 uploads tell us (19 Nov 2026)

> **Update: V5.1 scored 73** (T4 fit predicted ~85). A model trained on spliced copies scores well on our
> spliced tests for the wrong reason, so **our local tests cannot judge models trained on shuffled/spliced
> data.** They still rank models trained on normal recordings well. Leaderboard spread of the good models:
> 71-74, possibly within the leaderboard's own noise.

Reading time: about 10 minutes. Scripts and raw numbers: session notes; every test below can be rerun with `src/score.py`.

## 1. The evidence

Each upload was rebuilt **without the 6 holdout people** (S16-S18, f11-f13), so the local numbers are fair.
Then it was scored on 5 different ways of building test recordings. Quality is out of 80, speed out of 20.

| Upload | Leaderboard | T0 training order | T1 trimmed 15-35 min | T2 phase order shuffled | T3 order + lengths shuffled | T4 a person's sessions mixed | Local speed |
|---|---|---|---|---|---|---|---|
| V1 | 68 | 38.7 | 39.0 | 39.0 | 39.2 | 39.1 | 15.9 |
| V1.2 | 65 | 41.5 | 42.0 | 37.6 | 37.2 | 36.3 | 15.5 |
| V2 | 61 | **65.1** | 53.6 | 46.4 | 43.8 | 29.3 | 15.9 |
| V3 | 72 | 57.0 | 56.1 | 54.7 | 55.9 | 40.5 | 14.8 |
| V4 | 71 | 62.3 | 58.3 | 49.7 | 49.2 | 32.6 | 13.9 |
| TULUI | **74** | 58.9 | 58.5 | 55.2 | 55.6 | 39.5 | 14.8 |
| SHEH ⚠ | 54 | 67.4 | 64.9 | 57.3 | 52.8 | **28.1** | 15.1 |
| *rank correlation with the leaderboard, 6 uploads* | | −0.09 | +0.71 | +0.83 | +0.77 | +0.60 | |
| *rank correlation with the leaderboard, 7 uploads* | | **−0.43** | +0.07 | +0.14 | +0.50 | **+0.75** | |

⚠ SHEH: its file does not say which people it was trained on, and it gets Stress F1 0.92 on the holdout people in
training order (fair models: ~0.5), so it very likely saw them: its local numbers are inflated. The *pattern* still
counts: best of all on training order, worst of all when sessions are mixed, and lowest on the leaderboard.
Its zip also has the files inside a folder (`Submission v5/`) instead of at the root.

**Best predictor: T4 (sessions mixed).** Linear fit over the 6 fair-ish uploads: leaderboard ≈ 1.03 × (T4 quality + speed) + 14.5,
typical error ~4 points. T4 ranks the uploads V3, TULUI, V1, V1.2, V4, V2, SHEH; the leaderboard ranks them
TULUI, V3, V4, V1, V1.2, V2, SHEH. Only V4 is out of place. `score.py` now uses T4 as its main number ("MIXED").

## 2. What we can conclude

| Finding | Confidence | Evidence |
|---|---|---|
| Huawei's recordings do **not** keep the training phase order | **High** | The training-order test predicts the leaderboard worst (−0.09). V2 and V4, the models that learned the timetable, lose most. Huawei's own rules say so too. |
| Shuffled phase order is the closest local copy | Medium | T2 and T3 rank the uploads best (+0.83 / +0.77). For V2, V3, V4 and TULUI, leaderboard ≈ T2 quality + speed + 0 to 8. |
| Huawei's phase → class mapping is the same as ours | Medium | We re-scored with Cooldown = Rest, Warm-up = Rest, Baseline unscored, and post-task rest unscored. None fitted better than our mapping. |
| **Huawei's test punishes dependence on what else is in the recording** | Medium-high | T4 (a person's sessions mixed, so the phase mix of a recording changes) predicts the leaderboard best (+0.75; +0.90 among boosted-tree models). Training order predicts it in reverse (−0.43; −0.90). |
| V1 and V1.2 do better on the leaderboard than on most local tests | Explained by T4 | Their features use only the current 10 seconds, so they keep their score when the phase mix changes (T4: 39 and 36, level with V3/TULUI). Labels are not the reason: alternative phase→class mappings fit no better. |
| V1.2 (65) < V1 (68) | Medium | V1.2 assumes the first 60 s of each recording is rest. In shuffled recordings it often is not: T2/T3 quality drops from ~42 to ~37. |

**Rule of thumb from now on:** a model is better only if it wins on **T4 ("MIXED" in `score.py`)** without losing on T3.
A model that is much better on training order than on T4 has memorised sessions (V2, V4, SHEH): do not upload it.

## 3. V5 under the new rules: an outside review

### Strengths

- No time-since-start, phase-order or position features: compliant with the protocol rule.
- Trained with shuffled copies, so it cannot memorise the timetable. Best on T2/T3: 57.3 / 60.7, against TULUI 55.2 / 55.6.
- One model, 43 ms per call. Passes all packaging tests, including "no silent fallback to Rest".

### Weaknesses and loopholes (measured, not guessed)

1. **The whole recording leaks into every prediction (look-ahead).** We kept a recording up to minute *t* and replaced everything after *t* with another recording: **45-50 % of V5's predictions changed, even 8 minutes before *t***. The cause is the whole-recording features: `rec_*` summaries, value minus recording median (`_rel`), robust z (`_z`). For the jury, this counts as unlimited look-ahead.
2. **Sensitive to the mix of phases in a recording.** On T4, V5 falls to 41.5 with Stress F1 0.01. If Huawei's recordings mix phases differently from our sessions, V5 has nothing to fall back on.
3. **Possibly overfitted to our shuffling.** Training copies and the T3 test are spliced the same way, with abrupt jumps at phase borders. The model may learn "a jump = a new phase". If Huawei's recordings are continuous, this signal disappears. T3 is therefore an **upper** estimate.
4. **Stress is still the weak class** (F1 0.46-0.55; Rest 0.80, Activity 0.92). Stress tasks are short and their after-effect is labelled Rest.
5. **The local test is small:** 6 people, 17 source recordings. Differences under ~1-2 points are noise.
6. Smoothing (k = 1) and class bias (0) are not tuned for shuffled data.

### Fixes, tested

| Variant (trained without holdout people) | T2 | T3 | T4 | Average |
|---|---|---|---|---|
| V5 | 57.3 | 60.7 | 41.5 | 53.2 |
| **V5.1: trained with 3 kinds of shuffles (reordered, reordered at full length, sessions mixed)** | 56.9 | 60.8 | **52.6** | **56.8** |
| V5 without whole-recording features | 55.6 | 57.7 | 44.9 | 52.7 |
| V5, 5 random seeds averaged | 57.4 | 60.9 | 41.4 | 53.3 |

**V5.1** keeps V5's score where V5 is strong, and fixes its collapse when the phase mix changes (+11 on T4, the test that matches the
leaderboard best). It is the recommended upload. The T4 fit predicts ~85 for it, but V5.1 was trained on copies spliced like T4,
so that number is inflated: expect it to beat 74, not to reach 85.
Removing the whole-recording features fixes the look-ahead problem but costs ~3 points. Keep this for the jury version, and decide it closer to the final.

## 4. Next steps

1. Upload **V5.1** (`submissions/candidates/V5.1.zip`). It will tell us whether robustness to the phase mix matters on Huawei's data.
2. To explain V1's +10: upload a model that uses *only* local features (variant G). If it scores well above its local number, the "what else is in the recording" explanation is confirmed.
3. Before the finals: replace whole-recording and future-window features with past-only versions (rolling baselines over the last N minutes), and measure the cost with `score.py`.
