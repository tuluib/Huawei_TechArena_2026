# How Huawei evaluates: what our 7 uploads tell us (19 Nov 2026)

> **Update 20 Nov: 12 uploads.** TULUI35 (TULUI's recipe on all 35 people) = **75, best**; TULUI_fast = 74 (same
> predictions as TULUI, faster); V5 = 73; a GRU network (V27) = 69. Full table with local scores: `submissions/README.md`.
> Lessons: more training people helped (+1); speed-only changes did not show; sequence networks and big ensembles lose.

> **Update: V5.1 scored 73** (T4 fit predicted ~85). A model trained on spliced copies scores well on our
> spliced tests for the wrong reason, so **our local tests cannot judge models trained on shuffled/spliced
> data.** Earlier T4 ranking looked promising over seven uploads, but the later V5.1/TULUI comparison
> reversed the local and leaderboard order. Leaderboard spread of the good models: 71-74.

> **20 Nov calibration update:** on the fair holdout, T4 scores V5.1 about 12 quality points above TULUI,
> but the leaderboard ranks TULUI 74 and V5.1 73. Treat T4 as a robustness stress test, not a score
> predictor across training recipes. The earlier rank-correlation estimate came from only seven uploads.

> **Platform score decomposition:** the public leaderboard currently reports TULUI-fast at 74 with
> `run_time=87.81`. The official [topic description](https://huawei.agorize.com/en/challenges/2026-finland-tech-arena/pages/topic-description?lang=en)
> defines speed as `(1 - mean_runtime/0.3s) × 20`. Interpreting the leaderboard runtime value as milliseconds
> implies about 14.15 speed points and 59.85 quality points (macro-F1 ≈ 0.748). This is a calculation
> from the public total/runtime, not a score split reported by Agorize; use `experiments/platform_score_decompose.py`.

> **69-point reference archive:** `Last update result 69 Tului promised.zip` scores 73.7 on our full local
> T3-plus-speed diagnostic (T3 58.5/80, speed 15.2/20), versus the reported platform 69. The model
> metadata says `trained_on=all`, and its training-order holdout quality is 74.1/80, so local validation
> includes people used in training. This demonstrates the risk of in-sample evaluation; it does not show
> a 4.7-point bias for fair person-held-out models. Quick scoring gives 76.1; its one-off speed reading was
> 48.9 ms, while the full run and three repeated 50-call batches average 71.9 ms. Use the full local quality
> path and repeated timing when comparing candidates.

Reading time: about 10 minutes. Scripts and raw numbers: session notes; every test below can be rerun with `src/score.py`.

## 0. Official answers from Huawei (forum and topic page, 20 Nov 2026), verbatim

- **Look-ahead:** "The recommended look-ahead time is up to 30 seconds. Infinite look-ahead (i.e. whole recording
  analysis) practically scores zero for feasibility. How much the look-ahead time affects the scoring ultimately depends
  on how much benefit the look-ahead brings to the solution." "Look-ahead is the amount of time needed past a prediction
  point needed for the prediction."
- "The closer the model operates to real-time conditions, the better. The more a model relies on information from
  future time points (i.e. "seeing into the future"), the greater the penalty it will receive in terms of technical
  quality and real-world applicability during the final judging phase."
- **Protocol:** "Models should rely exclusively on physiological signals and must not use information about the protocol
  itself, such as the duration or structure of different phases. Any use of such information to gain an advantage will
  result in a significant penalty in the technical quality and real-world applicability scores during the final judging
  phase." "In the new validation dataset, tuning to the protocol is not possible anymore. Your solution should be
  functional for random free-living data and this is what the validation measures."
- **Metric:** "The macro-f1 is pooled over the whole validation dataset."
- **Timing:** "Timed run is based on the validation recordings of 15-35 minutes in duration."
- **Data:** "The validation dataset are entirely new subjects." "The validation dataset does not follow the protocols of
  the training dataset."
- Still open (asked on the forum): speed cap 0.3 s or 1.0 s, reference CPU, whether totals are rounded, how the
  validation recordings were built, which submission the jury reviews.

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

**Historical best fit: T4 (sessions mixed).** Linear fit over the original six fair-ish uploads: leaderboard ≈ 1.03 × (T4 quality + speed) + 14.5,
typical error ~4 points. That relationship failed for V5.1 versus TULUI, so it is no longer used as a score estimate. `score.py` now puts within-session phase reorder/length changes (T3) first and reports T4 as a separate stress test.

## 2. What we can conclude

| Finding | Confidence | Evidence |
|---|---|---|
| Huawei's recordings do **not** keep the training phase order | **High** | The training-order test predicts the leaderboard worst (−0.09). V2 and V4, the models that learned the timetable, lose most. Huawei's own rules say so too. |
| Shuffled phase order is the closest local copy | Medium | T2 and T3 rank the uploads best (+0.83 / +0.77). For V2, V3, V4 and TULUI, leaderboard ≈ T2 quality + speed + 0 to 8. |
| Huawei's phase → class mapping is the same as ours | Medium | We re-scored with Cooldown = Rest, Warm-up = Rest, Baseline unscored, and post-task rest unscored. None fitted better than our mapping. |
| **Huawei's test punishes dependence on what else is in the recording** | Low / unresolved | T4 made V5.1 look much stronger than TULUI, but the leaderboard order is reversed (73 vs 74). T4 diagnoses one kind of brittleness; it does not identify the hidden distribution. |
| V1 and V1.2 do better on the leaderboard than on most local tests | Explained by T4 | Their features use only the current 10 seconds, so they keep their score when the phase mix changes (T4: 39 and 36, level with V3/TULUI). Labels are not the reason: alternative phase→class mappings fit no better. |
| V1.2 (65) < V1 (68) | Medium | V1.2 assumes the first 60 s of each recording is rest. In shuffled recordings it often is not: T2/T3 quality drops from ~42 to ~37. |

The earlier T4 rule was too strong. A model that is much better on training order than on mixed sessions may be brittle (V2, V4, SHEH), but T4 cannot rank every later model recipe: V5.1 is the counterexample. Use local input variants to expose failure modes; use the platform to establish leaderboard gains.

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

1. V5.1 was uploaded on 19 Nov and scored **73**. TULUI remains the best at **74**.
2. `TULUI_fast.zip` was uploaded; the public leaderboard confirms it scored **74**, the same as the prior TULUI result. This confirms the speed-only candidate did not improve the public total despite the local timing gain.
3. The next candidate is `submissions/candidates/TULUI_shuffle_aug.zip`: shuffle augmentation on TULUI's 145 features won all five T3 holdout seeds by a mean 3.62 quality points. It remains unscored on the leaderboard.
4. Do not use T4 as a leaderboard predictor across model families. On 20 Nov, fair holdout scoring put V5.1 about 12 quality points above TULUI on generated mixed-session T4 recordings, while the leaderboard order is TULUI (74) above V5.1 (73). T4 remains useful as a robustness test only.
5. A 25% TULUI / 75% V5.1 probability blend improved the fair holdout's original and within-session scores slightly, but lost slightly on T4. That is not a robust enough gain to replace TULUI.
6. The public leaderboard exposes only each team's best score, not full submission history or a quality/speed split. Before jury review, continue the no-look-ahead work: replace global recording summaries, per-recording normalization, centered future context, and future lags with past-only rolling baselines, then measure the quality cost. The tested 27-feature restricted version lost substantial quality, so test feature families in stages rather than removing all context at once.
