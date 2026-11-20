# Plan to 79+ (written 2026-11-18)

> **Status 2026-11-20:** best leaderboard result is TULUI-fast (74); V5.1 scored 73. The next candidate
> is TULUI's 145-feature model retrained with shuffle augmentation (`submissions/candidates/TULUI_shuffle_aug.zip`): it won all five T3 holdout seeds by a mean 3.62 quality points and ran faster locally. Its platform score is unknown. New fair-holdout and CV findings, including the limitations of T4
> as a score predictor, are in [NEXT_MODEL_TO_77.md](NEXT_MODEL_TO_77.md) and [HUAWEI_EVALUATION.md](HUAWEI_EVALUATION.md).
> The notes below are historical and should not be read as the current submission schedule.

> **Update 2026-11-18 (evening), read first.** Official rules: the validation recordings are new people,
> one whole 15-35 min recording per run() call, and phases NOT in the training order or durations.
> Time-since-start / phase-order features fail there and are penalised by the jury; look-ahead (future
> data) is penalised at judging. Our old local test kept the training order, so it rewarded V4's
> timetable features (V4: 62.3 locally, 71 on the leaderboard). score.py now shuffles the holdout phases
> within each recording; that test ranks the uploads like the leaderboard (TULUI ~ V3 > V4 > V1).
> V5 = one model, no position features, trained with shuffled copies: 60.7 on the shuffled test vs
> TULUI 55.6 / V3 55.9 (upper estimate: training and test are shuffled the same way).
> Sections 2, 5 and 7 below describe V4 and are partly outdated.

Leaderboard today: **72** (V3). V4 scored **71** (2026-11-18): local gains did not carry over, to be analysed. Deadline: **1 December**. 3 submissions per day.

## 1. Where the points are

Score = 80 × macro-F1 (Rest / Activity / Stress, 10-second windows) + 20 × speed.

What cross-validation (person held out, `src/experiment.py`) says about V3:

| Problem | Size | Status |
|---|---|---|
| Stress F1 is 0.37 (Rest 0.74, Activity 0.86). Macro-F1 weighs it as much as the others. | biggest | improved to 0.46 |
| Rest at the start/end of exercise recordings and warm-up confused with Activity (~6,800 windows). No motion sensor; even BVP noise does not separate them. Only the position in the recording and the heart-rate trend do. | big | improved (position features + stage 2) |
| Short inputs (1-2 minute pieces) score ~0.60 against ~0.74 for whole recordings. | depends on the hidden test | open |
| Speed: V3 ~36-50 ms per call on a laptop = 16.7-18 / 20. The grader may be slower. | up to 2-4 points | open |

## 2. V4: what changed and what it scores

1. **Position features back in** (time since the input started, time left, input length). V3 dropped them because V2 crashed; trained on whole recordings *and* random pieces, the model learns when they mean something. Whole recordings 0.738 → 0.778, pieces unchanged or better.
2. **New features** (in `your_model.py`): pulse regularity and raw-BVP noise (separate the stress protocol from the exercise ones), skin-response size and 30 s / 120 s rates, heart rate against its recent minimum and running extremes, past-vs-future trends at 1 and 3 minutes. Small gain alone (+0.004).
3. **Stage 2** (the big one): a second LightGBM reads the stage-1 probabilities of the surrounding ±3 minutes (lags, centred and past/future averages, the whole input's class mix). It learns the protocol's shape (a task lasts 1-4 minutes, rest follows the cool-down) instead of a fixed moving average. Strict nested CV: **0.674 → 0.708 macro-F1, whole recordings 0.827, Stress 0.46**.
4. Both models live in **one ONNX file** (inputs `s1_X` / `s2_X`), so the submission is still `your_model.py` + `your_model.onnx`, with only numpy / pandas / scipy / onnxruntime at run time.

Local test bench on the 6 holdout people (model trained without them):

| How `run()` is called | V3 quality | V4 quality |
|---|---|---|
| Whole recordings | 57.0 | **62.3** |
| 10-minute pieces | 51.1 | 53.0 |
| 2-minute pieces | 49.0 | 49.5 |
| Everything glued into one call | 57.0 | 62.3 |

Total on the local bench: V3 ~74 → V4 ~77.7 (speed 15.4, to be won back).

## 3. What we tried that did not help (do not repeat)

- **LightGBM instead of sklearn's gradient boosting**: same accuracy (0.664 vs 0.663), 7× faster training. Kept for speed of experiments.
- **Feature selection**: 90-190 features all score the same (0.70-0.71); 60 loses 0.01, 20 loses 0.05. Selection is for speed only.
- Longer stage-2 context (±5-10 min) and a third stage: no gain.
- **Tului's submission** (`submissions/candidates/TULUI.zip`, V3 + 11 person-normalised features, one model): holdout bench 58.9 whole / 51.8 10-min / 47.7 2-min, 39.6 ms (V3: 57.0 / 51.1 / 49.0, 37.7 ms; V4: 62.3 / 53.0 / 49.5, 47.3 ms). Its 11 features added to V4: nested CV +0.004 (0.7079 → 0.7123) but holdout bench -0.8 on whole recordings: noise, not kept.
- `ml_ready_10s_dataset_v1.parquet`: unusable for training (no Rest class, labels are protocol names, Cooldown missing, `*_diff_from_rest` needs the true Rest phase = label leak, 5 s overlap, not on the scorer's 10 s grid).
- The fold split alone moves macro-F1 by ~0.008. **Changes smaller than 0.01 are noise** unless repeated.

## 4. The two unknowns that matter most

1. **How the hidden test calls `run()`** (whole files? pieces? one call?). V2 (82 locally) scored 61, V1 (54 locally) scored 68.
2. **What the leaderboard measures.** V1 is ~0.50 macro-F1 locally for *any* people (CV over all 35) but ~0.65 on the leaderboard, while V3's large local Stress advantage barely shows. Either the hidden data/labels differ or the grader is much slower (V1's tiny forest keeps its speed points, V3 loses them).

`src/eval_harness.py` (label mapping, 10 s scoring, speed formula) was written by us, not Huawei: only `adm_base_class.py` is official. **Get the official topic text, ask the organisers, and read the leaderboard's score breakdown (quality vs time) if Agorize shows it.**

## 5. Submission plan (≈ 39 slots left)

| When | Submit | Why |
|---|---|---|
| Next slot | **V4** (`train_v4.py all`) | Calibration: does +5 local quality show on the leaderboard? |
| Same day, if the leaderboard shows no quality/time split | V4 with stage 2 switched off | Measures what stage 2 is worth there |
| After the speed pass | V4-fast | Measures the speed exchange rate |
| Then | only models that beat the best by ≥ 0.01 locally in **every** input format | |
| Last 2 days | No new ideas; freeze, clean code for the jury | |

Record every upload in `submissions/README.md` (zip + leaderboard + local numbers).

## 6. Work split (4 people)

| Person | Owns | First tasks |
|---|---|---|
| **A: Validation and leaderboard** | `score.py`, `eval_harness.py`, submissions | Get the official rules (label mapping, timing, call format); run the calendar above; repeated CV (3 fold seeds) for any claimed gain. |
| **B: Features (stress)** | `your_model.py` features | Stress-specific ideas: SCR shape (rise time, recovery), beat detection from BVP (more HRV coverage than the E4's IBI), breathing rate from BVP amplitude modulation, per-task onset markers. Test each with `experiment.py cv all+time`, keep only those that also gain after stage 2. |
| **C: Models** | `experiment.py`, `train_v4.py` | Hyperparameter search for both stages (Optuna, overnight on the desktop); seed averaging; a small sequence model (1-D CNN / GRU over the window features, exported to ONNX) as an alternative stage 2. This is where the GPUs help. |
| **D: Speed and code quality** | inference path | `compute_features` is ~45 ms of ~55 ms: compute only the selected features, vectorise `_relative` (rolling median), fewer/larger trees. Target < 20 ms. Keep the code readable: the jury reads it. |

## 7. Commands

```
.venv\Scripts\activate
cd src
python experiment.py build                 # feature cache for the current your_model.py (~30 s)
python experiment.py cv all+time           # stage-1 CV, logged to experiments/log.csv (~1 min)
python experiment.py stack all+time        # + stage 2 (fast, slightly optimistic)
python experiment.py nested all+time       # strict two-stage CV (~5 min): the number to trust
python experiment.py ablate all+time       # drop one feature family at a time
python train_v4.py holdout                 # model without the 6 holdout people -> models/holdout/
python score.py ../models/holdout          # its honest local score, every input format + speed
python train_v4.py all                     # submission model
python package_submission.py               # tests + ../submission.zip
```

The feature cache is keyed by a hash of `your_model.py`, so editing a feature rebuilds it automatically.

## 8. Laptop in front, desktop for long runs

Tree models need no GPU: a full CV takes about a minute on a laptop CPU. The desktop is for overnight searches and the sequence model.

1. Desktop: install OpenSSH Server (Windows: Settings → System → Optional features; then `Start-Service sshd` and `Set-Service sshd -StartupType Automatic` in an admin PowerShell. Linux: `sudo apt install openssh-server`).
2. Both machines: install Tailscale (free) so the laptop reaches the desktop from any network, without opening the router.
3. Laptop: `ssh-keygen`, then copy `~/.ssh/id_ed25519.pub` to the desktop's `C:\Users\<you>\.ssh\authorized_keys` (for an admin account: `C:\ProgramData\ssh\administrators_authorized_keys`).
4. VS Code: install "Remote - SSH", *Connect to Host*, open the repo on the desktop. Editing and terminals then run on the desktop.
5. Long runs: Linux `tmux new -s train`, start the job, detach with Ctrl-B D. Windows: `Start-Process python -ArgumentList 'experiment.py','nested','all+time' -RedirectStandardOutput run.log -NoNewWindow`, or a scheduled task. Results land in `experiments/log.csv`.
6. Code moves through GitHub (`git pull` on the desktop before a run).
