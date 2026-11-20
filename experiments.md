# Huawei Tech Arena 2026 — Stage 0 calibration

## Scope

Ran `truth_eval.py` on 2026-11-20 with Python from `.venv`, one CPU affinity, and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. No model code was changed.

The evaluator found 35 subject groups across `data/AEROBIC`, `data/ANAEROBIC`, and `data/STRESS`. The longest recording was `data/STRESS/f01.parquet` (201,305 rows; 3145.375 s). This first evaluator pass checked `models/loso/<submission>/<subject>/` for already-produced fold artifacts and timed the packaged inference ZIPs; it did **not** yet invoke the available training recipes. Therefore its LOSO quality, total, and rank-match values are N/A, and this table is a speed audit only, not the requested calibration.

### Training-recipe audit correction (2026-11-20)

The project does contain training recipes, so the initial claim that fold training could not be reconstructed was too broad. Located recipes are `src/old/train_baseline.py` (V1), `src/old/train_v2.py` (V2), `src/old/train_robust.py` (V3 robust training), `src/train_v4.py` (V4), and `src/train_v5.py` (V5/V5.1). Newer recipes include `experiments/train_tului_aug.py`, `experiments/train_v6.py`, and `experiments/train_v7_interactions.py`. These have not yet been adapted or run as 35-fold LOSO by this script. The exact historical recipes for V1.2, original TULUI, SHEH, and the V27 “Last update 69” package were not found in the project’s Python training-script inventory; do not substitute one of the newer recipes and call it an exact reproduction.

## Executed table

The evaluator timed 50 full `run()` calls per package on the longest recording. Speed points are printed for both the 0.3 s cap on the current published competition topic page and the 1.0 s cap specified in the task prompt; each MET and speed value below is copied from the successful terminal run.

| Submission | Website Score | Local LOSO Quality (/80) | Local Speed (/20) | Local Total (/100) | MET (ms) | Rank Matches Website? |
|---|---:|---|---|---|---:|---|
| SubV1 | 68 | N/A (no LOSO folds) | 16.06 (0.3s); 18.82 (1.0s) | N/A (quality unavailable) | 59.107 | N/A |
| SubV1.2 | 65 | N/A (no LOSO folds) | 15.86 (0.3s); 18.76 (1.0s) | N/A (quality unavailable) | 62.075 | N/A |
| SubV2 | 61 | N/A (no LOSO folds) | 16.78 (0.3s); 19.03 (1.0s) | N/A (quality unavailable) | 48.259 | N/A |
| SubV3 | 72 | N/A (no LOSO folds) | 16.34 (0.3s); 18.90 (1.0s) | N/A (quality unavailable) | 54.881 | N/A |
| SubV4 | 71 | N/A (no LOSO folds) | 15.57 (0.3s); 18.67 (1.0s) | N/A (quality unavailable) | 66.440 | N/A |
| SubTULUI | 74 | N/A (no LOSO folds) | 15.98 (0.3s); 18.79 (1.0s) | N/A (quality unavailable) | 60.328 | N/A |
| SubTULUI Fast | 74 | N/A (no LOSO folds) | 17.02 (0.3s); 19.11 (1.0s) | N/A (quality unavailable) | 44.661 | N/A |
| SubV5 | 73 | N/A (no LOSO folds) | 15.70 (0.3s); 18.71 (1.0s) | N/A (quality unavailable) | 64.473 | N/A |
| SubV5.1 | 73 | N/A (no LOSO folds) | 16.05 (0.3s); 18.82 (1.0s) | N/A (quality unavailable) | 59.180 | N/A |
| SubSHEH | 54 | N/A (no LOSO folds) | 14.13 (0.3s); 18.24 (1.0s) | N/A (quality unavailable) | 88.073 | N/A |
| Last update 69 | 69 | N/A (no LOSO folds) | 16.40 (0.3s); 18.92 (1.0s) | N/A (quality unavailable) | 53.997 | N/A |
| Current candidate (unsubmitted) | N/A | N/A (no LOSO folds) | 17.37 (0.3s); 19.21 (1.0s) | N/A (quality unavailable) | 39.424 | N/A |

## Input discrepancies

- The prompt says there are nine historical submissions, then also supplies SHEH and “Last update 69”; this table includes all 11 scored packages found/provided, plus the unsubmitted current candidate.
- The prompt gives V5 a website score of 73. The older repository README says V5 was not uploaded. The table follows the score supplied in the current prompt.
- The task prompt specifies a 1.0 s speed cap. The current published topic page states a 0.3 s cap. Both speed computations are shown; confirm which rule applies to the current leaderboard before treating speed points as authoritative.

## Run failure and correction

The first run failed with `OSError: GetProcessAffinityMask failed` because the Windows API call had untyped pointer/handle arguments. The evaluator was corrected to declare 64-bit Windows API argument types, then the run above completed. This was an evaluator-only change.

### Attempted V2 LOSO follow-up

After locating the trainers, a 35-fold V2 LOSO run was started using `src/old/train_v2.py`, the archived V2 feature list, and the cached V2 feature dataset. It was manually interrupted after several minutes without producing a score; no V2 LOSO number is reported. Exact terminal output before interruption:

```text
dataset=14890 windows subjects=35 features=138
```

This run is incomplete, not a scored result. The slow/no-progress behavior needs to be diagnosed before launching the wider LOSO matrix.

### V2 LOSO preliminary run (superseded)

Re-ran V2 with per-fold progress and `OMP_NUM_THREADS=8` for local training speed (the submission timing benchmark remains one-core). The code used `src/old/train_v2.py`, the exact 138 feature names embedded in `submissionV2.zip`, the cached `v2_features.parquet`, 35 subject-disjoint GroupKFold splits (one held-out subject per split), and the archive's fixed smoothing width and class log-bias. No hyperparameter or bias tuning used held-out labels.

Exact terminal output from the preliminary run:

```text
V2 LOSO pooled macro-F1=0.86552861; quality80=69.24228872; seconds=801.5
```

This quality score is for the archived V2 recipe. V2 includes protocol-position/time features, so it is an evaluation-history row and not a compliant feature set for later model changes.

**Do not use that quality number in the final comparison.** The run smoothed only the scored rows in the training cache; the scorer runs predictions across unscored windows too, so the smoothing context was not identical. A 60-second feature-parity probe of archived and current V2 implementations failed because the archived extractor expects a whole recording (terminal `ValueError: all the input array dimensions except for the concatenation axis must match exactly ... index 0 has size 7 and index 44 has size 12`). On the full `STRESS/f01` recording the parity probe succeeded: `(316, 138)/(316, 138)`, identical window IDs and NaN masks, maximum absolute feature difference `0`.

### V2 corrected LOSO completed

Re-ran all 35 folds. For each held-out subject the evaluator fit V2 on the other 34 subjects, computed archived features for every window of every held-out recording (including unscored windows), applied the archive's smoothing and class bias, mapped the resulting window labels to input rows, and used `window_10s_mode` for pooled scoring. This matches the scorer's 10-second clock alignment and avoids smoothing across omitted unscored windows. No held-out labels were used for tuning.

Exact final terminal output:

```text
V2 corrected LOSO pooled windows=14890 F1=[0.892137177736728, 0.9309585751450871, 0.7750213857998289] macro=0.86603905 quality80=69.28312370 seconds=852.1
```

### V1 LOSO completed

V1 was retrained per held-out subject with the exact `src/old/train_baseline.py` RandomForest settings (60 trees, depth 8, balanced classes, seed 42) and its archived 12 mean/std features. Predictions were expanded back to input rows and scored with the local scorer's 10-second clock resampling and string tie order.

Exact final terminal output:

```text
V1 LOSO pooled windows=14890 F1=[0.5944889583740472, 0.7753063399041022, 0.24635761589403973] macro=0.53871764 quality80=43.09741104 seconds=74.1
```

The first V1 attempt failed at the window-scoring conversion because integer predictions were passed to a string-label mode helper. Exact terminal error: `KeyError: 1`. The evaluator was corrected to map predicted integer IDs to `Rest`/`Activity`/`Stress` before resampling; the successful result above is from the corrected run.

### V3 LOSO completed

Used cached robust training examples from `src/old/train_robust.py`, held out all rows for one subject at a time, trained with the same `train_v2.fit` HistGradientBoosting recipe, and scored full held-out recordings with the archived V3 134-feature implementation plus the package's fixed smoothing and bias. Archived/current feature parity on `STRESS/f01` had 316 matching windows, identical NaN masks, and max absolute difference 0.

Exact final terminal output:

```text
V3 LOSO pooled windows=14890 F1=[0.79315188762072, 0.9118054680609802, 0.5174880763116058] macro=0.74081514 quality80=59.26521152 seconds=1779.6
```

### V4 LOSO completed

Used the cached V4 feature matrix, which contains the archive's 190 features, and confirmed exact archived/current extractor parity on `STRESS/f01` (316 windows, matching window IDs and NaNs, maximum absolute feature difference 0). For each outer subject fold, stage 1 was trained with five group-held-out inner folds; stage 2 was fit on those inner out-of-fold probabilities, and the held-out full recordings were scored through both stages with archived post-processing.

Exact final terminal output:

```text
V4 LOSO pooled windows=14890 F1=[0.8809070595050897, 0.9481481481481482, 0.6795580110497238] macro=0.83620441 quality80=66.89635250 seconds=2367.6
```

The first V4 attempt failed because `experiment.build()` tried to start a Windows `ProcessPoolExecutor` from an inline `<stdin>` script. No score was produced. The corrected run loaded the already-built feature cache directly. Exact first error output:

```text
OSError: [Errno 22] Invalid argument: 'D:\\ALL\\STD\\Applications not Research Internship\\ML comp. Finland\\Huawei_TechArena_2026\\<stdin>'
BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore
```
