import os
import sys

# 1. Lock single-core execution BEFORE importing numpy/pandas/onnxruntime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pathlib
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.dirname(__file__))
from adm_base_class import ADM_Base
import your_model

REQUIRED_INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID_ACTIVITIES = {"Stress", "Activity", "Rest"}

# Official Validation Split (Holdout subjects from both Stage 1 'S' and Stage 2 'f')
# Tell your ML Architect NOT to train on these 6 subjects!
VAL_SUBJECTS = ["S16", "S17", "S18", "f11", "f12", "f13"]
PROTOCOLS = ["AEROBIC", "ANAEROBIC", "STRESS"]


def map_phase_to_activity(phase_series: pd.Series) -> pd.Series:
    """Maps raw Phase strings to 'Stress', 'Activity', 'Rest', or None (unscored)."""
    stress_phases = {"Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction"}
    activity_exact = {"Warm-up", "Sprint", "Cooldown"}
    rest_phases = {"Baseline", "Rest"}

    def _mapper(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        if val_str in stress_phases:
            return "Stress"
        if val_str in activity_exact or "rpm" in val_str.lower():
            return "Activity"
        if val_str in rest_phases:
            return "Rest"
        return None  # 'Undefined' or unlisted phases are not scored

    return phase_series.map(_mapper)


def window_10s_mode(series: pd.Series) -> pd.Series:
    """Computes the most frequent state in each 10-second window."""
    def _fast_mode(arr):
        valid = arr.dropna()
        if len(valid) == 0:
            return None
        vals, counts = np.unique(valid, return_counts=True)
        return vals[np.argmax(counts)]

    return series.resample("10s").apply(_fast_mode)


def run_validation_suite():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    print("=" * 65)
    print(" HUAWEI 2026 TECH ARENA - LOCAL VALIDATION & SCORING HARNESS")
    print("=" * 65)

    # --- 1. Instantiate Model & Verify Contract ---
    tracemalloc.start()
    model = your_model.get_model()
    assert isinstance(model, ADM_Base), "FAIL: Model must inherit from ADM_Base!"
    print("[PASS] Model successfully instantiated and inherits from ADM_Base.")

    all_y_true_10s = []
    all_y_pred_10s = []
    file_loaded_count = 0
    sample_df_for_speed = None

    # --- 2. Evaluate Quality Across Holdout Subjects ---
    print(f"\nEvaluating Quality on holdout subjects: {VAL_SUBJECTS}...")
    for protocol in PROTOCOLS:
        for subj in VAL_SUBJECTS:
            file_path = data_dir / protocol / f"{subj}.parquet"
            if not file_path.exists():
                continue

            df = pd.read_parquet(file_path)
            file_loaded_count += 1

            # Keep one representative file for the 50-run speed benchmark
            if sample_df_for_speed is None:
                sample_df_for_speed = df[REQUIRED_INPUT_COLS].copy()

            # Prepare clean sensor input (strip Phase and user_id)
            sensor_input = df[REQUIRED_INPUT_COLS].copy()

            # Run model prediction
            pred_df = model.run(sensor_input)

            # Contract checks on first file
            if file_loaded_count == 1:
                assert isinstance(pred_df, pd.DataFrame), "Output is not a DataFrame!"
                assert "Activity" in pred_df.columns, "Missing 'Activity' column!"
                assert pred_df.index.equals(sensor_input.index), "Output index mismatch!"
                invalid = set(pred_df["Activity"].dropna().unique()) - VALID_ACTIVITIES
                assert not invalid, f"Invalid activity predictions found: {invalid}"

            # Map ground truth Phase -> Activity and aggregate 10s windows
            true_activity = map_phase_to_activity(df["Phase"])
            true_10s = window_10s_mode(true_activity)
            pred_10s = window_10s_mode(pred_df["Activity"])

            # Filter out unscored ('Undefined' / None) 10-second windows
            scored_mask = true_10s.notna() & pred_10s.notna()
            all_y_true_10s.extend(true_10s[scored_mask].tolist())
            all_y_pred_10s.extend(pred_10s[scored_mask].tolist())

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_ram / (1024 ** 2)
    print(f"[PASS] Processed {file_loaded_count} validation recordings. Peak RAM: {peak_ram_mb:.1f} MB / 4096 MB")

    # --- 3. Compute Quality Score (0 - 80 points) ---
    macro_f1 = f1_score(
        all_y_true_10s,
        all_y_pred_10s,
        labels=["Stress", "Activity", "Rest"],
        average="macro",
        zero_division=0,
    )
    quality_score = macro_f1 * 80.0

    print("\n--- Per-Class 10s Window Performance ---")
    print(
        classification_report(
            all_y_true_10s,
            all_y_pred_10s,
            labels=["Stress", "Activity", "Rest"],
            digits=4,
            zero_division=0,
        )
    )

    # --- 4. Compute Speed Score (50-Run MET Benchmark, 0 - 20 points) ---
    print(f"Running 50-iteration Speed Benchmark on {len(sample_df_for_speed):,} rows...")
    run_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.run(sample_df_for_speed)
        t1 = time.perf_counter()
        run_times.append(t1 - t0)

    met = float(np.mean(run_times))
    std_t = float(np.std(run_times))
    speed_score = max(0.0, (1.0 - (met / 0.3)) * 20.0)

    print("=" * 65)
    print(f" QUALITY SCORE (80%): {quality_score:6.2f} / 80.00  (Macro-F1: {macro_f1:.4f})")
    print(f" SPEED SCORE   (20%): {speed_score:6.2f} / 20.00  (MET: {met*1000:.1f} ms ± {std_t*1000:.1f} ms)")
    print("-" * 65)
    print(f" TOTAL SIMULATED SCORE: {quality_score + speed_score:6.2f} / 100.00")
    print("=" * 65)


if __name__ == "__main__":
    run_validation_suite()