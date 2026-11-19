import os
import sys
import pathlib
import zipfile
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from adm_base_class import ADM_Base
import your_model

REQUIRED_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]
VALID_CLASSES = {"Stress", "Activity", "Rest"}


def make_synthetic_df(n_rows: int, fill_val: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 12:00:00", periods=n_rows, freq="15625us")  # 64 Hz
    data = {col: np.full(n_rows, fill_val, dtype=np.float64) for col in REQUIRED_COLS}
    return pd.DataFrame(data, index=idx)


def verify_output(df_in: pd.DataFrame, df_out: pd.DataFrame, test_name: str):
    assert isinstance(df_out, pd.DataFrame), f"[{test_name}] Output is not a DataFrame!"
    assert "Activity" in df_out.columns, f"[{test_name}] Missing 'Activity' column!"
    assert len(df_out) == len(df_in), f"[{test_name}] Row count mismatch ({len(df_out)} vs {len(df_in)})!"
    assert df_out.index.equals(df_in.index), f"[{test_name}] Index does not match input!"
    assert not df_out["Activity"].isna().any(), f"[{test_name}] Output contains NaN predictions!"
    invalid = set(df_out["Activity"].unique()) - VALID_CLASSES
    assert not invalid, f"[{test_name}] Invalid class labels returned: {invalid}"
    print(f"  [PASS] {test_name}")


def run_torture_tests_and_package():
    print("=" * 65)
    print(" HUAWEI 2026 - PRE-SUBMISSION TORTURE TEST & AUTO-PACKAGER")
    print("=" * 65)

    model = your_model.get_model()
    assert isinstance(model, ADM_Base), "FAIL: get_model() does not return an ADM_Base instance!"

    print("\n1. Running Edge-Case Torture Tests...")

    # Test 1: Sensor Dropout (100% NaNs across all columns)
    df_nan = make_synthetic_df(1280, np.nan)
    verify_output(df_nan, model.run(df_nan), "Test 1: Total Sensor Dropout (All NaNs)")

    # Test 2: Flatlined Zeros (Tests for division-by-zero in standard deviation / normalization)
    df_zero = make_synthetic_df(1280, 0.0)
    verify_output(df_zero, model.run(df_zero), "Test 2: Flatlined Zero Signals (Div-by-Zero Check)")

    # Test 3: Micro-Recording (Shorter than a single 10-second window, e.g., 15 rows)
    df_short = make_synthetic_df(15, 25.0)
    verify_output(df_short, model.run(df_short), "Test 3: Sub-Window Micro Stream (15 rows)")

    # Test 4: Odd/Prime Row Count (Tests window slicing & padding alignment: 1003 rows)
    df_odd = make_synthetic_df(1003, 10.0)
    verify_output(df_odd, model.run(df_odd), "Test 4: Non-Divisible Prime Row Count (1,003 rows)")

    # Test 5: Extreme Motion Artifact Spikes (Huge positive/negative outliers)
    df_spikes = make_synthetic_df(1280, 1.0)
    df_spikes.iloc[100:110] = 1e6
    df_spikes.iloc[500:510] = -1e6
    verify_output(df_spikes, model.run(df_spikes), "Test 5: Extreme Artifact Spikes (±1e6)")

    # Test 6: a real recording must be predicted by the model, not by the "Rest" fallback
    real = pd.read_parquet(pathlib.Path(__file__).resolve().parent.parent / "data" / "STRESS" / "S05.parquet")
    model.fallbacks = 0
    out = model.run(real[REQUIRED_COLS])
    verify_output(real[REQUIRED_COLS], out, "Test 6: Real recording (STRESS/S05)")
    assert model.fallbacks == 0, "FAIL: run() fell back to 'Rest' on a real recording (feature or ONNX error)!"
    assert out["Activity"].nunique() > 1, "FAIL: a whole real recording got a single class!"
    print("  [PASS] Test 6b: no fallback, several classes predicted")

    print("\n2. Packaging Submission Zip...")
    src_dir = pathlib.Path(__file__).resolve().parent
    project_root = src_dir.parent

    py_file = src_dir / "your_model.py"
    onnx_file = src_dir / "your_model.onnx"
    zip_path = project_root / "submission.zip"

    assert py_file.exists(), f"Missing {py_file}!"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write files at the root level of the zip (no nested folders)
        zf.write(py_file, arcname="your_model.py")
        if onnx_file.exists():
            zf.write(onnx_file, arcname="your_model.onnx")
            print("  [INFO] Packaged your_model.py + your_model.onnx")
        else:
            print("  [WARN] No your_model.onnx found in src/! Packaged your_model.py only.")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    assert size_mb <= 50.0, f"FAIL: submission.zip is {size_mb:.2f} MB (limit is 50.00 MB)!"

    print(f"  [PASS] Created '{zip_path.name}' ({size_mb:.3f} MB / 50.00 MB)")
    print("=" * 65)
    print(" STATUS: READY FOR AGORIZE UPLOAD")
    print("=" * 65)


if __name__ == "__main__":
    run_torture_tests_and_package()