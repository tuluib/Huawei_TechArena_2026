import os
import sys
import time
import pandas as pd

# This tells Python to look inside the exact same folder for your_model.py
sys.path.append(os.path.dirname(__file__))
import your_model

# Lock to a single CPU core to simulate Huawei's environment
os.environ["OMP_NUM_THREADS"] = "1"

def run_test_bench():
    print("--- Starting Local Test Bench ---")
    
    # Load the team's custom chip
    model = your_model.get_model()
    
    # Create 1,000 rows of fake sensor data to test the speed
    print("Generating synthetic test data...")
    fake_time_index = pd.date_range(start="2026-01-01", periods=1000, freq="ms")
    fake_data = pd.DataFrame({
        "BVP": [0.0] * 1000,
        "EDA": [0.0] * 1000,
        "HR": [60.0] * 1000,
        "IBI_dt": [0.0] * 1000,
        "IBI_duration": [0.0] * 1000,
        "Temperature": [36.5] * 1000
    }, index=fake_time_index)

    # Time the execution (simulating Huawei's 50-run MET benchmark)
    print("Running speed benchmark...")
    run_times = []
    
    for _ in range(50):
        start_time = time.perf_counter()
        output_df = model.run(fake_data)
        end_time = time.perf_counter()
        run_times.append(end_time - start_time)

    avg_time_seconds = sum(run_times) / len(run_times)
    
    print("\n--- TEST RESULTS ---")
    print(f"Output DataFrame Shape: {output_df.shape}")
    print(f"Output Columns: {list(output_df.columns)}")
    print(f"Mean Execution Time: {avg_time_seconds:.4f} seconds")
    
    if avg_time_seconds > 0.3:
        print("[FAIL] Model is too slow! Must be under 0.3 seconds.")
    else:
        print("[PASS] Execution speed is within Huawei limits.")

if __name__ == "__main__":
    run_test_bench()