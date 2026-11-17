import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx

from eval_harness import VAL_SUBJECTS, PROTOCOLS, REQUIRED_INPUT_COLS, map_phase_to_activity

BLOCK_SIZE = 640  # 10 seconds at 64 Hz
LABEL_TO_INT = {"Rest": 0, "Activity": 1, "Stress": 2}


def extract_block_features(df_input: pd.DataFrame) -> np.ndarray:
    arr = df_input[REQUIRED_INPUT_COLS].ffill().fillna(0.0).to_numpy(dtype=np.float32)
    arr = np.clip(arr, -1e4, 1e4)

    n_rows, n_cols = arr.shape
    n_blocks = max(1, int(np.ceil(n_rows / BLOCK_SIZE)))
    pad_len = (n_blocks * BLOCK_SIZE) - n_rows
    if pad_len > 0:
        arr = np.pad(arr, ((0, pad_len), (0, 0)), mode="edge")

    blocks = arr.reshape(n_blocks, BLOCK_SIZE, n_cols)
    means = blocks.mean(axis=1)
    stds = blocks.std(axis=1)
    return np.hstack([means, stds]).astype(np.float32)


def train_and_export():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    onnx_out_path = project_root / "src" / "your_model.onnx"

    X_list, y_list = [], []
    print("Loading training subjects (excluding validation holdout set)...")

    for protocol in PROTOCOLS:
        folder = data_dir / protocol
        if not folder.exists():
            continue
        for file_path in sorted(folder.glob("*.parquet")):
            subj = file_path.stem
            if subj in VAL_SUBJECTS:
                continue

            df = pd.read_parquet(file_path)
            feats = extract_block_features(df)

            mapped_phases = map_phase_to_activity(df["Phase"]).to_numpy()
            n_blocks = len(feats)

            for b in range(n_blocks):
                chunk = mapped_phases[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
                # Properly filter out both None and float NaN
                valid = [x for x in chunk if x in LABEL_TO_INT]
                if len(valid) > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    mode_label = vals[np.argmax(counts)]
                    X_list.append(feats[b])
                    y_list.append(LABEL_TO_INT[mode_label])

    X_train = np.vstack(X_list).astype(np.float32)
    y_train = np.array(y_list, dtype=np.int64)
    print(f"Training Baseline Random Forest on {len(X_train):,} valid ten-second windows...")

    clf = RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    initial_sample = X_train[:1]
    onx = to_onnx(
        clf,
        initial_sample,
        options={id(clf): {"zipmap": False}},
        target_opset=15,
    )
    with open(onnx_out_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"[SUCCESS] Saved trained ONNX model to: {onnx_out_path}")


if __name__ == "__main__":
    train_and_export()