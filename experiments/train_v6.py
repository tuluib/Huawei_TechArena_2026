"""Train a one-stage LightGBM candidate on original and windowed recordings.

This isolates V5.1's shuffle augmentation as a variable while retaining the current 186-feature
family. It is an experiment, not a jury-ready model: its feature set contains future/global context.

    python experiments/train_v6.py holdout
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E
from eval_harness import VAL_SUBJECTS


def main(train_on: str):
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    if train_on not in {"holdout", "all"}:
        raise ValueError("train_on must be 'holdout' or 'all'")
    data = E.build()
    if train_on == "holdout":
        data = data[~data["user"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    features = E.feature_sets()["all"]
    train = data[data["share"].to_numpy() >= 0.5]
    X = train[features].to_numpy(np.float32)
    y = train["y"].to_numpy()
    params = {**E.LGB_PARAMS, "n_jobs": os.cpu_count() or 1}
    print(f"Training {len(y):,} windows from {train['user'].nunique()} people; {len(features)} features")
    model = lgb.LGBMClassifier(**params).fit(X, y)
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, len(features)]))],
                           zipmap=False, target_opset=15)
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"version": "v6-no-shuffle", "features": features,
                             "smooth_k": 5, "log_bias": [0.0, -0.25, 0.25],
                             "trained_on": train_on, "training_data": "person-windowed-original"})
    out = ROOT / "models" / ("v6_holdout" if train_on == "holdout" else "v6_full")
    out.mkdir(parents=True, exist_ok=True)
    (out / "your_model.onnx").write_bytes(onx.SerializeToString())
    shutil.copy(ROOT / "src" / "your_model.py", out / "your_model.py")
    print(f"Saved {out / 'your_model.onnx'} ({(out / 'your_model.onnx').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "holdout")
