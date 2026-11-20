"""V5.1 training recipe plus 11 cross-sensor physiological interactions.

This tests whether TULUI's interaction features help the shuffled-training recipe that produced 73.
The candidate is saved separately and trained without the six-person holdout.

    python experiments/train_v7_interactions.py holdout
    python experiments/train_v7_interactions.py all
    python src/score.py v7_holdout --quick
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E
import score as S
import your_model as YM
from eval_harness import VAL_SUBJECTS


PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=60,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
              class_weight="balanced", verbose=-1, random_state=0)


def make_windows(recs, feats):
    xs, ys = [], []
    cols = [YM.ALL_FEATURES.index(f) for f in feats]
    for df in recs:
        F, row_wid = YM.compute_features(df[YM.INPUT_COLS])
        lab, share = E.window_labels(df, row_wid, len(F))
        keep = (lab >= 0) & (share >= 0.5)
        xs.append(F[keep][:, cols].astype(np.float32))
        ys.append(lab[keep])
    return np.vstack(xs), np.concatenate(ys)


def main(train_on: str):
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    feats = E.feature_sets()["all"]
    if train_on not in {"holdout", "all"}:
        raise ValueError("train_on must be 'holdout' or 'all'")
    recs = [pd.read_parquet(fp) for protocol in E.PROTOCOLS
            for fp in sorted((E.DATA / protocol).glob("*.parquet"))
            if train_on == "all" or fp.stem not in VAL_SUBJECTS]
    n_source = len(recs)
    shuffled = (S.shuffled_recordings(recs, seed=123, per_person=3, within=True)
                + S.shuffled_recordings(recs, seed=7, per_person=2, within=True, keep=(1.0, 1.0))
                + S.shuffled_recordings(recs, seed=9, per_person=6, within=False))
    n_augmented = len(shuffled)
    X2, y2 = make_windows(shuffled, feats)
    del shuffled
    X1, y1 = make_windows(recs, feats)
    del recs
    X, y = np.vstack((X1, X2)), np.concatenate((y1, y2))
    print(f"{n_source} source + {n_augmented} augmented recordings; {len(y):,} windows; "
          f"{len(feats)} features", flush=True)

    model = lgb.LGBMClassifier(**{**PARAMS, "n_jobs": os.cpu_count() or 1}).fit(X, y)
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, len(feats)]))],
                           zipmap=False, target_opset=15)
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"version": "v7-interactions", "features": feats, "smooth_k": 1,
                             "log_bias": [0.0, 0.0, 0.0], "trained_on": train_on,
                             "training": "v5.1 augmentations + TULUI physiological interactions"})
    out = ROOT / "models" / f"v7_{train_on}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "your_model.onnx").write_bytes(onx.SerializeToString())
    shutil.copy(ROOT / "src" / "your_model.py", out / "your_model.py")
    print(f"saved {out / 'your_model.onnx'} ({(out / 'your_model.onnx').stat().st_size / 1e6:.2f} MB)")
    if train_on == "all":
        import zipfile
        package = ROOT / "submissions" / "candidates" / "V7_interactions.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out / "your_model.py", arcname="your_model.py")
            z.write(out / "your_model.onnx", arcname="your_model.onnx")
        print(f"submission candidate: {package} ({package.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "holdout")
