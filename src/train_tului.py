"""Rebuild TULUI's recipe (leaderboard 74) and train it on more people.

    python train_tului.py holdout   # without the 6 holdout people -> models/tului_re_holdout/ (check vs TULUI)
    python train_tului.py all       # all 35 people -> models/tului35/ + submission.zip (the upload)

TULUI's own training script is not in the repo. Its ONNX metadata says: 145 features (its compute_features,
without t_s / t_frac / t_left / rec_dur), variant "robust" (recordings cut into 1, 2, 5, 10, 30-minute pieces
plus whole), missing values filled with training medians, smoothing 7, log-bias [0, -0.12, 0.2]. This script
follows that recipe with LightGBM, using TULUI's code with the faster, identical-output changes
(src/tului_model.py, shipped as your_model.py).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import experiment as E
from eval_harness import VAL_SUBJECTS

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "src" / "tului_model.py"  # TULUI_fast code = your_model.py of the TULUI35 upload
CHUNK_MINUTES = (1, 2, 5, 10, 30, None)
SMOOTH_K, LOG_BIAS = 7, [0.0, -0.12, 0.2]
PARAMS = dict(n_estimators=550, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced", verbose=-1,
              random_state=0)


def load_code():
    spec = importlib.util.spec_from_file_location("tului_code", CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def windows(mod, files, feats):
    """Feature rows + labels of whole recordings and of pieces of 1-30 minutes (random start, fixed seed)."""
    cols = [mod.ALL_FEATURES.index(f) for f in feats]
    X, y = [], []
    for fp in files:
        df = pd.read_parquet(fp)
        t = (df.index - df.index[0]).total_seconds().to_numpy()
        rng = np.random.default_rng(int(hashlib.md5(f"{fp.parent.name}/{fp.stem}".encode()).hexdigest()[:8], 16))
        for minutes in CHUNK_MINUTES:
            if minutes is None:
                groups = np.zeros(len(df), dtype=np.int64)
            else:
                size = 60.0 * minutes
                groups = np.floor((t + rng.uniform(0, size)) / size).astype(np.int64)
            b = np.flatnonzero(np.diff(groups)) + 1
            for s, e in zip(np.r_[0, b], np.r_[b, len(df)]):
                piece = df.iloc[s:e]
                if len(piece) < 64:
                    continue
                F, rw = mod.compute_features(piece[mod.INPUT_COLS])
                lab, share = E.window_labels(piece, rw, len(F))
                keep = (lab >= 0) & (share >= 0.5)
                X.append(F[keep][:, cols])
                y.append(lab[keep])
    return np.vstack(X).astype(np.float32), np.concatenate(y)


def main(train_on: str):
    import lightgbm as lgb
    import onnxruntime as ort
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    mod = load_code()
    feats = [f for f in mod.ALL_FEATURES if f not in ("t_s", "t_frac", "t_left", "rec_dur")]
    files = [fp for p in E.PROTOCOLS for fp in sorted((E.DATA / p).glob("*.parquet"))
             if train_on == "all" or fp.stem not in VAL_SUBJECTS]
    X, y = windows(mod, files, feats)
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    X = np.where(np.isfinite(X), X, medians)  # TULUI fills missing values with medians at run time
    print(f"{len(files)} recordings -> {len(y):,} windows (whole + pieces), {len(feats)} features")

    model = lgb.LGBMClassifier(**{**PARAMS, "n_jobs": os.cpu_count()}).fit(X, y)
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, len(feats)]))],
                           zipmap=False, target_opset=15)
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"features": feats, "medians": medians.tolist(), "smooth_k": SMOOTH_K,
                             "log_bias": LOG_BIAS, "variant": "robust", "trained_on": train_on,
                             "recipe": "train_tului.py (TULUI rebuilt with LightGBM)"})
    out = ROOT / "models" / ("tului_re_holdout" if train_on == "holdout" else "tului35")
    out.mkdir(parents=True, exist_ok=True)
    (out / "your_model.onnx").write_bytes(onx.SerializeToString())
    shutil.copy(CODE, out / "your_model.py")
    sess = ort.InferenceSession(str(out / "your_model.onnx"), providers=["CPUExecutionProvider"])
    pn = next(o.name for o in sess.get_outputs() if "prob" in o.name)
    diff = np.abs(sess.run([pn], {"X": X[:3000]})[0] - model.predict_proba(X[:3000])).max()
    print(f"saved {out}; max |onnx - lightgbm| {diff:.1e}")
    if train_on == "all":  # the upload: both files at the root of submission.zip
        import zipfile
        zpath = ROOT / "submission.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("your_model.py", "your_model.onnx"):
                z.write(out / f, f)
        print(f"wrote {zpath} (score it first: python score.py tului_re_holdout; then upload)")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("holdout", "all"):
        print("python train_tului.py holdout   |   python train_tului.py all")
        sys.exit(1)
    main(sys.argv[1])
