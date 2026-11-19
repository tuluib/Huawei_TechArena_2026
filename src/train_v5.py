"""Train V5.1: one LightGBM model, no position features, trained on recordings WITH SHUFFLED COPIES.

    python train_v5.py holdout   # without the 6 holdout people -> models/v5_holdout/ (then: python score.py v5_holdout)
    python train_v5.py all       # everyone -> src/your_model.onnx (the submission model)

Why: the official validation recordings "do not follow the training phase order or durations", and
features that exploit the protocol (time since start, phase order) are penalised by the jury. So:
- no t_s / t_frac / t_left / rec_dur features, and no second stage (V4's stage 2 learned the timetable);
- every training recording is used as is AND as 3 copies with its phases shuffled and shortened
  (score.shuffled_recordings, within the same session), so the model cannot learn the phase order.
On the shuffled holdout test this took the same model from 50.4 to 60.7 quality (V3 55.9, TULUI 55.6).
V5.1 adds full-length reordered copies and copies that mix a person's sessions: same score on reordered
recordings, +11 when the mix of phases changes (2026-11-19).
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

sys.path.append(os.path.dirname(__file__))
import experiment as E
import score as S
import your_model as ym
from eval_harness import VAL_SUBJECTS

SHUFFLED_COPIES = 3
SMOOTH_K = 1
PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=60, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced", verbose=-1,
              random_state=0)


def windows(recs, feats):
    """Feature rows + labels of every window that is mostly one scored state."""
    X, y = [], []
    for df in recs:
        F, rw = ym.compute_features(df[ym.INPUT_COLS])
        lab, share = E.window_labels(df, rw, len(F))
        keep = (lab >= 0) & (share >= 0.5)
        X.append(F[keep])
        y.append(lab[keep])
    cols = [ym.ALL_FEATURES.index(f) for f in feats]
    return np.vstack(X)[:, cols].astype(np.float32), np.concatenate(y)


def main(train_on: str):
    import lightgbm as lgb
    import onnxruntime as ort
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    feats = E.feature_sets()["all"]  # every feature except the position ones
    recs = [pd.read_parquet(fp) for p in E.PROTOCOLS for fp in sorted((E.DATA / p).glob("*.parquet"))
            if train_on == "all" or fp.stem not in VAL_SUBJECTS]
    # V5.1: three kinds of shuffled copies, so the model is robust to however Huawei built its recordings:
    # phases reordered and shortened (as in score.py), reordered at full length, and a person's sessions mixed
    # (changes the mix of phases in a recording, which V5 alone handled badly: 41.5 vs 52.6 quality).
    shuffled = (S.shuffled_recordings(recs, seed=123, per_person=SHUFFLED_COPIES, within=True)
                + S.shuffled_recordings(recs, seed=7, per_person=2, within=True, keep=(1.0, 1.0))
                + S.shuffled_recordings(recs, seed=9, per_person=6, within=False))
    X1, y1 = windows(recs, feats)
    X2, y2 = windows(shuffled, feats)
    X, y = np.vstack([X1, X2]), np.concatenate([y1, y2])
    print(f"{len(recs)} recordings ({len(y1):,} windows) + {len(shuffled)} shuffled copies ({len(y2):,} windows), "
          f"{len(feats)} features")

    model = lgb.LGBMClassifier(**{**PARAMS, "n_jobs": os.cpu_count()}).fit(X, y)
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, len(feats)]))],
                           zipmap=False, target_opset=15)
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"version": "v5.1", "features": feats, "smooth_k": SMOOTH_K, "log_bias": [0.0, 0.0, 0.0],
                             "trained_on": train_on, "shuffled_copies": SHUFFLED_COPIES})

    src = pathlib.Path(__file__).resolve().parent
    out = (src.parent / "models" / "v5_holdout" / "your_model.onnx" if train_on == "holdout"
           else src / "your_model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    if out.parent != src:  # make the folder a runnable submission for score.py
        shutil.copy(src / "your_model.py", out.parent / "your_model.py")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    pn = next(o.name for o in sess.get_outputs() if "prob" in o.name)
    diff = np.abs(sess.run([pn], {"X": X[:3000]})[0] - model.predict_proba(X[:3000])).max()
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB); max |onnx - lightgbm| {diff:.1e}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("holdout", "all"):
        print("Say which model to train:\n"
              "  python train_v5.py holdout   # test model -> models/v5_holdout/ (then: python score.py v5_holdout)\n"
              "  python train_v5.py all       # submission model -> src/your_model.onnx")
        sys.exit(1)
    main(sys.argv[1])
