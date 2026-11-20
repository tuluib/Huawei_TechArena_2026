"""Train a TULUI-feature LightGBM with V5.1/V7's shuffle augmentation.

This isolates the augmentation recipe from V7's extra feature columns. It uses
TULUI's own feature implementation and exact feature list, then trains with the
same subject split and augmentation generators used by the V7 experiment.

    python experiments/train_tului_aug.py holdout
    python experiments/train_tului_aug.py holdout wide
    python experiments/train_tului_aug.py holdout deep
    python src/score.py models/tului_aug_holdout TULUI --quick
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E
import score as S
from eval_harness import VAL_SUBJECTS

PARAMS = {
    "base": dict(n_estimators=400, learning_rate=0.04, num_leaves=15),
    "wide": dict(n_estimators=500, learning_rate=0.04, num_leaves=31),
    "deep": dict(n_estimators=800, learning_rate=0.025, num_leaves=15),
}
COMMON_PARAMS = dict(min_child_samples=60, subsample=0.7, subsample_freq=1,
                     colsample_bytree=0.5, reg_lambda=1.0,
                     class_weight="balanced", verbose=-1, random_state=0)


def load_tului():
    path = ROOT / "models" / "tului_fast" / "your_model.py"
    spec = importlib.util.spec_from_file_location("tului_aug_features", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.get_model()
    config = json.loads(model.session.get_modelmeta().custom_metadata_map["eh_config"])
    return module, config["features"]


def make_windows(recs, module, features):
    cols = [module.ALL_FEATURES.index(name) for name in features]
    xs, ys = [], []
    for df in recs:
        matrix, row_wid = module.compute_features(df[module.INPUT_COLS])
        label, share = E.window_labels(df, row_wid, len(matrix))
        keep = (label >= 0) & (share >= 0.5)
        xs.append(matrix[keep][:, cols].astype(np.float32))
        ys.append(label[keep])
    return np.vstack(xs), np.concatenate(ys)


def main(train_on: str, variant: str = "base"):
    import lightgbm as lgb
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    if train_on not in {"holdout", "all"}:
        raise ValueError("train_on must be 'holdout' or 'all'")
    if variant not in PARAMS:
        raise ValueError(f"variant must be one of {sorted(PARAMS)}")
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    module, features = load_tului()
    recs = [pd.read_parquet(fp) for protocol in E.PROTOCOLS
            for fp in sorted((E.DATA / protocol).glob("*.parquet"))
            if train_on == "all" or fp.stem not in VAL_SUBJECTS]
    n_source = len(recs)
    shuffled = (S.shuffled_recordings(recs, seed=123, per_person=3, within=True)
                + S.shuffled_recordings(recs, seed=7, per_person=2, within=True,
                                        keep=(1.0, 1.0))
                + S.shuffled_recordings(recs, seed=9, per_person=6, within=False))
    n_aug = len(shuffled)
    X_aug, y_aug = make_windows(shuffled, module, features)
    del shuffled
    X_src, y_src = make_windows(recs, module, features)
    del recs
    X, y = np.vstack((X_src, X_aug)), np.concatenate((y_src, y_aug))
    print(f"{n_source} source + {n_aug} augmented recordings; {len(features)} features; "
          f"{len(y):,} windows", flush=True)

    params = {**COMMON_PARAMS, **PARAMS[variant], "n_jobs": os.cpu_count() or 1}
    model = lgb.LGBMClassifier(**params).fit(X, y)
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, len(features)]))],
                           zipmap=False, target_opset=15)
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"version": f"tului-features-shuffle-aug-{variant}", "features": features,
                             "smooth_k": 1, "log_bias": [0.0, 0.0, 0.0],
                             "trained_on": train_on,
                             "training": f"V5.1 shuffle augmentation on TULUI features ({variant})"})
    suffix = "" if variant == "base" else f"_{variant}"
    out = ROOT / "models" / f"tului_aug{suffix}_{train_on}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "your_model.onnx").write_bytes(onx.SerializeToString())
    (out / "your_model.py").write_bytes((ROOT / "models" / "tului_fast" / "your_model.py").read_bytes())
    print(f"saved {out / 'your_model.onnx'} ({(out / 'your_model.onnx').stat().st_size / 1e6:.2f} MB)",
          flush=True)
    if train_on == "all":
        package = ROOT / "submissions" / "candidates" / f"TULUI_shuffle_aug{suffix}.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(out / "your_model.py", arcname="your_model.py")
            z.write(out / "your_model.onnx", arcname="your_model.onnx")
        print(f"submission candidate: {package} ({package.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "holdout",
         sys.argv[2] if len(sys.argv) > 2 else "base")
