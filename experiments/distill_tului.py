"""Compress TULUI into a smaller LightGBM student using teacher predictions.

This experiment targets the 20-point runtime component while preserving TULUI's decision boundary.
Training uses only the 29 non-holdout people. Soft teacher targets are mixed with ground-truth labels
to reduce hard-label overfitting; both are generated on original recordings and within-session shuffles.

    python experiments/distill_tului.py 300
    python src/score.py tului_distilled_300 --quick
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import score as S
from eval_harness import VAL_SUBJECTS


def load_teacher():
    folder = ROOT / "models" / "tului_fast"
    spec = importlib.util.spec_from_file_location("tului_teacher", folder / "your_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.get_model()
    meta = S.model_info(model)
    if meta.get("trained_on") != "holdout":
        raise RuntimeError("TULUI teacher must be trained without the holdout people")
    return model, module, meta


def main(n_estimators: int):
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    teacher, module, meta = load_teacher()
    source = [pd.read_parquet(p) for protocol in S.PROTOCOLS
              for p in sorted((S.DATA / protocol).glob("*.parquet")) if p.stem not in VAL_SUBJECTS]
    shuffled = S.shuffled_recordings(source, seed=123, per_person=2, within=True)
    train_recs = source + shuffled
    Xs, targets, teacher_labels, true_labels = [], [], [], []
    for i, df in enumerate(train_recs):
        F, row_wid = module.compute_features(df[module.INPUT_COLS])
        # Use the scorer's row-to-window mapping to omit undefined/transition windows.
        from experiment import window_labels
        lab, share = window_labels(df, row_wid, len(F))
        keep = (lab >= 0) & (share >= 0.5)
        if not keep.any():
            continue
        X_teacher = np.ascontiguousarray(F[:, teacher.cols])
        P = teacher.session.run([teacher.prob_name], {teacher.input_name: X_teacher})[0]
        P = module.smooth_probs(np.asarray(P, dtype=np.float64), teacher.smooth_k)
        scores = np.log(np.maximum(P, 1e-9)) + teacher.log_bias
        scores -= scores.max(axis=1, keepdims=True)
        P = np.exp(scores)
        P /= P.sum(axis=1, keepdims=True)
        Xs.append(F[keep][:, teacher.cols].astype(np.float32))
        target = 0.5 * P[keep] + 0.5 * np.eye(3, dtype=np.float64)[lab[keep]]
        targets.append(target.astype(np.float32))
        teacher_labels.append(P[keep].argmax(axis=1).astype(np.int64))
        true_labels.append(lab[keep].astype(np.int64))
        if (i + 1) % 20 == 0:
            print(f"teacher-labeled {i+1}/{len(train_recs)} recordings", flush=True)
    X, target = np.vstack(Xs), np.vstack(targets)
    teacher_y, true_y = np.concatenate(teacher_labels), np.concatenate(true_labels)
    print(f"Distillation set: {len(target):,} windows; teacher counts "
          f"{np.bincount(teacher_y, minlength=3).tolist()}; truth counts "
          f"{np.bincount(true_y, minlength=3).tolist()}")

    student = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
                                 n_jobs=os.cpu_count() or 1, random_state=0)
    X_weighted = np.repeat(X, 3, axis=0)
    y_weighted = np.tile(np.arange(3, dtype=np.int64), len(X))
    student.fit(X_weighted, y_weighted, sample_weight=target.reshape(-1))
    onx = convert_lightgbm(student,
                           initial_types=[("X", FloatTensorType([None, len(meta["features"])]))],
                           zipmap=False, target_opset=15)
    p = student.predict_proba(X[:5000])
    meta_out = onx.metadata_props.add()
    meta_out.key = "eh_config"
    meta_out.value = json.dumps({"version": f"tului-distill-soft-{n_estimators}",
                                 "features": meta["features"], "smooth_k": 1,
                                 "log_bias": [0.0, 0.0, 0.0], "trained_on": "holdout",
                                 "teacher": "TULUI", "n_estimators": n_estimators})
    out = ROOT / "models" / f"tului_distilled_soft_{n_estimators}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "your_model.onnx").write_bytes(onx.SerializeToString())
    shutil.copy(module.__file__, out / "your_model.py")
    print(f"train agreement: teacher={(student.predict(X[:5000]) == teacher_y[:5000]).mean():.4f}, "
          f"truth={(student.predict(X[:5000]) == true_y[:5000]).mean():.4f}; "
          f"probabilities shape {p.shape}")
    print(f"saved {out / 'your_model.onnx'} ({(out / 'your_model.onnx').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
