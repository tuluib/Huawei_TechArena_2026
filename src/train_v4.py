"""Train and export the two-stage model (V4) into a single ONNX file.

    python train_v4.py holdout [SET]   # without the 6 holdout people -> models/holdout/ (for score.py)
    python train_v4.py all [SET]       # everyone -> src/your_model.onnx (the submission model)

Stage 1: LightGBM on the window features (SET from experiment.feature_sets(), default "all+time").
Stage 2: LightGBM on the stage-1 probabilities of the surrounding windows (your_model.stage2_features).
Stage 2 is trained on OUT-OF-FOLD stage-1 probabilities (5-fold by person), so it learns from the kind
of predictions stage 1 makes on people it has never seen, which is the situation at test time.

Both models go into one ONNX file (inputs s1_X / s2_X). The feature list, smoothing width and class
bias are stored in its metadata. Output: `holdout` -> models/holdout/ (with a copy of your_model.py, so
score.py can run it); `all` -> src/your_model.onnx. Env EH_ONNX_OUT overrides the .onnx path.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))
import experiment as E
import your_model as ym
from eval_harness import VAL_SUBJECTS


def to_onnx(model, n_features: int, prefix: str):
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_lightgbm(model, initial_types=[("X", FloatTensorType([None, n_features]))],
                           zipmap=False, target_opset=15)
    return onnx.compose.add_prefix(onx, prefix)


def main(train_on: str, set_name: str = "all+time", s1_params=None, s2_params=None):
    import lightgbm as lgb
    import onnx
    import onnxruntime as ort

    data = E.build()
    if train_on == "holdout":
        data = data[~data["user"].isin(VAL_SUBJECTS)].reset_index(drop=True)
        data["rec"] = data["rec"].cat.remove_unused_categories()
    feats = E.feature_sets()[set_name]
    X = data[feats].to_numpy(np.float32)
    y = data["y"].to_numpy()
    ok = data["share"].to_numpy() >= 0.5
    print(f"{len(data):,} windows from {data['user'].nunique()} people, {len(feats)} features")

    s1 = {**E.LGB_PARAMS, **(s1_params or {}), "n_jobs": os.cpu_count()}
    s2 = {**E.LGB_PARAMS, **E.S2_PARAMS, **(s2_params or {}), "n_jobs": os.cpu_count()}
    P = E.oof_probs(data, feats, s1_params)                 # stage-1 out-of-fold
    m1 = lgb.LGBMClassifier(**s1).fit(X[ok], y[ok])         # stage 1 on everyone
    Z = np.vstack([ym.stage2_features(P[s:e]) for s, e in E.rec_slices(data)])
    m2 = lgb.LGBMClassifier(**s2).fit(Z[ok], y[ok])
    Q = E.stage2_oof(data, P)                               # for tuning smoothing + bias only
    score, k, bias = E.tune(data, Q, ks=(1, 3, 5))
    print(f"stage-2 OOF macro-F1 {score:.4f} (k={k}, bias={bias.tolist()})")

    merged = onnx.compose.merge_models(to_onnx(m1, len(feats), "s1_"), to_onnx(m2, Z.shape[1], "s2_"), io_map=[])
    meta = merged.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"two_stage": True, "features": feats, "smooth_k": int(k), "log_bias": bias.tolist(),
                             "set": set_name, "trained_on": train_on, "oof_macro_f1": round(float(score), 4)})
    onnx.checker.check_model(merged)
    src = pathlib.Path(__file__).resolve().parent
    default = (src.parent / "models" / "holdout" / "your_model.onnx" if train_on == "holdout"
               else src / "your_model.onnx")
    out = pathlib.Path(os.environ.get("EH_ONNX_OUT", default))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(merged.SerializeToString())
    if out.parent != src:  # make the folder a runnable submission for score.py
        shutil.copy(src / "your_model.py", out.parent / "your_model.py")

    # parity: ONNX == LightGBM for both stages (NaNs included)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    e1 = np.zeros((0, len(feats)), np.float32)
    e2 = np.zeros((0, Z.shape[1]), np.float32)
    d1 = np.abs(sess.run(["s1_probabilities"], {"s1_X": X[:3000], "s2_X": e2})[0] - m1.predict_proba(X[:3000])).max()
    d2 = np.abs(sess.run(["s2_probabilities"], {"s1_X": e1, "s2_X": Z[:3000]})[0] - m2.predict_proba(Z[:3000])).max()
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB); max |onnx - lightgbm| stage 1 {d1:.1e}, stage 2 {d2:.1e}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("holdout", "all"):
        print("Say which model to train:\n"
              "  python train_v4.py holdout   # test model, without the 6 holdout people -> models/holdout/ (then score.py)\n"
              "  python train_v4.py all       # submission model, everyone -> src/your_model.onnx")
        sys.exit(1)
    main(sys.argv[1], *(sys.argv[2:3] or []))
