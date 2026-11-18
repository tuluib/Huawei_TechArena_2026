"""Train the v2 model: cleaned-signal features + gradient-boosted trees, exported to ONNX.

    python train_v2.py cv                      # person-grouped cross-validation report
    python train_v2.py export full holdout     # train without the 6 validation people (honest local score)
    python train_v2.py export full all         # train on everyone (for the real submission)
    python train_v2.py export safe all         # same without the time-position features

Variants: "full" uses every feature; "safe" drops t_s / t_frac / t_left / rec_dur (no reliance on
the fixed protocol timetable).
Smoothing width and class bias are tuned on out-of-fold predictions and stored in the ONNX metadata.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.dirname(__file__))
import your_model as ym
from eval_harness import VAL_SUBJECTS, map_phase_to_activity

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "_cache" / "v2_features.parquet"
PROTOCOLS = ["AEROBIC", "ANAEROBIC", "STRESS"]
LABEL_TO_INT = {"Rest": 0, "Activity": 1, "Stress": 2}
TIME_ONLY = ["t_s", "t_frac", "t_left", "rec_dur"]
VARIANTS = {
    "full": ym.ALL_FEATURES,
    "safe": [f for f in ym.ALL_FEATURES if f not in TIME_ONLY],
}
PARAMS = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
              l2_regularization=1.0, max_features=0.5, class_weight="balanced", random_state=0)


def window_labels(df: pd.DataFrame, row_wid: np.ndarray, n: int):
    """Scorer-style label per window: most common scored state (ties: first by np.unique order)."""
    act = map_phase_to_activity(df["Phase"]).map(LABEL_TO_INT).to_numpy(dtype=np.float64)
    ok = np.isfinite(act)
    votes = np.stack([np.bincount(row_wid[ok & (act == c)], minlength=n)[:n] for c in range(3)], axis=1)
    total = np.bincount(row_wid, minlength=n)[:n]
    scored = votes.sum(axis=1)
    # np.unique sorts the label strings: Activity < Rest < Stress -> argmax tie order
    order = [1, 0, 2]
    lab = np.array(order)[votes[:, order].argmax(axis=1)]
    lab = np.where(scored > 0, lab, -1)
    share = scored / np.maximum(total, 1)
    return lab, share


def build_dataset(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)
    parts = []
    for p in PROTOCOLS:
        for fp in sorted((DATA / p).glob("*.parquet")):
            df = pd.read_parquet(fp)
            X, row_wid = ym.compute_features(df[ym.INPUT_COLS])
            lab, share = window_labels(df, row_wid, len(X))
            part = pd.DataFrame(X, columns=ym.ALL_FEATURES)
            part["y"], part["share"] = lab, share
            part["user"], part["rec"], part["win"] = fp.stem, f"{p}/{fp.stem}", np.arange(len(X))
            parts.append(part[part["y"] >= 0])
    data = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(exist_ok=True)
    data.to_parquet(CACHE)
    return data


def fit(tr: pd.DataFrame, feats, params=None):
    m = HistGradientBoostingClassifier(**(params or PARAMS))
    tr = tr[tr["share"] >= 0.5]  # train only on windows that are mostly one scored state
    return m.fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy())


def smooth_by_rec(df: pd.DataFrame, P: np.ndarray, k: int) -> np.ndarray:
    out = P.copy()
    for _, ix in df.groupby("rec", sort=False).indices.items():
        out[ix] = ym.smooth_probs(P[ix], k)  # rows are in window order within each recording
    return out


def oof_probs(data: pd.DataFrame, feats, params=None, folds: int = 7) -> np.ndarray:
    P = np.zeros((len(data), 3))
    for a, b in GroupKFold(folds).split(data, groups=data["user"]):
        m = fit(data.iloc[a], feats, params)
        P[b] = m.predict_proba(data.iloc[b][feats].to_numpy(np.float32))
    return P


def macro(y, pred):
    """Macro-F1 over the 3 classes (same as sklearn's, via a confusion matrix: much faster)."""
    cm = np.bincount(np.asarray(y) * 3 + np.asarray(pred), minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def tune(data: pd.DataFrame, P: np.ndarray):
    """Best smoothing width and log-bias on out-of-fold probabilities (coarse grid)."""
    y = data["y"].to_numpy()
    best = (-1, 1, np.zeros(3))
    grid = np.arange(-1.0, 1.01, 0.25)
    for k in (1, 3, 5, 7, 9, 13):
        L = np.log(np.maximum(smooth_by_rec(data, P, k), 1e-9))
        for b1 in grid:          # Activity
            for b2 in grid:      # Stress
                bias = np.array([0.0, b1, b2])
                s = macro(y, (L + bias).argmax(1))
                if s > best[0]:
                    best = (s, k, bias)
    return best


def cv_report():
    data = build_dataset()
    print(f"{len(data):,} windows, {data['user'].nunique()} people, classes "
          f"{np.bincount(data['y'], minlength=3).tolist()} (Rest, Activity, Stress)")
    y = data["y"].to_numpy()
    for name, feats in VARIANTS.items():
        t0 = time.time()
        P = oof_probs(data, feats)
        raw = macro(y, P.argmax(1))
        s, k, bias = tune(data, P)
        per = f1_score(y, (np.log(np.maximum(smooth_by_rec(data, P, k), 1e-9)) + bias).argmax(1),
                       labels=[0, 1, 2], average=None)
        print(f"[{name}] 7-fold person CV macro-F1: raw {raw:.3f} -> tuned {s:.3f} (k={k}, bias={bias.tolist()}) "
              f"per class R/A/S {per.round(3).tolist()}  [{time.time() - t0:.0f}s]")


def export(variant: str, train_on: str):
    data = build_dataset()
    feats = VARIANTS[variant]
    # tune post-processing on out-of-fold predictions of the training people only
    tr = data if train_on == "all" else data[~data["user"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    P = oof_probs(tr, feats)
    s, k, bias = tune(tr, P)
    print(f"[{variant}/{train_on}] OOF macro-F1 after tuning {s:.3f} (k={k}, bias={bias.tolist()})")
    export_trained(tr, feats, k, bias, s, variant, train_on)


def export_trained(tr, feats, k, bias, s, variant, train_on):
    """Fit on tr, convert to ONNX with the post-processing settings in its metadata, check it.

    Output path: env EH_ONNX_OUT, default src/your_model.onnx.
    """
    model = fit(tr, feats)

    from skl2onnx import to_onnx
    import skl2onnx.common.tree_ensemble as te
    if not getattr(te, "_eh_patched", False):  # newer sklearn stores the missing-value flag as bool
        _orig = te.add_node

        def _add_node(*a, nodes_missing_value_tracks_true=0, **kw):
            return _orig(*a, nodes_missing_value_tracks_true=int(nodes_missing_value_tracks_true), **kw)
        te.add_node, te._eh_patched = _add_node, True
    onx = to_onnx(model, tr[feats].to_numpy(np.float32)[:1],
                  options={id(model): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
    meta = onx.metadata_props.add()
    meta.key = "eh_config"
    meta.value = json.dumps({"features": feats, "smooth_k": int(k), "log_bias": bias.tolist(),
                             "variant": variant, "trained_on": train_on, "oof_macro_f1": round(float(s), 4)})
    out = pathlib.Path(os.environ.get("EH_ONNX_OUT", ROOT / "src" / "your_model.onnx"))
    out.write_bytes(onx.SerializeToString())

    # sanity check: ONNX probabilities == sklearn probabilities
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    Xs = tr[feats].to_numpy(np.float32)[:2000]
    pn = next(o.name for o in sess.get_outputs() if "prob" in o.name.lower())
    diff = np.abs(sess.run([pn], {sess.get_inputs()[0].name: Xs})[0] - model.predict_proba(Xs)).max()
    print(f"Saved {out} ({out.stat().st_size / 1e6:.2f} MB); max |onnx - sklearn| prob diff = {diff:.2e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        cv_report()
    elif cmd == "export":
        export(sys.argv[2], sys.argv[3])
    elif cmd == "rebuild":
        build_dataset(use_cache=False)
        print("features rebuilt")
