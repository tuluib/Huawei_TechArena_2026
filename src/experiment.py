"""Experiment pipeline: feature cache, person-grouped cross-validation, post-processing, logging.

The features come from your_model.compute_features (the same code the submission runs), so the
training data can never drift from what run() computes.

    python experiment.py build                 # feature cache for the current your_model.py (~1 min)
    python experiment.py cv [SET] [--tag T]    # 5-fold person CV with LightGBM, logged to experiments/log.csv
    python experiment.py ablate [SET]          # drop one feature group at a time
    python experiment.py importance [SET]      # LightGBM gain importance per feature

SET is a key of FEATURE_SETS (default "all"). Every number is out-of-fold: the person being scored
was never seen in training. Recordings are scored whole and cut into 1/2/5/10/30-minute pieces,
because we do not know how the hidden test calls run().
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
import your_model as ym
from eval_harness import map_phase_to_activity

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE_DIR = DATA / "_cache"
LOG = ROOT / "experiments" / "log.csv"
PROTOCOLS = ["AEROBIC", "ANAEROBIC", "STRESS"]
CLASS_NAMES = ["Rest", "Activity", "Stress"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_NAMES)}
CHUNK_MINUTES = (1, 2, 5, 10, 30, None)  # None = whole recording
TIME_ONLY = {"t_s", "t_frac", "t_left", "rec_dur"}  # meaningless when run() gets pieces

LGB_PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=60,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  class_weight="balanced", verbose=-1, random_state=0)


def feature_sets() -> dict[str, list[str]]:
    base = [f for f in ym.ALL_FEATURES if f not in TIME_ONLY]
    v3 = [f for f in base if f in V3_FEATURES]
    return {"all": base, "v3": v3, "all+time": base + sorted(TIME_ONLY), "v3+time": v3 + sorted(TIME_ONLY)}


# The 134 features of the V3 submission (to compare new features against).
V3_FEATURES = set(json.loads((ROOT / "experiments" / "v3_features.json").read_text()))


# --------------------------------------------------------------------------------------------
# Feature cache
# --------------------------------------------------------------------------------------------
CACHE_VERSION = "2"  # bump when the cache layout below changes


def code_hash() -> str:
    src = pathlib.Path(ym.__file__).read_bytes() + CACHE_VERSION.encode()
    return hashlib.sha1(src).hexdigest()[:10]


def window_phase(df: pd.DataFrame, row_wid: np.ndarray, n: int) -> np.ndarray:
    """Most common raw Phase name per window (for error analysis only)."""
    codes, names = pd.factorize(df["Phase"].astype(str))
    votes = np.zeros((n, len(names)), dtype=np.int64)
    np.add.at(votes, (row_wid, codes), 1)
    return np.asarray(names, dtype=object)[votes.argmax(axis=1)]


def window_labels(df: pd.DataFrame, row_wid: np.ndarray, n: int):
    """Scorer-style label per window (most common scored state) and the scored share of the window."""
    act = map_phase_to_activity(df["Phase"]).map(LABEL_TO_INT).to_numpy(dtype=np.float64)
    ok = np.isfinite(act)
    votes = np.stack([np.bincount(row_wid[ok & (act == c)], minlength=n)[:n] for c in range(3)], axis=1)
    total = np.bincount(row_wid, minlength=n)[:n]
    order = [1, 0, 2]  # the scorer's np.unique sorts strings: Activity < Rest < Stress (tie order)
    lab = np.array(order)[votes[:, order].argmax(axis=1)]
    lab = np.where(votes.sum(axis=1) > 0, lab, -1)
    return lab, votes.sum(axis=1) / np.maximum(total, 1)


def _recording_rows(path: str) -> pd.DataFrame:
    import warnings
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    fp = pathlib.Path(path)
    protocol, user = fp.parent.name, fp.stem
    df = pd.read_parquet(fp)
    t = (df.index - df.index[0]).total_seconds().to_numpy()
    seed = int(hashlib.md5(f"{protocol}/{user}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    parts = []
    for minutes in CHUNK_MINUTES:
        if minutes is None:
            groups = np.zeros(len(df), dtype=np.int64)
        else:
            size = 60.0 * minutes  # random phase: piece borders fall anywhere
            groups = np.floor((t + rng.uniform(0, size)) / size).astype(np.int64)
        bounds = np.flatnonzero(np.diff(groups)) + 1
        for ci, (s, e) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, len(df)])):
            piece = df.iloc[s:e]
            if len(piece) < 64:
                continue
            X, row_wid = ym.compute_features(piece[ym.INPUT_COLS])
            lab, share = window_labels(piece, row_wid, len(X))
            part = pd.DataFrame(X, columns=ym.ALL_FEATURES)
            part["y"], part["share"] = lab, share
            part["user"], part["protocol"] = user, protocol
            part["rec"] = f"{protocol}/{user}/{minutes}/{ci}"
            part["chunk"] = -1 if minutes is None else minutes
            part["win"] = np.arange(len(X))
            part["phase"] = window_phase(piece, row_wid, len(X))
            parts.append(part[part["y"] >= 0])
    return pd.concat(parts, ignore_index=True)


def build(force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"exp_{code_hash()}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    files = [str(fp) for p in PROTOCOLS for fp in sorted((DATA / p).glob("*.parquet"))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        parts = list(ex.map(_recording_rows, files))
    data = pd.concat(parts, ignore_index=True)
    for c in ("user", "protocol", "rec", "phase"):
        data[c] = data[c].astype("category")
    CACHE_DIR.mkdir(exist_ok=True)
    data.to_parquet(path)
    print(f"built {path.name}: {len(data):,} windows from {len(files)} recordings in {time.time() - t0:.0f}s")
    return data


# --------------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------------
def person_folds(data: pd.DataFrame, k: int = 5):
    """Fixed person -> fold assignment (balanced by window count) so experiments are paired."""
    sizes = data.groupby("user", observed=True).size().sort_values(ascending=False)
    load = np.zeros(k)
    fold_of = {}
    for u, n in sizes.items():
        f = int(load.argmin())
        fold_of[u] = f
        load[f] += n
    f = data["user"].map(fold_of).to_numpy()
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def oof_probs(data: pd.DataFrame, feats, params=None, folds=None, n_jobs: int = 0) -> np.ndarray:
    import lightgbm as lgb
    params = {**LGB_PARAMS, **(params or {}), "n_jobs": n_jobs or os.cpu_count()}
    P = np.zeros((len(data), 3))
    X = data[feats].to_numpy(np.float32)
    y = data["y"].to_numpy()
    train_ok = data["share"].to_numpy() >= 0.5  # windows that are mostly one scored state
    for a, b in folds or person_folds(data):
        a = a[train_ok[a]]
        m = lgb.LGBMClassifier(**params).fit(X[a], y[a])
        P[b] = m.predict_proba(X[b])
    return P


# --------------------------------------------------------------------------------------------
# Post-processing and scoring
# --------------------------------------------------------------------------------------------
def macro(y, pred) -> float:
    cm = np.bincount(np.asarray(y) * 3 + np.asarray(pred), minlength=9).reshape(3, 3).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * np.diag(cm) / np.maximum(denom, 1), 0.0)))


def per_class_f1(y, pred) -> np.ndarray:
    cm = np.bincount(np.asarray(y) * 3 + np.asarray(pred), minlength=9).reshape(3, 3).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.where(denom > 0, 2 * np.diag(cm) / np.maximum(denom, 1), 0.0)


def rec_slices(data: pd.DataFrame):
    """(start, end) of every piece; rows of one piece are contiguous and in window order."""
    r = data["rec"].cat.codes.to_numpy()
    b = np.flatnonzero(np.diff(r)) + 1
    return list(zip(np.r_[0, b], np.r_[b, len(r)]))


def smooth(data, P, k, slices=None):
    if k <= 1:
        return P
    out = P.copy()
    for s, e in slices or rec_slices(data):
        out[s:e] = ym.smooth_probs(P[s:e], k)
    return out


def tune(data, P, ks=(1, 3, 5, 7, 9, 13, 17)):
    """Best smoothing width and log-bias (Activity, Stress) for macro-F1 on out-of-fold probabilities."""
    y = data["y"].to_numpy()
    sl = rec_slices(data)
    best = (-1.0, 1, np.zeros(3))
    grid = np.arange(-1.0, 1.51, 0.25)
    for k in ks:
        L = np.log(np.maximum(smooth(data, P, k, sl), 1e-9))
        for b1 in grid:
            for b2 in grid:
                bias = np.array([0.0, b1, b2])
                s = macro(y, (L + bias).argmax(1))
                if s > best[0]:
                    best = (s, k, bias)
    return best


def report(data, P, tag="", feats=(), log=True, extra=None):
    s, k, bias = tune(data, P)
    pred = (np.log(np.maximum(smooth(data, P, k), 1e-9)) + bias).argmax(1)
    y = data["y"].to_numpy()
    chunk = data["chunk"].to_numpy()
    by_chunk = {c: macro(y[chunk == c], pred[chunk == c]) for c in sorted(np.unique(chunk))}
    f1 = per_class_f1(y, pred)
    print(f"[{tag}] macro-F1 {s:.4f}  (k={k}, bias A/S {bias[1]:+.2f}/{bias[2]:+.2f})  "
          f"F1 Rest/Act/Stress {f1[0]:.3f}/{f1[1]:.3f}/{f1[2]:.3f}")
    print("   by input: " + "  ".join(f"{'whole' if c == -1 else f'{c}m'} {v:.3f}" for c, v in by_chunk.items()))
    if log:
        LOG.parent.mkdir(exist_ok=True)
        new = not LOG.exists()
        with LOG.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "tag", "code", "n_feats", "macro", "f1_rest", "f1_act", "f1_stress", "k",
                            "bias_act", "bias_stress"] + [f"in_{'whole' if c == -1 else f'{c}m'}" for c in by_chunk]
                           + ["extra"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), tag, code_hash(), len(feats), f"{s:.4f}",
                        *(f"{v:.4f}" for v in f1), k, bias[1], bias[2], *(f"{v:.4f}" for v in by_chunk.values()),
                        extra or ""])
    return s, pred


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def cmd_cv(set_name, tag):
    data = build()
    feats = feature_sets()[set_name]
    t0 = time.time()
    P = oof_probs(data, feats)
    report(data, P, tag or set_name, feats, extra=f"{time.time() - t0:.0f}s")
    np.save(CACHE_DIR / f"oof_{set_name}.npy", P)
    return P


def feature_groups(feats):
    """Group features by prefix family, e.g. every hr_mean_* variant together."""
    groups = {}
    for f in feats:
        g = f.split("_lag")[0] if "_lag" in f else f
        for suffix in ("_rel", "_z", "_local", "_2m", "_5m", "_q10", "_q50", "_q90"):
            g = g.replace(suffix, "")
        groups.setdefault(g, []).append(f)
    return groups


def cmd_ablate(set_name):
    data = build()
    feats = feature_sets()[set_name]
    folds = person_folds(data)
    base, _ = report(data, oof_probs(data, feats, folds=folds), f"ablate base {set_name}", feats, log=False)
    for g, cols in sorted(feature_groups(feats).items()):
        rest = [f for f in feats if f not in cols]
        s, _ = report(data, oof_probs(data, rest, folds=folds), f"without {g} ({len(cols)})", rest, log=False)
        print(f"   -> change {s - base:+.4f}")


def cmd_importance(set_name):
    import lightgbm as lgb
    data = build()
    feats = feature_sets()[set_name]
    tr = data[data["share"] >= 0.5]
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "importance_type": "gain", "n_jobs": os.cpu_count()})
    m.fit(tr[feats].to_numpy(np.float32), tr["y"].to_numpy())
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).to_string())


# --------------------------------------------------------------------------------------------
# Stage 2: a second model that reads the first model's probabilities over the surrounding minutes
# --------------------------------------------------------------------------------------------
stage2_features = ym.stage2_features


S2_PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=100,
                 subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0)


def stage2_oof(data: pd.DataFrame, P: np.ndarray, folds=None) -> np.ndarray:
    import lightgbm as lgb
    Z = np.vstack([stage2_features(P[s:e]) for s, e in rec_slices(data)])
    y = data["y"].to_numpy()
    ok = data["share"].to_numpy() >= 0.5
    Q = np.zeros_like(P)
    for a, b in folds or person_folds(data):
        a = a[ok[a]]
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, **S2_PARAMS, "n_jobs": os.cpu_count()}).fit(Z[a], y[a])
        Q[b] = m.predict_proba(Z[b])
    return Q


def nested_two_stage(data: pd.DataFrame, feats, s1_params=None, s2_params=None, folds=None):
    """Strict two-stage CV: for each held-out fold, stage 2 is trained on stage-1 predictions made
    without that fold AND out-of-fold within the training people, exactly as at submission time."""
    import lightgbm as lgb
    X = data[feats].to_numpy(np.float32)
    y = data["y"].to_numpy()
    ok = data["share"].to_numpy() >= 0.5
    sl = rec_slices(data)
    s1 = {**LGB_PARAMS, **(s1_params or {}), "n_jobs": os.cpu_count()}
    s2 = {**LGB_PARAMS, **S2_PARAMS, **(s2_params or {}), "n_jobs": os.cpu_count()}
    Q = np.zeros((len(data), 3))
    P_outer = np.zeros((len(data), 3))
    for a, b in folds or person_folds(data):
        inner = data.iloc[a]
        P_in = np.zeros((len(a), 3))
        for ia, ib in person_folds(inner):
            ia = ia[ok[a][ia]]
            P_in[ib] = lgb.LGBMClassifier(**s1).fit(X[a][ia], y[a][ia]).predict_proba(X[a][ib])
        P_outer[b] = lgb.LGBMClassifier(**s1).fit(X[a][ok[a]], y[a][ok[a]]).predict_proba(X[b])
        P_tmp = np.zeros((len(data), 3))
        P_tmp[a], P_tmp[b] = P_in, P_outer[b]
        Z = np.vstack([stage2_features(P_tmp[s:e]) for s, e in sl])
        aa = a[ok[a]]
        Q[b] = lgb.LGBMClassifier(**s2).fit(Z[aa], y[aa]).predict_proba(Z[b])
    return P_outer, Q


def cmd_stack(set_name, tag):
    data = build()
    path = CACHE_DIR / f"oof_{set_name}.npy"
    P = np.load(path) if path.exists() else oof_probs(data, feature_sets()[set_name])
    report(data, P, f"stage 1 {set_name}", feature_sets()[set_name], log=False)
    Q = stage2_oof(data, P)
    report(data, Q, tag or f"stage 2 on {set_name}", feature_sets()[set_name])
    np.save(CACHE_DIR / f"oof2_{set_name}.npy", Q)


def cmd_nested(set_name, tag):
    data = build()
    feats = feature_sets()[set_name]
    t0 = time.time()
    P, Q = nested_two_stage(data, feats)
    report(data, P, f"nested stage 1 {set_name}", feats, log=False)
    report(data, Q, tag or f"nested 2-stage {set_name}", feats, extra=f"{time.time() - t0:.0f}s")
    np.save(CACHE_DIR / f"nested2_{set_name}.npy", Q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cv", "ablate", "importance", "stack", "nested"])
    ap.add_argument("set", nargs="?", default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "build":
        build(force=True)
    elif a.cmd == "cv":
        cmd_cv(a.set, a.tag)
    elif a.cmd == "nested":
        cmd_nested(a.set, a.tag)
    elif a.cmd == "stack":
        cmd_stack(a.set, a.tag)
    elif a.cmd == "ablate":
        cmd_ablate(a.set)
    else:
        cmd_importance(a.set)
