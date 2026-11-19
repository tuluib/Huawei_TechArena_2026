"""Old (V1-V3) trainer, kept for reference. It overwrites src/your_model.onnx with a model the current
your_model.py may not read: restore with `git checkout src/your_model.onnx`.

Train a model that works whatever input length run() receives.

The hidden test may call run() on pieces of a recording rather than whole recordings. Features
such as "time since start" or "value minus the recording median" then mean something different.
Here the training features are computed on recordings cut into pieces of random length
(1, 2, 5, 10, 30 minutes and whole), so the model learns every situation, and the time-position
features are not used.

    python train_robust.py export holdout   # without the 6 validation people (local checks)
    python train_robust.py export all       # everyone (submission)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src/
import train_v2 as T
import your_model as ym
from eval_harness import VAL_SUBJECTS

CACHE = T.DATA / "_cache" / "robust_features.parquet"
CHUNK_MINUTES = (1, 2, 5, 10, 30, None)  # None = whole recording
# V3's exact 134 features (your_model.py has more features since V4)
FEATURES = __import__("json").loads((T.ROOT / "experiments" / "v3_features.json").read_text())


def build(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)
    rng = np.random.default_rng(0)
    parts = []
    for p in T.PROTOCOLS:
        for fp in sorted((T.DATA / p).glob("*.parquet")):
            df = pd.read_parquet(fp)
            t = (df.index - df.index[0]).total_seconds().to_numpy()
            for minutes in CHUNK_MINUTES:
                if minutes is None:
                    groups = np.zeros(len(df), dtype=np.int64)
                else:
                    size = 60.0 * minutes  # random phase so chunk borders fall everywhere
                    groups = np.floor((t + rng.uniform(0, size)) / size).astype(np.int64)
                bounds = np.flatnonzero(np.diff(groups)) + 1
                for ci, (s, e) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, len(df)])):
                    piece = df.iloc[s:e]
                    if len(piece) < 64:
                        continue
                    X, row_wid = ym.compute_features(piece[ym.INPUT_COLS])
                    lab, share = T.window_labels(piece, row_wid, len(X))
                    part = pd.DataFrame(X, columns=ym.ALL_FEATURES)
                    part["y"], part["share"], part["user"] = lab, share, fp.stem
                    part["rec"] = f"{p}/{fp.stem}/{minutes}/{ci}"  # smoothing stays inside one input
                    part["chunk"] = -1 if minutes is None else minutes
                    parts.append(part[part["y"] >= 0])
            print(f"  {p}/{fp.stem}", flush=True)
    data = pd.concat(parts, ignore_index=True)
    data.to_parquet(CACHE)
    return data


def export(train_on: str):
    data = build()
    tr = data if train_on == "all" else data[~data["user"].isin(VAL_SUBJECTS)].reset_index(drop=True)
    print(f"{len(tr):,} training windows from {tr['user'].nunique()} people")
    P = T.oof_probs(tr, FEATURES, folds=5)
    s, k, bias = T.tune(tr, P)
    for c, g in tr.groupby("chunk").indices.items():
        L = np.log(np.maximum(T.smooth_by_rec(tr.iloc[g].reset_index(drop=True), P[g], k), 1e-9)) + bias
        print(f"  OOF macro-F1 for {'whole' if c == -1 else f'{c}-min'} inputs: {T.macro(tr['y'].to_numpy()[g], L.argmax(1)):.3f}")
    print(f"OOF macro-F1 overall {s:.3f} (k={k}, bias={bias.tolist()})")
    T.VARIANTS["robust"] = FEATURES
    T.build_dataset = lambda use_cache=True: data  # reuse train_v2's exporter on this dataset
    T.export_trained(tr, FEATURES, k, bias, s, "robust", train_on)


if __name__ == "__main__":
    if sys.argv[1] == "rebuild":
        build(use_cache=False)
    else:
        export(sys.argv[2])
