"""How good are the features? Person-grouped cross-validation, cleaned vs raw signals.

    python -m signal_processing.evaluate_features

Groups = user_id, so a person's recordings are never split between training and testing
(windows of the same person look alike; mixing them gives scores the leaderboard won't repeat).
The model is a quick reference classifier, not the final model (that's the ML Architect's job).
Score printed the way Huawei computes the quality part: 80 x macro-F1 over 10-second windows.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # evaluation runs on one core

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from signal_processing.features import FEATURE_NAMES, RELATIVE_FEATURES, extract_features
from signal_processing.labels import STATES, window_labels
from signal_processing.loading import DATA_DIR, list_recordings, load, model_input

CACHE_DIR = DATA_DIR / "_cache"


def build_dataset(clean: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Features + label + user_id + protocol for every scored window of every recording."""
    cache = CACHE_DIR / f"features_{'clean' if clean else 'raw'}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    parts, times = [], []
    for rec in list_recordings():
        df = load(rec)
        start = time.perf_counter()
        feats = extract_features(model_input(df), clean=clean)
        times.append((time.perf_counter() - start, len(df), rec.name))
        feats["state"] = window_labels(df).reindex(feats.index)
        feats["user_id"], feats["protocol"] = rec.user_id, rec.protocol
        parts.append(feats.reset_index())
    data = pd.concat(parts, ignore_index=True)
    data = data[data["state"].notna()].reset_index(drop=True)
    CACHE_DIR.mkdir(exist_ok=True)
    data.to_parquet(cache)
    slowest = max(times)
    print(f"[{'clean' if clean else 'raw'}] feature time per recording: "
          f"mean {np.mean([t for t, *_ in times]) * 1000:.0f} ms, slowest {slowest[0] * 1000:.0f} ms "
          f"({slowest[2]}, {slowest[1]:,} rows)")
    return data


def cross_validate(data: pd.DataFrame, features: list[str], n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions with person-grouped folds."""
    features = [f for f in features if data[f].notna().any()]  # e.g. SCR features without cleaning
    pred = np.empty(len(data), dtype=object)
    for train, test in GroupKFold(n_splits=n_splits).split(data, groups=data["user_id"]):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                               random_state=0)
        model.fit(data.loc[train, features], data.loc[train, "state"])
        pred[test] = model.predict(data.loc[test, features])
    return pred


def report(name: str, data: pd.DataFrame, pred: np.ndarray) -> float:
    f1 = f1_score(data["state"], pred, average="macro")
    per_class = f1_score(data["state"], pred, average=None, labels=list(STATES))
    print(f"{name:<38} macro-F1 {f1:.3f}  -> quality {80 * f1:5.1f}/80   "
          + "  ".join(f"{s} {v:.2f}" for s, v in zip(STATES, per_class)))
    return f1


def main() -> None:
    clean, raw = build_dataset(clean=True), build_dataset(clean=False)
    counts = clean["state"].value_counts()
    print(f"\n{len(clean):,} scored windows from {clean['user_id'].nunique()} people: "
          + ", ".join(f"{s} {counts.get(s, 0):,}" for s in STATES) + "\n")

    absolute = [f for f in FEATURE_NAMES if f not in RELATIVE_FEATURES]
    report("raw signals, all features", raw, cross_validate(raw, FEATURE_NAMES))
    report("cleaned, absolute features only", clean, cross_validate(clean, absolute))
    best = cross_validate(clean, FEATURE_NAMES)
    report("cleaned, all features", clean, best)

    print("\nConfusion matrix (rows = true, columns = predicted):")
    cm = confusion_matrix(clean["state"], best, labels=list(STATES))
    print(pd.DataFrame(cm, index=STATES, columns=STATES).to_string())
    print("\nPer protocol:")
    for protocol, part in clean.groupby("protocol"):
        print(f"  {protocol:<10} accuracy {np.mean(part['state'] == best[part.index]):.3f}")
    print()
    print(classification_report(clean["state"], best, digits=3))


if __name__ == "__main__":
    main()
