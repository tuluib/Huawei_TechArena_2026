"""Shared time base: seconds since the first sample and 10-second window ids.

Huawei scores predictions on 10-second windows (majority state per window), so features and
labels are both computed on the same grid: window k covers [t0 + 10k s, t0 + 10(k+1) s).
"""
import numpy as np
import pandas as pd

WINDOW_S = 10.0


def seconds_since_start(index: pd.DatetimeIndex) -> np.ndarray:
    """Float seconds since the first timestamp (works for any datetime64 resolution)."""
    return np.asarray((index - index[0]).total_seconds(), dtype=np.float64)


def window_ids(t: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    """Window number for each time in seconds."""
    return np.floor(t / window_s).astype(np.int64)
