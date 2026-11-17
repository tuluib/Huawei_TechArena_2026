"""Training labels: protocol phase -> target state, and the majority state per 10-second window.

Mapping from phases_to_activity.txt. Phases not listed (e.g. "Undefined") are not scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_processing.windows import WINDOW_S, seconds_since_start, window_ids

STATES = ("Rest", "Stress", "Activity")
STRESS_PHASES = {"Stroop", "TMCT", "Real opinion", "Opposite opinion", "Subtraction"}
ACTIVITY_PHASES = {"Warm-up", "Sprint", "Cooldown"}
REST_PHASES = {"Baseline", "Rest"}


def phase_to_state(phase: str) -> str | None:
    if phase in STRESS_PHASES:
        return "Stress"
    if phase in ACTIVITY_PHASES or "rpm" in phase:
        return "Activity"
    if phase in REST_PHASES:
        return "Rest"
    return None


def window_labels(df: pd.DataFrame, min_scored_share: float = 0.5) -> pd.Series:
    """Majority state of each 10-second window, aligned with extract_features().

    Windows where fewer than ``min_scored_share`` of the samples have a scored phase get NaN
    (they would not be scored, so they shouldn't be trained on).
    """
    mapping = {p: phase_to_state(p) for p in df["Phase"].unique()}
    codes = {s: i for i, s in enumerate(STATES)}
    state = df["Phase"].map(lambda p: codes.get(mapping[p], -1)).to_numpy()

    w = window_ids(seconds_since_start(df.index))
    n = w[-1] + 1
    total = np.bincount(w, minlength=n)
    votes = np.stack([np.bincount(w[state == i], minlength=n) for i in range(len(STATES))], axis=1)
    scored = votes.sum(axis=1)
    label = np.array(STATES, dtype=object)[votes.argmax(axis=1)]
    label[scored < min_scored_share * np.maximum(total, 1)] = None

    index = df.index[0] + pd.to_timedelta(np.arange(n) * WINDOW_S, unit="s")
    return pd.Series(label, index=index, name="state").iloc[np.flatnonzero(total > 0)]
