"""10-second window features from cleaned signals.

One row per scoring window. Three kinds of features:
1. Window features: computed from the 10 s itself (pulse rate, BVP amplitude, EDA level...).
2. Context features: the same quantities over a centred 60 s span. HRV and skin responses
   need more than 10 s of data to be meaningful.
3. Relative features: window value minus the recording's median. People differ a lot
   (EDA levels range 0.3-76 uS between recordings); changes within a person carry the signal.
   They assume run() receives a long stretch of one recording (see README).

All aggregation uses np.bincount over window ids instead of Python loops (speed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_processing.cleaning import FS_BVP, CleanSignals, Stream, clean_signals
from signal_processing.windows import WINDOW_S, window_ids

CONTEXT_WINDOWS = 6  # 6 x 10 s = 60 s centred context
PULSE_BAND_HZ = (0.6, 3.5)  # 36-210 bpm

WINDOW_FEATURES = [
    "bvp_log_rms", "bvp_pulse_bpm", "bvp_peak_ratio", "bvp_artifact_frac",
    "hr_mean", "hr_std", "hr_slope",
    "ibi_mean", "ibi_n",
    "eda_mean", "eda_slope", "eda_tonic", "eda_phasic_std",
    "temp_mean",
]
CONTEXT_FEATURES = ["hrv_rmssd_60s", "hrv_sdnn_60s", "scr_rate_60s", "temp_slope_60s", "eda_slope_60s",
                    "hr_mean_60s"]
RELATIVE_BASES = ["hr_mean", "bvp_pulse_bpm", "eda_tonic", "temp_mean", "bvp_log_rms", "ibi_mean",
                  "scr_rate_60s", "hrv_rmssd_60s"]
LOCAL_WINDOWS = 30  # 5-minute centred baseline: stress phases last 2-5 min between rest periods
RELATIVE_FEATURES = ([f"{b}_rel" for b in RELATIVE_BASES]  # minus the recording median
                     + [f"{b}_z" for b in RELATIVE_BASES]  # robust z-score within the recording
                     + [f"{b}_local" for b in RELATIVE_BASES])  # minus the surrounding 5 minutes
FEATURE_NAMES = WINDOW_FEATURES + CONTEXT_FEATURES + RELATIVE_FEATURES


def _window_stats(stream: Stream, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-window count, mean, std and linear slope (units per second) of a stream."""
    ok = ~np.isnan(stream.x)
    w = window_ids(stream.t[ok])
    x, t = stream.x[ok], stream.t[ok]
    keep = w < n
    w, x, t = w[keep], x[keep], t[keep]

    cnt = np.bincount(w, minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(w, x, n) / cnt
        var = np.bincount(w, x * x, n) / cnt - mean ** 2
        t_mean = np.bincount(w, t, n) / cnt
        cov_tx = np.bincount(w, t * x, n) / cnt - t_mean * mean
        var_t = np.bincount(w, t * t, n) / cnt - t_mean ** 2
        slope = np.where(cnt >= 3, cov_tx / var_t, np.nan)
    return cnt, mean, np.sqrt(np.maximum(var, 0.0)), slope


def _bvp_spectral(sig: CleanSignals, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pulse rate, spectral peak sharpness and log amplitude per window, all windows in one FFT.

    The BVP of each window is placed in a row of an (n, 640) matrix; the FFT along rows gives
    every window's spectrum at once. A clean pulse wave has one sharp peak at the heart rate;
    motion artifacts spread power over many frequencies, so peak_ratio doubles as a quality index.
    """
    per_win = int(WINDOW_S * FS_BVP)
    t, x = sig.bvp.t, sig.bvp.x
    ok = ~np.isnan(x)
    w = window_ids(t[ok])
    pos = np.clip(np.round((t[ok] - w * WINDOW_S) * FS_BVP).astype(np.int64), 0, per_win - 1)
    keep = w < n
    mat = np.zeros((n, per_win))
    mat[w[keep], pos[keep]] = x[ok][keep]

    cnt = np.bincount(w[keep], minlength=n)
    power = np.abs(np.fft.rfft(mat * np.hanning(per_win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(per_win, d=1.0 / FS_BVP)
    band = (freqs >= PULSE_BAND_HZ[0]) & (freqs <= PULSE_BAND_HZ[1])
    band_power = power[:, band]
    peak = band_power.argmax(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pulse_bpm = freqs[band][peak] * 60.0
        peak_ratio = band_power.max(axis=1) / band_power.sum(axis=1)
        log_rms = np.log1p(np.sqrt((mat ** 2).sum(axis=1) / np.maximum(cnt, 1)))
    empty = cnt < per_win // 2
    for a in (pulse_bpm, peak_ratio, log_rms):
        a[empty] = np.nan
    return pulse_bpm, peak_ratio, log_rms


def _hrv(sig: CleanSignals, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heart rate variability over a centred 60 s span.

    RMSSD uses only differences between directly consecutive valid beats (IBI_dt == duration);
    a missed beat in between would create a fake huge difference.
    """
    t, ibi = sig.ibi.t, sig.ibi.x
    ok = ~np.isnan(ibi)
    w_all = window_ids(t)
    keep = ok & (w_all < n)
    w = w_all[keep]
    cnt = np.bincount(w, minlength=n).astype(np.float64)
    s = np.bincount(w, ibi[keep], n)
    s2 = np.bincount(w, ibi[keep] ** 2, n)

    diff = np.diff(ibi, prepend=np.nan)
    pair = keep & sig.ibi_consecutive & ~np.isnan(diff)
    wd = w_all[pair]
    d_cnt = np.bincount(wd, minlength=n).astype(np.float64)
    d_ss = np.bincount(wd, diff[pair] ** 2, n)

    k = np.ones(CONTEXT_WINDOWS)
    roll = lambda a: np.convolve(a, k, mode="same")  # centred 60 s sums
    c60, s60, s260, dc60, dss60 = map(roll, (cnt, s, s2, d_cnt, d_ss))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ibi = s / cnt
        rmssd = np.where(dc60 >= 5, np.sqrt(dss60 / dc60), np.nan) * 1000.0  # ms
        sdnn = np.where(c60 >= 5, np.sqrt(np.maximum(s260 / c60 - (s60 / c60) ** 2, 0)), np.nan) * 1000.0
    return mean_ibi, cnt, rmssd, sdnn


def _rolling_mean(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rolling(CONTEXT_WINDOWS, center=True, min_periods=1).mean().to_numpy()


def _relative(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare a window value with the same person: recording median, robust z, local 5 min."""
    nan = np.full(len(a), np.nan)
    if not np.isfinite(a).any():
        return nan, nan, nan
    med = np.nanmedian(a)
    q75, q25 = np.nanpercentile(a, [75, 25])
    spread = q75 - q25
    z = (a - med) / spread if spread > 0 else nan
    local = a - pd.Series(a).rolling(LOCAL_WINDOWS, center=True, min_periods=3).median().to_numpy()
    return a - med, z, local


def extract_features(df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """Raw input DataFrame (the one run() receives) -> one feature row per 10-second window.

    The index is the start time of each window. Windows without any BVP are dropped.
    NaN means "not measurable here" (e.g. no beats detected); the model must handle it.
    """
    sig = clean_signals(df, clean=clean)
    n = int(sig.duration_s // WINDOW_S) + 1

    f: dict[str, np.ndarray] = {}
    f["bvp_pulse_bpm"], f["bvp_peak_ratio"], f["bvp_log_rms"] = _bvp_spectral(sig, n)
    art_w = window_ids(sig.bvp.t)
    f["bvp_artifact_frac"] = (np.bincount(art_w, sig.bvp_artifact.astype(float), n)
                              / np.maximum(np.bincount(art_w, minlength=n), 1))

    _, f["hr_mean"], f["hr_std"], f["hr_slope"] = _window_stats(sig.hr, n)
    f["ibi_mean"], f["ibi_n"], f["hrv_rmssd_60s"], f["hrv_sdnn_60s"] = _hrv(sig, n)

    _, f["eda_mean"], _, f["eda_slope"] = _window_stats(sig.eda, n)
    _, f["eda_tonic"], _, _ = _window_stats(Stream(sig.eda.t, sig.eda_tonic), n)
    _, _, f["eda_phasic_std"], _ = _window_stats(Stream(sig.eda.t, sig.eda_phasic), n)
    _, f["temp_mean"], _, _ = _window_stats(sig.temp, n)

    scr = np.bincount(window_ids(sig.scr_times), minlength=n)[:n].astype(np.float64)
    f["scr_rate_60s"] = np.convolve(scr, np.ones(CONTEXT_WINDOWS), mode="same")  # responses per minute
    temp_ctx = _rolling_mean(f["temp_mean"])
    f["temp_slope_60s"] = np.gradient(temp_ctx) / WINDOW_S * 60.0 if n > 1 else np.full(n, np.nan)  # C/min
    eda_ctx = _rolling_mean(f["eda_tonic"])
    f["eda_slope_60s"] = np.gradient(eda_ctx) / WINDOW_S * 60.0 if n > 1 else np.full(n, np.nan)
    f["hr_mean_60s"] = _rolling_mean(f["hr_mean"])

    for b in RELATIVE_BASES:
        f[f"{b}_rel"], f[f"{b}_z"], f[f"{b}_local"] = _relative(f[b])

    has_bvp = np.bincount(art_w, minlength=n)[:n] > 0
    index = sig.t0 + pd.to_timedelta(np.arange(n) * WINDOW_S, unit="s")
    out = pd.DataFrame({k: f[k] for k in FEATURE_NAMES}, index=index)
    out.index.name = "window_start"
    return out[has_bvp]
