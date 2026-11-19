"""Huawei Tech Arena 2026 - activity detection (Rest / Activity / Stress) from wristband signals.

Pipeline inside run():
1. Split the 64 Hz grid into per-sensor streams (BVP 64 Hz, EDA/temp 4 Hz, HR 1 Hz, IBI per beat)
   and clean each one (band-pass / low-pass filters, physiological range checks, spike rejection).
2. One feature row per 10-second window. Windows are aligned to the clock (10 s boundaries), the
   same grid the scorer uses (resample("10s")), not to the first sample.
3. Features: window values, 60 s context (HRV, skin responses, slopes), values relative to the same
   person (recording median, robust z, local 5-minute baseline), recording-level summaries and
   (optionally) the position in the recording.
4. Gradient-boosted trees (ONNX) -> class probabilities -> smoothed over neighbouring windows ->
   per-class bias tuned for macro-F1 -> one label per window, broadcast back to every row.

Everything the model needs besides this file (feature list, smoothing width, class bias) is stored
in the ONNX file's metadata, so a different .onnx can use a different feature subset.
Only numpy, pandas, scipy and onnxruntime are used.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy import signal

from adm_base_class import ADM_Base

CLASSES = np.array(["Rest", "Activity", "Stress"], dtype=object)
WINDOW_S = 10.0
FS_BVP = 64.0
FS_EDA = 4.0
MAX_GAP_S = 1.0

HR_RANGE = (35.0, 220.0)
IBI_RANGE = (0.27, 2.0)
TEMP_RANGE = (20.0, 42.0)
EDA_MIN = 0.02

SOS_BVP = signal.butter(2, [0.5, 4.0], btype="bandpass", fs=FS_BVP, output="sos")
SOS_EDA = signal.butter(2, 1.0, btype="lowpass", fs=FS_EDA, output="sos")
SOS_EDA_TONIC = signal.butter(2, 0.05, btype="lowpass", fs=FS_EDA, output="sos")

CONTEXT_WINDOWS = 6     # 60 s centred context
LOCAL_WINDOWS = 30      # 5-minute local baseline
PULSE_BAND_HZ = (0.6, 3.5)

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
RELATIVE_FEATURES = ([f"{b}_rel" for b in RELATIVE_BASES] + [f"{b}_z" for b in RELATIVE_BASES]
                     + [f"{b}_local" for b in RELATIVE_BASES])
LONG_BASES = ["hr_mean", "bvp_log_rms", "bvp_artifact_frac", "eda_tonic", "scr_rate_60s"]
LONG_FEATURES = [f"{b}_{w}m" for b in LONG_BASES for w in (2, 5)]  # centred 2 / 5 minute means
REC_BASES = ["hr_mean", "bvp_artifact_frac", "bvp_log_rms", "eda_tonic", "temp_mean"]
REC_FEATURES = [f"rec_{b}_q{q}" for b in REC_BASES for q in (10, 50, 90)] + ["rec_hr_range", "rec_dur"]
TIME_FEATURES = ["t_s", "t_frac", "t_left"]
# Neighbour windows: the same key features 30 s .. 3 min before/after (the shape of a phase).
LAG_BASES = ["hr_mean_local", "eda_tonic_local", "bvp_log_rms_local", "scr_rate_60s", "hrv_rmssd_60s_local",
             "temp_slope_60s", "bvp_artifact_frac", "hr_mean_rel"]
LAGS = (-18, -12, -6, -3, 3, 6, 12, 18)
LAG_FEATURES = [f"{b}_lag{L}" for L in LAGS for b in LAG_BASES]
# v4: pulse regularity / noise, skin-response size, heart rate against its recent extremes,
# and past-vs-future trends (onsets and ends of tasks).
EXTRA_FEATURES = ["bvp_periodicity", "bvp_vhf", "bvp_entropy", "scr_amp_60s", "scr_rate_30s", "scr_rate_120s",
                  "hr_vs_min3m", "hr_vs_runmax", "hr_vs_runmin", "eda_vs_min3m", "temp_vs_max3m"]
TREND_BASES = ["hr_mean", "eda_tonic", "scr_rate_60s", "temp_mean", "bvp_log_rms", "bvp_artifact_frac",
               "bvp_periodicity", "ibi_mean"]
TREND_FEATURES = [f"{b}_{s}" for b in TREND_BASES for s in ("d1m", "d3m", "vs_past3m", "vs_fut3m")]
REC2_FEATURES = [f"rec_{b}_q{q}" for b in ("bvp_periodicity", "bvp_vhf", "scr_rate_60s") for q in (10, 50, 90)]
ALL_FEATURES = (WINDOW_FEATURES + CONTEXT_FEATURES + RELATIVE_FEATURES + LONG_FEATURES
                + REC_FEATURES + TIME_FEATURES + LAG_FEATURES + EXTRA_FEATURES + TREND_FEATURES + REC2_FEATURES)
INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


# --------------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------------
def _segments(t: np.ndarray):
    breaks = np.flatnonzero(np.diff(t) > MAX_GAP_S) + 1
    return zip(np.r_[0, breaks], np.r_[breaks, len(t)])


def _interp_nan(x: np.ndarray) -> np.ndarray:
    bad = np.isnan(x)
    if not bad.any() or bad.all():
        return x
    x = x.copy()
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _filtfilt_segments(t, x, sos, drop_short: bool, one_pass: bool = False) -> np.ndarray:
    """Filter each gap-free segment (zero-phase, or a single causal pass when phase is irrelevant)."""
    out = np.full(len(x), np.nan)
    padlen = 3 * (2 * len(sos) + 1)
    for s, e in _segments(t):
        seg = x[s:e]
        bad = np.isnan(seg)
        if bad.all():
            continue
        if e - s > padlen:
            filtered = (signal.sosfilt(sos, _interp_nan(seg)) if one_pass
                        else signal.sosfiltfilt(sos, _interp_nan(seg)))
        elif drop_short:
            continue
        else:
            filtered = seg.copy()
        filtered[bad] = np.nan
        out[s:e] = filtered
    return out


def _reject_outside(x, lo, hi):
    x = x.copy()
    x[(x < lo) | (x > hi)] = np.nan
    return x


def _reject_spikes(x, kernel, max_dev, relative=False):
    if len(x) < kernel or np.isnan(x).all():
        return x
    med = signal.medfilt(_interp_nan(x), kernel)
    dev = np.abs(x - med)
    limit = max_dev * np.abs(med) if relative else max_dev
    x = x.copy()
    x[dev > limit] = np.nan
    return x


def _bvp_artifacts(t, x) -> np.ndarray:
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    block = np.floor(t).astype(np.int64)
    valid = ~np.isnan(x)
    n = block[-1] + 1
    cnt = np.bincount(block[valid], minlength=n)
    ss = np.bincount(block[valid], weights=x[valid] ** 2, minlength=n)
    rms = np.sqrt(ss / np.maximum(cnt, 1))
    typical = np.median(rms[cnt > 0]) if (cnt > 0).any() else 0.0
    bad_block = (cnt == 0) | (rms > 2.5 * typical) | (rms < 0.25 * typical)
    return bad_block[block] | ~valid


# --------------------------------------------------------------------------------------------
# Window features
# --------------------------------------------------------------------------------------------
def _wid(t):
    return np.floor(t / WINDOW_S).astype(np.int64)


def _window_stats(t, x, n):
    ok = ~np.isnan(x)
    t, x = t[ok], x[ok]
    w = _wid(t)
    keep = (w >= 0) & (w < n)
    w, x, t = w[keep], x[keep], t[keep]
    cnt = np.bincount(w, minlength=n).astype(np.float64)
    mean = np.bincount(w, x, n) / cnt
    var = np.bincount(w, x * x, n) / cnt - mean ** 2
    t_mean = np.bincount(w, t, n) / cnt
    cov_tx = np.bincount(w, t * x, n) / cnt - t_mean * mean
    var_t = np.bincount(w, t * t, n) / cnt - t_mean ** 2
    slope = np.where(cnt >= 3, cov_tx / var_t, np.nan)
    return cnt, mean, np.sqrt(np.maximum(var, 0.0)), slope


def _bvp_spectral(t, x, n, raw=None):
    per_win = int(WINDOW_S * FS_BVP)
    ok = ~np.isnan(x)
    w = _wid(t[ok])
    pos = np.clip(np.round((t[ok] - w * WINDOW_S) * FS_BVP).astype(np.int64), 0, per_win - 1)
    keep = (w >= 0) & (w < n)
    mat = np.zeros((n, per_win))
    mat[w[keep], pos[keep]] = x[ok][keep]
    cnt = np.bincount(w[keep], minlength=n)
    power = np.abs(np.fft.rfft(mat * np.hanning(per_win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(per_win, d=1.0 / FS_BVP)
    band = (freqs >= PULSE_BAND_HZ[0]) & (freqs <= PULSE_BAND_HZ[1])
    band_power = power[:, band]
    pulse_bpm = freqs[band][band_power.argmax(axis=1)] * 60.0
    peak_ratio = band_power.max(axis=1) / band_power.sum(axis=1)
    log_rms = np.log1p(np.sqrt((mat ** 2).sum(axis=1) / np.maximum(cnt, 1)))
    # pulse regularity: autocorrelation peak at 0.3-1.5 s lags (low when movement corrupts the pulse)
    ac = np.fft.irfft(power, axis=1)
    lo, hi = int(0.3 * FS_BVP), int(1.5 * FS_BVP)
    periodicity = ac[:, lo:hi].max(axis=1) / np.maximum(ac[:, 0], 1e-12)
    pn = band_power / np.maximum(band_power.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(pn * np.log(pn + 1e-12)).sum(axis=1)
    vhf = np.full(n, np.nan)
    if raw is not None:  # share of raw (unfiltered) power above 8 Hz: movement, not pulse
        rmat = np.zeros((n, per_win))
        rmat[w[keep], pos[keep]] = raw[ok][keep]
        rmat -= rmat.sum(axis=1, keepdims=True) / np.maximum(cnt, 1)[:, None]
        rp = np.abs(np.fft.rfft(rmat, axis=1)) ** 2
        vhf = np.log(rp[:, freqs > 8.0].sum(axis=1) / np.maximum(rp[:, freqs >= 0.5].sum(axis=1), 1e-12) + 1e-9)
    empty = cnt < per_win // 2
    for a in (pulse_bpm, peak_ratio, log_rms, periodicity, entropy, vhf):
        a[empty] = np.nan
    return pulse_bpm, peak_ratio, log_rms, periodicity, entropy, vhf


def _same(a, k):
    """Centred moving sum over k values; output always has len(a) (np.convolve 'same' does not
    when len(a) < k, which crashed short inputs)."""
    full = np.convolve(a, np.ones(k), mode="full")
    start = (k - 1) // 2
    return full[start:start + len(a)]


def _roll_sum(a, k):
    return _same(a, k)


def _nan_roll_mean(a, k):
    """Centred rolling mean that ignores NaN (min 1 valid value)."""
    ok = np.isfinite(a)
    s = _same(np.where(ok, a, 0.0), k)
    c = _same(ok.astype(np.float64), k)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _offset_mean(a, lo, hi):
    """Mean of a[i+lo .. i+hi] (inclusive offsets, clipped to the input), NaN ignored."""
    ok = np.isfinite(a)
    cs = np.r_[0.0, np.cumsum(np.where(ok, a, 0.0))]
    cc = np.r_[0.0, np.cumsum(ok)]
    i = np.arange(len(a))
    s = np.clip(i + lo, 0, len(a))
    e = np.clip(i + hi + 1, 0, len(a))
    c = cc[e] - cc[s]
    return np.where(c > 0, (cs[e] - cs[s]) / np.maximum(c, 1), np.nan)


def _past_extreme(a, k, fn):
    """Rolling min/max over the current and k-1 previous windows, NaN ignored."""
    r = pd.Series(a).rolling(k, min_periods=1)
    return (r.min() if fn == "min" else r.max()).to_numpy()


def _relative(a):
    nan = np.full(len(a), np.nan)
    if np.isfinite(a).sum() < 2:
        return nan, nan, nan
    q25, med, q75 = np.percentile(a[np.isfinite(a)], [25, 50, 75])
    spread = q75 - q25
    z = (a - med) / spread if spread > 0 else nan
    local = a - pd.Series(a).rolling(LOCAL_WINDOWS, center=True, min_periods=3).median().to_numpy()
    return a - med, z, local


def _qs(a, qs):
    v = a[np.isfinite(a)]
    return np.percentile(v, qs) if len(v) else np.full(len(qs), np.nan)


def compute_features(df: pd.DataFrame):
    """Input DataFrame -> (feature matrix [n_windows, len(ALL_FEATURES)], window id of every row).

    Window k covers [floor10(t0) + 10k, floor10(t0) + 10(k+1)) seconds: the scorer's grid.
    """
    idx = df.index
    origin = idx[0].floor("10s")
    t = np.asarray((idx - origin).total_seconds(), dtype=np.float64)
    offset = t[0]
    n = int(t[-1] // WINDOW_S) + 1
    row_wid = _wid(t)

    col = {c: df[c].to_numpy(dtype=np.float64) for c in INPUT_COLS}

    def stream(name):
        keep = np.isfinite(col[name])
        return t[keep], col[name][keep]

    bt, bx = stream("BVP")
    et, ex = stream("EDA")
    tt, tx = stream("Temperature")
    ht, hx = stream("HR")
    it, ix = stream("IBI_duration")
    ibi_dt = col["IBI_dt"][np.isfinite(col["IBI_duration"])]

    # ---- cleaning
    bx = np.clip(bx, -1e5, 1e5)
    bx_raw = bx
    bx = _filtfilt_segments(bt, bx, SOS_BVP, drop_short=True, one_pass=True)
    bvp_art = _bvp_artifacts(bt, bx)
    ex = np.where(ex < EDA_MIN, np.nan, ex)
    ex = _reject_spikes(ex, 5, 0.3, relative=True)
    ex = _filtfilt_segments(et, ex, SOS_EDA, drop_short=False)
    tonic = _filtfilt_segments(et, ex, SOS_EDA_TONIC, drop_short=False)
    phasic = ex - tonic
    tx = _reject_spikes(_reject_outside(tx, *TEMP_RANGE), 9, 0.5)
    hx = _reject_spikes(_reject_outside(hx, *HR_RANGE), 5, 15.0)
    ix_raw = ix
    ix = _reject_spikes(_reject_outside(ix, *IBI_RANGE), 5, 0.25, relative=True)
    consecutive = np.abs(ibi_dt - ix_raw) < 0.05

    scr_t = np.empty(0)
    scr_amp = np.empty(0)
    if len(phasic) > 4 and np.isfinite(phasic).any():
        min_rise = max(0.02, 0.005 * float(np.nanmedian(ex)))
        peaks, props = signal.find_peaks(np.nan_to_num(phasic), prominence=min_rise, distance=int(FS_EDA))
        scr_t = et[peaks]
        scr_amp = props["prominences"]

    # ---- window features
    f = {}
    (f["bvp_pulse_bpm"], f["bvp_peak_ratio"], f["bvp_log_rms"], f["bvp_periodicity"], f["bvp_entropy"],
     f["bvp_vhf"]) = _bvp_spectral(bt, bx, n, bx_raw)
    aw = _wid(bt)
    f["bvp_artifact_frac"] = (np.bincount(aw, bvp_art.astype(float), n)[:n]
                              / np.maximum(np.bincount(aw, minlength=n)[:n], 1))
    _, f["hr_mean"], f["hr_std"], f["hr_slope"] = _window_stats(ht, hx, n)

    # HRV
    ok = np.isfinite(ix)
    w_all = _wid(it)
    keep = ok & (w_all < n)
    w = w_all[keep]
    cnt = np.bincount(w, minlength=n).astype(np.float64)
    s1 = np.bincount(w, ix[keep], n)
    s2 = np.bincount(w, ix[keep] ** 2, n)
    diff = np.diff(ix, prepend=np.nan)
    pair = keep & consecutive & np.isfinite(diff)
    wd = w_all[pair]
    d_cnt = np.bincount(wd, minlength=n).astype(np.float64)
    d_ss = np.bincount(wd, diff[pair] ** 2, n)
    c60, s60, s260, dc60, dss60 = (_roll_sum(a, CONTEXT_WINDOWS) for a in (cnt, s1, s2, d_cnt, d_ss))
    f["ibi_mean"] = s1 / cnt
    f["ibi_n"] = cnt
    f["hrv_rmssd_60s"] = np.where(dc60 >= 5, np.sqrt(dss60 / np.maximum(dc60, 1)), np.nan) * 1000.0
    f["hrv_sdnn_60s"] = np.where(
        c60 >= 5, np.sqrt(np.maximum(s260 / np.maximum(c60, 1) - (s60 / np.maximum(c60, 1)) ** 2, 0)),
        np.nan) * 1000.0

    _, f["eda_mean"], _, f["eda_slope"] = _window_stats(et, ex, n)
    _, f["eda_tonic"], _, _ = _window_stats(et, tonic, n)
    _, _, f["eda_phasic_std"], _ = _window_stats(et, phasic, n)
    _, f["temp_mean"], _, _ = _window_stats(tt, tx, n)

    scr = np.bincount(_wid(scr_t), minlength=n)[:n].astype(np.float64)
    f["scr_rate_60s"] = _roll_sum(scr, CONTEXT_WINDOWS)
    f["scr_rate_30s"] = _roll_sum(scr, 3)
    f["scr_rate_120s"] = _roll_sum(scr, 12)
    f["scr_amp_60s"] = _roll_sum(np.bincount(_wid(scr_t), scr_amp, minlength=n)[:n], CONTEXT_WINDOWS)
    temp_ctx = _nan_roll_mean(f["temp_mean"], CONTEXT_WINDOWS)
    eda_ctx = _nan_roll_mean(f["eda_tonic"], CONTEXT_WINDOWS)
    if n > 1:
        f["temp_slope_60s"] = np.gradient(temp_ctx) / WINDOW_S * 60.0
        f["eda_slope_60s"] = np.gradient(eda_ctx) / WINDOW_S * 60.0
    else:
        f["temp_slope_60s"] = np.full(n, np.nan)
        f["eda_slope_60s"] = np.full(n, np.nan)
    f["hr_mean_60s"] = _nan_roll_mean(f["hr_mean"], CONTEXT_WINDOWS)

    for b in RELATIVE_BASES:
        f[f"{b}_rel"], f[f"{b}_z"], f[f"{b}_local"] = _relative(f[b])
    for b in LONG_BASES:
        for m in (2, 5):
            f[f"{b}_{m}m"] = _nan_roll_mean(f[b], 6 * m)

    for b in REC_BASES:
        for q, v in zip((10, 50, 90), _qs(f[b], [10, 50, 90])):
            f[f"rec_{b}_q{q}"] = np.full(n, v)
    f["rec_hr_range"] = f["rec_hr_mean_q90"] - f["rec_hr_mean_q10"]
    dur = float(t[-1] - offset)
    f["rec_dur"] = np.full(n, dur)
    centre = np.arange(n) * WINDOW_S + WINDOW_S / 2 - offset
    f["t_s"] = centre
    f["t_frac"] = centre / max(dur, 1.0)
    f["t_left"] = dur - centre

    hr = f["hr_mean"]
    f["hr_vs_min3m"] = hr - _past_extreme(hr, 18, "min")
    f["hr_vs_runmax"] = hr - np.fmax.accumulate(np.where(np.isfinite(hr), hr, -np.inf))
    f["hr_vs_runmin"] = hr - np.fmin.accumulate(np.where(np.isfinite(hr), hr, np.inf))
    f["eda_vs_min3m"] = f["eda_tonic"] - _past_extreme(f["eda_tonic"], 18, "min")
    f["temp_vs_max3m"] = f["temp_mean"] - _past_extreme(f["temp_mean"], 18, "max")
    for b in TREND_BASES:
        a = f[b]
        p1, n1 = _offset_mean(a, -6, -1), _offset_mean(a, 1, 6)
        p3, n3 = _offset_mean(a, -18, -1), _offset_mean(a, 1, 18)
        f[f"{b}_d1m"], f[f"{b}_d3m"] = n1 - p1, n3 - p3
        f[f"{b}_vs_past3m"], f[f"{b}_vs_fut3m"] = a - p3, a - n3
    for b in ("bvp_periodicity", "bvp_vhf", "scr_rate_60s"):
        for q, v in zip((10, 50, 90), _qs(f[b], [10, 50, 90])):
            f[f"rec_{b}_q{q}"] = np.full(n, v)

    for L in LAGS:  # value of window i+L (NaN past the ends)
        for b in LAG_BASES:
            a = np.full(n, np.nan)
            if L > 0 and L < n:
                a[:-L] = f[b][L:]
            elif L < 0 and -L < n:
                a[-L:] = f[b][:L]
            f[f"{b}_lag{L}"] = a

    X = np.column_stack([f[k] for k in ALL_FEATURES]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, row_wid


def smooth_probs(P: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average of class probabilities over k windows (edges use fewer windows)."""
    if k <= 1 or len(P) < 2:
        return P
    norm = _same(np.ones(len(P)), k)
    return np.column_stack([_same(P[:, c], k) / norm for c in range(P.shape[1])])


S2_LAGS = (-18, -12, -9, -6, -4, -2, -1, 1, 2, 4, 6, 9, 12, 18)
S2_WINDOWS = (3, 9, 19, 37)
S2_SPANS = ((-6, -1), (1, 6), (-18, -1), (1, 18), (-60, -1), (1, 60))


def stage2_features(P: np.ndarray) -> np.ndarray:
    """Stage-2 input: the stage-1 class probabilities of the surrounding windows of the same input.

    Lets the second model learn the shape of the protocol (a stress task lasts 1-4 minutes, rest
    follows the cool-down, ...) instead of a fixed moving average. Rows must be in window order."""
    n = len(P)
    cols = [P]
    for L in S2_LAGS:  # probabilities of window i+L
        a = np.full_like(P, np.nan)
        if 0 < L < n:
            a[:-L] = P[L:]
        elif 0 < -L < n:
            a[-L:] = P[:L]
        cols.append(a)
    for k in S2_WINDOWS:  # centred averages
        cols.append(smooth_probs(P, k))
    for lo, hi in S2_SPANS:  # averages of the past / future 1, 3 and 10 minutes
        cols.append(np.column_stack([_offset_mean(P[:, c], lo, hi) for c in range(P.shape[1])]))
    cols.append(np.tile(P.mean(axis=0), (n, 1)))  # class mix of the whole input
    pos = np.arange(n, dtype=np.float64)
    cols.append(np.column_stack([np.full(n, np.log1p(n)), pos, n - 1 - pos]))
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
class YourModel(ADM_Base):
    def load_model(self, onnx_file: pathlib.Path) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        meta = json.loads(self.session.get_modelmeta().custom_metadata_map["eh_config"])
        ins = {i.name: i for i in self.session.get_inputs()}
        outs = [o.name for o in self.session.get_outputs()]
        self.two_stage = bool(meta.get("two_stage", False))
        if self.two_stage:  # one file, two tree models: s1_* (features) and s2_* (stage-1 context)
            self.input_name, self.input2_name = "s1_X", "s2_X"
            self.prob_name, self.prob2_name = "s1_probabilities", "s2_probabilities"
            self.empty1 = np.zeros((0, ins["s1_X"].shape[1]), dtype=np.float32)
            self.empty2 = np.zeros((0, ins["s2_X"].shape[1]), dtype=np.float32)
        else:
            self.input_name = next(iter(ins))
            self.prob_name = next((o for o in outs if "prob" in o.lower()), outs[-1])
        self.cols = np.array([ALL_FEATURES.index(c) for c in meta["features"]], dtype=np.int64)
        self.smooth_k = int(meta.get("smooth_k", 1))
        self.log_bias = np.asarray(meta.get("log_bias", [0.0, 0.0, 0.0]), dtype=np.float64)

    def predict_windows(self, df: pd.DataFrame):
        X, row_wid = compute_features(df)
        X = np.ascontiguousarray(X[:, self.cols])
        if self.two_stage:
            P = self.session.run([self.prob_name], {self.input_name: X, self.input2_name: self.empty2})[0]
            Z = stage2_features(np.asarray(P, dtype=np.float64))
            P = self.session.run([self.prob2_name], {self.input_name: self.empty1, self.input2_name: Z})[0]
        else:
            P = self.session.run([self.prob_name], {self.input_name: X})[0]
        P = smooth_probs(np.asarray(P, dtype=np.float64), self.smooth_k)
        score = np.log(np.maximum(P, 1e-9)) + self.log_bias
        return score.argmax(axis=1), row_wid

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(input)
        if n_rows == 0:
            return pd.DataFrame({"Activity": pd.Series(dtype=object)}, index=input.index)
        data = input[INPUT_COLS]
        if not isinstance(data.index, pd.DatetimeIndex):  # no timestamps: assume 64 Hz
            data = data.set_axis(pd.date_range("2000-01-01", periods=n_rows, freq="15625us"))
        cls = np.zeros(n_rows, dtype=np.int8)  # 0 = "Rest"
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for s, e in _input_segments(data.index):
                try:
                    win_pred, row_wid = self.predict_windows(data.iloc[s:e])
                    cls[s:e] = win_pred[np.clip(row_wid, 0, len(win_pred) - 1)]
                except Exception:  # never crash the scorer; this piece keeps "Rest"
                    self.fallbacks = getattr(self, "fallbacks", 0) + 1  # checked by package_submission.py
        # explicit object dtype: pandas 3 would otherwise convert 100k+ strings to Arrow (slow)
        return pd.DataFrame(CLASSES[cls], index=input.index, columns=["Activity"], dtype=object, copy=False)


MAX_JUMP_NS = 300 * 10**9  # longer gaps than any inside a training recording (max 270 s)


def _input_segments(index: pd.DatetimeIndex):
    """Split the input where time goes backwards or jumps > 5 minutes.

    Several recordings glued into one input would otherwise create millions of empty windows
    (years apart) or negative window numbers (time going backwards), and mix two people's
    statistics. Each piece is handled as its own recording.
    """
    ns = index.asi8
    d = np.diff(ns)
    breaks = np.flatnonzero((d < 0) | (d > MAX_JUMP_NS)) + 1
    return zip(np.r_[0, breaks], np.r_[breaks, len(ns)])


def get_model() -> ADM_Base:
    model = YourModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model
