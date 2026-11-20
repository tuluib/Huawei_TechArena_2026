"""Huawei Tech Arena 2026 - activity detection (Rest / Activity / Stress) from wristband signals."""
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
    "eda_phasic_ratio", "scr_per_hr", "hr_temp_div",
    "hr_above_floor", "eda_above_floor", "bvp_rms_above_floor", "hrv_drop_from_peak",
    "hr_range_norm", "eda_range_norm", "eda_rel_slope", "hr_motion_gap",
]
CONTEXT_FEATURES = ["hrv_rmssd_60s", "hrv_sdnn_60s", "scr_rate_60s", "temp_slope_60s", "eda_slope_60s",
                    "hr_mean_60s"]
RELATIVE_BASES = ["hr_mean", "bvp_pulse_bpm", "eda_tonic", "temp_mean", "bvp_log_rms", "ibi_mean",
                  "scr_rate_60s", "hrv_rmssd_60s"]
RELATIVE_FEATURES = ([f"{b}_rel" for b in RELATIVE_BASES] + [f"{b}_z" for b in RELATIVE_BASES]
                     + [f"{b}_local" for b in RELATIVE_BASES])
LONG_BASES = ["hr_mean", "bvp_log_rms", "bvp_artifact_frac", "eda_tonic", "scr_rate_60s"]
LONG_FEATURES = [f"{b}_{w}m" for b in LONG_BASES for w in (2, 5)]
REC_BASES = ["hr_mean", "bvp_artifact_frac", "bvp_log_rms", "eda_tonic", "temp_mean"]
REC_FEATURES = [f"rec_{b}_q{q}" for b in REC_BASES for q in (10, 50, 90)] + ["rec_hr_range", "rec_dur"]
TIME_FEATURES = ["t_s", "t_frac", "t_left"]
LAG_BASES = ["hr_mean_local", "eda_tonic_local", "bvp_log_rms_local", "scr_rate_60s", "hrv_rmssd_60s_local",
             "temp_slope_60s", "bvp_artifact_frac", "hr_mean_rel"]
LAGS = (-18, -12, -6, -3, 3, 6, 12, 18)
LAG_FEATURES = [f"{b}_lag{L}" for L in LAGS for b in LAG_BASES]
ALL_FEATURES = (WINDOW_FEATURES + CONTEXT_FEATURES + RELATIVE_FEATURES + LONG_FEATURES
                + REC_FEATURES + TIME_FEATURES + LAG_FEATURES)
INPUT_COLS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


_TICKS_PER_S = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
_HANN = {}


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


def _regular_grid_start(t, ok, size):
    """Grid index of the first sample if all samples are present and exactly 1/64 s apart, else None.
    Then sample k sits at grid position g0 + k: the same window and slot the timestamp formula gives."""
    if len(t) < 2 or not ok.all():
        return None
    g0 = t[0] * FS_BVP
    if g0 != np.floor(g0) or g0 < 0 or g0 + len(t) > size:
        return None
    if not np.all(np.diff(t) == 1.0 / FS_BVP):
        return None
    return int(g0)


def _bvp_artifact_frac(t, x, n):
    """Share of BVP rows per 10 s window that are artifacts (= the old _bvp_artifacts + bincount per row,
    counted per second instead of per row: the same integer counts, so the same result)."""
    if len(t) == 0:
        return np.zeros(n)
    block = np.floor(t).astype(np.int64)
    valid = ~np.isnan(x)
    nb = block[-1] + 1
    rows = np.bincount(block, minlength=nb)
    cnt = np.bincount(block[valid], minlength=nb)
    ss = np.bincount(block[valid], weights=x[valid] ** 2, minlength=nb)
    rms = np.sqrt(ss / np.maximum(cnt, 1))
    typical = np.median(rms[cnt > 0]) if (cnt > 0).any() else 0.0
    bad_block = (cnt == 0) | (rms > 2.5 * typical) | (rms < 0.25 * typical)
    bad_rows = np.where(bad_block, rows, rows - cnt).astype(float)
    win = np.arange(nb) // int(WINDOW_S)
    return (np.bincount(win, bad_rows, n)[:n]
            / np.maximum(np.bincount(win, rows.astype(float), n)[:n], 1))


def _bvp_spectral(t, x, n):
    per_win = int(WINDOW_S * FS_BVP)
    ok = ~np.isnan(x)
    mat = np.zeros((n, per_win))
    g0 = _regular_grid_start(t, ok, n * per_win)
    if g0 is not None:  # usual case: every sample present on the 64 Hz grid -> place by position
        flat = mat.reshape(-1)
        flat[g0:g0 + len(x)] = x
        edges = np.arange(n + 1) * per_win
        cnt = np.clip(np.minimum(edges[1:], g0 + len(x)) - np.maximum(edges[:-1], g0), 0, None)
    else:  # gaps or missing samples: place every sample by its timestamp
        w = _wid(t[ok])
        pos = np.clip(np.round((t[ok] - w * WINDOW_S) * FS_BVP).astype(np.int64), 0, per_win - 1)
        keep = (w >= 0) & (w < n)
        mat[w[keep], pos[keep]] = x[ok][keep]
        cnt = np.bincount(w[keep], minlength=n)
    if per_win not in _HANN:
        _HANN[per_win] = np.hanning(per_win)
    power = np.abs(np.fft.rfft(mat * _HANN[per_win], axis=1)) ** 2
    freqs = np.fft.rfftfreq(per_win, d=1.0 / FS_BVP)
    band = (freqs >= PULSE_BAND_HZ[0]) & (freqs <= PULSE_BAND_HZ[1])
    band_power = power[:, band]
    pulse_bpm = freqs[band][band_power.argmax(axis=1)] * 60.0
    peak_ratio = band_power.max(axis=1) / band_power.sum(axis=1)
    log_rms = np.log1p(np.sqrt((mat ** 2).sum(axis=1) / np.maximum(cnt, 1)))
    empty = cnt < per_win // 2
    for a in (pulse_bpm, peak_ratio, log_rms):
        a[empty] = np.nan
    return pulse_bpm, peak_ratio, log_rms


def _same(a, k):
    full = np.convolve(a, np.ones(k), mode="full")
    start = (k - 1) // 2
    return full[start:start + len(a)]


def _roll_sum(a, k):
    return _same(a, k)


def _nan_roll_mean(a, k):
    ok = np.isfinite(a)
    s = _same(np.where(ok, a, 0.0), k)
    c = _same(ok.astype(np.float64), k)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _nan_roll_median(a: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(a).rolling(k, center=True, min_periods=3).median().to_numpy()


def _relative(a):
    nan = np.full(len(a), np.nan)
    v = a[np.isfinite(a)]
    if len(v) < 2:
        return nan, nan, nan
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    spread = q75 - q25
    z = (a - med) / spread if spread > 0 else nan
    local = a - _nan_roll_median(a, LOCAL_WINDOWS)
    return a - med, z, local


def _qs(a, qs):
    v = a[np.isfinite(a)]
    return np.percentile(v, qs) if len(v) else np.full(len(qs), np.nan)


def compute_features(df: pd.DataFrame):
    idx = df.index
    origin = idx[0].floor("10s")
    td = idx - origin  # integer ticks / ticks per second == total_seconds(), bit for bit, ~10x faster
    t = td.asi8 / _TICKS_PER_S[td.unit]
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

    bx = np.clip(bx, -1e5, 1e5)
    bx = _filtfilt_segments(bt, bx, SOS_BVP, drop_short=True, one_pass=True)
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
    if len(phasic) > 4 and np.isfinite(phasic).any():
        min_rise = max(0.02, 0.005 * float(np.nanmedian(ex)))
        peaks, _ = signal.find_peaks(np.nan_to_num(phasic), prominence=min_rise, distance=int(FS_EDA))
        scr_t = et[peaks]

    f = {}
    f["bvp_pulse_bpm"], f["bvp_peak_ratio"], f["bvp_log_rms"] = _bvp_spectral(bt, bx, n)
    f["bvp_artifact_frac"] = _bvp_artifact_frac(bt, bx, n)
    _, f["hr_mean"], f["hr_std"], f["hr_slope"] = _window_stats(ht, hx, n)

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
    temp_ctx = _nan_roll_mean(f["temp_mean"], CONTEXT_WINDOWS)
    eda_ctx = _nan_roll_mean(f["eda_tonic"], CONTEXT_WINDOWS)
    if n > 1:
        f["temp_slope_60s"] = np.gradient(temp_ctx) / WINDOW_S * 60.0
        f["eda_slope_60s"] = np.gradient(eda_ctx) / WINDOW_S * 60.0
    else:
        f["temp_slope_60s"] = np.full(n, np.nan)
        f["eda_slope_60s"] = np.full(n, np.nan)
    f["hr_mean_60s"] = _nan_roll_mean(f["hr_mean"], CONTEXT_WINDOWS)

    # Person-normalized physiological interactions & resting-floor anchors
    hr_q15, hr_q85 = _qs(f["hr_mean"], [15, 85])
    eda_q15, eda_q85 = _qs(f["eda_tonic"], [15, 85])
    bvp_q15 = _qs(f["bvp_log_rms"], [15])[0]
    hrv_q85 = _qs(f["hrv_rmssd_60s"], [85])[0]

    f["eda_phasic_ratio"] = f["eda_phasic_std"] / (np.maximum(f["eda_tonic"], 0.0) + 0.05)
    f["scr_per_hr"] = f["scr_rate_60s"] / np.maximum(f["hr_mean_60s"], 40.0)
    f["hr_above_floor"] = f["hr_mean"] - hr_q15
    f["eda_above_floor"] = f["eda_tonic"] - eda_q15
    f["bvp_rms_above_floor"] = f["bvp_log_rms"] - bvp_q15
    f["hrv_drop_from_peak"] = hrv_q85 - f["hrv_rmssd_60s"]
    f["hr_range_norm"] = f["hr_above_floor"] / (np.maximum(hr_q85 - hr_q15, 0.0) + 2.0)
    f["eda_range_norm"] = f["eda_above_floor"] / (np.maximum(eda_q85 - eda_q15, 0.0) + 0.05)
    f["eda_rel_slope"] = f["eda_slope_60s"] / (np.maximum(f["eda_tonic"], 0.0) + 0.05)
    f["hr_motion_gap"] = f["hr_above_floor"] - 20.0 * f["bvp_rms_above_floor"]

    for b in RELATIVE_BASES:
        f[f"{b}_rel"], f[f"{b}_z"], f[f"{b}_local"] = _relative(f[b])
    f["hr_temp_div"] = f["hr_mean_local"] - 10.0 * f["temp_slope_60s"]

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

    for L in LAGS:
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
    if k <= 1 or len(P) < 2:
        return P
    norm = _same(np.ones(len(P)), k)
    return np.column_stack([_same(P[:, c], k) / norm for c in range(P.shape[1])])


class YourModel(ADM_Base):
    def load_model(self, onnx_file: pathlib.Path) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outs = [o.name for o in self.session.get_outputs()]
        self.prob_name = next((o for o in outs if "prob" in o.lower()), outs[-1])
        meta = json.loads(self.session.get_modelmeta().custom_metadata_map["eh_config"])
        self.cols = np.array([ALL_FEATURES.index(c) for c in meta["features"]], dtype=np.int64)
        med = meta.get("medians")
        self.medians = (np.asarray(med, dtype=np.float32) if med is not None
                        else np.zeros(len(self.cols), dtype=np.float32))
        self.smooth_k = int(meta.get("smooth_k", 1))
        self.log_bias = np.asarray(meta.get("log_bias", [0.0, 0.0, 0.0]), dtype=np.float64)

    def predict_windows(self, df: pd.DataFrame):
        X, row_wid = compute_features(df)
        X_sub = np.ascontiguousarray(X[:, self.cols], dtype=np.float32)
        X_in = np.ascontiguousarray(np.where(np.isfinite(X_sub), X_sub, self.medians), dtype=np.float32)
        P = self.session.run([self.prob_name], {self.input_name: X_in})[0]
        P = smooth_probs(np.asarray(P, dtype=np.float64), self.smooth_k)
        score = np.log(np.maximum(P, 1e-9)) + self.log_bias
        return score.argmax(axis=1), row_wid

    def run(self, input: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(input)
        if n_rows == 0:
            return pd.DataFrame({"Activity": pd.Series(dtype=object)}, index=input.index)
        data = input  # compute_features reads only INPUT_COLS by name; no 6-column copy
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_axis(pd.date_range("2000-01-01", periods=n_rows, freq="15625us"))
        cls = np.zeros(n_rows, dtype=np.int8)  # 0 = "Rest"
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for s, e in _input_segments(data.index):
                try:
                    win_pred, row_wid = self.predict_windows(data.iloc[s:e])
                    cls[s:e] = win_pred[np.clip(row_wid, 0, len(win_pred) - 1)]
                except Exception:
                    pass
        return pd.DataFrame(CLASSES[cls], index=input.index, columns=["Activity"], dtype=object, copy=False)


MAX_JUMP_NS = 300 * 10**9


def _input_segments(index: pd.DatetimeIndex):
    ns = index.asi8
    d = np.diff(ns)
    breaks = np.flatnonzero((d < 0) | (d > MAX_JUMP_NS)) + 1
    return zip(np.r_[0, breaks], np.r_[breaks, len(ns)])


def get_model() -> ADM_Base:
    model = YourModel()
    model.load_model(pathlib.Path(__file__).with_suffix(".onnx"))
    return model