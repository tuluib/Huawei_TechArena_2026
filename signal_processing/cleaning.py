"""Cleaning of the raw wristband signals.

The input DataFrame puts every sensor on one 64 Hz time grid. Only BVP is sampled at 64 Hz;
EDA and temperature are 4 Hz, HR is 1 Hz and IBI is one value per detected beat, so those
columns are mostly NaN. Each signal is therefore taken out as its own stream (its native
samples only), cleaned with filters that suit that sensor, and kept with its own time axis.

Rules used everywhere:
- Never filter across a recording gap (> 1 s without samples): filters are applied per segment.
- Physically impossible values are rejected (set to NaN), not clipped.
- Filter coefficients are designed once at import time (speed matters: 20 % of the score).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from signal_processing.windows import seconds_since_start

FS_BVP = 64.0
FS_EDA = 4.0
FS_TEMP = 4.0
MAX_GAP_S = 1.0

# Plausible physiological ranges; anything outside is a sensor error.
HR_RANGE = (35.0, 220.0)
IBI_RANGE = (0.27, 2.0)  # seconds, i.e. 30-220 beats per minute
TEMP_RANGE = (20.0, 42.0)  # wrist skin temperature in Celsius
EDA_MIN = 0.02  # microsiemens; below this the electrodes have lost skin contact

# Butterworth filters as second-order sections (numerically stable).
# BVP: keep 0.5-4 Hz = 30-240 bpm pulse band; removes baseline wander and high-frequency noise.
SOS_BVP = signal.butter(2, [0.5, 4.0], btype="bandpass", fs=FS_BVP, output="sos")
# EDA: skin conductance changes slowly; everything above 1 Hz is electrical noise.
SOS_EDA = signal.butter(2, 1.0, btype="lowpass", fs=FS_EDA, output="sos")
# EDA tonic level (SCL): the very slow part (< 0.05 Hz). Phasic part (SCRs) = EDA - tonic.
SOS_EDA_TONIC = signal.butter(2, 0.05, btype="lowpass", fs=FS_EDA, output="sos")


@dataclass
class Stream:
    """One sensor in its native sampling: times in seconds since start, and values."""

    t: np.ndarray
    x: np.ndarray


@dataclass
class CleanSignals:
    """All cleaned signals of one recording."""

    t0: pd.Timestamp
    duration_s: float
    bvp: Stream  # band-pass filtered pulse wave
    bvp_artifact: np.ndarray  # bool per BVP sample: inside a 1-s block with abnormal amplitude
    eda: Stream  # low-pass filtered skin conductance
    eda_tonic: np.ndarray  # slow skin conductance level, same times as eda
    eda_phasic: np.ndarray  # fast responses, same times as eda
    scr_times: np.ndarray  # times of skin conductance responses (peaks of the phasic part)
    temp: Stream
    hr: Stream
    ibi: Stream  # beat time, beat-to-beat interval in seconds (NaN when rejected)
    ibi_consecutive: np.ndarray  # bool: this beat directly follows the previous one


def _stream(values: np.ndarray, t: np.ndarray) -> Stream:
    keep = ~np.isnan(values)
    return Stream(t[keep], values[keep])


def _segments(t: np.ndarray, max_gap_s: float = MAX_GAP_S):
    """Yield (start, end) index pairs of runs without time gaps."""
    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    yield from zip(np.r_[0, breaks], np.r_[breaks, len(t)])


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs (needed before filtering)."""
    bad = np.isnan(x)
    if not bad.any() or bad.all():
        return x
    x = x.copy()
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _filtfilt_segments(stream: Stream, sos: np.ndarray, short: str = "keep") -> np.ndarray:
    """Zero-phase filter each gap-free segment. Rejected samples (NaN) stay NaN.

    short="keep" returns too-short segments unfiltered, short="nan" drops them.
    """
    out = np.full(len(stream.x), np.nan)
    padlen = 3 * (2 * len(sos) + 1)  # what sosfiltfilt needs
    for s, e in _segments(stream.t):
        seg = stream.x[s:e]
        bad = np.isnan(seg)
        if bad.all():
            continue
        if e - s > padlen:
            filtered = signal.sosfiltfilt(sos, _interp_nan(seg))
        elif short == "keep":
            filtered = seg.copy()
        else:
            continue
        filtered[bad] = np.nan
        out[s:e] = filtered
    return out


def _reject_outside(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = x.copy()
    x[(x < lo) | (x > hi)] = np.nan
    return x


def _reject_spikes(x: np.ndarray, kernel: int, max_dev: float, relative: bool = False) -> np.ndarray:
    """Reject samples that jump away from their running median (single-sample spikes)."""
    if len(x) < kernel:
        return x
    med = signal.medfilt(_interp_nan(x), kernel)
    dev = np.abs(x - med)
    limit = max_dev * np.abs(med) if relative else max_dev
    x = x.copy()
    x[dev > limit] = np.nan
    return x


def _bvp_artifacts(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Flag 1-second blocks whose amplitude is far from the recording's typical amplitude.

    Motion and loose fit make the optical signal jump (very large) or flatten (very small).
    During exercise this happens a lot, so the artifact share is itself a useful feature.
    """
    block = np.floor(t).astype(np.int64)
    valid = ~np.isnan(x)
    n = block[-1] + 1 if len(block) else 0
    cnt = np.bincount(block[valid], minlength=n)
    ss = np.bincount(block[valid], weights=x[valid] ** 2, minlength=n)
    rms = np.sqrt(ss / np.maximum(cnt, 1))
    typical = np.median(rms[cnt > 0]) if (cnt > 0).any() else 0.0
    bad_block = (cnt == 0) | (rms > 2.5 * typical) | (rms < 0.25 * typical)
    return bad_block[block] | ~valid


def clean_signals(df: pd.DataFrame, clean: bool = True) -> CleanSignals:
    """Split the 64 Hz grid into per-sensor streams and clean them.

    ``clean=False`` only removes the mean of BVP and skips all filtering and rejection. It
    exists to measure how much the cleaning helps (see evaluate_features.py).
    """
    t = seconds_since_start(df.index)
    col = {c: df[c].to_numpy(dtype=np.float64) for c in
           ("BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature")}

    bvp = _stream(col["BVP"], t)
    eda = _stream(col["EDA"], t)
    temp = _stream(col["Temperature"], t)
    hr = _stream(col["HR"], t)
    ibi = _stream(col["IBI_duration"], t)
    ibi_dt = col["IBI_dt"][~np.isnan(col["IBI_duration"])]

    if clean:
        bvp.x = _filtfilt_segments(bvp, SOS_BVP, short="nan")
        bvp_artifact = _bvp_artifacts(bvp.t, bvp.x)

        eda.x = np.where(eda.x < EDA_MIN, np.nan, eda.x)
        eda.x = _reject_spikes(eda.x, kernel=5, max_dev=0.3, relative=True)
        eda.x = _filtfilt_segments(eda, SOS_EDA)
        tonic = _filtfilt_segments(eda, SOS_EDA_TONIC)

        temp.x = _reject_outside(temp.x, *TEMP_RANGE)
        temp.x = _reject_spikes(temp.x, kernel=9, max_dev=0.5)

        hr.x = _reject_outside(hr.x, *HR_RANGE)
        hr.x = _reject_spikes(hr.x, kernel=5, max_dev=15.0)

        ibi.x = _reject_outside(ibi.x, *IBI_RANGE)
        ibi.x = _reject_spikes(ibi.x, kernel=5, max_dev=0.25, relative=True)  # ectopic / missed beats
    else:
        bvp.x = bvp.x - np.nanmean(bvp.x) if len(bvp.x) else bvp.x
        bvp_artifact = np.zeros(len(bvp.x), dtype=bool)
        tonic = eda.x.copy()

    phasic = eda.x - tonic
    scr_times = np.empty(0)
    if len(phasic) > 4 and not np.isnan(phasic).all():
        # A skin conductance response: a rise of >= 0.02 uS or 0.5 % of the person's level
        # (people with high EDA have proportionally bigger responses), at most one per second.
        min_rise = max(0.02, 0.005 * float(np.nanmedian(eda.x)))
        peaks, _ = signal.find_peaks(np.nan_to_num(phasic), prominence=min_rise, distance=int(FS_EDA))
        scr_times = eda.t[peaks]

    consecutive = np.abs(ibi_dt - col["IBI_duration"][~np.isnan(col["IBI_duration"])]) < 0.05

    return CleanSignals(
        t0=df.index[0], duration_s=float(t[-1]) if len(t) else 0.0,
        bvp=bvp, bvp_artifact=bvp_artifact,
        eda=eda, eda_tonic=tonic, eda_phasic=phasic, scr_times=scr_times,
        temp=temp, hr=hr, ibi=ibi, ibi_consecutive=consecutive,
    )
