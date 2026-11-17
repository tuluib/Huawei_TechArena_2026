"""Figures: what the raw signals look like, what cleaning does, and which features separate the states.

    python -m signal_processing.explore            (needs matplotlib; offline only)

Writes PNGs to signal_processing/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signal_processing.cleaning import clean_signals
from signal_processing.evaluate_features import build_dataset
from signal_processing.labels import STATES, phase_to_state
from signal_processing.loading import list_recordings, load, model_input
from signal_processing.windows import seconds_since_start

OUT = Path(__file__).resolve().parent / "figures"
COLORS = {"Rest": "#9ecae1", "Stress": "#fc9272", "Activity": "#a1d99b"}


def _shade_states(ax, df: pd.DataFrame) -> None:
    """Colour the background by target state (minutes on the x axis)."""
    t = seconds_since_start(df.index) / 60
    state = df["Phase"].map(phase_to_state).to_numpy()
    change = np.flatnonzero(state[1:] != state[:-1]) + 1
    for s, e in zip(np.r_[0, change], np.r_[change, len(state)]):
        if state[s] in COLORS:
            ax.axvspan(t[s], t[e - 1], color=COLORS[state[s]], alpha=0.35, lw=0)


def _get(name: str):
    rec = next(r for r in list_recordings() if r.name == name)
    df = load(rec)
    return df, clean_signals(model_input(df)), clean_signals(model_input(df), clean=False)


def fig_recording(name: str) -> None:
    """Whole recording: each signal raw (grey) vs cleaned (colour), states shaded."""
    df, c, r = _get(name)
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    m = lambda s: s / 60

    ax = axes[0]
    ax.plot(m(r.hr.t), r.hr.x, color="0.7", lw=0.8, label="raw")
    ax.plot(m(c.hr.t), c.hr.x, color="C3", lw=1, label="cleaned")
    ax.set_ylabel("HR (bpm)")

    ax = axes[1]
    ax.plot(m(r.ibi.t), r.ibi.x * 1000, ".", color="0.6", ms=3, label="raw beats")
    rejected = np.isnan(c.ibi.x)
    ax.plot(m(c.ibi.t[rejected]), r.ibi.x[rejected] * 1000, "x", color="k", ms=4, label="rejected")
    ax.plot(m(c.ibi.t), c.ibi.x * 1000, ".", color="C0", ms=3, label="kept")
    ax.set_ylabel("IBI (ms)")

    ax = axes[2]
    ax.plot(m(r.eda.t), r.eda.x, color="0.7", lw=0.8, label="raw")
    ax.plot(m(c.eda.t), c.eda.x, color="C2", lw=1, label="filtered")
    ax.plot(m(c.eda.t), c.eda_tonic, color="k", lw=1, ls="--", label="tonic (SCL)")
    ax.plot(m(c.scr_times), np.interp(c.scr_times, c.eda.t, np.nan_to_num(c.eda.x)), "v", color="C1", ms=4,
            label="SCR peaks")
    ax.set_ylabel("EDA (uS)")

    ax = axes[3]
    ax.plot(m(r.temp.t), r.temp.x, color="0.7", lw=0.8, label="raw")
    ax.plot(m(c.temp.t), c.temp.x, color="C4", lw=1, label="cleaned")
    ax.set_ylabel("Skin temp (C)")

    ax = axes[4]
    blk = np.floor(c.bvp.t).astype(int)
    amp = pd.Series(np.abs(c.bvp.x)).groupby(blk).max()
    art = pd.Series(c.bvp_artifact).groupby(blk).mean()
    ax.plot(amp.index / 60, amp.to_numpy(), color="C0", lw=0.8, label="BVP amplitude (filtered, per s)")
    ax.fill_between(art.index / 60, 0, amp.max() * (art.to_numpy() > 0.5), color="k", alpha=0.25,
                    label="flagged artifact")
    ax.set_ylabel("BVP amplitude")
    ax.set_xlabel("minutes")

    for ax in axes:
        _shade_states(ax, df)
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"{name}: raw vs cleaned (background: Rest blue, Stress red, Activity green)")
    fig.tight_layout()
    fig.savefig(OUT / f"recording_{name.replace('/', '_')}.png", dpi=110)
    plt.close(fig)


def fig_bvp_zoom(name: str, start_s: float, seconds: float = 20.0) -> None:
    """20 s of BVP: raw vs band-pass filtered, and their spectra."""
    _, c, r = _get(name)
    sel = (r.bvp.t >= start_s) & (r.bvp.t < start_s + seconds)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(r.bvp.t[sel] - start_s, r.bvp.x[sel], color="0.6", lw=0.8, label="raw")
    a1.plot(c.bvp.t[sel] - start_s, c.bvp.x[sel], color="C0", lw=1.2, label="0.5-4 Hz band-pass")
    a1.set_xlabel("seconds")
    a1.legend(fontsize=8)
    a1.set_title(f"{name}: BVP at {start_s / 60:.1f} min")
    for x, lab, col in ((r.bvp.x[sel], "raw", "0.6"), (c.bvp.x[sel], "filtered", "C0")):
        x = np.nan_to_num(x - np.nanmean(x))
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        fr = np.fft.rfftfreq(len(x), 1 / 64)
        a2.semilogy(fr, p + 1e-9, color=col, lw=1, label=lab)
    a2.axvspan(0.5, 4, color="C0", alpha=0.1)
    a2.set_xlim(0, 10)
    a2.set_xlabel("Hz")
    a2.set_title("spectrum (pulse peak inside the shaded band)")
    fig.tight_layout()
    fig.savefig(OUT / f"bvp_zoom_{name.replace('/', '_')}.png", dpi=110)
    plt.close(fig)


def fig_feature_boxes() -> None:
    """Which features separate the three states? Box plots across all windows."""
    data = build_dataset(clean=True)
    feats = ["hr_mean_rel", "hrv_rmssd_60s", "bvp_log_rms_rel", "bvp_artifact_frac", "eda_tonic_local",
             "scr_rate_60s", "temp_mean_local", "temp_slope_60s"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, f in zip(axes.flat, feats):
        vals = [data.loc[data["state"] == s, f].dropna() for s in STATES]
        lo, hi = np.nanpercentile(data[f], [2, 98])
        ax.boxplot(vals, tick_labels=STATES, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="white"))
        for patch, s in zip(ax.patches, STATES):
            patch.set_facecolor(COLORS[s])
        ax.set_ylim(lo, hi)
        ax.set_title(f, fontsize=9)
    fig.suptitle("Feature values per state (all 14,757 scored windows)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_boxes.png", dpi=110)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig_recording("STRESS/S10")
    fig_recording("AEROBIC/S05")
    fig_bvp_zoom("STRESS/S10", start_s=300)
    fig_bvp_zoom("AEROBIC/S05", start_s=1500)
    fig_feature_boxes()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
