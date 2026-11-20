"""Local score of a model: quality on the 6 holdout people in every input format, plus speed.

    python score.py TULUI --quick                     # by name: looks in submissions/candidates/, submissions/, models/
    python score.py TULUI holdout --quick             # several side by side (holdout = our V4 test model)
    python score.py                                   # the current model (src/your_model.py + .onnx)
    python score.py ../submissions/submissionV4.zip   # a submission zip
    python score.py DIR                               # an unzipped submission folder
    python score.py A.zip B.zip ...                   # several models side by side
    python score.py --quick                           # fewer shuffled recordings (faster)

Quality = 80 x macro-F1 over 10-second windows (Rest / Activity / Stress), pooled over all recordings.
Primary diagnostic: T3 within-session recordings, with phases reordered and block lengths varied.
T2 (reordered full blocks) and T4 (same-person cross-session mixtures) are also reported. T4 is a
robustness stress test, not the headline score estimate: the V5.1/TULUI leaderboard ordering directly
contradicts the large T4 gap. No local variant has enough evidence to predict the hidden score reliably.
Speed = 20 x (1 - mean time per call / 0.3 s) over 50 calls on one 34-minute recording, single core.

The score is only honest for a model trained WITHOUT the holdout people
(S16, S17, S18, f11, f12, f13), e.g. after `python train_v4.py holdout`. A model trained on everyone
(what you upload) has seen them, so its quality here is inflated; a warning is printed.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"  # the grader uses a single core
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.simplefilter("ignore")

import importlib.util
import json
import pathlib
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

SRC = pathlib.Path(__file__).resolve().parent
sys.path.append(str(SRC))  # adm_base_class, eval_harness
from eval_harness import (PROTOCOLS, REQUIRED_INPUT_COLS, VAL_SUBJECTS, map_phase_to_activity,
                          window_10s_mode)

DATA = SRC.parent / "data"
LAB = {"Rest": 0, "Activity": 1, "Stress": 2}
FORMATS = [("whole", None), ("10-min pieces", 10), ("2-min pieces", 2), ("1-min pieces", 1)]
SPEED_FILE = ("AEROBIC", "S16")


# --------------------------------------------------------------------------------------------
# Loading models and data
# --------------------------------------------------------------------------------------------
def load_model(path: pathlib.Path):
    """Import your_model.py from a folder or a zip and call its get_model()."""
    if path.suffix == ".zip":
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="score_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    if not (path / "your_model.py").exists():  # zips packed with a folder inside (e.g. SHEH.zip)
        found = sorted(path.rglob("your_model.py"))
        if not found:
            sys.exit(f"No your_model.py in {path}")
        print(f"  NOTE: your_model.py is in a subfolder ({found[0].parent.name}/), not at the zip root")
        path = found[0].parent
    spec = importlib.util.spec_from_file_location(f"model_{abs(hash(str(path)))}", path / "your_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model()


def model_info(model) -> dict:
    try:
        return json.loads(model.session.get_modelmeta().custom_metadata_map["eh_config"])
    except Exception:
        return {}


def model_feature_summary(info: dict) -> str:
    """Report feature counts for both tabular and dual-rate sequence models."""
    features = info.get("features")
    if features:
        return f"{len(features)} features"
    n2 = len(info.get("feats_2s", []))
    n10 = len(info.get("feats_10s", []))
    if n2 or n10:
        return f"{n2 + n10} features ({n2} @ 2s + {n10} @ 10s)"
    return "feature count unknown"


def model_arch_summary(info: dict) -> str:
    if info.get("architecture"):
        return str(info["architecture"])
    if info.get("feats_2s") and info.get("feats_10s"):
        return "dual-rate sequence"
    return "two-stage" if info.get("two_stage", False) else "single-stage"


def holdout_recordings():
    return [pd.read_parquet(DATA / p / f"{u}.parquet")
            for p in PROTOCOLS for u in VAL_SUBJECTS if (DATA / p / f"{u}.parquet").exists()]


def pieces(df, minutes, seed):
    """Cut a recording into pieces of `minutes` (random start, fixed seed); None = whole."""
    if minutes is None:
        return [df]
    t = (df.index - df.index[0]).total_seconds().to_numpy()
    size = 60.0 * minutes
    g = np.floor((t + np.random.default_rng(seed).uniform(0, size)) / size).astype(np.int64)
    b = np.flatnonzero(np.diff(g)) + 1
    return [df.iloc[s:e] for s, e in zip(np.r_[0, b], np.r_[b, len(df)])]


def shuffled_recordings(recs, seed=0, per_person=4, within=False, keep=(0.3, 1.0), shuffle=True):
    """Recordings like the official validation set: 15-35 minutes of ONE person, phases in random
    order with random durations (the official set "does not follow the training phase order or
    durations"). Each phase block (Baseline, Stroop, 70 rpm, Rest, ...) is cut to a random 30-100 %
    of its length and the blocks are joined in random order on a 64 Hz grid.
    within=True (used by main): blocks stay inside their own recording (one session). This matched the
    leaderboard order of V1/V3/V4/TULUI (2026-11-18). within=False mixes a person's sessions; too harsh
    (Stress F1 ~0 for every model). per_person = shuffled copies per recording (within) or person.
    "Undefined" blocks are dropped. Fixed seed: every model gets the same recordings."""
    rng = np.random.default_rng(seed)
    by_user = {}
    for k, df in enumerate(recs):
        ph = df["Phase"].astype(str).to_numpy()
        cut = np.flatnonzero(ph[1:] != ph[:-1]) + 1
        # within=True: blocks are only shuffled inside their own recording (one session, one protocol)
        key = k if within else df["user_id"].iloc[0]
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(ph)]):
            if ph[s] != "Undefined" and e - s >= 64 * 20:  # keep blocks of 20 s or more
                by_user.setdefault(key, []).append(df.iloc[s:e])
    out = []
    for user, blocks in sorted(by_user.items()):
        for _ in range(per_person):
            target = rng.uniform(15, 35) * 60 * 64
            parts, total = [], 0
            for i in (rng.permutation(len(blocks)) if shuffle else range(len(blocks))):
                b = blocks[i]
                n = max(64 * 20, int(len(b) * rng.uniform(*keep)))
                start = rng.integers(0, len(b) - n + 1)
                parts.append(b.iloc[start:start + n])
                total += n
                if total >= target:
                    break
            rec = pd.concat(parts).iloc[:int(35 * 60 * 64)]  # at most 35 minutes
            rec.index = pd.Timestamp("2026-01-01 10:00:00") + pd.to_timedelta(np.arange(len(rec)) * 15625, unit="us")
            out.append(rec)
    return out


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
def f1_scores(yt, yp):
    cm = np.bincount(np.asarray(yt) * 3 + np.asarray(yp), minlength=9).reshape(3, 3).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.where(denom > 0, 2 * np.diag(cm) / np.maximum(denom, 1), 0.0)


def quality(model, recs, minutes=None):
    """Macro-F1 and per-class F1 (Rest, Activity, Stress) over every holdout recording.

    Predictions are put back by position (timestamps of different recordings can coincide)."""
    yt, yp = [], []
    for i, df in enumerate(recs):
        pred = np.concatenate([model.run(p[REQUIRED_INPUT_COLS])["Activity"].to_numpy()
                               for p in pieces(df, minutes, i)])
        true10 = window_10s_mode(map_phase_to_activity(df["Phase"]))
        pred10 = window_10s_mode(pd.Series(pred, index=df.index))
        ok = true10.notna() & pred10.notna()
        yt += [LAB[v] for v in true10[ok]]
        yp += [LAB[v] for v in pred10[ok]]
    f1 = f1_scores(yt, yp)
    return float(f1.mean()), f1


def time_per_call(model, df, runs=50) -> float:
    model.run(df)  # warm-up
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.run(df)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def find(name: str) -> pathlib.Path:
    """A path, or just a name: TULUI, holdout, submissionV4, ... (searched in the usual folders)."""
    p = pathlib.Path(name)
    if p.exists():
        return p
    root = SRC.parent
    for folder in (root / "submissions" / "candidates", root / "submissions", root / "models"):
        for cand in (folder / name, folder / f"{name}.zip"):
            if cand.exists():
                return cand
        for cand in folder.glob("*"):  # case-insensitive match, with or without .zip
            if cand.name.lower() in (name.lower(), f"{name.lower()}.zip"):
                return cand
    sys.exit(f"Model '{name}' not found. Put the zip in submissions/candidates/ or give its path.")


def main(args):
    quick = "--quick" in args
    paths = [find(a) for a in args if not a.startswith("--")] or [SRC]
    recs = holdout_recordings()
    tests = [("T3 shuffled within session (primary)",
              shuffled_recordings(recs, per_person=2 if quick else 4, within=True)),
             ("T2 reordered full phase blocks",
              shuffled_recordings(recs, per_person=1 if quick else 2, within=True, keep=(1.0, 1.0))),
             ("T4 mixed sessions (stress test)",
              shuffled_recordings(recs, per_person=1 if quick else 2, within=False)),
             ("training order (reference)", recs)]
    speed_df = pd.read_parquet(DATA / SPEED_FILE[0] / f"{SPEED_FILE[1]}.parquet")[REQUIRED_INPUT_COLS]

    rows = []
    for path in paths:
        name = "current model (src)" if path == SRC else path.name
        model = load_model(path)
        info = model_info(model)
        print(f"\n{name}")
        print(f"  trained on: {info.get('trained_on', 'unknown')}, {model_feature_summary(info)}, "
              f"architecture: {model_arch_summary(info)}")
        if info.get("trained_on") != "holdout":
            print("  WARNING: not trained with `holdout`: it has seen these people, quality is inflated.")
        q = {}
        for tname, trecs in tests:
            q[tname] = quality(model, trecs)
            s, f1 = q[tname]
            print(f"  {tname:34s} quality {80 * s:5.1f} / 80   (F1 Rest {f1[0]:.2f}  Activity {f1[1]:.2f}  "
                  f"Stress {f1[2]:.2f})", flush=True)
        sec = time_per_call(model, speed_df)
        speed = max(0.0, (1 - sec / 0.3) * 20)
        total = 80 * q[tests[0][0]][0] + speed
        print(f"  {'speed':34s} {speed:5.1f} / 20   ({1000 * sec:.1f} ms per call)")
        print(f"  T3 DIAGNOSTIC TOTAL (T3 quality + speed) {total:5.1f} / 100")
        rows.append((name, q, speed, total))

    if len(rows) > 1:
        print("\nSide by side:")
        print(f"  {'model':28s}{'T3':>8s}{'T2':>8s}{'T4':>8s}{'train':>8s}{'speed':>8s}{'T3+spd':>9s}")
        for name, q, speed, total in rows:
            print(f"  {name[:28]:28s}{80*q[tests[0][0]][0]:8.1f}{80*q[tests[1][0]][0]:8.1f}"
                  f"{80*q[tests[2][0]][0]:8.1f}{80*q[tests[3][0]][0]:8.1f}{speed:8.1f}{total:9.1f}")
        print("  (T3+speed is a local diagnostic only; leaderboard score is the ground truth.)")


if __name__ == "__main__":
    main(sys.argv[1:])
