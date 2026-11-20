"""Stage 0: calibrate the local evaluation against our real leaderboard (website) scores.

    python truth_eval.py            # every submission in SUBMISSIONS below

Quality (/80) = 80 x pooled macro-F1 over clock-aligned 10 s windows of the 6 holdout people
(S16, S17, S18, f11, f12, f13), computed with the scorer rules of eval_harness.py. Three ways to build
the test recordings (see score.shuffled_recordings):
  MIXED     a person's phases from all sessions, random order and lengths (best match to the leaderboard so far)
  SHUFFLED  phases shuffled inside each session
  ORDER     the original recordings (training protocol order)
Speed (/20) = max(0, (1 - MET) x 20), MET = mean seconds of 50 run() calls on the longest recording, measured
in a fresh process in an isolated temp folder (only your_model.py, your_model.onnx, adm_base_class.py) with
OMP/MKL/OPENBLAS threads = 1.

Quality is only fair for a model trained WITHOUT the 6 holdout people. Each submission below points to such a
copy where one exists; "fair" says whether it does.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
SRC = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.append(str(SRC))
import score as S

# name, website score, model folder used for the local score, fair (trained without the holdout people)?, note
SUBMISSIONS = [
    ("V1", 68, "models/v1", True, "uploaded file (its trainer skipped the holdout people)"),
    ("V1.2", 65, "models/v1_2", None, "uploaded file; training people unknown"),
    ("V2", 61, "models/v2_holdout", True, "retrained without holdout people, V2 code and features"),
    ("V3", 72, "models/v3_holdout", True, "retrained without holdout people, V3 code and features"),
    ("V4", 71, "models/holdout", True, "train_v4.py holdout"),
    ("TULUI", 74, "models/tului", True, "uploaded file (trained on 29 people)"),
    ("TULUI_fast", 74, "models/tului_fast", True, "uploaded file; same predictions as TULUI, faster code"),
    ("V5", 73, "models/v5_0_holdout", True, "rebuilt without holdout people, as uploaded"),
    ("V5.1", 73, "models/v5_holdout", True, "train_v5.py holdout (current src model's recipe)"),
    ("SHEH", 54, "models/sheh", False, "uploaded file; very likely trained on the holdout people"),
]


def longest_recording() -> pathlib.Path:
    files = [fp for p in S.PROTOCOLS for fp in (S.DATA / p).glob("*.parquet")]
    return max(files, key=lambda fp: pd.read_parquet(fp, columns=["HR"]).shape[0])


SPEED_CHILD = r'''
import os, sys, time, warnings
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
warnings.simplefilter("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import your_model
df = pd.read_parquet(sys.argv[1])[["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]]
m = your_model.get_model()
m.run(df)  # warm-up
ts = []
for _ in range(50):
    t0 = time.perf_counter(); m.run(df); ts.append(time.perf_counter() - t0)
print(repr(float(np.mean(ts))))
'''


def met_seconds(model_dir: pathlib.Path, recording: pathlib.Path) -> float:
    """50-run mean execution time in a fresh process, isolated folder, 1 thread."""
    with tempfile.TemporaryDirectory(prefix="truth_eval_") as tmp:
        tmp = pathlib.Path(tmp)
        for f in ("your_model.py", "your_model.onnx"):
            shutil.copy(model_dir / f, tmp / f)
        shutil.copy(SRC / "adm_base_class.py", tmp / "adm_base_class.py")
        (tmp / "speed_child.py").write_text(SPEED_CHILD, encoding="utf-8")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONWARNINGS": "ignore"}
        out = subprocess.run([sys.executable, "-I", "speed_child.py", str(recording)], cwd=tmp, env=env,
                             capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        return float(out.stdout.strip().splitlines()[-1])


def main():
    recs = S.holdout_recordings()
    tests = {"MIXED": S.shuffled_recordings(recs, per_person=4, within=False),
             "SHUFFLED": S.shuffled_recordings(recs, per_person=2, within=True),
             "ORDER": recs}
    longest = longest_recording()
    n_rows = pd.read_parquet(longest, columns=["HR"]).shape[0]
    print(f"Speed recording: {longest.parent.name}/{longest.name} ({n_rows:,} rows, {n_rows / 64 / 60:.1f} min)\n")

    rows = []
    for name, web, folder, fair, note in SUBMISSIONS:
        d = ROOT / folder
        model = S.load_model(d)
        q = {t: 80 * S.quality(model, r)[0] for t, r in tests.items()}
        met = met_seconds(d, longest)
        speed = max(0.0, (1.0 - met) * 20.0)
        rows.append(dict(Submission=name, Website=web, fair={True: "yes", False: "NO", None: "unknown"}[fair],
                         Q_MIXED=q["MIXED"], Q_SHUFFLED=q["SHUFFLED"], Q_ORDER=q["ORDER"], Speed=speed,
                         Total=q["MIXED"] + speed, MET_ms=1000 * met, note=note))
        print(f"  {name:11s} MIXED {q['MIXED']:5.1f}  SHUFFLED {q['SHUFFLED']:5.1f}  ORDER {q['ORDER']:5.1f}  "
              f"MET {1000 * met:6.1f} ms", flush=True)

    t = pd.DataFrame(rows)
    t["web_rank"] = t["Website"].rank(ascending=False, method="min").astype(int)
    t["local_rank"] = t["Total"].rank(ascending=False, method="min").astype(int)
    t["Rank Matches Website?"] = np.where(t["web_rank"] == t["local_rank"], "yes",
                                          "no (local #" + t["local_rank"].astype(str) + ")")
    show = t.rename(columns={"Website": "Website Score", "Q_MIXED": "Local Quality MIXED (/80)",
                             "Q_SHUFFLED": "Quality SHUFFLED", "Q_ORDER": "Quality ORDER",
                             "Speed": "Local Speed (/20)", "Total": "Local Total (/100)", "MET_ms": "MET (ms)"})
    cols = ["Submission", "Website Score", "fair", "Local Quality MIXED (/80)", "Quality SHUFFLED", "Quality ORDER",
            "Local Speed (/20)", "Local Total (/100)", "MET (ms)", "Rank Matches Website?"]
    print("\n" + show.sort_values("Website Score", ascending=False)[cols].round(1).to_string(index=False))

    print("\nRank correlation (Spearman) of local total with the website score:")
    for label, sub in [("all 10", t), ("fair copies only", t[t["fair"] == "yes"])]:
        for qcol in ("Q_MIXED", "Q_SHUFFLED", "Q_ORDER"):
            total = sub[qcol] + sub["Speed"]
            print(f"  {label:17s} quality {qcol[2:]:9s}: {total.rank().corr(sub['Website'].rank()):+.2f}")
    t.to_csv(ROOT / "experiments" / "truth_eval_stage0.csv", index=False)
    print(f"\nSaved {ROOT / 'experiments' / 'truth_eval_stage0.csv'}")


if __name__ == "__main__":
    main()
