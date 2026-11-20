"""Compare TULUI with the TULUI-feature shuffle-augmented model on T3 seeds.

Run from project root after training the holdout candidate:
    python experiments/evaluate_tului_aug.py
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd

import evaluate_blend as B
import score as S

ROOT = pathlib.Path(__file__).resolve().parents[1]


def score_one(recs, artifacts):
    ys, predictions = [], [[], []]
    for df in recs:
        true10 = S.window_10s_mode(S.map_phase_to_activity(df["Phase"]))
        probs = [B.probabilities(*artifact, df) for artifact in artifacts]
        preds = [pd.Series(np.asarray(["Rest", "Activity", "Stress"])[p.argmax(axis=1)],
                           index=df.index) for p in probs]
        pred10 = [S.window_10s_mode(p) for p in preds]
        ok = true10.notna()
        ys.extend(S.LAB[v] for v in true10[ok])
        for i in range(2):
            valid = ok & pred10[i].notna()
            predictions[i].extend(S.LAB[v] for v in pred10[i][valid])
    return [S.f1_scores(ys, p) for p in predictions]


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    candidate = ROOT / "models" / ("tului_aug_holdout" if variant == "base"
                                   else f"tului_aug_{variant}_holdout")
    paths = [S.find("TULUI"), candidate]
    artifacts = []
    for tag, path in zip(("tului", f"tului_aug_{variant}"), paths):
        model, module, meta = B.load_artifact(path, f"augseed_{tag}")
        if meta.get("trained_on") != "holdout":
            raise RuntimeError(f"{path} is not holdout-trained")
        artifacts.append((model, module, meta))
    base = S.holdout_recordings()
    rows = []
    for seed in range(5):
        recs = S.shuffled_recordings(base, seed=seed, per_person=2, within=True)
        scores = score_one(recs, artifacts)
        rows.append(scores)
        quality = [80 * score.mean() for score in scores]
        print(f"seed {seed}: TULUI={quality[0]:.2f}, augmented={quality[1]:.2f}, "
              f"delta={quality[1]-quality[0]:+.2f}; "
              f"class F1 TULUI={np.round(scores[0], 3)}, augmented={np.round(scores[1], 3)}",
              flush=True)
    a = np.asarray(rows)
    delta = 80 * (a[:, 1].mean(axis=1) - a[:, 0].mean(axis=1))
    avg_base, avg_aug = a[:, 0].mean(axis=0), a[:, 1].mean(axis=0)
    print(f"mean TULUI={80*avg_base.mean():.2f}; augmented={80*avg_aug.mean():.2f}; "
          f"delta={delta.mean():+.2f} +/- {delta.std(ddof=1):.2f}; wins={(delta > 0).sum()}/5")
    print(f"mean class F1 TULUI Rest/Activity/Stress={np.round(avg_base,3)}; "
          f"augmented={np.round(avg_aug,3)}; delta={np.round(avg_aug-avg_base,3)}")


if __name__ == "__main__":
    main()
