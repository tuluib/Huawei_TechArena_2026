"""Check whether V7 interactions complement TULUI on the fixed person holdout.

Run from project root with the project's environment active:
    python experiments/evaluate_v7_blend.py

Both artifacts must be trained without the six holdout people. This is a
candidate comparison, not a leaderboard score estimate.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

import evaluate_blend as B
import score as S

ROOT = pathlib.Path(__file__).resolve().parents[1]

def score_case(recs, artifacts, weights):
    ys = []
    guesses = {w: [] for w in weights}
    for df in recs:
        probs = [B.probabilities(*artifact, df) for artifact in artifacts]
        true10 = S.window_10s_mode(S.map_phase_to_activity(df["Phase"]))
        for w in weights:
            blended = w * probs[0] + (1.0 - w) * probs[1]
            pred = blended.argmax(axis=1)
            labels = pd.Series(np.array(["Rest", "Activity", "Stress"])[pred], index=df.index)
            pred10 = S.window_10s_mode(labels)
            ok = true10.notna() & pred10.notna()
            guesses[w].extend(S.LAB[v] for v in pred10[ok])
        ok = true10.notna()
        ys.extend(S.LAB[v] for v in true10[ok])
    return {w: 80 * S.f1_scores(ys, guesses[w]).mean() for w in weights}


def main():
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)  # TULUI weight; zero is V7.
    paths = [S.find("TULUI"), ROOT / "models" / "v7_holdout"]
    artifacts = []
    for tag, path in zip(("tului", "v7"), paths):
        model, module, meta = B.load_artifact(path, f"v7blend_{tag}")
        if meta.get("trained_on") != "holdout":
            raise RuntimeError(f"{path} is not holdout-trained")
        artifacts.append((model, module, meta))

    base = S.holdout_recordings()
    cases = [("Original", base)]
    for seed in range(5):
        cases.append((f"T3 within-session seed {seed}",
                      S.shuffled_recordings(base, seed=seed, per_person=2, within=True)))
    cases.append(("T4 mixed-session stress", S.shuffled_recordings(base, seed=0,
                                                                    per_person=2, within=False)))
    results = {w: [] for w in weights}
    for name, recs in cases:
        scores = score_case(recs, artifacts, weights)
        for w, score in scores.items():
            results[w].append(score)
        print(name + ": " + ", ".join(f"TULUI {w:.2f}={scores[w]:.2f}" for w in weights), flush=True)
    print("Mean T3 over five seeds:")
    for w in weights:
        print(f"  TULUI {w:.2f}: {np.mean(results[w][1:6]):.2f}")


if __name__ == "__main__":
    main()
