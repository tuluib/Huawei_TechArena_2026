"""Evaluate TULUI/V5.1 probability blends on the fixed person holdout.

Run from the project root with .venv active:
    python experiments/evaluate_blend.py

Both models must be holdout-trained. The script does not tune biases against this holdout.
"""
from __future__ import annotations

import json
import importlib.util
import pathlib
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import score as S


def load_artifact(path, tag):
    if path.suffix == ".zip":
        root = pathlib.Path(tempfile.mkdtemp(prefix=f"blend_{tag}_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
        found = sorted(root.rglob("your_model.py"))
        if not found:
            raise RuntimeError(f"No your_model.py in {path}")
        model_dir = found[0].parent
    else:
        model_dir = path
    spec = importlib.util.spec_from_file_location(f"blend_{tag}", model_dir / "your_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.get_model()
    return model, module, S.model_info(model)


def probabilities(model, module, meta, df):
    X, row_wid = module.compute_features(df[module.INPUT_COLS])
    names = model.session.get_inputs()
    xname = names[0].name
    p = model.session.run([model.prob_name], {xname: np.ascontiguousarray(X[:, model.cols])})[0]
    p = module.smooth_probs(np.asarray(p, dtype=np.float64), int(meta.get("smooth_k", 1)))
    bias = np.asarray(meta.get("log_bias", [0.0, 0.0, 0.0]), dtype=np.float64)
    logits = np.log(np.maximum(p, 1e-9)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    row_p = p[np.clip(row_wid, 0, len(p) - 1)]
    return row_p


def main():
    paths = [S.find("TULUI"), S.find("v5_holdout")]
    artifacts = []
    for tag, path in zip(("tului", "v5"), paths):
        model, module, meta = load_artifact(path, tag)
        if meta.get("trained_on") != "holdout":
            raise RuntimeError(f"{path.name} is not holdout-trained")
        artifacts.append((model, module, meta))

    base = S.holdout_recordings()
    cases = [
        ("original", base),
        ("T2 shuffled within session", S.shuffled_recordings(base, seed=0, per_person=2, within=True)),
        ("T4 mixed sessions", S.shuffled_recordings(base, seed=0, per_person=2, within=False)),
    ]
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight on TULUI; 0 = V5.1
    for label, recs in cases:
        ys, ps = [], {a: [] for a in alphas}
        for df in recs:
            pa, pb = [probabilities(*artifact, df) for artifact in artifacts]
            true10 = S.window_10s_mode(S.map_phase_to_activity(df["Phase"]))
            for alpha in alphas:
                pred = (alpha * pa + (1.0 - alpha) * pb).argmax(axis=1)
                pred_labels = pd.Series(np.array(["Rest", "Activity", "Stress"])[pred], index=df.index)
                pred10 = S.window_10s_mode(pred_labels)
                ok = true10.notna() & pred10.notna()
                ps[alpha].extend(S.LAB[v] for v in pred10[ok])
            ok = true10.notna()
            ys.extend(S.LAB[v] for v in true10[ok])
        print(f"\n{label}: {len(ys):,} scored windows")
        for alpha in alphas:
            f1 = S.f1_scores(ys, ps[alpha])
            print(f"  TULUI weight {alpha:.2f}: quality={80*f1.mean():5.1f}/80; "
                  f"F1 Rest/Activity/Stress={f1[0]:.3f}/{f1[1]:.3f}/{f1[2]:.3f}")


if __name__ == "__main__":
    main()
