"""Stage 0 package timing and LOSO fold-readiness audit.

This first pass measures the archived inference packages and checks for
prebuilt LOSO checkpoints. It does not yet invoke the training scripts to build
the folds, so it deliberately reports N/A for LOSO quality and total. Available
training recipes and missing historical variants are documented in experiments.md.

Run from the project root:
    python truth_eval.py

Speed column reports both thresholds because the user-provided prompt says
1.0 s, while the currently published official topic page says 0.3 s.
"""
from __future__ import annotations

import ctypes
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import zipfile

# Set thread limits before importing NumPy, Pandas, SciPy, or ONNX Runtime.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

import score as S


# Website scores supplied in the task. V5's score conflicts with the older
# repository README, which says V5 was not uploaded; keep the task value here
# and report the conflict instead of silently changing either record.
SUBMISSIONS = [
    ("SubV1", "submissions/submissionV1.zip", 68),
    ("SubV1.2", "submissions/V1.2(2nd trial65score).zip", 65),
    ("SubV2", "submissions/submissionV2.zip", 61),
    ("SubV3", "submissions/submissionV3.zip", 72),
    ("SubV4", "submissions/submissionV4.zip", 71),
    ("SubTULUI", "submissions/TULUI.zip", 74),
    ("SubTULUI Fast", "submissions/TULUI_fast.zip", 74),
    ("SubV5", "submissions/V5.zip", 73),
    ("SubV5.1", "submissions/V5.1.zip", 73),
    ("SubSHEH", "submissions/SHEH.zip", 54),
    ("Last update 69", "submissions/V27_dualrate_GRU.zip", 69),
    ("TULUI35", "submissions/TULUI35.zip", 75),
    ("Current candidate", "submissions/candidates/TULUI_shuffle_aug.zip", None),
]
PROTOCOLS = ("AEROBIC", "ANAEROBIC", "STRESS")
LOSOCV_ROOT = ROOT / "models" / "loso"


def pin_one_cpu() -> str:
    """Pin this benchmark process to one available logical CPU."""
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = os.sched_getaffinity(0)
        if not allowed:
            raise RuntimeError("No allowed CPU affinity mask")
        cpu = min(allowed)
        os.sched_setaffinity(0, {cpu})
        return f"CPU {cpu}"
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(process, ctypes.byref(process_mask),
                                               ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        mask = int(process_mask.value)
        if not mask:
            raise OSError("Current process has no allowed CPU")
        one_cpu = mask & -mask
        if not kernel32.SetProcessAffinityMask(process, ctypes.c_size_t(one_cpu)):
            raise ctypes.WinError(ctypes.get_last_error())
        return f"affinity mask {one_cpu:#x}"
    raise RuntimeError("Cannot pin this process to one CPU on this platform")


def subject_and_longest() -> tuple[list[str], pathlib.Path, float, pd.DataFrame]:
    subjects: set[str] = set()
    longest: tuple[float, pathlib.Path, pd.DataFrame] | None = None
    for protocol in PROTOCOLS:
        for path in sorted((ROOT / "data" / protocol).glob("*.parquet")):
            subjects.add(path.stem)
            frame = pd.read_parquet(path, columns=S.REQUIRED_INPUT_COLS)
            if len(frame) < 2:
                continue
            duration = (frame.index[-1] - frame.index[0]).total_seconds()
            if longest is None or duration > longest[0]:
                longest = (duration, path, frame)
    if longest is None:
        raise RuntimeError("No usable recordings found in the three protocol folders")
    return sorted(subjects), longest[1], longest[0], longest[2]


def artifact_metadata(model) -> dict:
    try:
        return json.loads(model.session.get_modelmeta().custom_metadata_map["eh_config"])
    except Exception:
        return {}


def loso_fold_models_exist(name: str, subjects: list[str]) -> bool:
    """Require one explicit, fold-trained package per held-out subject."""
    folder = LOSOCV_ROOT / name
    return folder.is_dir() and all(
        (folder / subject / "your_model.onnx").is_file()
        and (folder / subject / "your_model.py").is_file()
        for subject in subjects
    )


def benchmark(path: pathlib.Path, frame: pd.DataFrame) -> tuple[float | None, str]:
    """Load one submission in its own temp folder and measure 50 full calls."""
    temp = tempfile.TemporaryDirectory(prefix="truth_eval_")
    try:
        folder = pathlib.Path(temp.name)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(folder)
        else:
            shutil.copytree(path, folder, dirs_exist_ok=True)
        model = S.load_model(folder)
        output = model.run(frame)
        series = output["Activity"] if isinstance(output, pd.DataFrame) else output
        if len(series) != len(frame) or not series.index.equals(frame.index):
            return None, "FAILED: output length/index mismatch"
        if not pd.Series(series).isin(("Rest", "Activity", "Stress")).all():
            return None, "FAILED: invalid output label"

        times = []
        for _ in range(50):
            start = time.perf_counter()
            model.run(frame)
            times.append(time.perf_counter() - start)
        met_s = float(np.mean(times))
        meta = artifact_metadata(model)
        metadata_state = str(meta.get("trained_on", "unknown"))
        return met_s, metadata_state
    except Exception as exc:
        return None, f"FAILED: {type(exc).__name__}: {exc}"
    finally:
        temp.cleanup()


def speed_points(met_s: float, cap_s: float) -> float:
    return max(0.0, (1.0 - met_s / cap_s) * 20.0)


def main() -> None:
    cpu = pin_one_cpu()
    subjects, longest_path, duration_s, frame = subject_and_longest()
    print(f"CPU affinity: {cpu}; thread env vars: OMP=1 MKL=1 OPENBLAS=1")
    print(f"LOSO subject groups: {len(subjects)} across {', '.join(PROTOCOLS)}")
    print(f"Longest recording: {longest_path.relative_to(ROOT).as_posix()} "
          f"({len(frame):,} rows, {duration_s:.3f} s)")
    print("LOSO rule: omit a subject's files from every protocol, then score only that subject.")
    print("Fold checkpoint root:", LOSOCV_ROOT.relative_to(ROOT).as_posix())
    print("LOSO result: this timing-only pass did not train LOSO folds. The repo has "
          "training recipes, but they are not wired into this script yet. Local LOSO "
          "quality and total are N/A, not computed.")
    print("Speed formulas: current published topic page uses 0.3 s; prompt requests 1.0 s. "
          "Both are shown in Local Speed. Website source: "
          "https://huawei.agorize.com/en/challenges/2026-finland-tech-arena/pages/topic-description?lang=en")
    print("Count note: the nine listed historical entries plus SHEH and Last update 69 are 11; "
          "the current candidate is also shown, unscored.")
    print("Score note: V5=73 follows the current prompt; older submissions/README says V5 was not uploaded.")

    headers = ("Submission", "Website Score", "Local LOSO Quality (/80)", "Local Speed (/20)",
               "Local Total (/100)", "MET (ms)", "Rank Matches Website?")
    print("\t".join(headers))
    for name, relpath, website_score in SUBMISSIONS:
        path = ROOT / relpath
        if not path.is_file():
            print(f"{name}\t{website_score if website_score is not None else 'N/A'}\tN/A (archive missing)"
                  "\tN/A\tN/A\tFAILED: file missing\tN/A")
            continue

        fold_ok = loso_fold_models_exist(name, subjects)
        met_s, state = benchmark(path, frame)
        if met_s is None:
            speed_cell, met_cell = "N/A", state
        else:
            met_ms = met_s * 1000.0
            speed_cell = (f"{speed_points(met_s, 0.3):.2f} (0.3s); "
                          f"{speed_points(met_s, 1.0):.2f} (1.0s)")
            met_cell = f"{met_ms:.3f}"
        quality_cell = "N/A (no LOSO folds)" if not fold_ok else "PENDING"
        total_cell = "N/A (quality unavailable)" if not fold_ok else "PENDING"
        site_cell = "N/A (unsubmitted)" if website_score is None else str(website_score)
        print(f"{name}\t{site_cell}\t{quality_cell}\t{speed_cell}\t{total_cell}\t{met_cell}\tN/A")


if __name__ == "__main__":
    main()
