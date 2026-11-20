"""Estimate the hidden quality component from leaderboard total and runtime.

Agorize exposes an integer-like total and a `run_time` value. The public
evaluation rule is score = 80 * macro-F1 + 20 * (1 - mean_runtime/0.3s).
Leaderboard runtimes in this challenge are in milliseconds, so pass that
value as the second argument:

    python experiments/platform_score_decompose.py 74 87.811651

If the displayed total is rounded, the recovered quality and macro-F1 are
estimates within that rounding error.
"""
from __future__ import annotations

import sys


def decompose(total_score: float, runtime_ms: float) -> tuple[float, float, float]:
    if not 0.0 <= total_score <= 100.0:
        raise ValueError("total score must be in [0, 100]")
    if runtime_ms < 0.0:
        raise ValueError("runtime must be nonnegative")
    speed = max(0.0, 20.0 * (1.0 - runtime_ms / 300.0))
    quality = total_score - speed
    macro_f1 = quality / 80.0
    return quality, speed, macro_f1


def main(args: list[str]) -> None:
    if len(args) != 2:
        raise SystemExit(__doc__)
    total, runtime_ms = map(float, args)
    quality, speed, macro_f1 = decompose(total, runtime_ms)
    print(f"total score: {total:.2f}")
    print(f"estimated quality: {quality:.2f}/80")
    print(f"estimated speed: {speed:.2f}/20")
    print(f"estimated hidden macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
