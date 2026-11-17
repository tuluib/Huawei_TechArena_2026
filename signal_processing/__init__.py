"""Signal processing stage: raw wristband DataFrame -> clean signals -> 10-second window features.

Runtime code (used inside ``run()`` during evaluation) lives in ``windows``, ``cleaning`` and
``features`` and depends only on numpy, pandas and scipy. ``labels``, ``loading``,
``explore`` and ``evaluate_features`` are offline tools for training and analysis.
"""
from signal_processing.cleaning import CleanSignals, clean_signals
from signal_processing.features import FEATURE_NAMES, extract_features

__all__ = ["CleanSignals", "clean_signals", "extract_features", "FEATURE_NAMES"]
