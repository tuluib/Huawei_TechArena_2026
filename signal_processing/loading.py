"""Finding and loading the training recordings (offline only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROTOCOLS = ("STRESS", "AEROBIC", "ANAEROBIC")
INPUT_COLUMNS = ["BVP", "EDA", "HR", "IBI_dt", "IBI_duration", "Temperature"]


@dataclass(frozen=True)
class Recording:
    protocol: str
    user_id: str  # same person across protocols (S01 in STRESS and in AEROBIC)
    path: Path

    @property
    def name(self) -> str:
        return f"{self.protocol}/{self.user_id}"


def list_recordings(data_dir: Path = DATA_DIR) -> list[Recording]:
    return [Recording(p, f.stem, f) for p in PROTOCOLS for f in sorted((data_dir / p).glob("*.parquet"))]


def load(rec: Recording) -> pd.DataFrame:
    """Full recording including the Phase and user_id columns."""
    return pd.read_parquet(rec.path)


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the columns run() receives at evaluation time."""
    return df[INPUT_COLUMNS]
