from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitPaths:
    train_manifest: Path
    val_manifest: Path


def split_manifest(
    manifest_path: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> SplitPaths:
    """Split manifest by subject_id to avoid identity leakage."""
    df = pd.read_parquet(manifest_path)
    subjects = np.array(sorted(df["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(len(subjects) * val_fraction))
    val_subjects = set(subjects[:n_val])
    train_df = df[~df["subject_id"].astype(str).isin(val_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].astype(str).isin(val_subjects)].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_manifest.parquet"
    val_path = out_dir / "val_manifest.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    return SplitPaths(train_manifest=train_path, val_manifest=val_path)
