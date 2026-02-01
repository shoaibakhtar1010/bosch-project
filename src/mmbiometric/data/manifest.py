from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestRow:
    subject_id: str
    iris_path: Path
    fingerprint_path: Path


def _guess_modality(p: Path) -> Optional[str]:
    s = str(p).lower()
    if any(k in s for k in ["iris"]):
        return "iris"
    if any(k in s for k in ["finger", "fingerprint", "fp"]):
        return "fingerprint"
    return None


def _extract_subject_id(p: Path, subject_regex: str) -> str:
    """Extract subject id from path using a configurable regex.

    Default regex targets a sequence of digits; override in config if needed.
    """
    m = re.search(subject_regex, str(p))
    if not m:
        # fallback to parent folder
        return p.parent.name
    return m.group(1) if m.groups() else m.group(0)


def build_manifest(
    dataset_dir: Path,
    output_path: Path,
    subject_regex: str = r"(\d+)",
) -> Path:
    """Scan the dataset directory and build a (subject_id, iris_path, fingerprint_path) manifest.

    This is intentionally heuristic because Kaggle datasets often differ in folder naming.
    If pairing is wrong for your dataset layout, adjust the `subject_regex` or add a custom
    pairing strategy in this module.
    """
    dataset_dir = dataset_dir.resolve()
    files: list[Path] = [
        p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]

    iris_by_subject: dict[str, list[Path]] = {}
    fp_by_subject: dict[str, list[Path]] = {}

    for p in tqdm(files, desc="Scanning images"):
        modality = _guess_modality(p)
        if modality is None:
            continue
        sid = _extract_subject_id(p, subject_regex)
        if modality == "iris":
            iris_by_subject.setdefault(sid, []).append(p)
        else:
            fp_by_subject.setdefault(sid, []).append(p)

    rows: list[ManifestRow] = []
    for sid, iris_list in iris_by_subject.items():
        fp_list = fp_by_subject.get(sid, [])
        if not fp_list:
            continue
        # naive pairing: zip smallest list lengths (deterministic sort)
        iris_list = sorted(iris_list)
        fp_list = sorted(fp_list)
        n = min(len(iris_list), len(fp_list))
        for i in range(n):
            rows.append(ManifestRow(subject_id=sid, iris_path=iris_list[i], fingerprint_path=fp_list[i]))

    df = pd.DataFrame(
        {
            "subject_id": [r.subject_id for r in rows],
            "iris_path": [str(r.iris_path) for r in rows],
            "fingerprint_path": [str(r.fingerprint_path) for r in rows],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path
