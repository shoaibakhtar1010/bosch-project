from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import ray

from mmbiometric.distributed.ray_utils import init_ray

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class _Partial:
    subject_id: str
    modality: str
    path: str


def _guess_modality(path_str: str) -> Optional[str]:
    s = path_str.lower()
    if "iris" in s:
        return "iris"
    if any(k in s for k in ["finger", "fingerprint", "fp"]):
        return "fingerprint"
    return None


def _extract_subject_id(path_str: str, subject_regex: str) -> str:
    m = re.search(subject_regex, path_str)
    if not m:
        return Path(path_str).parent.name
    return m.group(1) if m.groups() else m.group(0)


@ray.remote
def _process_one(path_str: str, subject_regex: str) -> Optional[_Partial]:
    ext = Path(path_str).suffix.lower()
    if ext not in _IMAGE_EXTS:
        return None
    modality = _guess_modality(path_str)
    if modality is None:
        return None
    sid = _extract_subject_id(path_str, subject_regex)
    return _Partial(subject_id=sid, modality=modality, path=path_str)


def build_manifest_distributed(
    dataset_dir: Path,
    output_path: Path,
    subject_regex: str = r"(\d+)",
    num_cpus: int = 8,
) -> Path:
    """
    Scan dataset and write manifest.parquet with columns:
      subject_id, iris_path, fingerprint_path
    """
    init_ray(num_cpus=num_cpus, num_gpus=None)

    files = [
        str(p)
        for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]

    futures = [_process_one.remote(p, subject_regex) for p in files]
    results = ray.get(futures)

    iris_by_sid: dict[str, list[str]] = {}
    fp_by_sid: dict[str, list[str]] = {}

    for r in results:
        if r is None:
            continue
        if r.modality == "iris":
            iris_by_sid.setdefault(r.subject_id, []).append(r.path)
        else:
            fp_by_sid.setdefault(r.subject_id, []).append(r.path)

    rows: list[tuple[str, str, str]] = []
    for sid, iris_list in iris_by_sid.items():
        fp_list = fp_by_sid.get(sid, [])
        if not fp_list:
            continue
        iris_list.sort()
        fp_list.sort()
        n = min(len(iris_list), len(fp_list))
        for i in range(n):
            rows.append((sid, iris_list[i], fp_list[i]))

    df = pd.DataFrame(rows, columns=["subject_id", "iris_path", "fingerprint_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    ray.shutdown()
    return output_path
