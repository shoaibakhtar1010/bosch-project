from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub

from mmbiometric.distributed.ray_manifest import build_manifest_distributed
from mmbiometric.distributed.ray_train import RayTrainArgs, train_distributed


def preprocess_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, required=False, help="Dataset root path")
    p.add_argument("--output-dir", type=str, required=True, help="Run directory (manifest will be written here)")
    p.add_argument("--subject-regex", type=str, default=r"(\d+)")
    p.add_argument("--num-cpus", type=int, default=16)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        # fallback: download in CLI if dataset-dir not provided
        dataset_dir = Path(
            kagglehub.dataset_download("ninadmehendale/multimodal-iris-fingerprint-biometric-data")
        )

    build_manifest_distributed(
        dataset_dir=dataset_dir,
        output_path=output_dir / "manifest.parquet",
        subject_regex=args.subject_regex,
        num_cpus=args.num_cpus,
    )


def train_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset-dir", type=str, required=False, help="Dataset root path (for metadata only)")
    p.add_argument("--output-dir", type=str, required=True, help="Run directory containing manifest.parquet")
    p.add_argument("--subject-regex", type=str, default=r"(\d+)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--cpus-per-worker", type=int, default=4)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(".")

    train_distributed(
        RayTrainArgs(
            config_path=Path(args.config),
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            subject_regex=args.subject_regex,
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            cpus_per_worker=args.cpus_per_worker,
        )
    )
