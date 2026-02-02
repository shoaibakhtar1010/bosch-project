from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
from datetime import datetime, timezone


import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

from mmbiometric.config import load_config
from mmbiometric.data.dataset import MultimodalBiometricDataset
from mmbiometric.data.split import split_manifest
from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet
from mmbiometric.distributed.ray_utils import init_ray
from mmbiometric.utils.seed import seed_everything


@dataclass(frozen=True)
class RayTrainArgs:
    config_path: Path
    dataset_dir: Path
    output_dir: Path
    subject_regex: str
    num_workers: int
    use_gpu: bool
    cpus_per_worker: int


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _train_loop_per_worker(cfg: dict[str, Any]) -> None:
    config_path = Path(cfg["config_path"])
    dataset_dir = Path(cfg["dataset_dir"])
    output_dir = Path(cfg["output_dir"])
    subject_regex = str(cfg["subject_regex"])

    app_cfg = load_config(config_path)
    ctx = train.get_context()
    rank = ctx.get_world_rank()

    # reproducibility (different seed per rank but deterministic)
    seed_everything(app_cfg.seed + rank)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"

    # In best practice, preprocess runs before training.
    # As a guardrail, fail early if manifest missing.
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing manifest at {manifest_path}. Run mmbiometric-ray-preprocess first."
        )

    # split once on rank 0, others wait
    if rank == 0:
        split_manifest(
            manifest_path=manifest_path,
            out_dir=output_dir / "splits",
            val_fraction=app_cfg.data.val_fraction,
            seed=app_cfg.seed,
        )
    ctx.barrier()

    train_manifest = output_dir / "splits" / "train_manifest.parquet"
    val_manifest = output_dir / "splits" / "val_manifest.parquet"

    train_df = pd.read_parquet(train_manifest)
    labels = sorted(train_df["subject_id"].astype(str).unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    tfm = default_image_transform(app_cfg.data.image_size)
    train_ds = MultimodalBiometricDataset(train_manifest, tfm, tfm, label_to_idx)
    val_ds = MultimodalBiometricDataset(val_manifest, tfm, tfm, label_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=app_cfg.data.batch_size,
        shuffle=True,
        num_workers=app_cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=app_cfg.data.batch_size,
        shuffle=False,
        num_workers=app_cfg.data.num_workers,
        pin_memory=True,
    )

    # Important: wrap loader + model for DDP
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    model = MultimodalNet(
        backbone=app_cfg.model.backbone,
        embedding_dim=app_cfg.model.embedding_dim,
        num_classes=len(label_to_idx),
        dropout=app_cfg.model.dropout,
    )
    model = train.torch.prepare_model(model)

    device = train.torch.get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=app_cfg.train.lr, weight_decay=app_cfg.train.weight_decay
    )

    best_acc = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, app_cfg.train.epochs + 1):
        # -------- train --------
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            iris = batch.iris.to(device, non_blocking=True)
            fp = batch.fingerprint.to(device, non_blocking=True)
            y = batch.label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(iris, fp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        # -------- val --------
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                iris = batch.iris.to(device, non_blocking=True)
                fp = batch.fingerprint.to(device, non_blocking=True)
                y = batch.label.to(device, non_blocking=True)
                logits = model(iris, fp)
                loss = criterion(logits, y)

                val_loss_sum += float(loss.item())
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.numel())
                val_batches += 1

        # aggregate metrics across workers
        t = torch.tensor(
            [train_loss_sum, float(train_batches), val_loss_sum, float(val_batches), float(correct), float(total)],
            device=device,
            dtype=torch.float32,
        )
        _all_reduce_sum(t)

        train_loss = (t[0] / torch.clamp(t[1], min=1.0)).item()
        val_loss = (t[2] / torch.clamp(t[3], min=1.0)).item()
        val_acc = (t[4] / torch.clamp(t[5], min=1.0)).item()

        # rank0 saves best checkpoint + labels
        if rank == 0 and val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.module.state_dict(), "val_acc": best_acc}, best_path)
            (output_dir / "labels.json").write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

        # report metrics to Ray
        train.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

    # write metadata once
    # write metadata once
    if rank == 0:
        meta = {
            # model reconstruction contract (critical for decoupled inference)
            "backbone": app_cfg.model.backbone,
            "embedding_dim": app_cfg.model.embedding_dim,
            "dropout": app_cfg.model.dropout,
            "image_size": app_cfg.data.image_size,

            # training info
            "best_val_acc": best_acc,
            "epochs": app_cfg.train.epochs,
            "batch_size": app_cfg.data.batch_size,
            "seed": app_cfg.seed,

            # artifact locations
            "checkpoint_path": str(best_path),
            "labels_path": str(output_dir / "labels.json"),
            "manifest_path": str(manifest_path),
            "train_manifest": str(output_dir / "splits" / "train_manifest.parquet"),
            "val_manifest": str(output_dir / "splits" / "val_manifest.parquet"),

            # run provenance (useful in MLOps)
            "git_sha": os.environ.get("GITHUB_SHA"),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "ray_world_size": train.get_context().get_world_size(),
            "ray_use_gpu": bool(torch.cuda.is_available()),
        }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    (output_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train_distributed(args: RayTrainArgs) -> None:
    """
    Launch a Ray TorchTrainer job (DDP under the hood).
    """
    init_ray(num_cpus=None, num_gpus=None)

    scaling = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker={"CPU": args.cpus_per_worker},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_per_worker,
        train_loop_config={
            "config_path": str(args.config_path),
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "subject_regex": args.subject_regex,
        },
        scaling_config=scaling,
    )
    trainer.fit()
