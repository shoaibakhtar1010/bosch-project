from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray

from mmbiometric.distributed.ray_manifest import build_manifest_distributed
from mmbiometric.distributed.ray_train import RayTrainArgs, train_distributed


def _ray_init() -> None:
    """Initialize Ray using RAY_ADDRESS if present, otherwise start local."""
    address = os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")
    if ray.is_initialized():
        return
    if address:
        ray.init(address=address)
    else:
        ray.init()


def preprocess_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mmbiometric-ray-preprocess")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where manifest.parquet will be written",
    )
    parser.add_argument("--num-cpus", type=int, default=4, help="CPUs to use for preprocessing")
    args = parser.parse_args(argv)

    _ray_init()

    # IMPORTANT FIX:
    # build_manifest_distributed expects output_dir (directory), not output_path (file).
    build_manifest_distributed(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_cpus=args.num_cpus,
    )

    out_manifest = Path(args.output_dir) / "manifest.parquet"
    print(f"[OK] Wrote manifest: {out_manifest}")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mmbiometric-ray-train")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", required=True, help="Output directory (must contain manifest.parquet)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray Train workers")
    parser.add_argument("--cpus-per-worker", type=int, default=2, help="CPUs per worker")
    parser.add_argument(
        "--master-addr",
        help="Override MASTER_ADDR for torch distributed (useful on Windows multi-worker).",
    )
    parser.add_argument(
        "--gloo-ifname",
        help="Override GLOO_SOCKET_IFNAME for torch distributed (e.g. 'Wi-Fi').",
    )
    args = parser.parse_args(argv)
    config_value = args.config.strip()
    if not config_value:
        raise SystemExit("Config path is empty. Set --config or CONFIG_PATH to a valid YAML file.")
    config_path = Path(config_value).expanduser()
    if config_path.is_dir():
        raise SystemExit(f"Config path must be a file, got directory: {config_path}")

    _ray_init()

    ray_args = RayTrainArgs(
        config_path=str(config_path),
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        cpus_per_worker=args.cpus_per_worker,
        master_addr=args.master_addr,
        gloo_socket_ifname=args.gloo_ifname,
    )
    train_distributed(ray_args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmbiometric-ray")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--dataset-dir", required=True)
    p_pre.add_argument("--output-dir", required=True)
    p_pre.add_argument("--num-cpus", type=int, default=4)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument("--dataset-dir", required=True)
    p_tr.add_argument("--output-dir", required=True)
    p_tr.add_argument("--num-workers", type=int, default=2)
    p_tr.add_argument("--cpus-per-worker", type=int, default=2)

    ns, _ = parser.parse_known_args()
    if ns.cmd == "preprocess":
        preprocess_main()
    elif ns.cmd == "train":
        train_main()
    else:
        raise SystemExit(f"Unknown command: {ns.cmd}")


if __name__ == "__main__":
    main()
