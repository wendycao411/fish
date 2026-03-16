#!/usr/bin/env python3
"""Train Ultralytics YOLO on MegaFish unified dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import default_data_yaml, default_runs_root


def parse_batch(value: str) -> int | float:
    try:
        numeric = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("batch must be numeric, e.g. 16, 0.7, or -1") from e

    if numeric.is_integer():
        return int(numeric)
    return numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO for MegaFish")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Model checkpoint (e.g., yolo11m.pt or yolo26m.pt)")
    parser.add_argument("--data", type=Path, default=default_data_yaml(), help="Data YAML path")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=parse_batch, default=-1, help="Batch size as int/float, or -1 for autobatch")
    parser.add_argument("--device", type=str, default="0", help="Device id or 'cpu'")
    parser.add_argument("--project", type=Path, default=default_runs_root())
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from project/name/weights/last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.project / args.name
    last_ckpt = run_dir / "weights" / "last.pt"

    train_kwargs = dict(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
    )

    if args.resume:
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Requested --resume but checkpoint not found: {last_ckpt}")
        model = YOLO(str(last_ckpt.resolve()))
        train_kwargs["resume"] = True
    else:
        model = YOLO(args.model)

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
