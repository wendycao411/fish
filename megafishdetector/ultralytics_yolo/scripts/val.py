#!/usr/bin/env python3
"""Validate Ultralytics YOLO and save key metrics to JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import default_data_yaml, default_runs_root, save_json, to_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Ultralytics YOLO for MegaFish")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights .pt")
    parser.add_argument("--data", type=Path, default=default_data_yaml(), help="Data YAML path")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=str, default="auto")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", type=Path, default=default_runs_root())
    parser.add_argument("--name", type=str, default="val")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional explicit metrics JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        project=str(args.project.resolve()),
        name=args.name,
    )

    payload = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "results_dict": to_serializable(getattr(metrics, "results_dict", {})),
        "speed": to_serializable(getattr(metrics, "speed", {})),
    }

    out_json = args.out_json
    if out_json is None:
        out_json = Path(metrics.save_dir) / "metrics.json"
    save_json(out_json.resolve(), payload)
    print(f"Saved metrics: {out_json.resolve()}")


if __name__ == "__main__":
    main()
