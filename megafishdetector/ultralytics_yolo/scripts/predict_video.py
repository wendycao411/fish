#!/usr/bin/env python3
"""Run inference on video or image folder and export annotated outputs + CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import ensure_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with trained MegaFish Ultralytics model")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights .pt")
    parser.add_argument("--source", type=Path, required=True, help="Input video file or folder of images")
    parser.add_argument("--output", type=Path, required=True, help="Annotated output mp4 (video mode) or output directory (image mode)")
    parser.add_argument("--csv", type=Path, required=True, help="Output detections CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--fps", type=float, default=None, help="FPS override for image-folder mode")
    return parser.parse_args()


def write_csv_header(csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_sec", "x1", "y1", "x2", "y2", "conf"])


def append_rows(csv_path: Path, rows: list[list[float]]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def predict_video(model: YOLO, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ensure_dir(args.output.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        annotated = result.plot()
        writer.write(annotated)

        rows = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, conf):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                rows.append([frame_idx, frame_idx / fps, x1, y1, x2, y2, float(c)])
        append_rows(args.csv, rows)
        frame_idx += 1

    cap.release()
    writer.release()


def predict_images(model: YOLO, args: argparse.Namespace) -> None:
    image_paths = [p for p in sorted(args.source.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        raise RuntimeError(f"No images found in folder: {args.source}")

    out_dir = args.output
    ensure_dir(out_dir)
    fps = args.fps if args.fps is not None and args.fps > 0 else 1.0

    for frame_idx, image_path in enumerate(image_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result = model.predict(frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]

        annotated = result.plot()
        cv2.imwrite(str(out_dir / image_path.name), annotated)

        rows = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, conf):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                rows.append([frame_idx, frame_idx / fps, x1, y1, x2, y2, float(c)])
        append_rows(args.csv, rows)


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.csv = args.csv.resolve()

    model = YOLO(str(args.weights.resolve()))
    write_csv_header(args.csv)

    if args.source.is_file():
        predict_video(model, args)
    elif args.source.is_dir():
        predict_images(model, args)
    else:
        raise RuntimeError(f"Source not found: {args.source}")

    print(f"Annotated output: {args.output}")
    print(f"Detections CSV: {args.csv}")


if __name__ == "__main__":
    main()
