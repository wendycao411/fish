#!/usr/bin/env python3
"""Prepare CFDD in Ultralytics-compatible YOLO format.

This script:
1. Reads the CFDD COCO-style JSON annotations.
2. Uses the existing `yolov6_salmon/{train,valid,test}` split directories.
3. Writes YOLO txt labels to `output_dir/labels/{train,val,test}`.
4. Symlinks `output_dir/images/{train,val,test}` to the existing image split dirs.
5. Writes `output_dir/data.yaml` for Ultralytics training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CFDD YOLO labels and data.yaml")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("community-fish-detection-dataset/community_fish_detection_dataset.json"),
        help="Path to the CFDD COCO-style annotations JSON",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("community-fish-detection-dataset/yolov6_salmon"),
        help="Root containing CFDD split image dirs: train/, valid/, test/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("community-fish-detection-dataset/cfdd_yolo11"),
        help="Prepared YOLO dataset root to create",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing label files and data.yaml",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_split_lookup(split_root: Path) -> dict[str, str]:
    split_lookup: dict[str, str] = {}

    for src_name, dst_name in SPLIT_MAP.items():
        split_dir = split_root / src_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for image_path in split_dir.iterdir():
            if not image_path.is_file():
                continue
            split_lookup[image_path.name] = dst_name

    return split_lookup


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    wn = w / width
    hn = h / height
    return xc, yc, wn, hn


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    contents = "\n".join(
        [
            f"path: {dataset_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: fish",
            "",
        ]
    )
    path.write_text(contents, encoding="utf-8")


def symlink_split_dirs(split_root: Path, output_dir: Path) -> None:
    images_root = ensure_dir(output_dir / "images")
    for src_name, dst_name in SPLIT_MAP.items():
        src_dir = (split_root / src_name).resolve()
        dst_dir = images_root / dst_name
        if dst_dir.exists() or dst_dir.is_symlink():
            dst_dir.unlink()
        dst_dir.symlink_to(src_dir)


def main() -> None:
    args = parse_args()
    dataset_json = args.dataset_json.resolve()
    split_root = args.split_root.resolve()
    output_dir = args.output_dir.resolve()

    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset JSON: {dataset_json}")
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split root: {split_root}")

    data = load_json(dataset_json)
    images = data["images"]
    annotations = data["annotations"]

    split_lookup = build_split_lookup(split_root)

    image_meta: dict[str, dict] = {}
    for image in images:
        basename = Path(image["file_name"]).name
        split = split_lookup.get(basename)
        if split is None:
            continue
        image_meta[image["id"]] = {
            "basename": basename,
            "stem": Path(basename).stem,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "split": split,
        }

    labels_by_image: dict[str, list[str]] = defaultdict(list)
    skipped_annotations = 0
    for annot in annotations:
        meta = image_meta.get(annot["image_id"])
        if meta is None:
            continue

        width = meta["width"]
        height = meta["height"]
        bbox = annot.get("bbox")
        if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
            skipped_annotations += 1
            continue

        xc, yc, wn, hn = coco_bbox_to_yolo(bbox, width, height)
        xc = clamp01(xc)
        yc = clamp01(yc)
        wn = clamp01(wn)
        hn = clamp01(hn)
        if wn <= 0.0 or hn <= 0.0:
            skipped_annotations += 1
            continue

        labels_by_image[annot["image_id"]].append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    for split in SPLIT_MAP.values():
        ensure_dir(output_dir / "labels" / split)

    written_labels = 0
    empty_labels = 0
    for image_id, meta in image_meta.items():
        label_path = output_dir / "labels" / meta["split"] / f"{meta['stem']}.txt"
        if label_path.exists() and not args.overwrite:
            continue
        lines = labels_by_image.get(image_id, [])
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        written_labels += 1
        if not lines:
            empty_labels += 1

    symlink_split_dirs(split_root, output_dir)
    write_data_yaml(output_dir / "data.yaml", output_dir)

    print(f"[INFO] Prepared CFDD YOLO dataset at: {output_dir}")
    print(f"[INFO] Images with split assignment: {len(image_meta)}")
    print(f"[INFO] Label files written: {written_labels}")
    print(f"[INFO] Empty label files written: {empty_labels}")
    print(f"[INFO] Annotations skipped: {skipped_annotations}")
    print(f"[INFO] Data YAML: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
