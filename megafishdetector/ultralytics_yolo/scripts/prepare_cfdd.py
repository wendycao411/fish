#!/usr/bin/env python3
"""Prepare CFDD in Ultralytics-compatible YOLO format.

This script:
1. Reads the CFDD COCO-style JSON annotations.
2. Uses the existing `yolov6_salmon/{train,valid,test}` split directories.
3. Writes YOLO txt labels to `output_dir/labels/{train,val,test}`.
4. Symlinks image files into `output_dir/images/{train,val,test}`.
5. Writes `output_dir/data.yaml` for Ultralytics training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

LAYOUT_VERSION = "3"


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
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Seconds between progress updates while staging images/labels",
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


def copy_image_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def log_info(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    # Ensure progress messages appear in SLURM logs without long buffering delays.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

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
    log_info(f"[INFO] Loaded JSON with {len(images)} images and {len(annotations)} annotations")

    split_lookup = build_split_lookup(split_root)
    log_info(f"[INFO] Indexed split folders under: {split_root}")

    image_meta: dict[str, dict] = {}
    for image in images:
        basename = Path(image["file_name"]).name
        split = split_lookup.get(basename)
        if split is None:
            continue
        split_dir_name = next(k for k, v in SPLIT_MAP.items() if v == split)
        src_image_path = split_root / split_dir_name / basename
        image_meta[image["id"]] = {
            "basename": basename,
            "stem": Path(basename).stem,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "split": split,
            "src_image_path": src_image_path,
        }
    log_info(f"[INFO] Images matched to split folders: {len(image_meta)}")

    labels_by_image: dict[str, list[str]] = defaultdict(list)
    skipped_annotations = 0
    ann_start = time.time()
    ann_last_progress = ann_start
    total_annotations = len(annotations)
    log_info(f"[INFO] Annotation stage starting: total={total_annotations}, progress_interval={args.progress_interval:.1f}s")
    for idx, annot in enumerate(annotations, start=1):
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

        now = time.time()
        if (now - ann_last_progress) >= args.progress_interval:
            elapsed = max(now - ann_start, 1e-9)
            rate = idx / elapsed
            remaining = max(total_annotations - idx, 0)
            eta_seconds = (remaining / rate) if rate > 0 else 0.0
            pct = (idx / total_annotations * 100.0) if total_annotations > 0 else 100.0
            bar = render_progress_bar(idx, total_annotations)
            log_info(
                "[INFO] Annotation stage "
                f"{bar} {pct:6.2f}% ({idx}/{total_annotations}) "
                f"kept_images={len(labels_by_image)} skipped={skipped_annotations} "
                f"rate={rate:.1f} ann/s eta={format_duration(eta_seconds)}"
            )
            ann_last_progress = now

    ann_elapsed = max(time.time() - ann_start, 1e-9)
    ann_rate = total_annotations / ann_elapsed if total_annotations > 0 else 0.0
    log_info(
        "[INFO] Annotation stage "
        f"{render_progress_bar(total_annotations, total_annotations)} 100.00% "
        f"({total_annotations}/{total_annotations}) kept_images={len(labels_by_image)} "
        f"skipped={skipped_annotations} rate={ann_rate:.1f} ann/s eta=00:00"
    )

    for split in SPLIT_MAP.values():
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)
    log_info(f"[INFO] Output directories ready under: {output_dir}")

    written_labels = 0
    empty_labels = 0
    copied_images = 0
    missing_images = 0
    skipped_existing = 0
    total_images = len(image_meta)
    stage_start = time.time()
    last_progress = stage_start

    log_info(f"[INFO] Image stage starting: total={total_images}, progress_interval={args.progress_interval:.1f}s")

    def maybe_print_progress(processed: int, force: bool = False) -> None:
        nonlocal last_progress
        now = time.time()
        if not force and (now - last_progress) < args.progress_interval:
            return

        elapsed = max(now - stage_start, 1e-9)
        rate = processed / elapsed
        remaining = max(total_images - processed, 0)
        eta_seconds = (remaining / rate) if rate > 0 else 0.0
        pct = (processed / total_images * 100.0) if total_images > 0 else 100.0
        bar = render_progress_bar(processed, total_images)
        log_info(
            "[INFO] Image stage "
            f"{bar} {pct:6.2f}% ({processed}/{total_images}) "
            f"copied={copied_images} skipped={skipped_existing} "
            f"missing={missing_images} labels={written_labels} "
            f"rate={rate:.1f} img/s eta={format_duration(eta_seconds)}"
        )
        last_progress = now

    for idx, (image_id, meta) in enumerate(image_meta.items(), start=1):
        image_out_path = output_dir / "images" / meta["split"] / meta["basename"]
        label_path = output_dir / "labels" / meta["split"] / f"{meta['stem']}.txt"
        if image_out_path.exists() and label_path.exists() and not args.overwrite:
            skipped_existing += 1
            maybe_print_progress(idx)
            continue
        if not copy_image_file(meta["src_image_path"], image_out_path):
            missing_images += 1
            maybe_print_progress(idx)
            continue
        lines = labels_by_image.get(image_id, [])
        label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        copied_images += 1
        written_labels += 1
        if not lines:
            empty_labels += 1
        maybe_print_progress(idx)

    maybe_print_progress(total_images, force=True)

    write_data_yaml(output_dir / "data.yaml", output_dir)
    (output_dir / ".layout_version").write_text(LAYOUT_VERSION + "\n", encoding="utf-8")

    log_info(f"[INFO] Prepared CFDD YOLO dataset at: {output_dir}")
    log_info(f"[INFO] Images with split assignment: {len(image_meta)}")
    log_info(f"[INFO] Images copied: {copied_images}")
    log_info(f"[INFO] Label files written: {written_labels}")
    log_info(f"[INFO] Empty label files written: {empty_labels}")
    log_info(f"[INFO] Source images missing and skipped: {missing_images}")
    log_info(f"[INFO] Annotations skipped: {skipped_annotations}")
    log_info(f"[INFO] Data YAML: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
