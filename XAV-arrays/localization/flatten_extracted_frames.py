#!/usr/bin/env python3
"""
Flatten extracted frame folders into one directory.

Input layout:
  input_root/
    <video_id_1>/*.jpg
    <video_id_2>/*.jpg

Output layout:
  output_dir/
    <video_id_1>__<original_name>.jpg
    <video_id_2>__<original_name>.jpg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten extracted frame folders into one output folder.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/Users/wendycao/fish/XAV-arrays/localization/extracted_frames_200"),
        help="Root folder containing per-video subfolders with extracted images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/wendycao/fish/XAV-arrays/localization/extracted_frames_200_flat"),
        help="Single folder where all renamed images will be copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if output filename already exists.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def unique_output_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def flatten_frames(input_root: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for video_dir in sorted(input_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        for img_path in sorted(video_dir.rglob("*")):
            if not img_path.is_file() or not is_image_file(img_path):
                continue

            out_name = f"{video_id}__{img_path.name}"
            out_path = output_dir / out_name

            if out_path.exists() and not overwrite:
                out_path = unique_output_path(out_path)

            if out_path.exists() and overwrite:
                shutil.copy2(img_path, out_path)
            elif out_path.exists():
                skipped += 1
                continue
            else:
                shutil.copy2(img_path, out_path)
            copied += 1

    return copied, skipped


def main() -> None:
    args = parse_args()
    copied, skipped = flatten_frames(args.input_root, args.output_dir, args.overwrite)
    print(f"Done. copied={copied}, skipped={skipped}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
