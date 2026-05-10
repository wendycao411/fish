#!/usr/bin/env python3
"""Flatten extracted Darienne video frame folders into one directory."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import time
from pathlib import Path


DEFAULT_DATASET = Path("/Users/wendycao/fish/darienne_frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or move frames from darienne_frames/frames/<video-folder>/ into one "
            "flat folder, renaming each frame with the video name prefix."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--top-videos", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move frames instead of copying them. Empty source folders are removed afterward.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def safe_video_name(video_path: str) -> str:
    stem = Path(video_path).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def read_video_name_map(top_videos_csv: Path) -> dict[Path, str]:
    mapping: dict[Path, str] = {}
    if not top_videos_csv.exists():
        return mapping
    with top_videos_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_dir = row.get("frame_dir", "")
            video_path = row.get("video_path", "")
            if frame_dir and video_path:
                mapping[Path(frame_dir).resolve()] = safe_video_name(video_path)
    return mapping


def fallback_video_name(frame_dir: Path) -> str:
    name = frame_dir.name
    return re.sub(r"^rank\d+_", "", name)


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory is not empty: {output}\nUse --overwrite to write there.")
    output.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    frames_dir = args.frames_dir or args.dataset / "frames"
    top_videos_csv = args.top_videos or args.dataset / "top_videos.csv"
    output = args.output or args.dataset / "all_frames"

    if not frames_dir.exists():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")
    prepare_output(output, args.overwrite)

    video_name_by_dir = read_video_name_map(top_videos_csv)
    frame_dirs = sorted(path for path in frames_dir.iterdir() if path.is_dir())
    jobs: list[tuple[Path, Path]] = []
    for frame_dir in frame_dirs:
        video_name = video_name_by_dir.get(frame_dir.resolve(), fallback_video_name(frame_dir))
        for frame_path in sorted(frame_dir.glob("*.jpg")):
            destination = output / f"{video_name}_{frame_path.name}"
            jobs.append((frame_path, destination))

    if not jobs:
        raise SystemExit(f"No JPG frames found under {frames_dir}")

    started_at = time.monotonic()
    action = "Moved" if args.move else "Copied"
    for index, (source, destination) in enumerate(jobs, start=1):
        if destination.exists() and not args.overwrite:
            raise SystemExit(f"Destination already exists: {destination}\nUse --overwrite to replace it.")
        if args.move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)

        if index == 1 or index == len(jobs) or index % max(args.progress_every, 1) == 0:
            elapsed = time.monotonic() - started_at
            rate = index / elapsed if elapsed > 0 else 0.0
            remaining = (len(jobs) - index) / rate if rate > 0 else 0.0
            percent = 100.0 * index / len(jobs)
            print(
                f"{action} {index}/{len(jobs)} frames ({percent:5.1f}%) | "
                f"elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min",
                file=sys.stderr,
            )

    if args.move:
        for frame_dir in frame_dirs:
            try:
                frame_dir.rmdir()
            except OSError:
                pass

    print(f"{action} {len(jobs)} frames into {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
