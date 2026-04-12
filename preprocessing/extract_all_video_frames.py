#!/usr/bin/env python3
"""
Extract all decodable frames from synced videos into numbered image files.

Examples
--------
Extract one clip:
python preprocessing/extract_all_video_frames.py \
  --video /data/vision/beery/scratch/wendy/fish/synced_pairs/2780_FishCam01_.../2780_FishCam01_....mp4 \
  --output-dir /data/vision/beery/scratch/wendy/fish/processed/all_frames

Extract every mp4 under synced_pairs:
python preprocessing/extract_all_video_frames.py \
  --video-root /data/vision/beery/scratch/wendy/fish/synced_pairs \
  --output-dir /data/vision/beery/scratch/wendy/fish/processed/all_frames
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all decodable frames from videos into numbered JPG files."
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="Path to a specific mp4 file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="Optional root directory to scan recursively for mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-video frame folders will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output folder for a clip before extracting.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="ffmpeg JPEG quality scale. Lower is better. Default: 2",
    )
    return parser.parse_args()


def collect_videos(video_args: list[Path], video_root: Path | None) -> list[Path]:
    videos: list[Path] = []
    seen: set[Path] = set()

    for path in video_args:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            videos.append(resolved)

    if video_root is not None:
        for path in sorted(video_root.rglob("*.mp4")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                videos.append(resolved)

    if not videos:
        raise ValueError("No videos provided. Use --video and/or --video-root.")
    return videos


def extract_frames(video_path: Path, out_dir: Path, quality: int) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-q:v",
        str(quality),
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frame_count = len(list(out_dir.glob("frame_*.jpg")))
    stderr = result.stderr.strip()
    return frame_count, stderr


def main() -> int:
    args = parse_args()
    videos = collect_videos(args.video, args.video_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for video_path in videos:
        clip_name = video_path.stem
        clip_out_dir = args.output_dir / clip_name

        if clip_out_dir.exists():
            if args.overwrite:
                shutil.rmtree(clip_out_dir)
            elif any(clip_out_dir.glob("frame_*.jpg")):
                manifest_rows.append(
                    {
                        "clip_name": clip_name,
                        "video_path": str(video_path),
                        "frames_dir": str(clip_out_dir),
                        "frame_count": len(list(clip_out_dir.glob("frame_*.jpg"))),
                        "status": "skipped_existing",
                        "ffmpeg_stderr_tail": "",
                    }
                )
                continue

        frame_count, stderr = extract_frames(video_path, clip_out_dir, args.quality)
        manifest_rows.append(
            {
                "clip_name": clip_name,
                "video_path": str(video_path),
                "frames_dir": str(clip_out_dir),
                "frame_count": frame_count,
                "status": "ok" if frame_count > 0 else "no_frames",
                "ffmpeg_stderr_tail": stderr[-2000:] if stderr else "",
            }
        )
        print(f"{clip_name}: extracted {frame_count} frames")

    manifest_path = args.output_dir / "frame_extraction_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_name",
                "video_path",
                "frames_dir",
                "frame_count",
                "status",
                "ffmpeg_stderr_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
