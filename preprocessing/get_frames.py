#!/usr/bin/env python3
"""
Extract frames from videos while preserving the original video frame indices
in the output filenames.

Examples
--------
Extract one video:
python preprocessing/get_frames.py \
  --video /data/vision/beery/scratch/wendy/fish/synced_pairs/2780_FishCam01_.../2780_FishCam01_....mp4 \
  --output-dir /data/vision/beery/scratch/wendy/fish/processed/frame_extractions

Extract all videos under a root:
python preprocessing/get_frames.py \
  --video-root /data/vision/beery/scratch/wendy/fish/synced_pairs \
  --output-dir /data/vision/beery/scratch/wendy/fish/processed/frame_extractions
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos and name them with original frame indices."
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="Path to a specific video file. Can be passed multiple times.",
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
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames. Default: 95",
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


def extract_frames_for_video(
    video_path: Path,
    out_dir: Path,
    jpeg_quality: int,
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    written = 0
    failed_reads = 0
    while True:
        next_frame_idx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        ok, frame = cap.read()
        if not ok or frame is None:
            # If the decoder has reached the end cleanly, stop. Otherwise count it as a failed read
            # and stop as well, because OpenCV cannot always recover past corruption.
            if cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                failed_reads += 1
            break

        out_path = out_dir / f"frame_{next_frame_idx:06d}.jpg"
        success = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not success:
            raise RuntimeError(f"Failed to write frame {next_frame_idx} for {video_path}")
        written += 1

    cap.release()
    return written, failed_reads


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
                        "failed_reads": 0,
                        "status": "skipped_existing",
                    }
                )
                continue

        written, failed_reads = extract_frames_for_video(
            video_path=video_path,
            out_dir=clip_out_dir,
            jpeg_quality=args.jpeg_quality,
        )
        manifest_rows.append(
            {
                "clip_name": clip_name,
                "video_path": str(video_path),
                "frames_dir": str(clip_out_dir),
                "frame_count": written,
                "failed_reads": failed_reads,
                "status": "ok" if written > 0 else "no_frames",
            }
        )
        print(f"{clip_name}: extracted {written} frames")

    manifest_path = args.output_dir / "frame_extraction_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_name",
                "video_path",
                "frames_dir",
                "frame_count",
                "failed_reads",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
