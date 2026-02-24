#!/usr/bin/env python3
"""
Extract frames per video using localization detections:
1) one frame at the midpoint of each detected event
2) random non-event frames to reach a target count

"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract event-midpoint and random frames from synced videos."
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("/Users/wendycao/fish/XAV-arrays/localization/out_synced_pairs"),
        help="Root folder containing per-video localizations_merged_filtered.csv files.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/Users/wendycao/fish/synced_pairs"),
        help="Root folder containing per-video subfolders with mp4 files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Users/wendycao/fish/XAV-arrays/localization/extracted_frames_200"),
        help="Where extracted frames and manifests are written.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=200,
        help="Target number of extracted frames per video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible random-frame sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing jpg files in output folders.",
    )
    parser.add_argument(
        "--min-video-number",
        type=int,
        default=2729,
        help="Only process videos whose folder name starts with a numeric ID >= this value.",
    )
    return parser.parse_args()


def find_csv_files(csv_root: Path) -> List[Path]:
    return sorted(csv_root.glob("*/localizations_merged_filtered.csv"))


def parse_leading_video_number(video_id: str) -> int | None:
    head = video_id.split("_", 1)[0]
    if head.isdigit():
        return int(head)
    return None


def read_event_midpoints_seconds(csv_path: Path) -> List[float]:
    mids: List[float] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_min_offset", "time_max_offset"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{csv_path}: missing required columns {sorted(required)}; found {reader.fieldnames}"
            )

        for row in reader:
            try:
                t0 = float(row["time_min_offset"])
                t1 = float(row["time_max_offset"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(t0) or not math.isfinite(t1):
                continue
            if t1 < t0:
                t0, t1 = t1, t0
            mids.append((t0 + t1) / 2.0)
    return mids


def find_video_file(video_dir: Path) -> Path | None:
    preferred = video_dir / f"{video_dir.name}.mp4"
    if preferred.exists():
        return preferred
    mp4s = sorted(video_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def open_video_props(video_path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0.0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video properties for {video_path}: fps={fps}, frames={total_frames}")
    return fps, total_frames


def clamp_frame_idx(frame_idx: int, total_frames: int) -> int:
    return max(0, min(total_frames - 1, frame_idx))


def evenly_subsample(sorted_unique_values: Sequence[int], k: int) -> List[int]:
    n = len(sorted_unique_values)
    if k >= n:
        return list(sorted_unique_values)
    if k <= 0:
        return []
    if k == 1:
        return [sorted_unique_values[n // 2]]

    sampled: List[int] = []
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        sampled.append(sorted_unique_values[idx])

    seen = set()
    deduped = []
    for value in sampled:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    if len(deduped) == k:
        return deduped

    for value in sorted_unique_values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
        if len(deduped) == k:
            break
    return deduped


def choose_frame_indices(
    midpoint_seconds: Sequence[float],
    fps: float,
    total_frames: int,
    target_frames: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    if target_frames <= 0:
        return [], []

    target_frames = min(target_frames, total_frames)

    event_frame_set = {
        clamp_frame_idx(int(round(mid_sec * fps)), total_frames) for mid_sec in midpoint_seconds
    }
    event_frames = sorted(event_frame_set)
    event_selected = evenly_subsample(event_frames, min(len(event_frames), target_frames))

    needed_random = target_frames - len(event_selected)
    if needed_random <= 0:
        return event_selected, []

    blocked = set(event_selected)
    random_pool = [idx for idx in range(total_frames) if idx not in blocked]
    if needed_random >= len(random_pool):
        random_selected = sorted(random_pool)
    else:
        random_selected = sorted(rng.sample(random_pool, needed_random))

    return event_selected, random_selected


def extract_frames(
    video_path: Path,
    output_dir: Path,
    event_frames: Sequence[int],
    random_frames: Sequence[int],
    overwrite: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    frame_to_source: Dict[int, str] = {}
    for idx in event_frames:
        frame_to_source[idx] = "event_mid"
    for idx in random_frames:
        if idx not in frame_to_source:
            frame_to_source[idx] = "random"

    ordered = sorted(frame_to_source.items(), key=lambda x: x[0])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video for extraction: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    written = 0
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "time_seconds", "source", "image_file"])

        for frame_idx, source in ordered:
            out_name = f"{frame_idx:06d}_{source}.jpg"
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                writer.writerow([frame_idx, frame_idx / fps, source, out_name])
                written += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            writer.writerow([frame_idx, frame_idx / fps, source, out_name])
            written += 1

    cap.release()
    return written


def process_one_video(
    csv_path: Path,
    video_root: Path,
    output_root: Path,
    target_frames: int,
    rng: random.Random,
    overwrite: bool,
) -> Tuple[str, int, int, int]:
    video_id = csv_path.parent.name
    video_dir = video_root / video_id
    if not video_dir.exists():
        raise FileNotFoundError(f"No video folder for {video_id}: {video_dir}")

    video_path = find_video_file(video_dir)
    if video_path is None:
        raise FileNotFoundError(f"No mp4 file found in {video_dir}")

    midpoint_seconds = read_event_midpoints_seconds(csv_path)
    fps, total_frames = open_video_props(video_path)
    event_frames, random_frames = choose_frame_indices(
        midpoint_seconds=midpoint_seconds,
        fps=fps,
        total_frames=total_frames,
        target_frames=target_frames,
        rng=rng,
    )

    out_dir = output_root / video_id
    written = extract_frames(video_path, out_dir, event_frames, random_frames, overwrite=overwrite)
    return video_id, len(event_frames), len(random_frames), written


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_files = find_csv_files(args.csv_root)
    if not csv_files:
        raise FileNotFoundError(f"No localization CSV files found under {args.csv_root}")

    filtered_csv_files = []
    skipped_by_number = 0
    for csv_path in csv_files:
        video_id = csv_path.parent.name
        number = parse_leading_video_number(video_id)
        if number is None or number < args.min_video_number:
            skipped_by_number += 1
            continue
        filtered_csv_files.append(csv_path)

    if not filtered_csv_files:
        raise RuntimeError(
            f"No videos meet min-video-number={args.min_video_number} under {args.csv_root}"
        )

    ok = 0
    failed = 0
    for csv_path in filtered_csv_files:
        try:
            video_id, n_event, n_random, n_written = process_one_video(
                csv_path=csv_path,
                video_root=args.video_root,
                output_root=args.output_root,
                target_frames=args.target_frames,
                rng=rng,
                overwrite=args.overwrite,
            )
            print(
                f"[OK] {video_id}: event_mid={n_event}, random={n_random}, total_written={n_written}"
            )
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {csv_path.parent.name}: {exc}")
            failed += 1

    print(
        f"Done. videos_ok={ok}, videos_failed={failed}, skipped_by_number={skipped_by_number}, "
        f"output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
