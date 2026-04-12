#!/usr/bin/env python3
"""
Extract surrounding video frames for unmatched audio-localization cases.

This script:
- reads `audio_motion_unmatched.csv`
- keeps `unmatched_in_frame` rows by default
- merges nearby unmatched rows from the same clip to avoid duplicate review windows
- extracts +/- N context frames from the source synced video
- writes a manifest plus one folder per merged event
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnmatchedRow:
    clip_name: str
    frame_idx: int
    status: str
    split: str
    row_number: int
    frame_stem: str
    image_path: str
    visualization_path: str
    source_csv: str
    time_seconds: float
    pixel_x: float
    pixel_y: float
    nearest_box_distance_px: float


@dataclass(frozen=True)
class EventGroup:
    clip_name: str
    start_frame: int
    end_frame: int
    center_frame: int
    source_rows: tuple[UnmatchedRow, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract surrounding frames for unmatched audio-localization events."
    )
    parser.add_argument(
        "--unmatched-csv",
        type=Path,
        default=Path(
            "/data/vision/beery/scratch/wendy/fish/processed/audio_motion_mismatch_reports/audio_motion_unmatched.csv"
        ),
        help="Path to audio_motion_unmatched.csv",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/synced_pairs"),
        help="Root containing per-clip synced mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write review folders and manifest CSV.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Number of frames before and after the event center to extract.",
    )
    parser.add_argument(
        "--statuses",
        nargs="*",
        default=("unmatched_in_frame",),
        help="Statuses from the unmatched CSV to include. Default: unmatched_in_frame",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Optional cap on merged events to process. 0 means all.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        raise ValueError("Missing integer value")
    return int(float(value))


def load_rows(csv_path: Path, allowed_statuses: set[str]) -> list[UnmatchedRow]:
    rows: list[UnmatchedRow] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", ""))
            if status not in allowed_statuses:
                continue
            rows.append(
                UnmatchedRow(
                    clip_name=str(row["clip_name"]),
                    frame_idx=parse_int(row["frame_idx"]),
                    status=status,
                    split=str(row.get("split", "")),
                    row_number=parse_int(row["row_number"]),
                    frame_stem=str(row.get("frame_stem", "")),
                    image_path=str(row.get("image_path", "")),
                    visualization_path=str(row.get("visualization_path", "")),
                    source_csv=str(row.get("source_csv", "")),
                    time_seconds=parse_float(row.get("time_seconds")),
                    pixel_x=parse_float(row.get("pixel_x")),
                    pixel_y=parse_float(row.get("pixel_y")),
                    nearest_box_distance_px=parse_float(row.get("nearest_box_distance_px")),
                )
            )
    return rows


def merge_rows(rows: list[UnmatchedRow], radius: int) -> list[EventGroup]:
    by_clip: dict[str, list[UnmatchedRow]] = defaultdict(list)
    for row in rows:
        by_clip[row.clip_name].append(row)

    groups: list[EventGroup] = []
    max_gap = radius * 2

    for clip_name, clip_rows in sorted(by_clip.items()):
        clip_rows.sort(key=lambda r: (r.frame_idx, r.row_number))
        current: list[UnmatchedRow] = []
        current_start = -1
        current_end = -1

        for row in clip_rows:
            row_start = row.frame_idx - radius
            row_end = row.frame_idx + radius
            if not current:
                current = [row]
                current_start = row_start
                current_end = row_end
                continue

            if row.frame_idx - current[-1].frame_idx <= max_gap or row_start <= current_end:
                current.append(row)
                current_start = min(current_start, row_start)
                current_end = max(current_end, row_end)
                continue

            groups.append(finalize_group(clip_name, current, current_start, current_end))
            current = [row]
            current_start = row_start
            current_end = row_end

        if current:
            groups.append(finalize_group(clip_name, current, current_start, current_end))

    return groups


def finalize_group(
    clip_name: str,
    rows: list[UnmatchedRow],
    start_frame: int,
    end_frame: int,
) -> EventGroup:
    frame_values = sorted(row.frame_idx for row in rows)
    center_frame = int(round(sum(frame_values) / len(frame_values)))
    return EventGroup(
        clip_name=clip_name,
        start_frame=max(0, start_frame),
        end_frame=max(0, end_frame),
        center_frame=max(0, center_frame),
        source_rows=tuple(rows),
    )


def find_video_path(video_root: Path, clip_name: str) -> Path:
    candidate = video_root / clip_name / f"{clip_name}.mp4"
    if candidate.exists():
        return candidate
    matches = sorted((video_root / clip_name).glob("*.mp4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No mp4 found for clip {clip_name} under {video_root}")


def get_video_frame_count(video_path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != "N/A"]
    for value in values:
        try:
            n = int(value)
        except ValueError:
            continue
        if n > 0:
            return n
    return 0


def extract_single_frame(video_path: Path, frame_idx: int, out_path: Path) -> None:
    select_expr = f"select=eq(n\\,{frame_idx})"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        select_expr,
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def write_group_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "event_id",
        "clip_name",
        "center_frame",
        "window_start",
        "window_end",
        "source_frame_count",
        "source_row_numbers",
        "source_frame_indices",
        "source_visualizations",
        "video_path",
        "event_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.radius < 0:
        raise ValueError("--radius must be >= 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.unmatched_csv, set(args.statuses))
    groups = merge_rows(rows, args.radius)
    if args.max_events > 0:
        groups = groups[: args.max_events]

    manifest_rows: list[dict[str, object]] = []
    for event_idx, group in enumerate(groups, start=1):
        video_path = find_video_path(args.video_root, group.clip_name)
        total_frames = get_video_frame_count(video_path)
        if total_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {video_path}")

        window_start = max(0, group.center_frame - args.radius)
        window_end = min(total_frames - 1, group.center_frame + args.radius)
        event_dir = args.output_dir / f"{event_idx:04d}__{group.clip_name}__f{group.center_frame:06d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(window_start, window_end + 1):
            rel = frame_idx - group.center_frame
            rel_tag = f"{rel:+03d}"
            out_name = f"{rel_tag}__frame_{frame_idx:06d}.jpg"
            extract_single_frame(video_path, frame_idx, event_dir / out_name)

        manifest_rows.append(
            {
                "event_id": event_idx,
                "clip_name": group.clip_name,
                "center_frame": group.center_frame,
                "window_start": window_start,
                "window_end": window_end,
                "source_frame_count": len(group.source_rows),
                "source_row_numbers": ";".join(str(r.row_number) for r in group.source_rows),
                "source_frame_indices": ";".join(str(r.frame_idx) for r in group.source_rows),
                "source_visualizations": ";".join(r.visualization_path for r in group.source_rows if r.visualization_path),
                "video_path": str(video_path),
                "event_dir": str(event_dir),
            }
        )

    manifest_path = args.output_dir / "review_manifest.csv"
    write_group_manifest(manifest_path, manifest_rows)
    print(f"Wrote {manifest_path}")
    print(f"Processed merged events: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
