#!/usr/bin/env python3
"""
Collect nearest neighboring frames already present in the train/val dataset
for each unmatched audio-localization event.

This avoids decoding the source mp4s and is useful when some synced videos are
corrupted or partially unreadable.

Output layout per event:
- event_dir/images/*.jpg
- event_dir/heatmaps/*_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFrame:
    clip_name: str
    frame_idx: int
    split: str
    stem: str
    image_path: Path
    heatmap_path: Path


@dataclass(frozen=True)
class UnmatchedRow:
    clip_name: str
    frame_idx: int
    row_number: int
    split: str
    frame_stem: str
    visualization_path: str


@dataclass(frozen=True)
class EventGroup:
    clip_name: str
    center_frame: int
    source_rows: tuple[UnmatchedRow, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect neighboring train/val dataset frames for unmatched events."
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
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/vision/beery/scratch/wendy/fish/processed/extracted_frames_200_heatmap_yolo_separate"
        ),
        help="Dataset root containing images/, labels/, and heatmaps/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write neighboring dataset-frame review folders.",
    )
    parser.add_argument(
        "--neighbors-per-side",
        type=int,
        default=5,
        help="How many nearest dataset frames to keep on each side of the event frame.",
    )
    parser.add_argument(
        "--statuses",
        nargs="*",
        default=("unmatched_in_frame",),
        help="Statuses from the unmatched CSV to include. Default: unmatched_in_frame",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=10,
        help="Merge unmatched rows within this many frames of each other.",
    )
    return parser.parse_args()


def parse_int(value: str | None) -> int:
    if value is None or value == "":
        raise ValueError("Missing integer value")
    return int(float(value))


def load_unmatched_rows(csv_path: Path, allowed_statuses: set[str]) -> list[UnmatchedRow]:
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
                    row_number=parse_int(row["row_number"]),
                    split=str(row.get("split", "")),
                    frame_stem=str(row.get("frame_stem", "")),
                    visualization_path=str(row.get("visualization_path", "")),
                )
            )
    return rows


def merge_rows(rows: list[UnmatchedRow], merge_gap: int) -> list[EventGroup]:
    by_clip: dict[str, list[UnmatchedRow]] = defaultdict(list)
    for row in rows:
        by_clip[row.clip_name].append(row)

    groups: list[EventGroup] = []
    for clip_name, clip_rows in sorted(by_clip.items()):
        clip_rows.sort(key=lambda r: (r.frame_idx, r.row_number))
        current: list[UnmatchedRow] = []
        for row in clip_rows:
            if not current:
                current = [row]
                continue
            if row.frame_idx - current[-1].frame_idx <= merge_gap:
                current.append(row)
                continue
            groups.append(finalize_group(clip_name, current))
            current = [row]
        if current:
            groups.append(finalize_group(clip_name, current))
    return groups


def finalize_group(clip_name: str, rows: list[UnmatchedRow]) -> EventGroup:
    center = int(round(sum(r.frame_idx for r in rows) / len(rows)))
    return EventGroup(clip_name=clip_name, center_frame=center, source_rows=tuple(rows))


def load_dataset_frames(dataset_root: Path) -> dict[str, list[DatasetFrame]]:
    by_clip: dict[str, list[DatasetFrame]] = defaultdict(list)
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        heatmap_dir = dataset_root / "heatmaps" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.jpg")):
            stem = image_path.stem
            clip_name, rest = stem.split("__", 1)
            frame_idx = int(rest.split("_", 1)[0])
            by_clip[clip_name].append(
                DatasetFrame(
                    clip_name=clip_name,
                    frame_idx=frame_idx,
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    heatmap_path=heatmap_dir / f"{stem}_heatmap.png",
                )
            )
    for frames in by_clip.values():
        frames.sort(key=lambda f: f.frame_idx)
    return dict(by_clip)


def choose_neighbors(
    frames: list[DatasetFrame],
    center_frame: int,
    neighbors_per_side: int,
) -> list[DatasetFrame]:
    before = [f for f in frames if f.frame_idx < center_frame]
    after = [f for f in frames if f.frame_idx > center_frame]
    exact = [f for f in frames if f.frame_idx == center_frame]

    selected = before[-neighbors_per_side:] + exact + after[:neighbors_per_side]
    return selected


def safe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "clip_name",
                "center_frame",
                "source_row_numbers",
                "source_frame_indices",
                "source_visualizations",
                "neighbor_frame_idx",
                "delta_from_center",
                "split",
                "image_path",
                "heatmap_path",
                "copied_image_path",
                "copied_heatmap_path",
                "event_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_unmatched_rows(args.unmatched_csv, set(args.statuses))
    groups = merge_rows(rows, args.merge_gap)
    frames_by_clip = load_dataset_frames(args.dataset_root)

    manifest_rows: list[dict[str, object]] = []
    for event_idx, group in enumerate(groups, start=1):
        clip_frames = frames_by_clip.get(group.clip_name, [])
        if not clip_frames:
            continue
        neighbors = choose_neighbors(clip_frames, group.center_frame, args.neighbors_per_side)
        event_dir = args.output_dir / f"{event_idx:04d}__{group.clip_name}__f{group.center_frame:06d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        image_dir = event_dir / "images"
        heatmap_dir = event_dir / "heatmaps"
        image_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for frame in neighbors:
            delta = frame.frame_idx - group.center_frame
            prefix = f"{delta:+05d}__{frame.frame_idx:06d}__{frame.split}"
            copied_image_path = image_dir / f"{prefix}.jpg"
            copied_heatmap_path = heatmap_dir / f"{prefix}_heatmap.png"
            safe_copy(frame.image_path, copied_image_path)
            safe_copy(frame.heatmap_path, copied_heatmap_path)
            manifest_rows.append(
                {
                    "event_id": event_idx,
                    "clip_name": group.clip_name,
                    "center_frame": group.center_frame,
                    "source_row_numbers": ";".join(str(r.row_number) for r in group.source_rows),
                    "source_frame_indices": ";".join(str(r.frame_idx) for r in group.source_rows),
                    "source_visualizations": ";".join(
                        r.visualization_path for r in group.source_rows if r.visualization_path
                    ),
                    "neighbor_frame_idx": frame.frame_idx,
                    "delta_from_center": delta,
                    "split": frame.split,
                    "image_path": str(frame.image_path),
                    "heatmap_path": str(frame.heatmap_path),
                    "copied_image_path": str(copied_image_path),
                    "copied_heatmap_path": str(copied_heatmap_path),
                    "event_dir": str(event_dir),
                }
            )

    manifest_path = args.output_dir / "dataset_neighbor_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote {manifest_path}")
    print(f"Processed merged events: {len(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
