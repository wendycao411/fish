#!/usr/bin/env python3
"""Build 10-frame danger_rocks CVAT batches from batch_005 onward.

This reads the existing danger_rocks `cvat_batches` manifests, keeps only 10
centered frames per localization window, removes duplicate frame timestamps
caused by overlapping windows, and regroups the surviving images into new
5000-image batches under `danger_rocks_localization_windows/batches_10`.

The output directory is written with hardlinks by default so the new batch set
is fast to materialize without duplicating image data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "danger_rocks_localization_windows" / "cvat_batches"
DEFAULT_OUTPUT_ROOT = REPO / "danger_rocks_localization_windows" / "batches_10"


@dataclass(frozen=True)
class ManifestRow:
    batch_name: str
    window_index: int
    local_frame_index: int
    frame_timestamp_utc: str
    localization_timestamp_utc: str
    video_path: str
    source_path: Path
    row: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 10-frame danger_rocks CVAT batches from existing batch_005+ manifests.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-batch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--keep-frames", type=int, default=10)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def batch_number(name: str) -> int:
    if not name.startswith("batch_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return -1


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def center_rows(rows: list[dict[str, str]], keep_frames: int) -> list[dict[str, str]]:
    if keep_frames <= 0:
        raise SystemExit("--keep-frames must be at least 1")
    if len(rows) <= keep_frames:
        return rows
    start = max(0, (len(rows) - keep_frames) // 2)
    end = start + keep_frames
    return rows[start:end]


def source_path_for_row(batch_dir: Path, row: dict[str, str]) -> Path:
    filename = row.get("filename")
    if filename:
        return batch_dir / filename
    return Path(row.get("batch_image_path") or row.get("image_path") or "")


def copy_or_link(source: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "hardlink":
        try:
            os.link(source, dest)
            return
        except OSError:
            shutil.copy2(source, dest)
            return
    shutil.copy2(source, dest)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.keep_frames < 1:
        raise SystemExit("--keep-frames must be at least 1")
    if not args.source_root.exists():
        raise SystemExit(f"Source root does not exist: {args.source_root}")

    source_batches = [
        path for path in sorted(args.source_root.iterdir())
        if path.is_dir() and batch_number(path.name) >= args.start_batch
    ]
    if not source_batches:
        raise SystemExit(f"No source batches found at or after batch_{args.start_batch:03d}")

    grouped_rows: list[ManifestRow] = []
    seen_rows = 0
    for batch_dir in source_batches:
        manifest_path = batch_dir / "manifest.csv"
        if not manifest_path.exists():
            continue
        for row in read_manifest(manifest_path):
            seen_rows += 1
            window_index_raw = row.get("window_index", "")
            local_frame_raw = row.get("local_frame_index", "")
            frame_timestamp = row.get("frame_timestamp_utc", "")
            localization_timestamp = row.get("localization_timestamp_utc", "")
            video_path = row.get("video_path", "")
            try:
                window_index = int(window_index_raw)
                local_frame_index = int(local_frame_raw)
            except ValueError:
                continue

            source_path = source_path_for_row(batch_dir, row)
            if not source_path.exists():
                continue

            grouped_rows.append(
                ManifestRow(
                    batch_name=batch_dir.name,
                    window_index=window_index,
                    local_frame_index=local_frame_index,
                    frame_timestamp_utc=frame_timestamp,
                    localization_timestamp_utc=localization_timestamp,
                    video_path=video_path,
                    source_path=source_path,
                    row=dict(row),
                )
            )

    if not grouped_rows:
        raise SystemExit("No manifest rows were loaded from the requested batches")

    windows: dict[tuple[str, int], list[ManifestRow]] = defaultdict(list)
    for item in grouped_rows:
        windows[(item.batch_name, item.window_index)].append(item)

    selected_rows: list[ManifestRow] = []
    for key in sorted(windows.keys(), key=lambda item: (batch_number(item[0]), item[1])):
        rows = sorted(windows[key], key=lambda item: item.local_frame_index)
        selected_rows.extend(
            ManifestRow(
                batch_name=item.batch_name,
                window_index=item.window_index,
                local_frame_index=item.local_frame_index,
                frame_timestamp_utc=item.frame_timestamp_utc,
                localization_timestamp_utc=item.localization_timestamp_utc,
                video_path=item.video_path,
                source_path=item.source_path,
                row=item.row,
            )
            for item in center_rows(rows, args.keep_frames)
        )

    selected_rows.sort(
        key=lambda item: (
            item.video_path,
            item.frame_timestamp_utc,
            item.localization_timestamp_utc,
            item.batch_name,
            item.window_index,
            item.local_frame_index,
        )
    )

    deduped_rows: list[ManifestRow] = []
    seen_frames: set[tuple[str, str]] = set()
    overlap_frames_dropped = 0
    for item in selected_rows:
        frame_key = (item.video_path, item.frame_timestamp_utc)
        if frame_key in seen_frames:
            overlap_frames_dropped += 1
            continue
        seen_frames.add(frame_key)
        deduped_rows.append(item)

    if args.dry_run:
        summary = {
            "source_batches": len(source_batches),
            "source_rows": seen_rows,
            "selected_rows_before_dedup": len(selected_rows),
            "kept_rows": len(deduped_rows),
            "overlap_frames_dropped": overlap_frames_dropped,
            "output_batches": (len(deduped_rows) + args.batch_size - 1) // args.batch_size,
            "output_root": str(args.output_root),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows_by_batch: list[list[dict[str, object]]] = []
    batch_count = (len(deduped_rows) + args.batch_size - 1) // args.batch_size
    for _ in range(batch_count):
        manifest_rows_by_batch.append([])

    for index, item in enumerate(deduped_rows):
        batch_zero = index // args.batch_size
        batch_name = f"batch_{batch_zero + 1:03d}"
        batch_dir = args.output_root / batch_name
        dest = batch_dir / item.source_path.name
        copy_or_link(item.source_path, dest, args.mode)

        row = dict(item.row)
        row["batch"] = batch_name
        row["batch_image_path"] = str(dest)
        row["filename"] = dest.name
        manifest_rows_by_batch[batch_zero].append(row)

    summary_rows: list[dict[str, object]] = []
    for batch_zero, rows in enumerate(manifest_rows_by_batch):
        batch_name = f"batch_{batch_zero + 1:03d}"
        batch_dir = args.output_root / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            write_csv(batch_dir / "manifest.csv", rows)
        summary_rows.append(
            {
                "batch": batch_name,
                "image_count": len(rows),
                "folder": str(batch_dir),
                "manifest": str(batch_dir / "manifest.csv"),
            }
        )

    write_csv(args.output_root / "batch_summary.csv", summary_rows)
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "source_root": str(args.source_root),
                "output_root": str(args.output_root),
                "start_batch": args.start_batch,
                "keep_frames": args.keep_frames,
                "batch_size": args.batch_size,
                "source_batches": len(source_batches),
                "source_rows": seen_rows,
                "selected_rows_before_dedup": len(selected_rows),
                "kept_rows": len(deduped_rows),
                "overlap_frames_dropped": overlap_frames_dropped,
                "output_batches": batch_count,
                "mode": args.mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "kept_rows": len(deduped_rows),
                "overlap_frames_dropped": overlap_frames_dropped,
                "output_batches": batch_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())