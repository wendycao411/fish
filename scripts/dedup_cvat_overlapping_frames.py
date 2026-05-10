#!/usr/bin/env python3
"""Remove repeated source frames caused by overlapping localization windows."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path("/Users/wendycao/fish/taylor_islet_localization_windows/cvat_batches")
DEFAULT_MANIFEST = DEFAULT_ROOT.parent / "manifest.csv"
DEFAULT_SUMMARY = DEFAULT_ROOT.parent / "summary.json"
DEFAULT_BATCH_SUMMARY = DEFAULT_ROOT / "batch_summary.csv"

TOP_MANIFEST_FIELDS = [
    "frame_index",
    "window_index",
    "local_frame_index",
    "frame_timestamp_utc",
    "localization_timestamp_utc",
    "seconds_from_localization",
    "video_offset_sec",
    "image_path",
    "video_path",
    "localization_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove duplicate frame timestamps across CVAT batches.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def backup_file(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Batch root does not exist: {args.root}")

    batch_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("batch_"))
    if not batch_dirs:
        raise SystemExit(f"No batch folders found under {args.root}")

    last_kept_frame_time: dict[str, datetime] = {}
    kept_master_rows: list[dict[str, object]] = []
    kept_batch_rows: dict[Path, list[dict[str, object]]] = {}
    batch_summary_rows: list[dict[str, object]] = []
    overlap_frames_dropped = 0

    for batch_dir in batch_dirs:
        manifest_path = batch_dir / "manifest.csv"
        if not manifest_path.exists():
            continue

        batch_rows: list[dict[str, object]] = []
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_path = row.get("video_path", "")
                frame_timestamp_raw = row.get("frame_timestamp_utc", "")
                frame_timestamp = parse_timestamp(frame_timestamp_raw) if frame_timestamp_raw else None
                last_frame_timestamp = last_kept_frame_time.get(video_path)
                if frame_timestamp is not None and last_frame_timestamp is not None and frame_timestamp <= last_frame_timestamp:
                    overlap_frames_dropped += 1
                    batch_image_path = row.get("batch_image_path") or row.get("image_path")
                    if batch_image_path:
                        Path(batch_image_path).unlink(missing_ok=True)
                    continue

                if frame_timestamp is not None:
                    last_kept_frame_time[video_path] = frame_timestamp
                batch_rows.append(row)
                kept_master_rows.append({field: row.get(field, "") for field in TOP_MANIFEST_FIELDS})

        kept_batch_rows[batch_dir] = batch_rows
        batch_summary_rows.append(
            {
                "batch": batch_dir.name,
                "image_count": len(batch_rows),
                "folder": str(batch_dir),
                "manifest": str(manifest_path),
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "batches": len(batch_dirs),
                    "kept_frames": len(kept_master_rows),
                    "overlap_frames_dropped": overlap_frames_dropped,
                },
                indent=2,
            )
        )
        return 0

    if args.manifest.exists():
        backup_file(args.manifest)
        write_csv(args.manifest, kept_master_rows, TOP_MANIFEST_FIELDS)

    for batch_dir in batch_dirs:
        manifest_path = batch_dir / "manifest.csv"
        rows = kept_batch_rows.get(batch_dir)
        if rows is None or not manifest_path.exists():
            continue
        backup_file(manifest_path)
        write_csv(manifest_path, rows, list(rows[0].keys()) if rows else [])

    if args.batch_summary.exists():
        backup_file(args.batch_summary)
        write_csv(args.batch_summary, batch_summary_rows, ["batch", "image_count", "folder", "manifest"])

    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        summary["actual_frames"] = len(kept_master_rows)
        summary["overlap_frames_dropped"] = overlap_frames_dropped
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "batches": len(batch_dirs),
                "kept_frames": len(kept_master_rows),
                "overlap_frames_dropped": overlap_frames_dropped,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())