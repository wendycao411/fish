#!/usr/bin/env python3
"""Unified deduplication utility for CVAT manifests with multiple scopes.

Usage:
  python scripts/deduplicate_cvat_manifest.py --scope {filename,intra-batch,video-offset,overlap} [--roots DIR ...] [--apply] [--dataset DATASET]

Modes:
  filename (default)
    - Dedupes by filename across all batches in a root.
    - Keeps first occurrence (in batch alphabetical order).

  intra-batch
    - Removes duplicate rows within each batch's manifest (same filename).

  video-offset
    - Dedupes by (video_path, video_offset_sec) pair across manifests.
    - More semantic than filename (matches source frames).

  overlap
    - Removes repeated frames from overlapping localization windows.
    - Compares (video_path, frame_timestamp_utc) pairs.
    - Scans batches in order, keeping first frame timestamp seen per video.

Default roots depend on dataset:
  - darienne: darienne_localization_windows/cvat_batches, darienne_danger_rocks_localization_windows/cvat_batches
  - taylor: taylor_islet_localization_windows/cvat_batches
  - danger: danger_rocks_localization_windows/cvat_batches
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "darienne"


def get_default_roots(dataset: str) -> list[Path]:
    if dataset == "taylor":
        return [REPO_ROOT / "taylor_islet_localization_windows" / "cvat_batches"]
    elif dataset == "danger":
        return [REPO_ROOT / "danger_rocks_localization_windows" / "cvat_batches"]
    else:  # darienne
        return [
            REPO_ROOT / "darienne_localization_windows" / "cvat_batches",
            REPO_ROOT / "darienne_danger_rocks_localization_windows" / "cvat_batches",
        ]


def gather_batches(root: Path) -> list[Path]:
    if not root.exists():
        return []
    batches = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("batch_")]
    return batches


def gather_manifests(roots: list[Path]) -> list[Path]:
    manifests = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("batch_*/manifest.csv")):
            manifests.append(p)
    return manifests


def parse_timestamp(value: str) -> datetime:
    """Parse ISO 8601 timestamp with optional Z suffix."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def update_batch_summary(root: Path, counts_by_batch: dict[str, int]) -> None:
    summary = root / "batch_summary.csv"
    if not summary.exists():
        return
    rows = []
    with summary.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            batch = row.get("batch")
            if batch in counts_by_batch:
                row["image_count"] = str(counts_by_batch[batch])
            rows.append(row)
    with summary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ========== Dedup by filename (across batches) ==========
def dedup_by_filename(roots: list[Path], apply: bool = False) -> int:
    """Deduplicate by filename across all batches in a root."""
    all_batches = []
    for root in roots:
        all_batches.extend(gather_batches(root))

    if not all_batches:
        print("No batch_* directories found.")
        return 0

    print(f"Found {len(all_batches)} batch directories.")

    # Map basename -> list of full paths
    name_map: dict[str, list[Path]] = defaultdict(list)
    for batch in all_batches:
        for p in sorted(batch.iterdir()):
            if not p.is_file() or p.name in {"manifest.csv", "batch_summary.csv"}:
                continue
            name_map[p.name].append(p)

    duplicates = {name: paths for name, paths in name_map.items() if len(paths) > 1}
    print(f"Found {len(duplicates)} duplicated basenames across batches.")

    to_remove: list[Path] = []
    keep_for_batch: dict[Path, set[str]] = {batch: set() for batch in all_batches}

    # Keep first occurrence per duplicate basename
    for name, paths in duplicates.items():
        paths_sorted = sorted(paths, key=lambda p: all_batches.index(p.parent) if p.parent in all_batches else 0)
        keeper = paths_sorted[0]
        keep_for_batch[keeper.parent].add(name)
        to_remove.extend(paths_sorted[1:])

    # Mark non-duplicated files as keep
    for name, paths in name_map.items():
        if len(paths) == 1:
            keep_for_batch[paths[0].parent].add(name)

    # Report and apply removals
    removal_counts: dict[Path, int] = defaultdict(int)
    for p in to_remove:
        removal_counts[p.parent] += 1
        if apply:
            p.unlink()

    for batch, count in sorted(removal_counts.items()):
        print(f"  {batch.name}: removing {count} duplicate files")

    if to_remove and not apply:
        print(f"Dry-run: {len(to_remove)} files would be removed. Re-run with --apply to remove.")
        return 0

    # Update manifests
    counts_by_batch: dict[str, int] = {}
    for batch in all_batches:
        manifest = batch / "manifest.csv"
        if not manifest.exists():
            continue
        keep_fnames = keep_for_batch.get(batch, set())
        out_rows = []
        with manifest.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            for row in reader:
                fname = row.get("filename") or Path(row.get("batch_image_path", "")).name
                if fname in keep_fnames:
                    out_rows.append(row)

        if apply:
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(out_rows)
        counts_by_batch[batch.name] = len(out_rows)

    # Update batch_summary.csv per root
    if apply:
        for root in roots:
            update_batch_summary(root, counts_by_batch)
        print("Updated manifests and batch_summary.csv files.")

    return len(to_remove)


# ========== Dedup within each batch manifest ==========
def dedup_intra_batch(roots: list[Path], apply: bool = False) -> int:
    """Remove duplicate rows within each batch's manifest."""
    all_batches = []
    for root in roots:
        all_batches.extend(gather_batches(root))

    if not all_batches:
        return 0

    total_removed = 0
    counts_by_batch = {}

    for batch in all_batches:
        manifest = batch / "manifest.csv"
        if not manifest.exists():
            continue

        with manifest.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        seen = set()
        out_rows = []
        removed_count = 0

        for r in rows:
            fname = r.get("filename") or Path(r.get("batch_image_path", "")).name
            if fname in seen:
                removed_count += 1
                continue
            seen.add(fname)
            out_rows.append(r)

        if removed_count > 0:
            print(f"{batch.name}: removing {removed_count} duplicate rows ({len(rows)} -> {len(out_rows)})")
            total_removed += removed_count
            counts_by_batch[batch.name] = len(out_rows)

            if apply:
                bak = batch / "manifest.csv.bak"
                if not bak.exists():
                    manifest.rename(bak)
                with manifest.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(out_rows)

    if total_removed == 0:
        print("No intra-manifest duplicates found.")
        return 0

    if apply:
        for root in roots:
            update_batch_summary(root, counts_by_batch)
        print("Applied changes and updated batch_summary.csv.")
    else:
        print(f"Dry-run: {total_removed} duplicate rows would be removed. Re-run with --apply.")

    return total_removed


# ========== Dedup by (video_path, video_offset_sec) ==========
def dedup_by_video_offset(roots: list[Path], apply: bool = False) -> int:
    """Deduplicate frames by (video_path, video_offset_sec) pair."""
    manifests = gather_manifests(roots)
    if not manifests:
        print("No manifests found.")
        return 0

    # Build index: (video_path, video_offset_sec) -> list of (manifest, batch, filename, batch_image_path)
    idx = defaultdict(list)
    for m in manifests:
        batch = m.parent
        with m.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                video = (row.get("video_path") or "").strip()
                offset = (row.get("video_offset_sec") or "").strip()
                if not video or not offset:
                    continue
                key = (video, offset)
                entry = {
                    "manifest": m,
                    "batch": batch,
                    "filename": row.get("filename") or Path(row.get("batch_image_path", "")).name,
                    "batch_image_path": row.get("batch_image_path") or "",
                    "row": row,
                }
                idx[key].append(entry)

    # Plan removals: keep first occurrence per key
    removals = []
    for key, entries in idx.items():
        if len(entries) <= 1:
            continue
        entries_sorted = sorted(entries, key=lambda e: (e["batch"].name, e["filename"]))
        for e in entries_sorted[1:]:
            removals.append((e["manifest"], e["batch"], e["filename"], e["batch_image_path"]))

    if not removals:
        print("No duplicates by (video_path, video_offset_sec) found.")
        return 0

    print(f"Found {len(removals)} duplicate (video_path, video_offset_sec) frames to remove.")

    if apply:
        # Group by manifest and apply
        removals_by_manifest = defaultdict(set)
        for manifest, batch, fname, batch_image_path in removals:
            removals_by_manifest[manifest].add(fname)

        removed_by_batch = defaultdict(list)
        for manifest, fnames in removals_by_manifest.items():
            batch = manifest.parent
            bak = batch / "manifest.csv.bak"
            if not bak.exists():
                manifest.rename(bak)

            with bak.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames
                rows = list(reader)

            out_rows = []
            for r in rows:
                fname = r.get("filename") or Path(r.get("batch_image_path", "")).name
                if fname in fnames:
                    removed_by_batch[batch].append(fname)
                    # Try to remove the file
                    bip = r.get("batch_image_path")
                    if bip:
                        try:
                            Path(bip).unlink()
                        except Exception as e:
                            print(f"Warning: could not remove {bip}: {e}", file=sys.stderr)
                    continue
                out_rows.append(r)

            with manifest.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(out_rows)

        # Update batch_summary.csv per root
        for root in roots:
            counts_by_batch = {}
            for batch in gather_batches(root):
                manifest = batch / "manifest.csv"
                if manifest.exists():
                    with manifest.open("r") as fh:
                        counts_by_batch[batch.name] = sum(1 for _ in fh) - 1
            update_batch_summary(root, counts_by_batch)

        print("Applied changes.")
    else:
        print("Dry-run: Re-run with --apply to remove files and update manifests.")

    return len(removals)


# ========== Dedup by overlapping frames ==========
def dedup_by_overlap(roots: list[Path], apply: bool = False) -> int:
    """Remove repeated frames from overlapping localization windows."""
    manifests = gather_manifests(roots)
    if not manifests:
        print("No manifests found.")
        return 0

    last_kept_frame_time: dict[str, datetime] = {}
    removed_files: list[Path] = []
    overlap_count = 0

    for manifest in manifests:
        batch = manifest.parent
        with manifest.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        out_rows = []
        for row in rows:
            video_path = row.get("video_path", "")
            frame_timestamp_raw = row.get("frame_timestamp_utc", "")
            frame_timestamp = parse_timestamp(frame_timestamp_raw)

            last_frame_timestamp = last_kept_frame_time.get(video_path)
            if frame_timestamp is not None and last_frame_timestamp is not None and frame_timestamp <= last_frame_timestamp:
                overlap_count += 1
                batch_image_path = row.get("batch_image_path") or row.get("image_path")
                if batch_image_path:
                    removed_files.append(Path(batch_image_path))
                continue

            if frame_timestamp is not None:
                last_kept_frame_time[video_path] = frame_timestamp
            out_rows.append(row)

        if apply and len(out_rows) < len(rows):
            bak = batch / "manifest.csv.bak"
            if not bak.exists():
                manifest.rename(bak)
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys() if rows else [])
                writer.writeheader()
                writer.writerows(out_rows)

    if overlap_count == 0:
        print("No overlapping frames found.")
        return 0

    print(f"Found {overlap_count} overlapping frames to remove.")

    if apply:
        for p in removed_files:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: could not remove {p}: {e}", file=sys.stderr)

        # Update batch_summary.csv per root
        for root in roots:
            counts_by_batch = {}
            for batch in gather_batches(root):
                manifest = batch / "manifest.csv"
                if manifest.exists():
                    with manifest.open("r") as fh:
                        counts_by_batch[batch.name] = sum(1 for _ in fh) - 1
            update_batch_summary(root, counts_by_batch)
        print("Applied changes.")
    else:
        print("Dry-run: Re-run with --apply to remove files and update manifests.")

    return overlap_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified deduplication utility for CVAT manifests.",
        epilog="Modes: filename (across batches), intra-batch (within batch), video-offset (by source frame), overlap (by timestamp)",
    )
    parser.add_argument(
        "--scope",
        choices=["filename", "intra-batch", "video-offset", "overlap"],
        default="filename",
        help="Deduplication scope/mode (default: filename)",
    )
    parser.add_argument(
        "--dataset",
        choices=["darienne", "taylor", "danger"],
        default=DEFAULT_DATASET,
        help="Dataset to deduplicate (determines default roots)",
    )
    parser.add_argument("--roots", type=Path, nargs="*", help="Override default roots")
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise dry-run")
    args = parser.parse_args(argv)

    roots = args.roots if args.roots else get_default_roots(args.dataset)
    roots = [Path(r).expanduser().resolve() for r in roots if Path(r).expanduser().resolve().exists()]

    if not roots:
        print("No valid roots found; exiting.")
        return 1

    print(f"Dedup mode: {args.scope}")
    print(f"Roots: {roots}")
    print()

    if args.scope == "filename":
        return dedup_by_filename(roots, args.apply)
    elif args.scope == "intra-batch":
        return dedup_intra_batch(roots, args.apply)
    elif args.scope == "video-offset":
        return dedup_by_video_offset(roots, args.apply)
    elif args.scope == "overlap":
        return dedup_by_overlap(roots, args.apply)
    else:
        print(f"Unknown scope: {args.scope}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
