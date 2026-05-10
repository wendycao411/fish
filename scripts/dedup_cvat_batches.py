#!/usr/bin/env python3
"""Deduplicate image files across CVAT batch folders and update manifests.

Usage: python scripts/dedup_cvat_batches.py [--roots DIR ...] [--apply]

Default roots:
 - darienne_localization_windows/cvat_batches
 - darienne_danger_rocks_localization_windows/cvat_batches

The script finds duplicate basenames across all batches, keeps the first occurrence
(in batch alphabetical order), removes other copies, and updates each batch's
`manifest.csv` and `batch_summary.csv` counts. By default the script runs as
"dry-run" and will only print what would be changed. Pass `--apply` to perform
file removals and manifest updates.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO_ROOT / "darienne_localization_windows" / "cvat_batches",
    REPO_ROOT / "darienne_danger_rocks_localization_windows" / "cvat_batches",
]


def gather_batches(root: Path) -> list[Path]:
    if not root.exists():
        return []
    batches = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("batch_")]
    return batches


def read_manifest_count(batch_root: Path) -> int:
    manifest = batch_root / "manifest.csv"
    if not manifest.exists():
        # count jpg files
        return len([p for p in batch_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    # count rows
    with manifest.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1


def update_manifest(batch_root: Path, keep_filenames: set[str]):
    manifest = batch_root / "manifest.csv"
    if not manifest.exists():
        return 0
    out_rows = []
    with manifest.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return 0
        for row in reader:
            fname = row.get("filename") or Path(row.get("batch_image_path", "")).name
            if fname not in keep_filenames:
                # row removed
                continue
            out_rows.append(row)
    # overwrite manifest
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return len(out_rows)


def update_batch_summary(root: Path, counts_by_batch: dict[str, int]):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--apply", action="store_true", help="Actually remove duplicate files and update manifests")
    args = parser.parse_args(argv)

    # Resolve and filter roots
    roots: list[Path] = [Path(r).expanduser().resolve() for r in args.roots]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("No cvat_batches roots found; nothing to do.")
        return 0

    # Gather batches in order
    all_batches: list[Path] = []
    for root in roots:
        all_batches.extend(gather_batches(root))

    if not all_batches:
        print("No batch_* directories found under provided roots.")
        return 0

    print(f"Found {len(all_batches)} batch directories.")

    # Map basename -> list of (batch_dir, full_path)
    name_map: dict[str, list[Path]] = defaultdict(list)
    for batch in all_batches:
        for p in sorted(batch.iterdir()):
            if not p.is_file():
                continue
            if p.name in {"manifest.csv", "batch_summary.csv"}:
                continue
            name_map[p.name].append(p)

    duplicates = {name: paths for name, paths in name_map.items() if len(paths) > 1}
    print(f"Found {len(duplicates)} duplicated basenames across batches.")

    to_remove: list[Path] = []
    keep_for_batch: dict[Path, set[str]] = {batch: set() for batch in all_batches}
    # Decide which to keep: first occurrence in all_batches order
    for name, paths in duplicates.items():
        # sort paths by batch order according to all_batches
        paths_sorted = sorted(paths, key=lambda p: all_batches.index(p.parent) if p.parent in all_batches else 0)
        keeper = paths_sorted[0]
        keep_for_batch[keeper.parent].add(name)
        for p in paths_sorted[1:]:
            to_remove.append(p)
    # For non-duplicated names, mark as keep
    for name, paths in name_map.items():
        if len(paths) == 1:
            p = paths[0]
            keep_for_batch[p.parent].add(name)

    # Report planned removals
    removal_counts: dict[Path, int] = defaultdict(int)
    for p in to_remove:
        removal_counts[p.parent] += 1

    if not to_remove:
        print("No duplicates to remove.")
        return 0

    print("Planned removals by batch:")
    for batch in all_batches:
        cnt = removal_counts.get(batch, 0)
        if cnt:
            print(f" - {batch}: remove {cnt} files")

    if not args.apply:
        print("Dry-run: no files will be removed. Re-run with --apply to perform changes.")
        return 0

    # Perform removals
    removed_by_batch: dict[Path, list[str]] = defaultdict(list)
    for p in to_remove:
        try:
            p.unlink()
            removed_by_batch[p.parent].append(p.name)
        except Exception as exc:
            print(f"Error removing {p}: {exc}", file=sys.stderr)

    # Update manifests and summary counts
    counts_by_batch_name: dict[str, int] = {}
    for batch in all_batches:
        kept = keep_for_batch.get(batch, set())
        new_count = update_manifest(batch, kept)
        counts_by_batch_name[batch.name] = new_count

    # Update batch_summary.csv for each root
    for root in roots:
        update_batch_summary(root, counts_by_batch_name)

    # Print summary
    total_removed = sum(len(v) for v in removed_by_batch.values())
    print(f"Removed {total_removed} duplicate files.")
    for batch, items in removed_by_batch.items():
        print(f" - {batch.name}: removed {len(items)} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
