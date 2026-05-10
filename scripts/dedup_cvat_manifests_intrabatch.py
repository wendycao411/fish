#!/usr/bin/env python3
"""Remove duplicate rows inside each CVAT batch manifest (same `filename`).

Usage: python scripts/dedup_cvat_manifests_intrabatch.py [--roots DIR ...] [--apply]

Default roots are the two darienne cvat_batches directories.
The script makes a backup of each manifest as `manifest.csv.bak` before overwriting.
When run with --apply it will write deduped manifests and update `batch_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

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


def dedup_manifest(manifest_path: Path) -> tuple[int,int,list[str]]:
    # returns (orig_count, new_count, removed_filenames)
    if not manifest_path.exists():
        return (0, 0, [])
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    seen = set()
    out_rows = []
    removed = []
    for r in rows:
        fname = r.get("filename") or Path(r.get("batch_image_path", "")).name
        if fname in seen:
            removed.append(fname)
            continue
        seen.add(fname)
        out_rows.append(r)
    return (len(rows), len(out_rows), removed, fieldnames)


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
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("No roots found; exiting.")
        return 0

    all_batches = []
    for root in roots:
        all_batches.extend(gather_batches(root))

    total_removed = 0
    counts_by_batch = {}
    for batch in all_batches:
        manifest = batch / "manifest.csv"
        if not manifest.exists():
            # skip
            continue
        orig_count, new_count, removed, fieldnames = dedup_manifest(manifest)
        if not removed:
            continue
        print(f"{batch.name}: would remove {len(removed)} duplicate rows (orig {orig_count} -> {new_count})")
        total_removed += len(removed)
        counts_by_batch[batch.name] = new_count
        if args.apply:
            # backup
            bak = batch / "manifest.csv.bak"
            if not bak.exists():
                manifest.rename(bak)
            # write deduped manifest
            with bak.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or fieldnames
                rows = list(reader)
            seen = set()
            out_rows = []
            for r in rows:
                fname = r.get("filename") or Path(r.get("batch_image_path", "")).name
                if fname in seen:
                    continue
                seen.add(fname)
                out_rows.append(r)
            with manifest.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(out_rows)
    if total_removed == 0:
        print("No intra-manifest duplicates found.")
        return 0
    print(f"Total duplicate manifest rows to remove: {total_removed}")
    if args.apply:
        # update summaries per root
        for root in roots:
            update_batch_summary(root, counts_by_batch)
        print("Applied changes and updated batch_summary.csv entries.")
    else:
        print("Dry-run: nothing written. Re-run with --apply to apply changes.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
