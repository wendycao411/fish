#!/usr/bin/env python3
"""Regroup remaining danger_rocks CVAT batch images into ~5000-image batches.

- Backs up existing `darienne_danger_rocks_localization_windows/cvat_batches` to a timestamped folder.
- Reads all existing `manifest.csv` files in the backup and collects rows whose files still exist.
- Moves image files into new `cvat_batches/batch_###` folders of `--batch-size` (default 5000).
- Updates per-batch `manifest.csv` and writes `batch_summary.csv`.

Usage: python3 scripts/regroup_danger_cvat_batches.py [--batch-size N] [--apply]
Default is dry-run: shows what would be done. Use `--apply` to move files and write outputs.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
DANGER_ROOT = REPO / "darienne_danger_rocks_localization_windows"
CVAT_BATCHES = DANGER_ROOT / "cvat_batches"


def load_manifest_rows(src_batches: Path):
    rows = []
    manifests = sorted(src_batches.glob('batch_*/manifest.csv'))
    for m in manifests:
        with m.open('r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # determine file path to check
                bip = row.get('batch_image_path') or row.get('image_path') or row.get('filename')
                if not bip:
                    continue
                rows.append((row, bip))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)

    if not CVAT_BATCHES.exists():
        print(f"No cvat_batches found at {CVAT_BATCHES}")
        return 1

    # backup
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    backup_dir = CVAT_BATCHES.with_name(f"cvat_batches.bak.{ts}")
    if args.apply:
        print(f"Backing up {CVAT_BATCHES} -> {backup_dir}")
        shutil.move(str(CVAT_BATCHES), str(backup_dir))
    else:
        print(f"Dry-run: would back up {CVAT_BATCHES} -> {backup_dir}")
        backup_dir = CVAT_BATCHES

    # load rows from backup_dir
    rows_with_paths = load_manifest_rows(backup_dir)
    # filter to existing files
    existing = []
    missing = 0
    for row, bip in rows_with_paths:
        p = Path(bip)
        if p.exists():
            existing.append((row, p))
        else:
            missing += 1
    print(f"Found {len(rows_with_paths)} manifest rows; {len(existing)} existing files, {missing} missing")

    if not existing:
        print('No files to regroup')
        return 0

    # prepare new cvat_batches dir
    if args.apply:
        CVAT_BATCHES.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Dry-run: would create {CVAT_BATCHES}")

    # move into new batches
    batch_size = args.batch_size
    total = len(existing)
    batch_index = 0
    batch_rows = []
    current_rows = []
    moved = 0

    def new_batch_dir(idx: int) -> Path:
        return CVAT_BATCHES / f"batch_{idx:03d}"

    for i, (row, p) in enumerate(existing):
        if i % batch_size == 0:
            batch_index += 1
            if current_rows:
                batch_rows.append((batch_index - 1, current_rows))
            current_rows = []
        # update row to point to new batch filename (we will compute dest)
        current_rows.append((row, p))
    if current_rows:
        batch_rows.append((batch_index, current_rows))

    print(f"Will create {len(batch_rows)} batches with batch_size {batch_size}")

    summary_rows = []
    for idx, items in batch_rows:
        batch_name = f"batch_{idx:03d}"
        batch_dir = CVAT_BATCHES / batch_name
        if args.apply:
            batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for j, (row, p) in enumerate(items, start=1):
            dest = batch_dir / p.name
            if args.apply:
                try:
                    shutil.move(str(p), str(dest))
                except Exception as e:
                    print(f"Error moving {p} -> {dest}: {e}")
                    continue
            # update row fields
            row['batch'] = batch_name
            row['batch_image_path'] = str(dest)
            row['filename'] = dest.name
            manifest_rows.append(row)
            moved += 1
        # write manifest
        if args.apply and manifest_rows:
            mn = batch_dir / 'manifest.csv'
            with mn.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)
        summary_rows.append({'batch': batch_name, 'image_count': len(manifest_rows), 'folder': str(batch_dir), 'manifest': str(batch_dir / 'manifest.csv')})

    # write batch_summary.csv
    if args.apply:
        bs = CVAT_BATCHES / 'batch_summary.csv'
        with bs.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Moved {moved} files into {len(batch_rows)} batches and wrote batch_summary.csv")
    else:
        print(f"Dry-run: would move {moved} files into {len(batch_rows)} batches and write batch_summary.csv")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
