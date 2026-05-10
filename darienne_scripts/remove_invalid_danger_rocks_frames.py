#!/usr/bin/env python3
"""Remove frames and manifest rows for videos not in valid overlay_videos.

This operates on:
 - darienne_danger_rocks_localization_windows/cvat_batches/*/manifest.csv
 - darienne_danger_rocks_localization_windows/manifest.csv
 - darienne_danger_rocks_localization_windows/all_frames (if present)

It finds valid videos by listing filenames in darienne_heatmap_overlays/danger/overlay_videos
and matching manifest `video_path` stem against those filenames (stem without extension).

Backups: per-batch manifest -> manifest.csv.bak (if not present); root manifest -> manifest.csv.bak

Usage: python3 scripts/remove_invalid_danger_rocks_frames.py [--apply]
Default: dry-run (no deletions). Use --apply to delete files and update manifests.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict
import sys

REPO = Path(__file__).resolve().parents[1]
DANGER_ROOT = REPO / "darienne_danger_rocks_localization_windows"
OVERLAYS_DIR = REPO / "darienne_heatmap_overlays" / "danger" / "overlay_videos"
CVAT_BATCHES = DANGER_ROOT / "cvat_batches"
ALL_FRAMES = DANGER_ROOT / "all_frames"
ROOT_MANIFEST = DANGER_ROOT / "manifest.csv"
BATCH_SUMMARY = CVAT_BATCHES / "batch_summary.csv"


def load_valid_video_stems() -> set[str]:
    stems = set()
    if not OVERLAYS_DIR.exists():
        return stems
    for p in OVERLAYS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            stems.add(p.stem)
    return stems


def process_manifests(valid_stems: set[str], apply: bool) -> dict:
    manifests = list(CVAT_BATCHES.glob('batch_*/manifest.csv'))
    removed = defaultdict(int)
    removed_files = []
    for manifest in manifests:
        batch = manifest.parent
        with manifest.open('r', encoding='utf-8') as fh:
            reader = list(csv.DictReader(fh))
            fieldnames = reader[0].keys() if reader else []
        keep = []
        to_remove_names = []
        for row in reader:
            vid = row.get('video_path','')
            if not vid:
                keep.append(row)
                continue
            stem = Path(vid).stem
            if stem in valid_stems:
                keep.append(row)
            else:
                removed[batch.name] += 1
                to_remove_names.append(row.get('batch_image_path') or row.get('image_path') or row.get('filename'))
        if not to_remove_names:
            continue
        print(f"Batch {batch.name}: would remove {len(to_remove_names)} rows/files")
        if apply:
            # backup
            bak = batch / 'manifest.csv.bak'
            if not bak.exists():
                manifest.rename(bak)
            # write new manifest
            with manifest.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
                writer.writeheader()
                writer.writerows(keep)
            # remove files
            for bip in to_remove_names:
                if not bip:
                    continue
                p = Path(bip)
                try:
                    if p.exists():
                        p.unlink()
                        removed_files.append(str(p))
                except Exception as e:
                    print(f"Error removing {p}: {e}", file=sys.stderr)
    return {"removed_counts": dict(removed), "removed_files": removed_files}


def process_root_manifest(valid_stems: set[str], apply: bool) -> dict:
    if not ROOT_MANIFEST.exists():
        return {"root_removed": 0}
    with ROOT_MANIFEST.open('r', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = rows[0].keys() if rows else []
    keep = []
    removed = 0
    removed_files = []
    for row in rows:
        vid = row.get('video_path','')
        stem = Path(vid).stem if vid else ''
        if stem in valid_stems:
            keep.append(row)
        else:
            removed += 1
            bip = row.get('image_path')
            if bip:
                removed_files.append(bip)
    print(f"Root manifest: would remove {removed} rows/files")
    if apply and removed>0:
        bak = ROOT_MANIFEST.with_suffix('.csv.bak')
        if not bak.exists():
            ROOT_MANIFEST.rename(bak)
        with ROOT_MANIFEST.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(keep)
        for bip in removed_files:
            p = Path(bip)
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                print(f"Error removing {p}: {e}", file=sys.stderr)
    return {"root_removed": removed, "root_removed_files": removed_files}


def update_batch_summary(apply: bool):
    if not BATCH_SUMMARY.exists():
        return
    # recompute counts
    rows = []
    with BATCH_SUMMARY.open('r', encoding='utf-8') as fh:
        reader = list(csv.DictReader(fh))
        fieldnames = reader[0].keys() if reader else []
    new_rows = []
    for row in reader:
        batch = row.get('batch')
        batch_dir = CVAT_BATCHES / batch
        manifest = batch_dir / 'manifest.csv'
        if manifest.exists():
            with manifest.open('r', encoding='utf-8') as mfh:
                cnt = sum(1 for _ in mfh) - 1
        else:
            cnt = len([p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}])
        row['image_count'] = str(cnt)
        new_rows.append(row)
    print("Would update batch_summary.csv counts")
    if apply:
        with BATCH_SUMMARY.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(new_rows)


def remove_all_frames_in_all_frames(removed_files:list[str], apply: bool):
    # If ALL_FRAMES exists, remove any files that refer to invalid videos
    if not ALL_FRAMES.exists():
        return 0
    removed=0
    for p in ALL_FRAMES.glob('*.jpg'):
        # crude check: if filename contains any valid stem keep, else remove
        if not any(stem in p.name for stem in load_valid_video_stems()):
            removed+=1
            print(f"Would remove {p}")
            if apply:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Error removing {p}: {e}", file=sys.stderr)
    return removed


def main(argv: list[str]|None=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    valid = load_valid_video_stems()
    if not valid:
        print(f"No valid videos found in {OVERLAYS_DIR}")
        return 1
    print(f"Found {len(valid)} valid video stems")
    # process manifests
    result = process_manifests(valid, args.apply)
    root_result = process_root_manifest(valid, args.apply)
    removed_all = remove_all_frames_in_all_frames(result.get('removed_files', []), args.apply)
    update_batch_summary(args.apply)
    total_removed = sum(result.get('removed_counts', {}).values()) + root_result.get('root_removed',0) + removed_all
    print(f"Total rows/files removed: {total_removed}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
