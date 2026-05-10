#!/usr/bin/env python3
"""Deduplicate frames by (video_path, video_offset_sec) across CVAT manifests.

Usage: python scripts/dedup_by_video_offset.py [--roots DIR ...] [--apply]

Default roots: darienne_localization_windows/cvat_batches and darienne_danger_rocks_localization_windows/cvat_batches

Dry-run by default. Pass --apply to remove duplicate image files and update manifests and batch_summary.csv.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO_ROOT / "darienne_localization_windows" / "cvat_batches",
    REPO_ROOT / "darienne_danger_rocks_localization_windows" / "cvat_batches",
]


def gather_manifests(roots: list[Path]) -> list[Path]:
    manifests = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob('batch_*/manifest.csv')):
            manifests.append(p)
    return manifests


def build_index(manifests: list[Path]):
    idx = defaultdict(list)
    for m in manifests:
        batch = m.parent
        with m.open('r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                video = (row.get('video_path') or '').strip()
                off = (row.get('video_offset_sec') or '').strip()
                if not video or not off:
                    continue
                key = (video, off)
                entry = {
                    'manifest': m,
                    'batch': batch,
                    'row': row,
                    'filename': row.get('filename') or Path(row.get('batch_image_path','')).name,
                    'batch_image_path': row.get('batch_image_path') or '',
                }
                idx[key].append(entry)
    return idx


def plan_actions(idx: dict):
    # for keys with multiple entries, keep first, remove others
    removals = []
    for key, entries in idx.items():
        if len(entries) <= 1:
            continue
        # sort entries by batch name and then by filename to have deterministic choice
        entries_sorted = sorted(entries, key=lambda e: (e['batch'].name, e['filename']))
        keeper = entries_sorted[0]
        for e in entries_sorted[1:]:
            removals.append((e['manifest'], e['batch'], e['filename'], e['batch_image_path']))
    return removals


def apply_removals(removals: list[tuple[Path,Path,str,str]]):
    # removals: list of (manifest_path, batch_dir, filename, batch_image_path)
    removed_by_batch = defaultdict(list)
    # group removals per manifest for manifest editing
    removals_by_manifest = defaultdict(set)
    for manifest, batch, fname, batch_image_path in removals:
        removals_by_manifest[manifest].add(fname)
    # edit manifests with backups
    for manifest, fnames in removals_by_manifest.items():
        batch = manifest.parent
        bak = batch / 'manifest.csv.bak'
        if not bak.exists():
            manifest.rename(bak)
        with bak.open('r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        out_rows = []
        for r in rows:
            fname = r.get('filename') or Path(r.get('batch_image_path','')).name
            if fname in fnames:
                removed_by_batch[batch].append(fname)
                # attempt to remove file
                bip = r.get('batch_image_path')
                if bip:
                    try:
                        p = Path(bip)
                        if p.exists():
                            p.unlink()
                    except Exception as e:
                        print(f"Error removing file {bip}: {e}", file=sys.stderr)
                continue
            out_rows.append(r)
        # write deduped manifest
        with manifest.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
    return removed_by_batch


def update_batch_summary_for_roots(roots: list[Path]):
    # recompute counts from manifest.csv for each batch and update the root's batch_summary.csv
    for root in roots:
        if not root.exists():
            continue
        counts = {}
        for batch in sorted(root.iterdir()):
            if not batch.is_dir():
                continue
            manifest = batch / 'manifest.csv'
            if manifest.exists():
                with manifest.open('r', encoding='utf-8') as fh:
                    cnt = sum(1 for _ in fh) - 1
            else:
                cnt = len([p for p in batch.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}])
            counts[batch.name] = cnt
        summary = root / 'batch_summary.csv'
        if not summary.exists():
            continue
        rows = []
        with summary.open('r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            for row in reader:
                b = row.get('batch')
                if b in counts:
                    row['image_count'] = str(counts[b])
                rows.append(row)
        with summary.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', type=Path, nargs='*', default=DEFAULT_ROOTS)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    manifests = gather_manifests(roots)
    if not manifests:
        print('No manifests found; exiting')
        return 0
    print(f'Found {len(manifests)} manifests')
    idx = build_index(manifests)
    dup_keys = [k for k,v in idx.items() if len(v)>1]
    print(f'Found {len(dup_keys)} duplicated (video,offset) keys')
    # show sample
    sample = list(dup_keys)[:10]
    for k in sample:
        entries = idx[k]
        print('KEY:', k)
        for e in entries[:6]:
            print('  ->', e['batch'].name, e['filename'])
    removals = plan_actions(idx)
    print(f'Planned {len(removals)} duplicate removals (keeping first occurrence per key)')
    if not removals:
        return 0
    # report counts per batch
    from collections import defaultdict
    c=defaultdict(int)
    for _,batch,fname,bip in removals:
        c[batch]+=1
    print('Removals by batch (sample):')
    for b,n in list(c.items())[:40]:
        print(' -', b.name, n)

    if not args.apply:
        print('Dry-run: no files or manifests changed. Re-run with --apply to apply changes.')
        return 0

    removed_by_batch = apply_removals(removals)
    update_batch_summary_for_roots(roots)
    total_removed = sum(len(v) for v in removed_by_batch.values())
    print(f'Applied removals: removed {total_removed} files and updated manifests/summary')
    for b,items in removed_by_batch.items():
        print(' -', b.name, len(items))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
