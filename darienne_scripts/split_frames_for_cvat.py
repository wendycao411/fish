#!/usr/bin/env python3
"""Split a flat frame folder into CVAT-sized batch folders."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path


DEFAULT_SOURCE = Path("/Users/wendycao/fish/darienne_localization_windows/all_frames")
DEFAULT_OUTPUT = Path("/Users/wendycao/fish/darienne_localization_windows/cvat_batches")
DEFAULT_MANIFEST = Path("/Users/wendycao/fish/darienne_localization_windows/manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split image frames into CVAT batch folders.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum images per batch. Overrides --batches when provided.",
    )
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    parser.add_argument(
        "--link",
        action="store_true",
        help="Create hardlinks instead of copying. Use when source and output are on the same filesystem.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {output}\nUse --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def read_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = row.get("image_path", "")
            if image_path:
                rows[Path(image_path).name] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def batch_index(item_index: int, total_items: int, batches: int) -> int:
    return min((item_index * batches) // total_items, batches - 1)


def main() -> int:
    args = parse_args()
    if args.batches < 1:
        raise SystemExit("--batches must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.move and args.link:
        raise SystemExit("Choose only one of --move or --link")
    if not args.source.exists():
        raise SystemExit(f"Source folder does not exist: {args.source}")

    images = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No images found in {args.source}")

    batch_count = (
        ((len(images) + args.batch_size - 1) // args.batch_size)
        if args.batch_size is not None
        else args.batches
    )

    prepare_output(args.output, args.overwrite)
    manifest_by_name = read_manifest(args.manifest)
    batch_rows: list[list[dict[str, object]]] = [[] for _ in range(batch_count)]
    summary_rows: list[dict[str, object]] = []
    batch_dirs = [args.output / f"batch_{index:03d}" for index in range(1, batch_count + 1)]
    for batch_dir in batch_dirs:
        batch_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.monotonic()
    action = "Moved" if args.move else ("Linked" if args.link else "Copied")
    for item_index, source_path in enumerate(images):
        batch_zero = (
            min(item_index // args.batch_size, batch_count - 1)
            if args.batch_size is not None
            else batch_index(item_index, len(images), batch_count)
        )
        batch_number = batch_zero + 1
        destination = batch_dirs[batch_zero] / source_path.name
        if args.move:
            shutil.move(str(source_path), str(destination))
        elif args.link:
            destination.unlink(missing_ok=True)
            destination.hardlink_to(source_path)
        else:
            shutil.copy2(source_path, destination)

        row = dict(manifest_by_name.get(source_path.name, {}))
        row["batch"] = f"batch_{batch_number:03d}"
        row["batch_image_path"] = str(destination)
        row["filename"] = source_path.name
        batch_rows[batch_zero].append(row)

        done = item_index + 1
        if done == 1 or done == len(images) or done % max(args.progress_every, 1) == 0:
            elapsed = time.monotonic() - started_at
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (len(images) - done) / rate if rate > 0 else 0.0
            percent = 100.0 * done / len(images)
            print(
                f"{action} {done}/{len(images)} images ({percent:5.1f}%) | "
                f"elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min",
                file=sys.stderr,
            )

    for batch_number, rows in enumerate(batch_rows, start=1):
        batch_name = f"batch_{batch_number:03d}"
        batch_dir = args.output / batch_name
        write_csv(batch_dir / "manifest.csv", rows)
        summary_rows.append(
            {
                "batch": batch_name,
                "image_count": len(rows),
                "folder": str(batch_dir),
                "manifest": str(batch_dir / "manifest.csv"),
            }
        )

    write_csv(args.output / "batch_summary.csv", summary_rows)
    print(f"{action} {len(images)} images into {batch_count} batches at {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
