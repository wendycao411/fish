#!/usr/bin/env python3
"""Recursively convert raw .h264 videos to .mp4 with ffmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/Volumes/My Passport/Lancaster_AV-array_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find .h264 files recursively and create matching .mp4 files next "
            "to the originals."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Input frame rate to assume for raw .h264 streams. Default: 30",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .mp4 outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without running ffmpeg.",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help=(
            "Re-encode with H.264/AAC-compatible MP4 settings instead of "
            "copying the original video stream."
        ),
    )
    return parser.parse_args()


def iter_h264_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".h264"
    )


def build_ffmpeg_command(
    source: Path,
    destination: Path,
    fps: float,
    overwrite: bool,
    reencode: bool,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-framerate", f"{fps:g}", "-i", str(source)])

    if reencode:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
    else:
        command.extend(["-c:v", "copy", "-movflags", "+faststart"])

    command.append(str(destination))
    return command


def convert_file(
    source: Path,
    fps: float,
    overwrite: bool,
    dry_run: bool,
    reencode: bool,
) -> bool:
    destination = source.with_suffix(".mp4")

    if destination.exists() and not overwrite:
        print(f"skip existing: {destination}")
        return True

    command = build_ffmpeg_command(source, destination, fps, overwrite, reencode)

    if dry_run:
        print(f"would convert: {source} -> {destination}")
        return True

    print(f"converting: {source} -> {destination}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode == 0:
        return True

    print(f"failed: {source}", file=sys.stderr)
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return False


def main() -> int:
    args = parse_args()
    root = args.root.expanduser()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on PATH.", file=sys.stderr)
        return 2

    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 2

    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 2

    files = iter_h264_files(root)
    if not files:
        print(f"No .h264 files found under: {root}")
        return 0

    print(f"Found {len(files)} .h264 file(s) under: {root}")
    failures = 0
    for source in files:
        if not convert_file(
            source=source,
            fps=args.fps,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            reencode=args.reencode,
        ):
            failures += 1

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
