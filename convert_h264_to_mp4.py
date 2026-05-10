#!/usr/bin/env python3
"""Recursively convert raw .h264 videos to .mp4 with ffmpeg."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/Volumes/My Passport/Lancaster_AV-array_data")
DEFAULT_CAMERA = "FishCam03"
DEFAULT_TARGET_DURATION_SECONDS = 300.0
FRAME_RATE_PATTERN = re.compile(r"(?:^|_)fr-(\d+(?:\.\d+)?)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find .h264 files recursively and create matching .mp4 files. By "
            "default outputs are saved next to the originals."
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
        default=None,
        help=(
            "Input frame rate to use for every raw .h264 stream. If set, this "
            "overrides duration-based FPS calculation."
        ),
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=30.0,
        help="Frame rate to use when a file has no fr-N filename tag. Default: 30",
    )
    parser.add_argument(
        "--target-duration-seconds",
        type=float,
        default=DEFAULT_TARGET_DURATION_SECONDS,
        help=(
            "Derive each input FPS from its frame count so the output has "
            f"this duration. Default: {DEFAULT_TARGET_DURATION_SECONDS:g} "
            "seconds (5 minutes)."
        ),
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
            "Re-encode with H.264/AAC-compatible MP4 settings. This is now "
            "the default because stream-copying these raw H.264 files can "
            "produce incorrect durations."
        ),
    )
    parser.add_argument(
        "--copy-video",
        action="store_true",
        help=(
            "Copy the original H.264 video stream instead of re-encoding. "
            "Use only if you have verified the resulting MP4 duration."
        ),
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"Only convert videos for this camera label. Default: {DEFAULT_CAMERA}",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Convert every .h264 file, including non-FishCam03 folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Save .mp4 files under this writable directory, mirroring paths "
            "relative to the input root. Default: save next to each .h264 file."
        ),
    )
    return parser.parse_args()


def infer_frame_rate(source: Path, fallback_fps: float) -> float:
    match = FRAME_RATE_PATTERN.search(source.stem)
    if match is None:
        return fallback_fps
    return float(match.group(1))


def count_video_frames(source: Path) -> int | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(source),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output or output == "N/A":
        return None

    try:
        return int(output)
    except ValueError:
        return None


def frame_rate_for_source(
    source: Path,
    fps: float | None,
    default_fps: float,
    target_duration_seconds: float | None,
) -> float | None:
    if fps is not None:
        return fps

    if target_duration_seconds is not None:
        frame_count = count_video_frames(source)
        if frame_count is None:
            return None
        return frame_count / target_duration_seconds

    return infer_frame_rate(source, default_fps)


def path_matches_camera(path: Path, camera: str) -> bool:
    camera_lower = camera.lower()
    return any(camera_lower in part.lower() for part in path.parts)


def iter_h264_files(root: Path, camera: str | None) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".h264"
        and (camera is None or path_matches_camera(path, camera))
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


def destination_for(source: Path, root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return source.with_suffix(".mp4")

    relative_source = source.relative_to(root)
    return output_root / relative_source.with_suffix(".mp4")


def convert_file(
    source: Path,
    root: Path,
    output_root: Path | None,
    fps: float | None,
    default_fps: float,
    target_duration_seconds: float | None,
    overwrite: bool,
    dry_run: bool,
    reencode: bool,
    progress: str,
) -> bool:
    destination = destination_for(source, root, output_root)

    if destination.exists() and not overwrite:
        print(f"{progress} skip existing: {destination}")
        return True

    input_fps = frame_rate_for_source(source, fps, default_fps, target_duration_seconds)
    if input_fps is None:
        print(f"failed to count frames for duration-based conversion: {source}", file=sys.stderr)
        return False

    if input_fps <= 0:
        print(f"invalid input FPS for {source}: {input_fps:g}", file=sys.stderr)
        return False

    command = build_ffmpeg_command(source, destination, input_fps, overwrite, reencode)

    if dry_run:
        print(f"{progress} would convert at {input_fps:g} fps: {source} -> {destination}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"{progress} converting at {input_fps:g} fps: {source} -> {destination}")
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
    output_root = args.output_root.expanduser() if args.output_root is not None else None

    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on PATH.", file=sys.stderr)
        return 2

    if args.target_duration_seconds is not None:
        if args.target_duration_seconds <= 0:
            print("--target-duration-seconds must be greater than 0.", file=sys.stderr)
            return 2
        if shutil.which("ffprobe") is None:
            print("ffprobe was not found on PATH.", file=sys.stderr)
            return 2

    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 2

    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 2

    camera = None if args.all_cameras else args.camera
    files = iter_h264_files(root, camera)
    if not files:
        camera_note = "" if camera is None else f" for {camera}"
        print(f"No .h264 files found{camera_note} under: {root}")
        return 0

    camera_note = "all cameras" if camera is None else camera
    print(f"Found {len(files)} .h264 file(s) for {camera_note} under: {root}")
    failures = 0
    total = len(files)
    reencode = args.reencode or not args.copy_video
    for index, source in enumerate(files, start=1):
        if not convert_file(
            source=source,
            root=root,
            output_root=output_root,
            fps=args.fps,
            default_fps=args.default_fps,
            target_duration_seconds=args.target_duration_seconds,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            reencode=reencode,
            progress=f"[{index}/{total}]",
        ):
            failures += 1

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
