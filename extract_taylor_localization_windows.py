#!/usr/bin/env python3
"""Extract 10-second frame windows centered on Taylor FishCam03 localization points."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import build_darienne_dataset as dataset


DEFAULT_TAYLOR_ROOT = dataset.DEFAULT_ROOT / "FishCam03_birdseye_Taylor_Islet"
DEFAULT_OUTPUT = Path("/Users/wendycao/fish/darienne_localization_windows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save frames from windows centered on localization points in Taylor "
            "FishCam03 videos."
        )
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_TAYLOR_ROOT)
    parser.add_argument("--localizations", type=Path, default=dataset.DEFAULT_LOCALIZATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-videos", type=int, default=100)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--default-duration-sec", type=float, default=299.5)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--drop-overlap-frames",
        action="store_true",
        help="Drop repeated source frames when overlapping windows produce the same video timestamp.",
    )
    parser.add_argument(
        "--overlay-videos-dir",
        type=Path,
        default=None,
        help="Optional directory with overlay videos; when set, only localizations whose source video has an overlay there will be used.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def prepare_output(output: Path, overwrite: bool, dry_run: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {output}\nUse --overwrite to replace it.")
        if not dry_run:
            shutil.rmtree(output)
    (output / "all_frames").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def expected_window_frames(duration_sec: float, fps: float) -> int:
    return int(duration_sec * fps + 0.5)


def extract_window(sample: dataset.Sample, start_sec: float, duration_sec: float, fps: float, temp_dir: Path) -> None:
    pattern = temp_dir / "frame_%06d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            str(sample.video.path),
            "-vf",
            f"fps={fps:g}",
            "-q:v",
            "2",
            "-y",
            str(pattern),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    prepare_output(args.output, args.overwrite, args.dry_run)

    print(f"Discovering Taylor FishCam03 videos under {args.video_root}", file=sys.stderr)
    videos = dataset.discover_videos(
        args.video_root,
        include_invalid=args.include_invalid,
        probe_videos=False,
        default_duration_sec=args.default_duration_sec,
    )
    if not videos:
        raise SystemExit("No Taylor FishCam03 videos found.")

    print(f"Reading localizations from {args.localizations}", file=sys.stderr)
    samples = dataset.read_localization_samples(args.localizations, videos)
    # If overlay_videos_dir is provided, restrict samples to videos that have overlays
    if args.overlay_videos_dir:
        allowed: set[str] = set()
        try:
            for p in sorted(args.overlay_videos_dir.iterdir()):
                if not p.is_file():
                    continue
                if "Invalid" in p.parts:
                    continue
                # map overlay filename like '<orig>_heatmap_overlay.mp4' -> '<orig>.mp4'
                m = re.sub(r"_heatmap_overlay(?=\.[^.]+$)", "", p.name)
                allowed.add(m)
        except FileNotFoundError:
            raise SystemExit(f"Overlay videos directory not found: {args.overlay_videos_dir}")
        samples = [s for s in samples if s.video.path.name in allowed]
    counts = Counter(sample.video.path for sample in samples)
    if not counts:
        raise SystemExit("No localization timestamps matched Taylor FishCam03 videos.")

    top_paths = [path for path, _ in counts.most_common(args.top_videos)]
    rank_by_path = {path: rank for rank, path in enumerate(top_paths, start=1)}
    selected = sorted(
        [sample for sample in samples if sample.video.path in rank_by_path],
        key=lambda sample: (rank_by_path[sample.video.path], sample.timestamp_utc, sample.offset_sec),
    )

    half_window = args.window_sec / 2.0
    window_rows: list[dict[str, object]] = []
    expected_total_frames = 0
    for window_index, sample in enumerate(selected, start=1):
        start_sec = max(0.0, sample.offset_sec - half_window)
        end_sec = min(sample.video.duration_sec, sample.offset_sec + half_window)
        duration_sec = max(0.0, end_sec - start_sec)
        expected_frames = expected_window_frames(duration_sec, args.fps)
        expected_total_frames += expected_frames
        window_rows.append(
            {
                "window_index": window_index,
                "video_rank": rank_by_path[sample.video.path],
                "video_localization_count": counts[sample.video.path],
                "localization_timestamp_utc": sample.timestamp_utc.isoformat().replace("+00:00", "Z"),
                "window_start_utc": (
                    sample.video.start_utc + timedelta(seconds=start_sec)
                ).isoformat().replace("+00:00", "Z"),
                "window_end_utc": (
                    sample.video.start_utc + timedelta(seconds=end_sec)
                ).isoformat().replace("+00:00", "Z"),
                "video_offset_center_sec": round(sample.offset_sec, 6),
                "video_offset_start_sec": round(start_sec, 6),
                "window_duration_sec": round(duration_sec, 6),
                "expected_frames": expected_frames,
                "localization_ids": ";".join(sample.localization_ids),
                "x_m": sample.x_m,
                "y_m": sample.y_m,
                "z_m": sample.z_m,
                "video_path": str(sample.video.path),
            }
        )

    summary = {
        "video_root": str(args.video_root),
        "localizations": str(args.localizations),
        "output": str(args.output),
        "dry_run": args.dry_run,
        "top_videos_requested": args.top_videos,
        "top_videos_used": len(top_paths),
        "videos_found": len(videos),
        "videos_with_localizations": len(counts),
        "matched_localizations": len(samples),
        "selected_windows": len(selected),
        "window_sec": args.window_sec,
        "fps": args.fps,
        "expected_frames": expected_total_frames,
        "default_duration_sec": args.default_duration_sec,
        "include_invalid": args.include_invalid,
    }

    write_csv(args.output / "windows.csv", window_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 0

    frame_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    frame_total = 0
    started_at = time.monotonic()
    frames_dir = args.output / "all_frames"
    overlap_frames_dropped = 0
    last_kept_frame_time: dict[Path, datetime] = {}

    for window_index, sample in enumerate(selected, start=1):
        start_sec = max(0.0, sample.offset_sec - half_window)
        end_sec = min(sample.video.duration_sec, sample.offset_sec + half_window)
        duration_sec = max(0.0, end_sec - start_sec)
        video_stem = safe_name(sample.video.path.stem)
        loc_stamp = dataset.safe_timestamp(sample.timestamp_utc)

        with tempfile.TemporaryDirectory(prefix="darienne_window_") as temp_name:
            temp_dir = Path(temp_name)
            try:
                extract_window(sample, start_sec, duration_sec, args.fps, temp_dir)
            except subprocess.CalledProcessError as exc:
                failures.append(
                    {
                        "window_index": window_index,
                        "localization_timestamp_utc": sample.timestamp_utc.isoformat().replace("+00:00", "Z"),
                        "video_path": str(sample.video.path),
                        "error": str(exc),
                    }
                )
                continue

            for local_frame_index, temp_frame in enumerate(sorted(temp_dir.glob("frame_*.jpg")), start=1):
                frame_time_sec = start_sec + ((local_frame_index - 1) / args.fps)
                frame_timestamp = sample.video.start_utc + timedelta(seconds=frame_time_sec)
                frame_stamp = dataset.safe_timestamp(frame_timestamp)
                last_frame_timestamp = last_kept_frame_time.get(sample.video.path)
                if args.drop_overlap_frames and last_frame_timestamp is not None and frame_timestamp <= last_frame_timestamp:
                    overlap_frames_dropped += 1
                    temp_frame.unlink(missing_ok=True)
                    continue
                last_kept_frame_time[sample.video.path] = frame_timestamp
                output_name = (
                    f"{video_stem}_loc_{loc_stamp}_"
                    f"win{window_index:05d}_frame{local_frame_index:04d}_{frame_stamp}.jpg"
                )
                output_path = frames_dir / output_name
                shutil.move(str(temp_frame), output_path)
                frame_total += 1
                frame_rows.append(
                    {
                        "frame_index": frame_total,
                        "window_index": window_index,
                        "local_frame_index": local_frame_index,
                        "frame_timestamp_utc": frame_timestamp.isoformat().replace("+00:00", "Z"),
                        "localization_timestamp_utc": sample.timestamp_utc.isoformat().replace("+00:00", "Z"),
                        "seconds_from_localization": round(frame_time_sec - sample.offset_sec, 6),
                        "video_offset_sec": round(frame_time_sec, 6),
                        "image_path": str(output_path),
                        "video_path": str(sample.video.path),
                        "localization_ids": ";".join(sample.localization_ids),
                    }
                )

        if (
            window_index == 1
            or window_index == len(selected)
            or window_index % max(args.progress_every, 1) == 0
        ):
            elapsed = time.monotonic() - started_at
            rate = frame_total / elapsed if elapsed > 0 else 0.0
            remaining = (expected_total_frames - frame_total) / rate if rate > 0 else 0.0
            percent = 100.0 * window_index / len(selected)
            print(
                f"Windows {window_index}/{len(selected)} ({percent:5.1f}%) | "
                f"frames {frame_total}/{expected_total_frames} | "
                f"elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min",
                file=sys.stderr,
            )

    write_csv(args.output / "manifest.csv", frame_rows)
    write_csv(args.output / "failures.csv", failures)
    summary["actual_frames"] = frame_total
    summary["overlap_frames_dropped"] = overlap_frames_dropped
    summary["failed_windows"] = len(failures)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
