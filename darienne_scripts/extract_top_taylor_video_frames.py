#!/usr/bin/env python3
"""Extract every frame from Taylor FishCam03 videos ranked by localization count."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

import darienne_scripts.build_darienne_dataset as dataset


DEFAULT_TAYLOR_ROOT = dataset.DEFAULT_ROOT / "FishCam03_birdseye_Taylor_Islet"
DEFAULT_OUTPUT = Path("/Users/wendycao/fish/darienne_frames")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save all frames from the top Taylor FishCam03 videos by localization count."
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_TAYLOR_ROOT)
    parser.add_argument("--localizations", type=Path, default=dataset.DEFAULT_LOCALIZATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-videos", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--default-duration-sec", type=float, default=299.5)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_output(output: Path, overwrite: bool, dry_run: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output directory is not empty: {output}\nUse --overwrite to replace it.")
        if not dry_run:
            shutil.rmtree(output)
    (output / "frames").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frame_count_for(video: dataset.Video, fps: float) -> int:
    return int(video.duration_sec * fps + 0.5)


def run_ffmpeg_extract(
    video: dataset.Video,
    output_dir: Path,
    fps: float,
    video_index: int,
    total_videos: int,
    total_done_before: int,
    total_expected: int,
    started_at: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video.path),
        "-vf",
        f"fps={fps:g}",
        "-q:v",
        "2",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        str(pattern),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_print = 0.0
    current_frame = 0
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=", 1)[1])
            except ValueError:
                pass

        now = time.monotonic()
        if now - last_print < 2.0 and "progress=end" not in line:
            continue
        last_print = now

        total_done = min(total_done_before + current_frame, total_expected)
        elapsed = now - started_at
        rate = total_done / elapsed if elapsed > 0 else 0.0
        remaining = (total_expected - total_done) / rate if rate > 0 else 0.0
        video_percent = min(100.0, 100.0 * current_frame / max(frame_count_for(video, fps), 1))
        total_percent = 100.0 * total_done / max(total_expected, 1)
        print(
            f"Video {video_index}/{total_videos} {video_percent:5.1f}% | "
            f"overall {total_done}/{total_expected} ({total_percent:5.1f}%) | "
            f"elapsed {elapsed/60:.1f} min | ETA {remaining/60:.1f} min",
            file=sys.stderr,
        )

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command, stderr=stderr)


def video_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


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
    localization_samples = dataset.read_localization_samples(args.localizations, videos)
    counts = Counter(sample.video.path for sample in localization_samples)
    if not counts:
        raise SystemExit("No localization timestamps matched Taylor FishCam03 videos.")

    top_paths = [path for path, _ in counts.most_common(args.top_videos)]
    videos_by_path = {video.path: video for video in videos}
    selected_videos = [videos_by_path[path] for path in top_paths]
    expected_counts = [frame_count_for(video, args.fps) for video in selected_videos]
    total_expected = sum(expected_counts)

    video_rows: list[dict[str, object]] = []
    for rank, video in enumerate(selected_videos, start=1):
        video_rows.append(
            {
                "video_rank": rank,
                "localization_count": counts[video.path],
                "expected_frames": frame_count_for(video, args.fps),
                "fps": args.fps,
                "video_start_utc": video.start_utc.isoformat().replace("+00:00", "Z"),
                "video_duration_sec": video.duration_sec,
                "video_path": str(video.path),
                "frame_dir": str(args.output / "frames" / f"rank{rank:03d}_{video_id(video.path)}"),
            }
        )

    summary = {
        "video_root": str(args.video_root),
        "localizations": str(args.localizations),
        "output": str(args.output),
        "dry_run": args.dry_run,
        "top_videos_requested": args.top_videos,
        "top_videos_used": len(selected_videos),
        "fps": args.fps,
        "videos_found": len(videos),
        "videos_with_localizations": len(counts),
        "matched_localizations": len(localization_samples),
        "expected_frames": total_expected,
        "default_duration_sec": args.default_duration_sec,
        "include_invalid": args.include_invalid,
    }
    write_csv(args.output / "top_videos.csv", video_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 0

    started_at = time.monotonic()
    total_done_before = 0
    failures: list[dict[str, object]] = []
    for rank, video in enumerate(selected_videos, start=1):
        frame_dir = args.output / "frames" / f"rank{rank:03d}_{video_id(video.path)}"
        try:
            run_ffmpeg_extract(
                video,
                frame_dir,
                args.fps,
                rank,
                len(selected_videos),
                total_done_before,
                total_expected,
                started_at,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(
                {
                    "video_rank": rank,
                    "video_path": str(video.path),
                    "error": str(exc),
                }
            )
        actual_count = len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0
        video_rows[rank - 1]["actual_frames"] = actual_count
        total_done_before += actual_count

    write_csv(args.output / "top_videos.csv", video_rows)
    write_csv(args.output / "failures.csv", failures)
    summary["actual_frames"] = total_done_before
    summary["failed_videos"] = len(failures)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
