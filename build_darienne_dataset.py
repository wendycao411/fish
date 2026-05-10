#!/usr/bin/env python3
"""Build a spaced FishCam03 frame dataset from localization tables and videos."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(
    "/Users/wendycao/Library/CloudStorage/OneDrive-Personal/"
    "audio-visual-data/Test Data for Rupa"
)
DEFAULT_LOCALIZATIONS = DEFAULT_ROOT / "All_Localizations_Daylight_LA_filtered_2_1_FS"
DEFAULT_OUTPUT = Path("/Users/wendycao/fish/DarienneDataset2000")

VIDEO_RE = re.compile(r"FishCam03_(\d{8}T\d{6}(?:\.\d+)?)Z")


@dataclass(frozen=True)
class Video:
    path: Path
    start_utc: datetime
    duration_sec: float
    width: int
    height: int
    fps: float

    @property
    def end_utc(self) -> datetime:
        return self.start_utc + timedelta(seconds=self.duration_sec)


@dataclass(frozen=True)
class Sample:
    label: str
    timestamp_utc: datetime
    video: Video
    offset_sec: float
    localization_ids: tuple[str, ...] = ()
    nearest_localization_sec: float | None = None
    x_m: str = ""
    y_m: str = ""
    z_m: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FishCam03 frame dataset with half localization frames and "
            "half random non-localization frames."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--localizations", type=Path, default=DEFAULT_LOCALIZATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-frames", type=int, default=2000)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=30.0,
        help="Minimum spacing between selected samples, measured in absolute UTC time.",
    )
    parser.add_argument(
        "--negative-exclusion-sec",
        type=float,
        default=2.0,
        help="Reject random frames this close to any localization in the same video.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--default-duration-sec",
        type=float,
        default=299.5,
        help=(
            "Assumed video duration used during discovery. The FishCam03 files checked "
            "were 299.5 seconds long."
        ),
    )
    parser.add_argument(
        "--probe-videos",
        action="store_true",
        help="Use ffprobe for every video's exact duration/fps/size. Slower on OneDrive folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the dataset and write manifests, but do not extract images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include videos in directories named Invalid.",
    )
    parser.add_argument(
        "--overlay-summary",
        type=Path,
        default=None,
        help="Optional CSV overlay summary with `video_path` and `status` columns to restrict videos to status==ok.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print extraction progress every N frames.",
    )
    return parser.parse_args()


def parse_video_timestamp(path: Path) -> datetime | None:
    match = VIDEO_RE.search(path.name)
    if not match:
        return None
    raw = match.group(1)
    fmt = "%Y%m%dT%H%M%S.%f" if "." in raw else "%Y%m%dT%H%M%S"
    return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout


def parse_fps(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = float(denominator)
        return float(numerator) / denominator_float if denominator_float else 0.0
    return float(value)


def probe_video(path: Path) -> tuple[float, int, int, float]:
    output = run_text(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration,width,height",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ]
    )
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return (
        float(fields["duration"]),
        int(fields["width"]),
        int(fields["height"]),
        parse_fps(fields.get("avg_frame_rate", "0")),
    )


def parse_resolution(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d+)x(\d+)_", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def parse_filename_fps(path: Path) -> float:
    match = re.search(r"_fr-(\d+(?:\.\d+)?)_", path.name)
    if match:
        return float(match.group(1))
    return 0.0


def discover_videos(
    root: Path,
    include_invalid: bool,
    probe_videos: bool,
    default_duration_sec: float,
    allowed_video_names: set[str] | None = None,
) -> list[Video]:
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
        and "FishCam03" in str(p)
        and (include_invalid or "Invalid" not in p.parts)
        and (allowed_video_names is None or p.name in allowed_video_names)
    )
    videos: list[Video] = []
    for path in paths:
        start = parse_video_timestamp(path)
        if start is None:
            continue
        if probe_videos:
            try:
                duration, width, height, fps = probe_video(path)
            except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
                print(f"Warning: skipping unreadable video: {path} ({exc})", file=sys.stderr)
                continue
        else:
            width, height = parse_resolution(path)
            fps = parse_filename_fps(path)
            duration = default_duration_sec
        videos.append(Video(path, start, duration, width, height, fps))
    return sorted(videos, key=lambda v: (v.start_utc, str(v.path)))


def parse_localization_datetime(row: dict[str, str]) -> datetime | None:
    # Prefer ISO-style UTC timestamp if present (used in per-video CSVs)
    iso_val = (row.get("localization_timestamp_utc") or row.get("timestamp") or "").strip()
    if iso_val:
        try:
            # Normalize trailing Z to +00:00 for fromisoformat
            if iso_val.endswith("Z"):
                iso = iso_val[:-1] + "+00:00"
            else:
                iso = iso_val
            return datetime.fromisoformat(iso).astimezone(timezone.utc)
        except ValueError:
            pass

    # Fall back to legacy columns and formats
    value = (row.get("Begin Date Time") or "").strip()
    if value:
        normalized = re.sub(r"\s+", " ", value)
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    date_value = (row.get("Begin Date") or "").strip()
    time_value = (row.get("Begin Clock Time") or "").strip()
    if date_value and time_value:
        normalized = f"{date_value} {time_value}"
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def find_video(videos: list[Video], timestamp: datetime) -> Video | None:
    starts = [video.start_utc for video in videos]
    index = bisect.bisect_right(starts, timestamp) - 1
    if index < 0:
        return None
    video = videos[index]
    if video.start_utc <= timestamp < video.end_utc:
        return video
    return None


def read_localization_samples(localization_dir: Path, videos: list[Video]) -> list[Sample]:
    samples: list[Sample] = []
    paths = sorted(p for p in localization_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"})
    for table_path in paths:
        delim = "\t" if table_path.suffix.lower() == ".txt" else ","
        with table_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delim)
            for row_number, row in enumerate(reader, start=2):
                timestamp = parse_localization_datetime(row)
                if timestamp is None:
                    continue
                video = find_video(videos, timestamp)
                if video is None:
                    continue
                offset = (timestamp - video.start_utc).total_seconds()
                localization_id = f"{table_path.name}:{row_number}"
                # support multiple possible column names for coordinates
                x_val = (row.get("x_m") or row.get("x") or "").strip()
                y_val = (row.get("y_m") or row.get("y") or "").strip()
                z_val = (row.get("z_m") or row.get("z") or "").strip()
                samples.append(
                    Sample(
                        label="localization",
                        timestamp_utc=timestamp,
                        video=video,
                        offset_sec=offset,
                        localization_ids=(localization_id,),
                        x_m=x_val,
                        y_m=y_val,
                        z_m=z_val,
                    )
                )
    return sorted(samples, key=lambda s: s.timestamp_utc)


def read_overlay_summary(summary_csv: Path) -> set[str]:
    names: set[str] = set()
    try:
        with summary_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                status = (row.get("status") or "").strip().lower()
                if status == "ok":
                    path = (row.get("video_path") or "").strip()
                    if path:
                        names.add(Path(path).name)
    except FileNotFoundError:
        print(f"Warning: overlay summary not found: {summary_csv}", file=sys.stderr)
    return names


def choose_spaced(
    samples: Iterable[Sample],
    wanted: int,
    min_gap_sec: float,
    rng: random.Random,
) -> list[Sample]:
    candidates = list(samples)
    rng.shuffle(candidates)
    selected: list[Sample] = []
    selected_times: list[datetime] = []
    for sample in candidates:
        if len(selected) >= wanted:
            break
        if not far_from_selected(selected_times, sample.timestamp_utc, min_gap_sec):
            continue
        selected.append(sample)
        bisect.insort(selected_times, sample.timestamp_utc)
    return sorted(selected, key=lambda s: s.timestamp_utc)


def nearest_delta_sec(sorted_times: list[datetime], timestamp: datetime) -> float | None:
    if not sorted_times:
        return None
    index = bisect.bisect_left(sorted_times, timestamp)
    deltas: list[float] = []
    if index < len(sorted_times):
        deltas.append(abs((sorted_times[index] - timestamp).total_seconds()))
    if index:
        deltas.append(abs((sorted_times[index - 1] - timestamp).total_seconds()))
    return min(deltas) if deltas else None


def far_from_selected(sorted_selected: list[datetime], timestamp: datetime, min_gap_sec: float) -> bool:
    delta = nearest_delta_sec(sorted_selected, timestamp)
    return delta is None or delta >= min_gap_sec


def make_random_samples(
    videos: list[Video],
    localization_samples: list[Sample],
    existing_samples: list[Sample],
    wanted: int,
    min_gap_sec: float,
    exclusion_sec: float,
    rng: random.Random,
) -> list[Sample]:
    loc_times_by_video: dict[Path, list[datetime]] = {}
    for sample in localization_samples:
        loc_times_by_video.setdefault(sample.video.path, []).append(sample.timestamp_utc)
    for times in loc_times_by_video.values():
        times.sort()

    selected_times = sorted(sample.timestamp_utc for sample in existing_samples)
    selected_random: list[Sample] = []
    weights = [max(video.duration_sec, 0.0) for video in videos]
    attempts = max(10000, wanted * 500)

    for _ in range(attempts):
        if len(selected_random) >= wanted:
            break
        video = rng.choices(videos, weights=weights, k=1)[0]
        if video.duration_sec <= 1.0:
            continue
        offset = rng.uniform(0.5, video.duration_sec - 0.5)
        timestamp = video.start_utc + timedelta(seconds=offset)

        nearest_loc = nearest_delta_sec(loc_times_by_video.get(video.path, []), timestamp)
        if nearest_loc is not None and nearest_loc < exclusion_sec:
            continue
        if not far_from_selected(selected_times, timestamp, min_gap_sec):
            continue

        sample = Sample(
            label="random",
            timestamp_utc=timestamp,
            video=video,
            offset_sec=offset,
            nearest_localization_sec=nearest_loc,
        )
        selected_random.append(sample)
        bisect.insort(selected_times, timestamp)

    return sorted(selected_random, key=lambda s: s.timestamp_utc)


def sample_to_row(sample: Sample, image_path: Path | None, index: int) -> dict[str, str | int | float]:
    return {
        "dataset_index": index,
        "label": sample.label,
        "timestamp_utc": sample.timestamp_utc.isoformat().replace("+00:00", "Z"),
        "video_start_utc": sample.video.start_utc.isoformat().replace("+00:00", "Z"),
        "video_offset_sec": round(sample.offset_sec, 6),
        "video_path": str(sample.video.path),
        "image_path": str(image_path) if image_path else "",
        "localization_ids": ";".join(sample.localization_ids),
        "nearest_localization_sec": (
            "" if sample.nearest_localization_sec is None else round(sample.nearest_localization_sec, 6)
        ),
        "x_m": sample.x_m,
        "y_m": sample.y_m,
        "z_m": sample.z_m,
        "video_width": sample.video.width,
        "video_height": sample.video.height,
        "video_fps": round(sample.video.fps, 6),
    }


def safe_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"


def extract_frame(sample: Sample, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{sample.offset_sec:.6f}",
            "-i",
            str(sample.video.path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(image_path),
        ],
        check=True,
    )


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        existing = list(output.iterdir())
        if existing:
            raise SystemExit(
                f"Output directory already exists and is not empty: {output}\n"
                "Use --overwrite to add/replace dataset files there."
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "images" / "localization").mkdir(parents=True, exist_ok=True)
    (output / "images" / "random").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    prepare_output(args.output, args.overwrite)

    positive_target = round(args.total_frames * args.positive_fraction)
    random_target = args.total_frames - positive_target

    print(f"Discovering FishCam03 videos under {args.root}", file=sys.stderr)
    videos = discover_videos(
        args.root,
        args.include_invalid,
        args.probe_videos,
        args.default_duration_sec,
    )
    if not videos:
        raise SystemExit("No FishCam03 videos found.")

    print(f"Reading localizations from {args.localizations}", file=sys.stderr)
    localization_samples = read_localization_samples(args.localizations, videos)
    if not localization_samples:
        raise SystemExit("No localizations matched FishCam03 video time ranges.")

    positives = choose_spaced(localization_samples, positive_target, args.min_gap_sec, rng)
    randoms = make_random_samples(
        videos,
        localization_samples,
        positives,
        random_target,
        args.min_gap_sec,
        args.negative_exclusion_sec,
        rng,
    )
    samples = positives + randoms
    rng.shuffle(samples)

    manifest_rows: list[dict[str, object]] = []
    extract_start = time.monotonic()
    for index, sample in enumerate(samples, start=1):
        image_name = f"{index:05d}_{sample.label}_{safe_timestamp(sample.timestamp_utc)}.jpg"
        image_path = args.output / "images" / sample.label / image_name
        if not args.dry_run:
            extract_frame(sample, image_path)
            if index == 1 or index == len(samples) or index % max(args.progress_every, 1) == 0:
                elapsed = time.monotonic() - extract_start
                per_frame = elapsed / index
                remaining = per_frame * (len(samples) - index)
                percent = 100.0 * index / len(samples)
                print(
                    f"Extracted {index}/{len(samples)} frames "
                    f"({percent:5.1f}%) | elapsed {elapsed/60:.1f} min | "
                    f"ETA {remaining/60:.1f} min",
                    file=sys.stderr,
                )
        manifest_rows.append(sample_to_row(sample, image_path, index))

    write_csv(args.output / "manifest.csv", manifest_rows)
    write_csv(
        args.output / "localizations_used.csv",
        [sample_to_row(sample, None, index) for index, sample in enumerate(positives, start=1)],
    )

    summary = {
        "root": str(args.root),
        "localizations": str(args.localizations),
        "output": str(args.output),
        "dry_run": args.dry_run,
        "seed": args.seed,
        "total_requested": args.total_frames,
        "positive_target": positive_target,
        "random_target": random_target,
        "total_written": len(samples),
        "localization_frames": len(positives),
        "random_frames": len(randoms),
        "matched_localization_candidates": len(localization_samples),
        "fishcam03_videos": len(videos),
        "min_gap_sec": args.min_gap_sec,
        "negative_exclusion_sec": args.negative_exclusion_sec,
        "default_duration_sec": args.default_duration_sec,
        "probe_videos": args.probe_videos,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)

    if len(positives) < positive_target:
        print(
            f"Warning: only {len(positives)} spaced localization samples were available "
            f"for target {positive_target}. Lower --min-gap-sec to include more.",
            file=sys.stderr,
        )
    if len(randoms) < random_target:
        print(
            f"Warning: only {len(randoms)} random samples were available for target {random_target}. "
            "Lower --min-gap-sec or --negative-exclusion-sec to include more.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
