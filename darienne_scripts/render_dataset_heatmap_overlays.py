#!/usr/bin/env python
"""Render heatmap overlay videos for videos used in Darienne localization datasets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path("/Users/wendycao/fish")
DEFAULT_HOMOGRAPHY = REPO_ROOT / "XAV-arrays/localization/darienne_affine_H.npy"
DEFAULT_HEATMAP_SCRIPT = REPO_ROOT / "XAV-arrays/localization/heatmap_overlay_errors.py"

DATASET_DEFAULTS = {
    "taylor": REPO_ROOT / "darienne_localization_windows",
    "danger": REPO_ROOT / "darienne_danger_rocks_localization_windows",
}


def load_overlay_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("heatmap_overlay_errors", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load overlay module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each source video represented in a dataset windows.csv, render a full-video "
            "localization heatmap overlay."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["taylor", "danger", "both"],
        default="both",
        help="Dataset to render.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Custom dataset root containing windows.csv. Use only with --dataset taylor or danger.",
    )
    parser.add_argument("--homography-npy", type=Path, default=DEFAULT_HOMOGRAPHY)
    parser.add_argument("--heatmap-script", type=Path, default=DEFAULT_HEATMAP_SCRIPT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "darienne_heatmap_overlays")
    parser.add_argument("--mode", choices=["trail", "cumulative"], default="trail")
    parser.add_argument(
        "--trail-sec",
        type=float,
        default=10.0,
        help="Trailing heatmap duration in seconds. Use 10 to match localization-window context.",
    )
    parser.add_argument("--sigma-px", type=float, default=24.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--time-offset-sec",
        type=float,
        default=-5.0,
        help="Shift localization times on video timeline (negative = earlier, positive = later). Default is -5.0s.",
    )
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit per video.")
    parser.add_argument(
        "--frame-progress-every",
        type=int,
        default=100,
        help="Print progress every N frames while rendering each video. Use 0 to disable.",
    )
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-debug-frame", action="store_true")
    return parser.parse_args()


def safe_stem(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def iter_dataset_roots(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.dataset_root is not None:
        if args.dataset == "both":
            raise SystemExit("--dataset-root can only be used with --dataset taylor or --dataset danger")
        return [(args.dataset, args.dataset_root)]
    if args.dataset == "both":
        return [("taylor", DATASET_DEFAULTS["taylor"]), ("danger", DATASET_DEFAULTS["danger"])]
    return [(args.dataset, DATASET_DEFAULTS[args.dataset])]


def read_windows(windows_csv: Path) -> list[dict[str, str]]:
    with windows_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_video(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        video_path = row.get("video_path", "")
        if not video_path:
            continue
        grouped.setdefault(video_path, []).append(row)
    return grouped


def write_video_localizations(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uuid",
        "time_min_offset",
        "time_max_offset",
        "x",
        "y",
        "z",
        "source_selection_file",
        "localization_timestamp_utc",
        "window_index",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r.get("video_offset_center_sec", "0") or 0)):
            t0 = float(row["video_offset_center_sec"])
            loc_id = row.get("localization_ids", "")
            source_selection_file = loc_id.split(":", 1)[0] if loc_id else ""
            writer.writerow(
                {
                    "uuid": loc_id,
                    "time_min_offset": f"{t0:.6f}",
                    "time_max_offset": f"{t0 + 0.1:.6f}",
                    "x": row.get("x_m", ""),
                    "y": row.get("y_m", ""),
                    "z": row.get("z_m", ""),
                    "source_selection_file": source_selection_file,
                    "localization_timestamp_utc": row.get("localization_timestamp_utc", ""),
                    "window_index": row.get("window_index", ""),
                }
            )


def render_dataset(
    label: str,
    dataset_root: Path,
    args: argparse.Namespace,
    overlay_module,
    homography,
) -> dict[str, object]:
    windows_csv = dataset_root / "windows.csv"
    if not windows_csv.exists():
        raise FileNotFoundError(f"Missing windows.csv: {windows_csv}")

    rows = read_windows(windows_csv)
    grouped = group_by_video(rows)
    videos = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    out_dir = args.output_root / label
    csv_dir = out_dir / "per_video_localizations"
    video_dir = out_dir / "overlay_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    started_at = time.monotonic()

    for index, (video_path_str, video_rows) in enumerate(videos, start=1):
        video_path = Path(video_path_str)
        stem = safe_stem(video_path)
        loc_csv = csv_dir / f"{stem}_localizations.csv"
        out_video = video_dir / f"{stem}_heatmap_overlay.mp4"

        write_video_localizations(video_rows, loc_csv)

        status = "ok"
        if out_video.exists() and not args.overwrite:
            status = "skipped_exists"
        else:
            try:
                overlay_module.make_time_varying_heat_overlay_video_errors(
                    video_path=video_path,
                    localizations_csv=loc_csv,
                    H=homography,
                    out_video_path=out_video,
                    mode=args.mode,
                    window_sec=args.trail_sec,
                    sigma_px=args.sigma_px,
                    alpha=args.alpha,
                    decay_sec=0.0,
                    hard_cut=True,
                    time_offset_sec=args.time_offset_sec,
                    max_frames=None if args.max_frames <= 0 else args.max_frames,
                    debug=not args.no_debug_frame,
                    progress_every_frames=args.frame_progress_every,
                    progress_prefix=f"[{label} video {index}/{len(videos)}]",
                )
                if args.with_audio:
                    overlay_module.mux_audio_from_source_video(out_video, video_path)
            except Exception as exc:
                status = f"failed: {exc}"

        elapsed = time.monotonic() - started_at
        per_video = elapsed / index
        remaining = per_video * (len(videos) - index)
        percent = 100.0 * index / max(len(videos), 1)
        print(
            f"[{label}] {progress_bar(index, len(videos))} "
            f"{index}/{len(videos)} ({percent:5.1f}%) {status} | "
            f"events={len(video_rows)} | elapsed={elapsed/60:.1f} min | ETA={remaining/60:.1f} min | "
            f"{video_path.name}",
            file=sys.stderr,
        )

        summary_rows.append(
            {
                "dataset": label,
                "video_index": index,
                "localization_count": len(video_rows),
                "video_path": str(video_path),
                "localizations_csv": str(loc_csv),
                "overlay_video": str(out_video),
                "time_offset_sec": args.time_offset_sec,
                "status": status,
            }
        )

    summary_csv = out_dir / "overlay_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "dataset": label,
        "dataset_root": str(dataset_root),
        "videos": len(videos),
        "summary_csv": str(summary_csv),
        "overlay_dir": str(video_dir),
    }


def main() -> int:
    args = parse_args()
    overlay_module = load_overlay_module(args.heatmap_script)
    homography = overlay_module.load_homography(args.homography_npy)

    summaries = []
    for label, dataset_root in iter_dataset_roots(args):
        summaries.append(render_dataset(label, dataset_root, args, overlay_module, homography))

    print("Completed overlay jobs:")
    for summary in summaries:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
