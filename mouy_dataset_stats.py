#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<id>\d+)_FishCam(?P<cam>\d+)_(?P<ts>\d{8}T\d{6})\.(?P<frac>\d+)Z_.*\.mp4$"
)
AUDIO_RE = re.compile(
    r"^(?P<prefix>.*?)(AMAR173)\.(?P<channel>\d+)\.(?P<ts>\d{8}T\d{6})Z\.wav$"
)
SYNCED_AUDIO_RE = re.compile(r"^AMAR173\.(?P<channel>\d+)\.\d{8}T\d{6}Z\.wav$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stats and plots for 2019-09-15_HornbyIsland_AMAR_07-HI dataset."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--synced-pairs",
        type=Path,
        default=None,
        help="Folder containing per-video synced audio pairs (used to count videos with audio).",
    )
    parser.add_argument(
        "--top-down-min-id",
        type=int,
        default=2729,
        help="Video IDs >= this are top-down; below are side-view.",
    )
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def short_video_label(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_rows: list[dict] = []
    audio_rows: list[dict] = []

    for path in root.iterdir():
        if path.suffix.lower() == ".mp4":
            m = VIDEO_RE.match(path.name)
            if not m:
                continue
            vid = int(m.group("id"))
            cam = f"FishCam{m.group('cam')}"
            ts = m.group("ts")
            view = "top" if vid >= args.top_down_min_id else "side"
            video_rows.append(
                {
                    "filename": path.name,
                    "video_id": vid,
                    "camera": cam,
                    "timestamp": ts,
                    "view": view,
                }
            )
        elif path.suffix.lower() == ".wav":
            m = AUDIO_RE.match(path.name)
            if not m:
                continue
            channel = int(m.group("channel"))
            ts = m.group("ts")
            audio_rows.append(
                {
                    "filename": path.name,
                    "channel": channel,
                    "timestamp": ts,
                }
            )

    # Basic counts
    total_videos = len(video_rows)
    total_audio = len(audio_rows)

    # Video stats
    videos_by_camera = Counter(r["camera"] for r in video_rows)
    videos_by_view = Counter(r["view"] for r in video_rows)
    videos_by_day = Counter(r["timestamp"][:8] for r in video_rows)

    # Audio stats
    audio_by_channel = Counter(r["channel"] for r in audio_rows)
    audio_by_day = Counter(r["timestamp"][:8] for r in audio_rows)

    # Matching logic
    video_ts = {r["timestamp"] for r in video_rows}
    audio_ts = {r["timestamp"] for r in audio_rows}

    audio_video_match_ts = sorted(video_ts & audio_ts)

    # Video-video matches across cameras (same timestamp with >=2 cameras)
    videos_by_ts = defaultdict(set)
    for r in video_rows:
        videos_by_ts[r["timestamp"]].add(r["camera"])
    video_multi_cam_ts = [ts for ts, cams in videos_by_ts.items() if len(cams) >= 2]

    # Audio channel completeness per timestamp
    audio_channels_by_ts = defaultdict(set)
    for r in audio_rows:
        audio_channels_by_ts[r["timestamp"]].add(r["channel"])
    audio_full_channel_ts = [ts for ts, chans in audio_channels_by_ts.items() if len(chans) >= 6]

    # Write CSVs
    write_csv(video_rows, out_dir / "videos.csv")
    write_csv(audio_rows, out_dir / "audio.csv")

    match_rows = [
        {"metric": "total_videos", "value": total_videos},
        {"metric": "total_audio_recordings", "value": total_audio},
        {"metric": "unique_video_timestamps", "value": len(video_ts)},
        {"metric": "unique_audio_timestamps", "value": len(audio_ts)},
        {"metric": "audio_video_matching_timestamps", "value": len(audio_video_match_ts)},
        {"metric": "multi_camera_video_timestamps", "value": len(video_multi_cam_ts)},
        {"metric": "audio_timestamps_with_6_channels", "value": len(audio_full_channel_ts)},
    ]

    # Synced-pairs stats (videos that have audio matches)
    synced_video_names = set()
    synced_audio_counts: dict[str, int] = {}
    if args.synced_pairs is not None and args.synced_pairs.exists():
        for child in args.synced_pairs.iterdir():
            if child.is_dir():
                synced_video_names.add(child.name)
                channels = set()
                for item in child.iterdir():
                    if not item.is_file():
                        continue
                    if "_chunk" in item.name:
                        continue
                    m = SYNCED_AUDIO_RE.match(item.name)
                    if not m:
                        continue
                    channels.add(int(m.group("channel")))
                if channels:
                    synced_audio_counts[child.name] = len(channels)
            elif child.suffix.lower() == ".mp4":
                synced_video_names.add(child.stem)

    if synced_video_names:
        matched = sum(1 for r in video_rows if Path(r["filename"]).stem in synced_video_names)
        match_rows.append({"metric": "videos_with_synced_audio", "value": matched})
        match_rows.append({"metric": "videos_without_synced_audio", "value": total_videos - matched})
        if synced_audio_counts:
            counts = list(synced_audio_counts.values())
            match_rows.append({"metric": "avg_audio_channels_per_video", "value": float(sum(counts)) / len(counts)})
    write_csv(match_rows, out_dir / "summary_metrics.csv")

    # Daily counts CSV
    daily_rows = []
    all_days = sorted(set(videos_by_day) | set(audio_by_day))
    for day in all_days:
        daily_rows.append(
            {
                "day": day,
                "video_count": videos_by_day.get(day, 0),
                "audio_count": audio_by_day.get(day, 0),
            }
        )
    write_csv(daily_rows, out_dir / "daily_counts.csv")

    # Plots (optional if matplotlib available)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def bar_plot(counter: Counter, title: str, path: Path) -> None:
            labels = list(counter.keys())
            values = [counter[k] for k in labels]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, values, color="#4c72b0")
            ax.set_title(title)
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)

        bar_plot(videos_by_camera, "Videos by Camera", out_dir / "videos_by_camera.png")
        bar_plot(videos_by_view, "Videos by View (side vs top)", out_dir / "videos_by_view.png")
        bar_plot(audio_by_channel, "Audio by Channel", out_dir / "audio_by_channel.png")

        # Daily counts plot
        if daily_rows:
            days = [r["day"] for r in daily_rows]
            vcounts = [r["video_count"] for r in daily_rows]
            acounts = [r["audio_count"] for r in daily_rows]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(days, vcounts, marker="o", label="Videos")
            ax.plot(days, acounts, marker="o", label="Audio")
            ax.set_title("Daily Counts")
            ax.set_ylabel("Count")
            ax.set_xlabel("Day")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / "daily_counts.png", dpi=200)
            plt.close(fig)

        # Synced audio plot
        if synced_video_names:
            matched = sum(1 for r in video_rows if Path(r["filename"]).stem in synced_video_names)
            unmatched = total_videos - matched
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["with_audio", "without_audio"], [matched, unmatched], color=["#55a868", "#c44e52"])
            ax.set_title("Videos With Synced Audio")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(out_dir / "videos_with_audio.png", dpi=200)
            plt.close(fig)

        # Per-video audio channel count (ignore chunk files)
        if synced_video_names:
            # Include all videos, even those with zero audio channels.
            rows = []
            for r in sorted(video_rows, key=lambda x: x["filename"]):
                stem = Path(r["filename"]).stem
                count = synced_audio_counts.get(stem, 0)
                rows.append(
                    {
                        "video": stem,
                        "short_label": short_video_label(r["filename"]),
                        "audio_channel_count": count,
                    }
                )
            write_csv(rows, out_dir / "video_audio_channel_counts.csv")

            labels = [r["short_label"] for r in rows]
            values = [r["audio_channel_count"] for r in rows]
            fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))
            ax.bar(labels, values, color="#4c72b0")
            ax.set_title("Audio Channels per Video (Synced, No Chunks)")
            ax.set_ylabel("Audio Channel Count")
            ax.set_xlabel("Video")
            ax.tick_params(axis="x", labelrotation=90)
            fig.tight_layout()
            fig.savefig(out_dir / "video_audio_channel_counts.png", dpi=200)
            plt.close(fig)
    except Exception as exc:
        print(f"Skipping plots (matplotlib unavailable): {exc}")

    # Print summary
    print("Summary:")
    for row in match_rows:
        print(f"- {row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
