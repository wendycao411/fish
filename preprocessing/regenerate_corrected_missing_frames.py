#!/usr/bin/env python3
"""Regenerate only the missing frames implied by a corrected localization timestamp.

The script reads a localization CSV, shifts each timestamp earlier by ``offset_sec``,
computes the original and corrected frame windows, and writes only the corrected frames
that were not already covered by the original labeled window.

Window convention
-----------------
For a window size of ``N`` frames, the inclusive frame range is:

``start = center - floor(N / 2)``
``end = start + N - 1``

So for ``N = 10``, the window is ``center - 5`` through ``center + 4``.

Outputs
-------
``output-root/``
  ``<video_name>/``
    ``localization_<row_index>/``
      ``frame_000123.jpg``
      ``metadata.csv``
      ``frame_actions.csv``
      ``contact_sheet.jpg``  (optional debug output)
  ``corrected_missing_frames_metadata.csv``

Dry runs do not write any files.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXT = ".jpg"
DEFAULT_JPEG_QUALITY = 95
FRAME_NAME_RE = re.compile(r"^frame_(\d+)$")


@dataclass(frozen=True)
class FrameAction:
    frame_index: int
    action: str
    reason: str
    output_path: str


@dataclass(frozen=True)
class LocalizationSummary:
    row_index: int
    localization_key: str
    source_row_indices: str
    video_name: str
    video_path: str
    original_time_sec: float
    corrected_time_sec: float
    fps: float
    old_center_frame: int
    corrected_center_frame: int
    old_window_start: int
    old_window_end: int
    corrected_window_start: int
    corrected_window_end: int
    old_window_source: str
    extracted_frame_indices: str
    skipped_frame_indices: str
    extracted_count: int
    skipped_count: int
    skipped_negative_count: int
    skipped_covered_count: int
    skipped_overlap_count: int
    skipped_exists_count: int
    reason_skipped: str
    localization_dir: str
    metadata_csv: str
    frame_actions_csv: str
    contact_sheet_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate corrected frame windows for fish localizations and write only the "
            "frames missing from the original labeled window."
        )
    )
    parser.add_argument("--localizations-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--old-frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp-col", required=True)
    parser.add_argument("--video-col", required=True)
    parser.add_argument(
        "--localization-group-col",
        default="localization_ids",
        help="Column that identifies one scientist-labeled localization event. Default: localization_ids.",
    )
    parser.add_argument(
        "--event-offset-col",
        default="seconds_from_localization",
        help="Column giving the offset from the localization event timestamp. Default: seconds_from_localization.",
    )
    parser.add_argument(
        "--video-contains",
        default=None,
        help="Only process videos whose resolved path or stem contains this substring.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--offset-sec", type=float, default=4.0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--debug-contact-sheet",
        action="store_true",
        help="Save a side-by-side contact sheet for each localization.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Alias for --debug-contact-sheet.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for saved frames. Default: 95.",
    )
    return parser.parse_args()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def parse_float(value: object, *, field_name: str, row_index: int) -> float:
    try:
        if pd.isna(value):
            raise ValueError
        return float(value)
    except Exception as exc:
        raise ValueError(f"Row {row_index}: could not parse {field_name!r} as float: {value!r}") from exc


def parse_timestamp(value: object, *, field_name: str, row_index: int) -> float:
    try:
        if pd.isna(value):
            raise ValueError
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            raise ValueError
        return float(parsed.timestamp())
    except Exception as exc:
        raise ValueError(f"Row {row_index}: could not parse {field_name!r} as a timestamp: {value!r}") from exc


def window_bounds(center_frame: int, window_size: int) -> tuple[int, int]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    start = center_frame - (window_size // 2)
    end = start + window_size - 1
    return start, end


def window_frames(center_frame: int, window_size: int) -> list[int]:
    start, end = window_bounds(center_frame, window_size)
    return list(range(start, end + 1))


def load_rows(
    csv_path: Path,
    timestamp_col: str,
    video_col: str,
    localization_group_col: str,
    event_offset_col: str,
) -> list[dict[str, object]]:
    df = pd.read_csv(csv_path)
    missing = [col for col in (timestamp_col, video_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {missing}")

    rows: list[dict[str, object]] = []
    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        localization_key = str(row[localization_group_col]) if localization_group_col in df.columns else f"row_{row_index:06d}"
        event_offset = parse_float(row[event_offset_col], field_name=event_offset_col, row_index=row_index) if event_offset_col in df.columns else 0.0
        timestamp_sec = parse_timestamp(row[timestamp_col], field_name=timestamp_col, row_index=row_index)
        video_offset_sec = parse_float(row["video_offset_sec"], field_name="video_offset_sec", row_index=row_index) if "video_offset_sec" in df.columns else timestamp_sec
        rows.append(
            {
                "row_index": row_index,
                "localization_key": localization_key,
                "timestamp_sec": timestamp_sec,
                "video_offset_sec": video_offset_sec,
                "event_offset_sec": event_offset,
                "event_time_sec": video_offset_sec - event_offset,
                "video_value": str(row[video_col]),
                "raw_row": row.to_dict(),
            }
        )
    return rows


def resolve_video_path(video_value: str, video_root: Path) -> Path:
    candidate = Path(video_value)
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        direct = video_root / candidate
        if direct.exists():
            return direct.resolve()

    search_names: list[str] = []
    basename = candidate.name
    stem = candidate.stem
    if basename:
        search_names.append(basename)
    if stem and stem != basename:
        search_names.append(stem)
    if stem:
        search_names.append(f"{stem}.mp4")
    if basename and not basename.endswith(".mp4"):
        search_names.append(f"{basename}.mp4")

    seen: set[Path] = set()
    for name in search_names:
        for match in sorted(video_root.rglob(name)):
            resolved = match.resolve()
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                return resolved

    raise FileNotFoundError(f"Could not resolve video {video_value!r} under {video_root}")


def get_video_fps(video_path: Path, fps_override: float | None) -> float:
    if fps_override is not None:
        if fps_override <= 0:
            raise ValueError("--fps must be positive")
        return float(fps_override)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for FPS lookup: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise RuntimeError(f"Could not read FPS from video: {video_path}")
        return fps
    finally:
        cap.release()


def collect_frame_indices(directory: Path) -> set[int]:
    frame_indices: set[int] = set()
    if not directory.exists():
        return frame_indices

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        match = FRAME_NAME_RE.match(path.stem)
        if match:
            frame_indices.add(int(match.group(1)))
    return frame_indices


def find_old_localization_dir(old_frame_root: Path, video_name: str, row_index: int) -> Path | None:
    row_names = [f"localization_{row_index:06d}", f"localization_{row_index}", f"{row_index:06d}", f"{row_index}"]
    video_candidates = [old_frame_root / video_name, old_frame_root / safe_component(video_name), old_frame_root]

    for video_root in video_candidates:
        for row_name in row_names:
            candidate = video_root / row_name
            if candidate.is_dir():
                return candidate

    return None


def read_frame_at(cap: cv2.VideoCapture, frame_index: int, frame_count: int) -> object | None:
    if frame_index < 0 or frame_index >= frame_count:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def save_frame(frame: object, out_path: Path, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not success:
        raise RuntimeError(f"Failed to write {out_path}")


def make_panel(frame: object | None, label_lines: list[str], title: str, max_side: int = 720) -> np.ndarray:
    if frame is None:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        canvas = frame.copy()

    h, w = canvas.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        canvas = cv2.resize(canvas, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    h, w = canvas.shape[:2]
    header_h = 32 + 20 * len(label_lines)
    panel = np.zeros((header_h + h + 12, w + 24, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    panel[header_h : header_h + h, 12 : 12 + w] = canvas

    y = 24
    cv2.putText(panel, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    y += 20
    for line in label_lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1)
        y += 20
    return panel


def build_contact_sheet(
    old_frame: object | None,
    corrected_frame: object | None,
    old_lines: list[str],
    corrected_lines: list[str],
    out_path: Path,
) -> None:
    left = make_panel(old_frame, old_lines, "Old center frame")
    right = make_panel(corrected_frame, corrected_lines, "Corrected center frame")

    height = max(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    if right.shape[0] != height:
        right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    sheet = np.concatenate([left, right], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_video_group(
    video_name: str,
    video_path: Path,
    localization_groups: list[list[dict[str, object]]],
    args: argparse.Namespace,
) -> tuple[list[LocalizationSummary], Counter[str]]:
    counters: Counter[str] = Counter()
    summaries: list[LocalizationSummary] = []
    occupied_corrected_frames: set[int] = set()
    video_dir = args.output_root / video_name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] Could not open video: {video_path}")
        counters["videos_unopenable"] += 1
        fps_value = float(args.fps) if args.fps is not None else float("nan")

        for group in localization_groups:
            representative = group[0]
            row_index = int(representative["row_index"])
            localization_key = str(representative["localization_key"])
            source_row_indices = ";".join(str(int(row["row_index"])) for row in group)
            original_time_sec = float(representative["event_time_sec"])
            corrected_time_sec = original_time_sec - float(args.offset_sec)
            old_center_frame = int(round(original_time_sec * (fps_value if np.isfinite(fps_value) else 1.0)))
            corrected_center_frame = int(round(corrected_time_sec * (fps_value if np.isfinite(fps_value) else 1.0)))
            old_window_start, old_window_end = window_bounds(old_center_frame, args.window_size)
            corrected_window_start, corrected_window_end = window_bounds(corrected_center_frame, args.window_size)
            summaries.append(
                LocalizationSummary(
                    row_index=row_index,
                    localization_key=localization_key,
                    source_row_indices=source_row_indices,
                    video_name=video_name,
                    video_path=str(video_path),
                    original_time_sec=original_time_sec,
                    corrected_time_sec=corrected_time_sec,
                    fps=fps_value,
                    old_center_frame=old_center_frame,
                    corrected_center_frame=corrected_center_frame,
                    old_window_start=old_window_start,
                    old_window_end=old_window_end,
                    corrected_window_start=corrected_window_start,
                    corrected_window_end=corrected_window_end,
                    old_window_source="unavailable_video",
                    extracted_frame_indices="",
                    skipped_frame_indices="",
                    extracted_count=0,
                    skipped_count=0,
                    skipped_negative_count=0,
                    skipped_covered_count=0,
                    skipped_overlap_count=0,
                    skipped_exists_count=0,
                    reason_skipped="video_unopenable",
                    localization_dir=str(video_dir),
                    metadata_csv=str(video_dir / "metadata.csv"),
                    frame_actions_csv=str(video_dir / "frame_actions.csv"),
                    contact_sheet_path=str(video_dir / "contact_sheet.jpg"),
                )
            )
        return summaries, counters

    try:
        fps = get_video_fps(video_path, args.fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[video] {video_name}: fps={fps:.3f}, frames={frame_count}")

        video_frame_actions: list[dict[str, object]] = []

        for group in localization_groups:
            representative = group[0]
            row_index = int(representative["row_index"])
            localization_key = str(representative["localization_key"])
            source_row_indices = ";".join(str(int(row["row_index"])) for row in group)
            original_time_sec = float(representative["event_time_sec"])
            corrected_time_sec = original_time_sec - float(args.offset_sec)
            old_center_frame = int(round(original_time_sec * fps))
            corrected_center_frame = int(round(corrected_time_sec * fps))
            old_window_start, old_window_end = window_bounds(old_center_frame, args.window_size)
            corrected_window_start, corrected_window_end = window_bounds(corrected_center_frame, args.window_size)

            old_window_frames = set(window_frames(old_center_frame, args.window_size))
            corrected_window = window_frames(corrected_center_frame, args.window_size)
            old_window_source = "computed_from_event_time"

            output_frame_indices: list[int] = []
            skipped_frame_indices: list[int] = []
            frame_actions: list[FrameAction] = []

            if corrected_time_sec < 0:
                counters["rows_with_negative_corrected_time"] += 1

            for frame_index in corrected_window:
                out_path = video_dir / f"frame_{frame_index:06d}{IMAGE_EXT}"
                if frame_index < 0:
                    counters["frames_skipped_negative"] += 1
                    skipped_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(frame_index=frame_index, action="skipped", reason="negative_frame_index", output_path=str(out_path))
                    )
                    continue

                if frame_index in old_window_frames:
                    counters["frames_skipped_covered"] += 1
                    skipped_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(
                            frame_index=frame_index,
                            action="skipped",
                            reason="inside_old_labeled_window",
                            output_path=str(out_path),
                        )
                    )
                    continue

                if frame_index in occupied_corrected_frames:
                    counters["frames_skipped_covered"] += 1
                    skipped_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(
                            frame_index=frame_index,
                            action="skipped",
                            reason="overlaps_previous_localization_window",
                            output_path=str(out_path),
                        )
                    )
                    continue

                if out_path.exists() and not args.overwrite:
                    counters["frames_skipped_existing"] += 1
                    skipped_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(frame_index=frame_index, action="skipped", reason="already_exists", output_path=str(out_path))
                    )
                    continue

                if args.dry_run:
                    counters["frames_would_extract"] += 1
                    output_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(frame_index=frame_index, action="would_extract", reason="dry_run", output_path=str(out_path))
                    )
                    continue

                frame = read_frame_at(cap, frame_index, frame_count)
                if frame is None:
                    counters["frames_failed_read"] += 1
                    skipped_frame_indices.append(frame_index)
                    frame_actions.append(
                        FrameAction(frame_index=frame_index, action="skipped", reason="video_read_failed", output_path=str(out_path))
                    )
                    continue

                save_frame(frame, out_path, args.jpeg_quality)
                counters["frames_extracted"] += 1
                occupied_corrected_frames.add(frame_index)
                output_frame_indices.append(frame_index)
                frame_actions.append(
                    FrameAction(frame_index=frame_index, action="extracted", reason="missing_from_old_window", output_path=str(out_path))
                )

            summary_row = {
                "row_index": row_index,
                "localization_key": localization_key,
                "source_row_indices": source_row_indices,
                "video_name": video_name,
                "video_path": str(video_path),
                "original_time_sec": float(f"{original_time_sec:.6f}"),
                "corrected_time_sec": float(f"{corrected_time_sec:.6f}"),
                "fps": float(f"{fps:.6f}"),
                "old_center_frame": old_center_frame,
                "corrected_center_frame": corrected_center_frame,
                "old_window_start": old_window_start,
                "old_window_end": old_window_end,
                "corrected_window_start": corrected_window_start,
                "corrected_window_end": corrected_window_end,
                "old_window_source": old_window_source,
                "extracted_frame_indices": ";".join(str(idx) for idx in output_frame_indices),
                "skipped_frame_indices": ";".join(str(idx) for idx in skipped_frame_indices),
                "extracted_count": len(output_frame_indices),
                "skipped_count": len(skipped_frame_indices),
                "skipped_negative_count": sum(1 for action in frame_actions if action.reason == "negative_frame_index"),
                "skipped_covered_count": sum(1 for action in frame_actions if action.reason == "inside_old_labeled_window"),
                "skipped_overlap_count": sum(1 for action in frame_actions if action.reason == "overlaps_previous_localization_window"),
                "skipped_exists_count": sum(1 for action in frame_actions if action.reason == "already_exists"),
                "reason_skipped": (
                    "all_corrected_frames_skipped"
                    if len(output_frame_indices) == 0 and len(skipped_frame_indices) > 0 and not args.dry_run
                    else ("dry_run" if args.dry_run else "")
                ),
                "localization_dir": str(video_dir),
                "metadata_csv": str(video_dir / "metadata.csv"),
                "frame_actions_csv": str(video_dir / "frame_actions.csv"),
                "contact_sheet_path": str(video_dir / f"contact_sheet_{localization_key}.jpg"),
            }

            summaries.append(
                LocalizationSummary(
                    row_index=summary_row["row_index"],
                    localization_key=summary_row["localization_key"],
                    source_row_indices=summary_row["source_row_indices"],
                    video_name=summary_row["video_name"],
                    video_path=summary_row["video_path"],
                    original_time_sec=summary_row["original_time_sec"],
                    corrected_time_sec=summary_row["corrected_time_sec"],
                    fps=summary_row["fps"],
                    old_center_frame=summary_row["old_center_frame"],
                    corrected_center_frame=summary_row["corrected_center_frame"],
                    old_window_start=summary_row["old_window_start"],
                    old_window_end=summary_row["old_window_end"],
                    corrected_window_start=summary_row["corrected_window_start"],
                    corrected_window_end=summary_row["corrected_window_end"],
                    old_window_source=summary_row["old_window_source"],
                    extracted_frame_indices=summary_row["extracted_frame_indices"],
                    skipped_frame_indices=summary_row["skipped_frame_indices"],
                    extracted_count=summary_row["extracted_count"],
                    skipped_count=summary_row["skipped_count"],
                    skipped_negative_count=summary_row["skipped_negative_count"],
                    skipped_covered_count=summary_row["skipped_covered_count"],
                    skipped_overlap_count=summary_row["skipped_overlap_count"],
                    skipped_exists_count=summary_row["skipped_exists_count"],
                    reason_skipped=summary_row["reason_skipped"],
                    localization_dir=summary_row["localization_dir"],
                    metadata_csv=summary_row["metadata_csv"],
                    frame_actions_csv=summary_row["frame_actions_csv"],
                    contact_sheet_path=summary_row["contact_sheet_path"],
                )
            )

            video_frame_actions.extend(
                {
                    "row_index": row_index,
                    "localization_key": localization_key,
                    "frame_index": action.frame_index,
                    "action": action.action,
                    "reason": action.reason,
                    "output_path": action.output_path,
                }
                for action in frame_actions
            )

            print(
                f"[loc {localization_key}] extracted={len(output_frame_indices)} skipped={len(skipped_frame_indices)} "
                f"old={old_window_start}..{old_window_end} corrected={corrected_window_start}..{corrected_window_end}"
            )

        if not args.dry_run:
            video_dir.mkdir(parents=True, exist_ok=True)
            write_csv(video_dir / "metadata.csv", [summary.__dict__ for summary in summaries], list(LocalizationSummary.__annotations__.keys()))
            write_csv(
                video_dir / "frame_actions.csv",
                video_frame_actions,
                ["row_index", "localization_key", "frame_index", "action", "reason", "output_path"],
            )

            if args.debug_contact_sheet or args.save_overlays:
                first_group = localization_groups[0]
                representative = first_group[0]
                sample_old_frame = read_frame_at(cap, int(round(float(representative["event_time_sec"]) * fps)), frame_count)
                sample_corrected_frame = read_frame_at(cap, int(round((float(representative["event_time_sec"]) - float(args.offset_sec)) * fps)), frame_count)
                old_lines = [
                    f"video={video_name}",
                    f"localization_key={representative['localization_key']}",
                    f"timestamp_sec={float(representative['event_time_sec']):.6f}",
                ]
                corrected_lines = [
                    f"corrected_time_sec={float(representative['event_time_sec']) - float(args.offset_sec):.6f}",
                    f"offset_sec={args.offset_sec:.3f}",
                ]
                build_contact_sheet(
                    sample_old_frame,
                    sample_corrected_frame,
                    old_lines,
                    corrected_lines,
                    video_dir / "contact_sheet.jpg",
                )
    finally:
        cap.release()

    return summaries, counters


def main() -> int:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")

    rows = load_rows(
        args.localizations_csv,
        args.timestamp_col,
        args.video_col,
        args.localization_group_col,
        args.event_offset_col,
    )
    print(f"Read {len(rows)} localization rows from {args.localizations_csv}")

    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    resolved_videos: dict[str, Path] = {}
    resolved_by_value: dict[str, Path] = {}
    for row in rows:
        video_value = str(row["video_value"])
        if video_value in resolved_by_value:
            video_path = resolved_by_value[video_value]
        else:
            video_path = resolve_video_path(video_value, args.video_root)
            resolved_by_value[video_value] = video_path
        video_name = safe_component(video_path.stem)
        if args.video_contains and args.video_contains not in video_name and args.video_contains not in str(video_path):
            continue
        resolved_videos[video_name] = video_path
        grouped[video_name][str(row["localization_key"])].append(row)

    print(f"Resolved {len(resolved_videos)} videos under {args.video_root}")

    all_summaries: list[LocalizationSummary] = []
    total_counts: Counter[str] = Counter()
    for video_name, localization_groups_map in sorted(grouped.items(), key=lambda item: item[0]):
        localization_groups = sorted(
            localization_groups_map.values(),
            key=lambda group: float(group[0]["event_time_sec"]),
        )
        summaries, counters = process_video_group(video_name, resolved_videos[video_name], localization_groups, args)
        all_summaries.extend(summaries)
        total_counts.update(counters)

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        combined_fieldnames = list(LocalizationSummary.__annotations__.keys())
        combined_rows = [summary.__dict__ for summary in all_summaries]
        write_csv(args.output_root / "corrected_missing_frames_metadata.csv", combined_rows, combined_fieldnames)

    print(f"Videos processed: {len(resolved_videos)}")
    print(f"Rows with corrected_time_sec < 0: {total_counts.get('rows_with_negative_corrected_time', 0)}")
    if args.dry_run:
        print(f"Frames that would be extracted: {total_counts.get('frames_would_extract', 0)}")
    else:
        print(f"Frames extracted: {total_counts.get('frames_extracted', 0)}")
    print(f"Frames skipped inside old window: {total_counts.get('frames_skipped_covered', 0)}")
    print(f"Frames skipped because they already existed: {total_counts.get('frames_skipped_existing', 0)}")
    print(f"Frames skipped because frame index was negative: {total_counts.get('frames_skipped_negative', 0)}")
    print(f"Videos that could not be opened: {total_counts.get('videos_unopenable', 0)}")
    if args.dry_run:
        print("Dry run complete: no files were written.")
    else:
        print(f"Wrote combined metadata to {args.output_root / 'corrected_missing_frames_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())