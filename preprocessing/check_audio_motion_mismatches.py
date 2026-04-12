#!/usr/bin/env python3
"""
Check whether audio localizations that correspond to the heatmap dataset frames
line up with YOLO labels in that same dataset.

This version is stdlib-only so it can run in minimal environments. It uses:
- dataset_root/images/{train,val}/*.jpg
- dataset_root/labels/{train,val}/*.txt
- dataset_root/heatmaps/{train,val}/*_heatmap.png (presence check only)
- per-clip localization CSVs to recover the original sound localizations

It writes:
- audio_motion_unmatched.csv
- audio_motion_off_frame.csv
- audio_motion_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SIZE_RE = re.compile(r"_(\d+)x(\d+)_")


@dataclass(frozen=True)
class Box:
    class_id: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float


@dataclass(frozen=True)
class DatasetFrame:
    split: str
    stem: str
    clip_name: str
    frame_idx: int
    width: int
    height: int
    image_path: str
    label_path: str
    heatmap_path: str
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class LocalizationRecord:
    clip_name: str
    source_csv: str
    row_number: int
    frame_idx: int
    time_seconds: float
    time_min_offset: float
    time_max_offset: float
    confidence: float
    x: float
    y: float
    x_err_span: float
    y_err_span: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heatmap-dataset audio localizations against YOLO labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/vision/beery/scratch/wendy/fish/processed/extracted_frames_200_heatmap_yolo_separate"
        ),
        help="Dataset root containing images/, labels/, and heatmaps/.",
    )
    parser.add_argument(
        "--localizations-root",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/XAV-arrays/localization/out_synced_pairs"),
        help="Root containing per-clip localization CSVs such as */localizations_merged_filtered.csv.",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/XAV-arrays/localization/top_down_H.npy"),
        help="3x3 homography .npy used to project x/y world coordinates into image pixels.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS. If omitted, try parsing it from the clip name token like '_fr-10_'.",
    )
    parser.add_argument(
        "--pixel-tolerance",
        type=float,
        default=150.0,
        help="Max allowed distance in pixels from projected sound point to nearest YOLO box.",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help="Allow matching a localization to dataset frames within +/- this many frame indices.",
    )
    parser.add_argument(
        "--coord-cols",
        nargs=2,
        default=("x", "y"),
        metavar=("X_COL", "Y_COL"),
        help="Localization CSV columns to project into image space.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "val"),
        help="Dataset splits to inspect. Default: train val",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output CSV reports.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Write one visualization image for each unmatched_in_frame localization.",
    )
    parser.add_argument(
        "--sound-box-size",
        type=int,
        default=100,
        help="Side length in pixels for the sound-localization box in the visualization.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_frame_idx_from_stem(stem: str) -> int:
    frame_part = stem.split("__", 1)[1]
    return int(frame_part.split("_", 1)[0])


def parse_clip_name_from_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


def parse_size_from_name(name: str) -> tuple[int, int]:
    match = SIZE_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse WIDTHxHEIGHT from filename: {name}")
    return int(match.group(1)), int(match.group(2))


def parse_fps_from_clip_name(clip_name: str) -> float | None:
    marker = "_fr-"
    idx = clip_name.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    end = clip_name.find("_", start)
    token = clip_name[start:] if end < 0 else clip_name[start:end]
    try:
        return float(token)
    except ValueError:
        return None


def yolo_boxes_from_label_file(label_path: Path, width: int, height: int) -> tuple[Box, ...]:
    if not label_path.exists():
        return tuple()

    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}: {line}")
        class_id = int(parts[0])
        cx = float(parts[1]) * width
        cy = float(parts[2]) * height
        bw = float(parts[3]) * width
        bh = float(parts[4]) * height
        boxes.append(
            Box(
                class_id=class_id,
                xtl=cx - bw / 2.0,
                ytl=cy - bh / 2.0,
                xbr=cx + bw / 2.0,
                ybr=cy + bh / 2.0,
            )
        )
    return tuple(boxes)


def load_dataset_frames(dataset_root: Path, splits: tuple[str, ...]) -> list[DatasetFrame]:
    frames: list[DatasetFrame] = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        heatmap_dir = dataset_root / "heatmaps" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = image_path.stem
            clip_name = parse_clip_name_from_stem(stem)
            frame_idx = parse_frame_idx_from_stem(stem)
            width, height = parse_size_from_name(image_path.name)
            label_path = label_dir / f"{stem}.txt"
            heatmap_path = heatmap_dir / f"{stem}_heatmap.png"
            boxes = yolo_boxes_from_label_file(label_path, width, height)
            frames.append(
                DatasetFrame(
                    split=split,
                    stem=stem,
                    clip_name=clip_name,
                    frame_idx=frame_idx,
                    width=width,
                    height=height,
                    image_path=str(image_path),
                    label_path=str(label_path),
                    heatmap_path=str(heatmap_path),
                    boxes=boxes,
                )
            )
    if not frames:
        raise FileNotFoundError(f"No dataset images found under {dataset_root}")
    return frames


def read_npy_matrix(path: Path) -> list[list[float]]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not an .npy file")
        major = f.read(1)[0]
        minor = f.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}.{minor} in {path}")
        header = f.read(header_len).decode("latin1")
        meta = ast.literal_eval(header)
        descr = meta["descr"]
        shape = meta["shape"]
        fortran_order = meta["fortran_order"]
        if fortran_order:
            raise ValueError(f"Fortran-order arrays are not supported: {path}")
        if len(shape) != 2:
            raise ValueError(f"Expected 2D matrix in {path}, found shape {shape}")
        rows, cols = int(shape[0]), int(shape[1])
        dtype_sizes = {"<f4": 4, "<f8": 8, "|f4": 4, "|f8": 8}
        if descr not in dtype_sizes:
            raise ValueError(f"Unsupported dtype {descr} in {path}")
        item_size = dtype_sizes[descr]
        raw = f.read(rows * cols * item_size)
        fmt = "<" + ("f" if item_size == 4 else "d") * (rows * cols)
        values = struct.unpack(fmt, raw)
        matrix: list[list[float]] = []
        for row_idx in range(rows):
            start = row_idx * cols
            matrix.append(list(values[start : start + cols]))
        return matrix


def world_to_pixel(x: float, y: float, homography: list[list[float]]) -> tuple[float, float]:
    h00, h01, h02 = homography[0]
    h10, h11, h12 = homography[1]
    h20, h21, h22 = homography[2]
    px = h00 * x + h01 * y + h02
    py = h10 * x + h11 * y + h12
    pw = h20 * x + h21 * y + h22
    if not math.isfinite(pw) or abs(pw) < 1e-12:
        return float("nan"), float("nan")
    return px / pw, py / pw


def load_localizations(
    localizations_root: Path,
    coord_cols: tuple[str, str],
    fps_by_clip: dict[str, float],
) -> list[LocalizationRecord]:
    c1_name, c2_name = coord_cols
    localizations: list[LocalizationRecord] = []
    csv_paths = sorted(localizations_root.glob("*/localizations_merged_filtered.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No localization CSVs found under {localizations_root}")

    for csv_path in csv_paths:
        clip_name = csv_path.parent.name
        fps = fps_by_clip.get(clip_name)
        if fps is None:
            continue
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = {"time_min_offset", c1_name, c2_name}
            missing = sorted(required - fieldnames)
            if missing:
                raise KeyError(f"{csv_path} is missing required columns: {missing}")
            for row_number, row in enumerate(reader, start=2):
                t0 = parse_float(row.get("time_min_offset"))
                t1 = parse_float(row.get("time_max_offset"))
                if not math.isfinite(t0):
                    continue
                if not math.isfinite(t1):
                    t1 = t0
                if t1 < t0:
                    t0, t1 = t1, t0
                x = parse_float(row.get(c1_name))
                y = parse_float(row.get(c2_name))
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                t_mid = (t0 + t1) / 2.0
                localizations.append(
                    LocalizationRecord(
                        clip_name=clip_name,
                        source_csv=str(csv_path),
                        row_number=row_number,
                        frame_idx=int(round(t_mid * fps)),
                        time_seconds=t_mid,
                        time_min_offset=t0,
                        time_max_offset=t1,
                        confidence=parse_float(row.get("confidence"))
                        if math.isfinite(parse_float(row.get("confidence")))
                        else 1.0,
                        x=x,
                        y=y,
                        x_err_span=parse_float(row.get(f"{c1_name}_err_span")),
                        y_err_span=parse_float(row.get(f"{c2_name}_err_span")),
                    )
                )
    return localizations


def box_distance_px(u: float, v: float, box: Box) -> float:
    dx = 0.0
    dy = 0.0
    if u < box.xtl:
        dx = box.xtl - u
    elif u > box.xbr:
        dx = u - box.xbr
    if v < box.ytl:
        dy = box.ytl - v
    elif v > box.ybr:
        dy = v - box.ybr
    return math.hypot(dx, dy)


def best_box_match(u: float, v: float, boxes: tuple[Box, ...]) -> tuple[Box | None, float]:
    best_box: Box | None = None
    best_distance = float("inf")
    for box in boxes:
        distance = box_distance_px(u, v, box)
        if distance < best_distance:
            best_box = box
            best_distance = distance
    return best_box, best_distance


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def save_unmatched_visualization(
    *,
    image_path: str,
    out_path: Path,
    sound_x: float,
    sound_y: float,
    sound_box_size: int,
    label_box: Box,
    distance_px: float,
) -> None:
    half = sound_box_size / 2.0
    sx1 = sound_x - half
    sy1 = sound_y - half
    sx2 = sound_x + half
    sy2 = sound_y + half

    draw_commands = [
        f"rectangle {sx1:.2f},{sy1:.2f} {sx2:.2f},{sy2:.2f}",
        f"circle {sound_x:.2f},{sound_y:.2f} {sound_x + 6.0:.2f},{sound_y:.2f}",
    ]
    label_draw = f"rectangle {label_box.xtl:.2f},{label_box.ytl:.2f} {label_box.xbr:.2f},{label_box.ybr:.2f}"
    label_text = f"dist={distance_px:.1f}px"

    cmd = [
        "convert",
        image_path,
        "-stroke",
        "#00FFFF",
        "-strokewidth",
        "4",
        "-fill",
        "none",
        "-draw",
        draw_commands[0],
        "-draw",
        draw_commands[1],
        "-stroke",
        "#FF2D2D",
        "-strokewidth",
        "4",
        "-fill",
        "none",
        "-draw",
        label_draw,
        "-fill",
        "#FFFFFF",
        "-undercolor",
        "#000000B0",
        "-pointsize",
        "26",
        "-annotate",
        "+20+40",
        label_text,
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    splits = tuple(args.splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "unmatched_visualizations"
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    homography = read_npy_matrix(args.homography)
    dataset_frames = load_dataset_frames(args.dataset_root, splits)

    frames_by_clip: dict[str, list[DatasetFrame]] = defaultdict(list)
    fps_by_clip: dict[str, float] = {}
    missing_heatmaps = 0
    for frame in dataset_frames:
        frames_by_clip[frame.clip_name].append(frame)
        if not Path(frame.heatmap_path).exists():
            missing_heatmaps += 1
        if args.fps is not None:
            fps_by_clip[frame.clip_name] = args.fps
        elif frame.clip_name not in fps_by_clip:
            parsed_fps = parse_fps_from_clip_name(frame.clip_name)
            if parsed_fps is not None:
                fps_by_clip[frame.clip_name] = parsed_fps

    unresolved_fps = sorted(clip for clip in frames_by_clip if clip not in fps_by_clip)
    if unresolved_fps:
        raise ValueError(
            "Could not infer FPS for clip(s); rerun with --fps. "
            f"Examples: {unresolved_fps[:5]}"
        )

    for clip_frames in frames_by_clip.values():
        clip_frames.sort(key=lambda f: f.frame_idx)

    localizations = load_localizations(
        localizations_root=args.localizations_root,
        coord_cols=tuple(args.coord_cols),
        fps_by_clip=fps_by_clip,
    )
    localizations_by_clip: dict[str, list[LocalizationRecord]] = defaultdict(list)
    for loc in localizations:
        localizations_by_clip[loc.clip_name].append(loc)

    unmatched_rows: list[dict[str, object]] = []
    off_frame_rows: list[dict[str, object]] = []
    summary_counts: Counter[str] = Counter()

    for clip_name, clip_frames in sorted(frames_by_clip.items()):
        clip_localizations = localizations_by_clip.get(clip_name, [])
        if not clip_localizations:
            continue
        frames_by_index: dict[int, list[DatasetFrame]] = defaultdict(list)
        for frame in clip_frames:
            frames_by_index[frame.frame_idx].append(frame)

        for loc in clip_localizations:
            candidate_frames: list[DatasetFrame] = []
            for delta in range(-args.frame_tolerance, args.frame_tolerance + 1):
                candidate_frames.extend(frames_by_index.get(loc.frame_idx + delta, []))
            if not candidate_frames:
                continue

            for frame in candidate_frames:
                u, v = world_to_pixel(loc.x, loc.y, homography)
                if not math.isfinite(u) or not math.isfinite(v):
                    continue
                in_frame = 0 <= u < frame.width and 0 <= v < frame.height
                if not in_frame:
                    summary_counts["off_frame"] += 1
                    off_frame_rows.append(
                        {
                            "status": "off_frame",
                            "split": frame.split,
                            "clip_name": clip_name,
                            "frame_idx": frame.frame_idx,
                            "frame_stem": frame.stem,
                            "image_path": frame.image_path,
                            "heatmap_path": frame.heatmap_path,
                            "source_csv": loc.source_csv,
                            "row_number": loc.row_number,
                            "time_seconds": loc.time_seconds,
                            "time_min_offset": loc.time_min_offset,
                            "time_max_offset": loc.time_max_offset,
                            "confidence": loc.confidence,
                            "world_x": loc.x,
                            "world_y": loc.y,
                            "pixel_x": u,
                            "pixel_y": v,
                            "frame_width": frame.width,
                            "frame_height": frame.height,
                            "x_err_span": loc.x_err_span,
                            "y_err_span": loc.y_err_span,
                        }
                    )
                    continue

                if not frame.boxes:
                    summary_counts["no_boxes_in_frame"] += 1
                    unmatched_rows.append(
                        {
                            "status": "no_boxes_in_frame",
                            "split": frame.split,
                            "clip_name": clip_name,
                            "frame_idx": frame.frame_idx,
                            "frame_stem": frame.stem,
                            "image_path": frame.image_path,
                            "label_path": frame.label_path,
                            "heatmap_path": frame.heatmap_path,
                            "source_csv": loc.source_csv,
                            "row_number": loc.row_number,
                            "time_seconds": loc.time_seconds,
                            "time_min_offset": loc.time_min_offset,
                            "time_max_offset": loc.time_max_offset,
                            "confidence": loc.confidence,
                            "world_x": loc.x,
                            "world_y": loc.y,
                            "pixel_x": u,
                            "pixel_y": v,
                            "nearest_box_distance_px": "",
                            "notes": "Frame has no YOLO labels.",
                        }
                    )
                    continue

                best_box, best_distance = best_box_match(u, v, frame.boxes)
                assert best_box is not None
                if best_distance <= args.pixel_tolerance:
                    summary_counts["matched"] += 1
                else:
                    summary_counts["unmatched_in_frame"] += 1
                    vis_path = ""
                    if args.save_visualizations:
                        vis_name = (
                            f"{frame.split}__{frame.stem}__row{loc.row_number}"
                            "__unmatched.jpg"
                        )
                        out_path = vis_dir / vis_name
                        save_unmatched_visualization(
                            image_path=frame.image_path,
                            out_path=out_path,
                            sound_x=u,
                            sound_y=v,
                            sound_box_size=args.sound_box_size,
                            label_box=best_box,
                            distance_px=best_distance,
                        )
                        vis_path = str(out_path)

                    unmatched_rows.append(
                        {
                            "status": "unmatched_in_frame",
                            "split": frame.split,
                            "clip_name": clip_name,
                            "frame_idx": frame.frame_idx,
                            "frame_stem": frame.stem,
                            "image_path": frame.image_path,
                            "label_path": frame.label_path,
                            "heatmap_path": frame.heatmap_path,
                            "source_csv": loc.source_csv,
                            "row_number": loc.row_number,
                            "time_seconds": loc.time_seconds,
                            "time_min_offset": loc.time_min_offset,
                            "time_max_offset": loc.time_max_offset,
                            "confidence": loc.confidence,
                            "world_x": loc.x,
                            "world_y": loc.y,
                            "pixel_x": u,
                            "pixel_y": v,
                            "nearest_box_distance_px": best_distance,
                            "nearest_box_class_id": best_box.class_id,
                            "nearest_box_xtl": best_box.xtl,
                            "nearest_box_ytl": best_box.ytl,
                            "nearest_box_xbr": best_box.xbr,
                            "nearest_box_ybr": best_box.ybr,
                            "visualization_path": vis_path,
                            "notes": "",
                        }
                    )

    summary_rows = [
        {"metric": "dataset_frames", "value": len(dataset_frames)},
        {"metric": "clips_in_dataset", "value": len(frames_by_clip)},
        {"metric": "localizations_loaded", "value": len(localizations)},
        {"metric": "pixel_tolerance", "value": args.pixel_tolerance},
        {"metric": "frame_tolerance", "value": args.frame_tolerance},
        {"metric": "missing_heatmap_png_files", "value": missing_heatmaps},
        {"metric": "matched", "value": summary_counts["matched"]},
        {"metric": "unmatched_in_frame", "value": summary_counts["unmatched_in_frame"]},
        {"metric": "no_boxes_in_frame", "value": summary_counts["no_boxes_in_frame"]},
        {"metric": "off_frame", "value": summary_counts["off_frame"]},
    ]

    unmatched_csv = args.output_dir / "audio_motion_unmatched.csv"
    off_frame_csv = args.output_dir / "audio_motion_off_frame.csv"
    summary_csv = args.output_dir / "audio_motion_summary.csv"

    write_csv(
        unmatched_csv,
        unmatched_rows,
        [
            "status",
            "split",
            "clip_name",
            "frame_idx",
            "frame_stem",
            "image_path",
            "label_path",
            "heatmap_path",
            "source_csv",
            "row_number",
            "time_seconds",
            "time_min_offset",
            "time_max_offset",
            "confidence",
            "world_x",
            "world_y",
            "pixel_x",
            "pixel_y",
            "nearest_box_distance_px",
            "nearest_box_class_id",
            "nearest_box_xtl",
            "nearest_box_ytl",
            "nearest_box_xbr",
            "nearest_box_ybr",
            "visualization_path",
            "notes",
        ],
    )
    write_csv(
        off_frame_csv,
        off_frame_rows,
        [
            "status",
            "split",
            "clip_name",
            "frame_idx",
            "frame_stem",
            "image_path",
            "heatmap_path",
            "source_csv",
            "row_number",
            "time_seconds",
            "time_min_offset",
            "time_max_offset",
            "confidence",
            "world_x",
            "world_y",
            "pixel_x",
            "pixel_y",
            "frame_width",
            "frame_height",
            "x_err_span",
            "y_err_span",
        ],
    )
    write_csv(summary_csv, summary_rows, ["metric", "value"])

    print(f"Wrote {summary_csv}")
    print(f"Wrote {unmatched_csv} ({len(unmatched_rows)} rows)")
    print(f"Wrote {off_frame_csv} ({len(off_frame_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
