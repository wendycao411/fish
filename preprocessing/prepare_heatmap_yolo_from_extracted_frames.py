#!/usr/bin/env python3
"""
Build a YOLO dataset from extracted frames by overlaying audio-localization heatmaps.

Inputs:
- extracted frame folders + CVAT annotations.xml
- localization CSV per clip
- homography matrix
- source video per clip (for FPS)

Outputs:
- YOLO-style dataset:
    out_dir/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrameRecord:
    rel_image_name: str
    clip_name: str
    frame_file: str
    frame_idx: int
    width: int
    height: int
    boxes: list[tuple[str, float, float, float, float]]


def world_to_pixel(c1: float, c2: float, homography: np.ndarray) -> tuple[float, float]:
    pt = np.array([[[float(c1), float(c2)]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(pt, homography.astype(np.float32))[0, 0]
    return float(uv[0]), float(uv[1])


def add_gaussian_inplace(
    heat_hw: np.ndarray,
    u: float,
    v: float,
    sigma_u: float = 20.0,
    sigma_v: float = 20.0,
    weight: float = 1.0,
    cutoff: float = 3.0,
) -> None:
    if (not np.isfinite(weight)) or weight <= 0:
        return
    if (not np.isfinite(sigma_u)) or sigma_u <= 0:
        sigma_u = 20.0
    if (not np.isfinite(sigma_v)) or sigma_v <= 0:
        sigma_v = 20.0

    heat_h, heat_w = heat_hw.shape
    ru = int(max(2, round(cutoff * float(sigma_u))))
    rv = int(max(2, round(cutoff * float(sigma_v))))

    u0 = max(0, int(np.floor(u)) - ru)
    u1 = min(heat_w - 1, int(np.floor(u)) + ru)
    v0 = max(0, int(np.floor(v)) - rv)
    v1 = min(heat_h - 1, int(np.floor(v)) + rv)
    if u0 >= u1 or v0 >= v1:
        return

    xs = np.arange(u0, u1 + 1, dtype=np.float32)
    ys = np.arange(v0, v1 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = (xx - float(u)) / max(1e-6, float(sigma_u))
    dy = (yy - float(v)) / max(1e-6, float(sigma_v))
    g = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)
    heat_hw[v0 : v1 + 1, u0 : u1 + 1] += float(weight) * g


def normalize_heatmap(h: np.ndarray, percentile: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    h = np.asarray(h, dtype=np.float32)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    pos = h[h > 0]
    if pos.size > 0:
        hi = float(np.percentile(pos, percentile))
    else:
        hi = 0.0
    if hi <= eps:
        hi = float(np.max(h))
        if hi <= eps:
            return np.zeros_like(h, dtype=np.float32)
    h = np.clip(h / hi, 0.0, 1.0)
    return h.astype(np.float32)


def overlay_heatmap(
    frame_bgr: np.ndarray,
    heat_hw: np.ndarray,
    alpha: float = 0.45,
    cmap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    h = normalize_heatmap(heat_hw)
    heat_u8 = np.clip(255.0 * h, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cmap)
    if heat_color.shape[:2] != frame_bgr.shape[:2]:
        heat_color = cv2.resize(heat_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_color, alpha, 0)


def heatmap_to_color_image(heat_hw: np.ndarray, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    h = normalize_heatmap(heat_hw)
    heat_u8 = np.clip(255.0 * h, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cmap)


def parse_frame_idx(frame_file: str) -> int:
    stem = Path(frame_file).stem
    token = stem.split("_", 1)[0]
    return int(token)


def parse_xml_annotations(xml_path: Path) -> tuple[list[FrameRecord], list[str]]:
    root = ET.parse(xml_path).getroot()
    image_nodes = root.findall(".//image")
    records: list[FrameRecord] = []
    all_labels: set[str] = set()

    for node in image_nodes:
        rel = node.attrib["name"]
        width = int(node.attrib["width"])
        height = int(node.attrib["height"])

        if "__" not in rel:
            # fallback: if name is already a relative path.
            rel_path = Path(rel)
            clip_name = rel_path.parent.name
            frame_file = rel_path.name
        else:
            clip_name, frame_file = rel.split("__", 1)

        boxes: list[tuple[str, float, float, float, float]] = []
        for box in node.findall("box"):
            label = str(box.attrib["label"])
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            boxes.append((label, xtl, ytl, xbr, ybr))
            all_labels.add(label)

        frame_idx = parse_frame_idx(frame_file)
        records.append(
            FrameRecord(
                rel_image_name=rel,
                clip_name=clip_name,
                frame_file=frame_file,
                frame_idx=frame_idx,
                width=width,
                height=height,
                boxes=boxes,
            )
        )

    return records, sorted(all_labels)


def build_heatmaps_for_requested_frames(
    frame_indices: set[int],
    frame_h: int,
    frame_w: int,
    fps: float,
    localizations_csv: Path,
    homography: np.ndarray,
    coord_cols: tuple[str, str] = ("x", "y"),
    sigma_px: float = 20.0,
    use_error_spans: bool = True,
) -> dict[int, np.ndarray]:
    df = pd.read_csv(localizations_csv)
    c1_name, c2_name = coord_cols
    required = ["time_min_offset", c1_name, c2_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {localizations_csv}: {missing}")

    heatmaps: dict[int, np.ndarray] = {
        idx: np.zeros((frame_h, frame_w), dtype=np.float32) for idx in frame_indices
    }

    for _, row in df.iterrows():
        t = float(row["time_min_offset"])
        frame_idx = int(round(t * fps))
        if frame_idx not in heatmaps:
            continue

        c1 = float(row[c1_name])
        c2 = float(row[c2_name])
        if not np.isfinite(c1) or not np.isfinite(c2):
            continue

        u, v = world_to_pixel(c1, c2, homography)
        if not (0 <= u < frame_w and 0 <= v < frame_h):
            continue

        conf = float(row.get("confidence", 1.0))
        if not np.isfinite(conf):
            conf = 1.0
        conf = float(np.clip(conf, 0.0, 1.0))

        su = sv = float(sigma_px)
        if use_error_spans:
            e1 = row.get(f"{c1_name}_err_span", np.nan)
            e2 = row.get(f"{c2_name}_err_span", np.nan)
            if np.isfinite(e1) and float(e1) > 0:
                su = max(su, float(sigma_px) * float(e1))
            if np.isfinite(e2) and float(e2) > 0:
                sv = max(sv, float(sigma_px) * float(e2))

        add_gaussian_inplace(heatmaps[frame_idx], u, v, sigma_u=su, sigma_v=sv, weight=conf)

    for idx in list(heatmaps.keys()):
        heatmaps[idx] = normalize_heatmap(heatmaps[idx])

    return heatmaps


def yolo_line_from_box(
    class_id: int, xtl: float, ytl: float, xbr: float, ybr: float, width: int, height: int
) -> str | None:
    bw = max(0.0, xbr - xtl)
    bh = max(0.0, ybr - ytl)
    if bw <= 1e-6 or bh <= 1e-6:
        return None
    cx = xtl + bw / 2.0
    cy = ytl + bh / 2.0
    x = np.clip(cx / width, 0.0, 1.0)
    y = np.clip(cy / height, 0.0, 1.0)
    w = np.clip(bw / width, 0.0, 1.0)
    h = np.clip(bh / height, 0.0, 1.0)
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def write_data_yaml(out_dir: Path, class_names: list[str]) -> None:
    yaml_path = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
    ]
    yaml_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("extracted_frames_200"),
        help="Root containing clip folders and annotations.xml",
    )
    parser.add_argument(
        "--annotations-xml",
        type=Path,
        default=Path("extracted_frames_200/annotations.xml"),
        help="CVAT XML annotation file",
    )
    parser.add_argument(
        "--localizations-root",
        type=Path,
        default=Path("XAV-arrays/localization/out_synced_pairs"),
        help="Per-clip localization folder root",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("synced_pairs"),
        help="Per-clip source videos root",
    )
    parser.add_argument(
        "--homography-npy",
        type=Path,
        default=Path("XAV-arrays/localization/top_down_H.npy"),
        help="3x3 homography .npy path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("processed/extracted_frames_200_heatmap_yolo"),
        help="Output YOLO dataset directory",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument(
        "--compose-mode",
        choices=["overlay", "separate"],
        default="overlay",
        help="overlay: save fused image only; separate: save original image + separate heatmap image",
    )
    parser.add_argument("--sigma-px", type=float, default=20.0)
    parser.add_argument("--coord-cols", nargs=2, default=("x", "y"))
    parser.add_argument("--no-error-spans", action="store_true")
    parser.add_argument(
        "--save-heatmap-npy",
        action="store_true",
        help="When compose-mode=separate, also save per-frame raw heatmap as .npy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of frames to process (0 means all)",
    )
    args = parser.parse_args()

    homography = np.load(args.homography_npy).astype(np.float32)
    if homography.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography, got {homography.shape}")

    records, class_names = parse_xml_annotations(args.annotations_xml)
    if not records:
        raise RuntimeError(f"No records parsed from {args.annotations_xml}")
    if not class_names:
        class_names = ["fish"]

    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    rng = random.Random(args.seed)
    ordered = list(records)
    rng.shuffle(ordered)
    if args.limit and args.limit > 0:
        ordered = ordered[: args.limit]

    n_val = int(round(len(ordered) * args.val_ratio))
    val_set = {rec.rel_image_name for rec in ordered[:n_val]}

    for split in ("train", "val"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        if args.compose_mode == "separate":
            (args.out_dir / "heatmaps" / split).mkdir(parents=True, exist_ok=True)
            if args.save_heatmap_npy:
                (args.out_dir / "heatmaps_npy" / split).mkdir(parents=True, exist_ok=True)

    # Group records by clip to compute heatmaps clip-wise.
    by_clip: dict[str, list[FrameRecord]] = {}
    for rec in ordered:
        by_clip.setdefault(rec.clip_name, []).append(rec)

    processed = 0
    for clip_name, clip_records in sorted(by_clip.items()):
        video_path = args.videos_root / clip_name / f"{clip_name}.mp4"
        loc_csv = args.localizations_root / clip_name / "localizations_merged_filtered.csv"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for clip {clip_name}: {video_path}")
        if not loc_csv.exists():
            raise FileNotFoundError(f"Missing localization CSV for clip {clip_name}: {loc_csv}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS for {video_path}: {fps}")

        frame_h = clip_records[0].height
        frame_w = clip_records[0].width
        requested_indices = {rec.frame_idx for rec in clip_records}

        heatmaps = build_heatmaps_for_requested_frames(
            frame_indices=requested_indices,
            frame_h=frame_h,
            frame_w=frame_w,
            fps=fps,
            localizations_csv=loc_csv,
            homography=homography,
            coord_cols=tuple(args.coord_cols),
            sigma_px=args.sigma_px,
            use_error_spans=not args.no_error_spans,
        )

        for rec in clip_records:
            img_path = args.frames_root / rec.clip_name / rec.frame_file
            if not img_path.exists():
                raise FileNotFoundError(f"Missing extracted frame: {img_path}")
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {img_path}")

            heat = heatmaps.get(rec.frame_idx)
            if heat is None:
                heat = np.zeros((rec.height, rec.width), dtype=np.float32)
            split = "val" if rec.rel_image_name in val_set else "train"
            out_stem = f"{rec.clip_name}__{Path(rec.frame_file).stem}"
            out_img = args.out_dir / "images" / split / f"{out_stem}.jpg"
            out_lbl = args.out_dir / "labels" / split / f"{out_stem}.txt"

            if args.compose_mode == "overlay":
                image_to_save = overlay_heatmap(frame, heat, alpha=args.overlay_alpha)
            else:
                image_to_save = frame
            ok = cv2.imwrite(str(out_img), image_to_save)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_img}")

            if args.compose_mode == "separate":
                out_heat = args.out_dir / "heatmaps" / split / f"{out_stem}_heatmap.png"
                heat_color = heatmap_to_color_image(heat)
                ok_heat = cv2.imwrite(str(out_heat), heat_color)
                if not ok_heat:
                    raise RuntimeError(f"Failed to write heatmap image: {out_heat}")
                if args.save_heatmap_npy:
                    out_heat_npy = args.out_dir / "heatmaps_npy" / split / f"{out_stem}_heatmap.npy"
                    np.save(out_heat_npy, heat.astype(np.float32))

            yolo_lines: list[str] = []
            for label, xtl, ytl, xbr, ybr in rec.boxes:
                class_id = label_to_id[label]
                line = yolo_line_from_box(class_id, xtl, ytl, xbr, ybr, rec.width, rec.height)
                if line is not None:
                    yolo_lines.append(line)
            out_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            processed += 1

        print(f"[clip] {clip_name}: {len(clip_records)} frames processed")

    write_data_yaml(args.out_dir, class_names)
    print(f"[ok] dataset written to: {args.out_dir}")
    print(f"[ok] processed frames: {processed}")
    print(f"[ok] classes: {class_names}")
    print(f"[ok] compose_mode: {args.compose_mode}")


if __name__ == "__main__":
    main()
