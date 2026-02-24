#!/usr/bin/env python3
"""Prepare a unified MegaFish dataset in Ultralytics YOLO format.

This script is additive and does not modify the legacy YOLOv5 pipeline.
It can optionally call existing converters in dataset_utils and then unify their
outputs into:
  images/{train,val,test}
  labels/{train,val,test}
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml

from common import ensure_dir, repo_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "tr": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
    "testing": "test",
    "te": "test",
}

DATASET_REGISTRY = {
    "aims_ozfish": {
        "converter": "process_aims_ozfish_to_yolo.py",
        "manual": "Download AIMS OzFish raw data, then run dataset_utils/process_aims_ozfish_to_yolo.py",
    },
    "deepfish": {
        "converter": "process_deepfish_to_yolo.py",
        "manual": "Download DeepFish raw data, then run dataset_utils/process_deepfish_to_yolo.py",
    },
    "fathomnet": {
        "converter": "download_and_process_fathomnet.py",
        "manual": "Install fathomnet API creds/env, then run dataset_utils/download_and_process_fathomnet.py",
    },
    "lfitw": {
        "converter": "process_lfitw_to_yolo.py",
        "manual": "Download NOAA Labelled Fishes In The Wild, then run dataset_utils/process_lfitw_to_yolo.py",
    },
    "noaa_estuary": {
        "converter": "process_noaa_estuary_to_yolo.py",
        "manual": "Download NOAA Estuary data, then run dataset_utils/process_noaa_estuary_to_yolo.py",
    },
    "viame_fishtrack": {
        "converter": "process_viame_fishtrack_to_yolo.py",
        "manual": "Download VIAME FishTrack22 collection and extracted frames, then run dataset_utils/process_viame_fishtrack_to_yolo.py",
    },
}


@dataclass
class Sample:
    dataset: str
    image_path: Path
    label_path: Optional[Path]
    split_hint: Optional[str]


@dataclass
class PrepareStats:
    images_total: int = 0
    labels_missing: int = 0
    boxes_total: int = 0
    boxes_invalid: int = 0


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Build unified MegaFish Ultralytics dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "datasets" / "megafish_ultra",
        help="Output dataset root (default: megafishdetector/datasets/megafish_ultra)",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=root / "datasets" / "processed",
        help="Root containing per-dataset processed YOLO folders",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "datasets" / "raw",
        help="Root containing per-dataset raw downloads used by converter scripts",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_REGISTRY.keys()),
        help="Comma-separated datasets to include",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow converter scripts that download data (notably FathomNet)",
    )
    parser.add_argument(
        "--run-converters",
        action="store_true",
        help="Call existing dataset_utils converter scripts if processed outputs are missing",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir if it exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic autosplit fallback",
    )
    parser.add_argument(
        "--split-weights",
        type=str,
        default="0.9,0.1,0.0",
        help="Fallback autosplit weights for train,val,test",
    )
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(sorted(IMAGE_EXTS)),
        help="Allowed image extensions (comma-separated)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid labels instead of skipping invalid lines",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Render sample overlays and print bbox stats",
    )
    parser.add_argument("--sanity-samples", type=int, default=24)
    parser.add_argument(
        "--sanity-out",
        type=Path,
        default=None,
        help="Override sanity output dir (default: <output-dir>/out/sanity)",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=root / "ultralytics_yolo" / "data" / "megafish.yaml",
        help="Path to write Ultralytics data yaml",
    )
    return parser.parse_args()


def parse_weights(weights: str) -> Tuple[float, float, float]:
    vals = [float(x.strip()) for x in weights.split(",")]
    if len(vals) != 3:
        raise ValueError("--split-weights must have 3 values: train,val,test")
    if any(v < 0 for v in vals):
        raise ValueError("--split-weights must be non-negative")
    if sum(vals) <= 0:
        raise ValueError("--split-weights sum must be > 0")
    return vals[0], vals[1], vals[2]


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def find_processed_dir(dataset: str, processed_root: Path) -> Path:
    candidates = [
        processed_root / dataset,
        processed_root / f"{dataset}_yolo",
        processed_root / f"{dataset.upper()}_YOLO",
        processed_root / dataset.replace("_", ""),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def run_converter_if_requested(dataset: str, args: argparse.Namespace, processed_dir: Path) -> bool:
    if processed_dir.exists() or not args.run_converters:
        return processed_dir.exists()

    converter = DATASET_REGISTRY[dataset]["converter"]
    converter_path = repo_root() / "dataset_utils" / converter
    if not converter_path.exists():
        print(f"[WARN] Missing converter script for {dataset}: {converter_path}")
        return False

    if dataset == "fathomnet":
        if not args.download:
            print("[WARN] Skipping FathomNet converter without --download")
            return False
        cmd = ["python3", str(converter_path), str(processed_dir)]
    else:
        raw_dir = args.raw_root / dataset
        if not raw_dir.exists():
            print(f"[WARN] Missing raw dataset dir for {dataset}: {raw_dir}")
            return False
        cmd = ["python3", str(converter_path), str(raw_dir), str(processed_dir)]

    print(f"[INFO] Running converter: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] Converter failed for {dataset} (exit={result.returncode})")
        return False
    return processed_dir.exists()


def list_images(root: Path, image_exts: set[str]) -> List[Path]:
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in image_exts
    ]
    return files


def infer_split_from_parts(parts: Sequence[str]) -> Optional[str]:
    for token in parts:
        key = token.lower()
        if key in SPLIT_ALIASES:
            return SPLIT_ALIASES[key]
    return None


def resolve_label_path(dataset_dir: Path, image_path: Path) -> Optional[Path]:
    # Try common patterns used by existing converters.
    stem = image_path.stem
    name = image_path.name

    candidates = []
    rel = image_path.relative_to(dataset_dir)
    parts = list(rel.parts)

    if "images" in parts:
        i = parts.index("images")
        prefix = parts[:i]
        suffix = parts[i + 1 :]
        if suffix:
            split_part = suffix[0]
            after = suffix[1:]
            candidates.append(dataset_dir.joinpath(*prefix, "labels", split_part, *after).with_suffix(".txt"))
            candidates.append(dataset_dir.joinpath(*prefix, "labels", split_part, *after[:-1], f"{name}.txt"))
        candidates.append(dataset_dir.joinpath(*prefix, "labels", *suffix).with_suffix(".txt"))

    candidates.extend(
        [
            dataset_dir / "labels" / f"{stem}.txt",
            dataset_dir / "labels" / f"{name}.txt",
            image_path.with_suffix(".txt"),
        ]
    )

    for c in candidates:
        if c.exists():
            return c
    return None


def collect_samples_for_dataset(dataset: str, dataset_dir: Path, image_exts: set[str]) -> List[Sample]:
    if not dataset_dir.exists():
        return []

    # Prefer known image roots to avoid pulling previews/groundtruth assets.
    image_roots = [d for d in [dataset_dir / "images", dataset_dir / "frames", dataset_dir / "JPEGImages"] if d.exists()]
    if not image_roots:
        image_roots = [dataset_dir]

    seen: set[Path] = set()
    samples: List[Sample] = []
    for root in image_roots:
        for image_path in list_images(root, image_exts):
            if "groundtruth" in {p.lower() for p in image_path.parts}:
                continue
            if image_path in seen:
                continue
            seen.add(image_path)
            split_hint = infer_split_from_parts(image_path.relative_to(dataset_dir).parts)
            label_path = resolve_label_path(dataset_dir, image_path)
            samples.append(Sample(dataset=dataset, image_path=image_path, label_path=label_path, split_hint=split_hint))

    return sorted(samples, key=lambda s: str(s.image_path))


def yolo_autosplit(n: int, weights: Tuple[float, float, float], seed: int) -> List[str]:
    random.seed(seed)
    idx = random.choices([0, 1, 2], weights=weights, k=n)
    names = ["train", "val", "test"]
    return [names[i] for i in idx]


def parse_and_collapse_labels(label_path: Optional[Path], strict: bool) -> Tuple[List[str], int, int, Dict[str, float]]:
    lines_out: List[str] = []
    kept = 0
    invalid = 0
    stats = {
        "xc_min": math.inf,
        "xc_max": -math.inf,
        "yc_min": math.inf,
        "yc_max": -math.inf,
        "w_min": math.inf,
        "w_max": -math.inf,
        "h_min": math.inf,
        "h_max": -math.inf,
        "ar_min": math.inf,
        "ar_max": -math.inf,
    }

    if label_path is None or not label_path.exists():
        return lines_out, kept, invalid, stats

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return lines_out, kept, invalid, stats

    for ln, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            invalid += 1
            if strict:
                raise ValueError(f"Invalid label format at {label_path}:{ln}: {line}")
            continue

        try:
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            invalid += 1
            if strict:
                raise
            continue

        vals = [x, y, w, h]
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            invalid += 1
            if strict:
                raise ValueError(f"NaN/Inf label at {label_path}:{ln}")
            continue
        if any(v < 0.0 or v > 1.0 for v in vals):
            invalid += 1
            if strict:
                raise ValueError(f"Out-of-range label at {label_path}:{ln}: {vals}")
            continue

        lines_out.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept += 1

        stats["xc_min"] = min(stats["xc_min"], x)
        stats["xc_max"] = max(stats["xc_max"], x)
        stats["yc_min"] = min(stats["yc_min"], y)
        stats["yc_max"] = max(stats["yc_max"], y)
        stats["w_min"] = min(stats["w_min"], w)
        stats["w_max"] = max(stats["w_max"], w)
        stats["h_min"] = min(stats["h_min"], h)
        stats["h_max"] = max(stats["h_max"], h)
        if h > 0:
            ar = w / h
            stats["ar_min"] = min(stats["ar_min"], ar)
            stats["ar_max"] = max(stats["ar_max"], ar)

    return lines_out, kept, invalid, stats


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def update_global_stats(global_stats: Dict[str, float], local_stats: Dict[str, float]) -> None:
    for k, v in local_stats.items():
        if not math.isfinite(v):
            continue
        if k.endswith("_min"):
            global_stats[k] = min(global_stats.get(k, math.inf), v)
        elif k.endswith("_max"):
            global_stats[k] = max(global_stats.get(k, -math.inf), v)


def create_sanity_outputs(
    entries: List[Tuple[Path, Path, List[str]]],
    out_dir: Path,
    sample_count: int,
) -> None:
    ensure_dir(out_dir)
    n = min(sample_count, len(entries))
    for i in range(n):
        image_path, _, lines = entries[i]
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lines:
            _, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"{i:04d}_{image_path.stem}.jpg"), img)


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    ensure_dir(path.parent)
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "fish"},
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.processed_root = args.processed_root.resolve()
    args.raw_root = args.raw_root.resolve()
    args.yaml_path = args.yaml_path.resolve()

    image_exts = {x.strip().lower() if x.strip().startswith(".") else f".{x.strip().lower()}" for x in args.image_exts.split(",")}
    weights = parse_weights(args.split_weights)

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    for split in ("train", "val", "test"):
        ensure_dir(args.output_dir / "images" / split)
        ensure_dir(args.output_dir / "labels" / split)

    datasets = [normalize_dataset_name(d) for d in args.datasets.split(",") if d.strip()]
    missing_msgs: List[str] = []

    stats = PrepareStats()
    global_bbox_stats = {
        "xc_min": math.inf,
        "xc_max": -math.inf,
        "yc_min": math.inf,
        "yc_max": -math.inf,
        "w_min": math.inf,
        "w_max": -math.inf,
        "h_min": math.inf,
        "h_max": -math.inf,
        "ar_min": math.inf,
        "ar_max": -math.inf,
    }

    sanity_entries: List[Tuple[Path, Path, List[str]]] = []

    for dataset in datasets:
        if dataset not in DATASET_REGISTRY:
            print(f"[WARN] Unknown dataset '{dataset}', skipping")
            continue

        processed_dir = find_processed_dir(dataset, args.processed_root)
        available = processed_dir.exists() or run_converter_if_requested(dataset, args, processed_dir)
        if not available:
            missing_msgs.append(f"- {dataset}: not found at {processed_dir}. {DATASET_REGISTRY[dataset]['manual']}")
            continue

        samples = collect_samples_for_dataset(dataset, processed_dir, image_exts)
        if not samples:
            missing_msgs.append(f"- {dataset}: found {processed_dir} but no images discovered")
            continue

        unsplit_idx = [i for i, s in enumerate(samples) if s.split_hint is None]
        autosplits = yolo_autosplit(len(unsplit_idx), weights, args.seed)
        autosplit_map = {idx: autosplits[j] for j, idx in enumerate(unsplit_idx)}

        print(f"[INFO] {dataset}: {len(samples)} images")

        for i, sample in enumerate(samples):
            split = sample.split_hint if sample.split_hint in {"train", "val", "test"} else autosplit_map[i]
            rel_name = f"{dataset}__{sample.image_path.stem}{sample.image_path.suffix.lower()}"
            out_img = args.output_dir / "images" / split / rel_name
            out_lbl = args.output_dir / "labels" / split / f"{dataset}__{sample.image_path.stem}.txt"

            link_or_copy(sample.image_path, out_img, copy_files=args.copy)

            lines, kept, invalid, local_stats = parse_and_collapse_labels(sample.label_path, strict=args.strict)
            out_lbl.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            stats.images_total += 1
            stats.boxes_total += kept
            stats.boxes_invalid += invalid
            if sample.label_path is None or not sample.label_path.exists():
                stats.labels_missing += 1

            update_global_stats(global_bbox_stats, local_stats)
            if args.sanity_check:
                sanity_entries.append((sample.image_path, out_lbl, lines))

    write_data_yaml(args.yaml_path, args.output_dir)

    if args.sanity_check:
        sanity_out = args.sanity_out.resolve() if args.sanity_out else args.output_dir / "out" / "sanity"
        create_sanity_outputs(sanity_entries, sanity_out, args.sanity_samples)
        print(f"[INFO] sanity overlays: {sanity_out}")

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = len(list((args.output_dir / "images" / split).glob("*")))

    print("[INFO] Unified dataset ready")
    print(f"[INFO] output-dir: {args.output_dir}")
    print(f"[INFO] yaml: {args.yaml_path}")
    print(f"[INFO] split counts: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(
        "[INFO] label stats: "
        f"images={stats.images_total} missing_label_files={stats.labels_missing} "
        f"boxes_kept={stats.boxes_total} boxes_invalid_skipped={stats.boxes_invalid}"
    )

    def fmt(v: float) -> str:
        return f"{v:.6f}" if math.isfinite(v) else "n/a"

    print(
        "[INFO] bbox ranges: "
        f"xc=[{fmt(global_bbox_stats['xc_min'])},{fmt(global_bbox_stats['xc_max'])}] "
        f"yc=[{fmt(global_bbox_stats['yc_min'])},{fmt(global_bbox_stats['yc_max'])}] "
        f"w=[{fmt(global_bbox_stats['w_min'])},{fmt(global_bbox_stats['w_max'])}] "
        f"h=[{fmt(global_bbox_stats['h_min'])},{fmt(global_bbox_stats['h_max'])}] "
        f"aspect(w/h)=[{fmt(global_bbox_stats['ar_min'])},{fmt(global_bbox_stats['ar_max'])}]"
    )

    if missing_msgs:
        print("[WARN] Some datasets were skipped:")
        for msg in missing_msgs:
            print(msg)


if __name__ == "__main__":
    main()
