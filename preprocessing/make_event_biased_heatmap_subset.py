#!/usr/bin/env python3
"""Create a small event-biased subset of the heatmap YOLO dataset.

The subset preserves the source train/val proportion, prioritizes frames whose
saved heatmap has max value > threshold, and fills the remainder with random
non-event frames when there are too few positive examples.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sample:
    split: str
    stem: str
    clip_id: str
    frame_idx: int
    image_path: Path
    label_path: Path
    heatmap_npy_path: Path
    heatmap_png_path: Path | None
    heatmap_max: float
    is_event: bool
    num_boxes: int
    total_box_area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 250-image event-biased heatmap subset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/processed/extracted_frames_200_heatmap_yolo_separate"),
        help="Source heatmap YOLO dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/processed/extracted_frames_200_heatmap_yolo_separate_subset250"),
        help="Output subset directory",
    )
    parser.add_argument("--subset-size", type=int, default=250, help="Total number of images to select")
    parser.add_argument(
        "--event-ratio",
        type=float,
        default=0.8,
        help="Target fraction of event images when enough exist",
    )
    parser.add_argument(
        "--heatmap-threshold",
        type=float,
        default=0.1,
        help="Heatmap max threshold above which a frame counts as an event (defaults assume saved heatmaps are scaled to 0-10)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir if it exists")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def parse_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"(.+)__(\d+)_([^_]+(?:_[^_]+)*)$", stem)
    if not m:
        return stem, -1
    return m.group(1), int(m.group(2))


def label_stats(label_path: Path) -> tuple[int, float]:
    num_boxes = 0
    total_box_area = 0.0
    if not label_path.exists():
        return num_boxes, total_box_area

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue
        num_boxes += 1
        total_box_area += max(0.0, bw) * max(0.0, bh)
    return num_boxes, total_box_area


def list_samples(root: Path, split: str, heatmap_threshold: float) -> list[Sample]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    hm_npy_dir = root / "heatmaps_npy" / split
    hm_png_dir = root / "heatmaps" / split

    samples: list[Sample] = []
    for image_path in sorted(img_dir.glob("*.jpg")):
        stem = image_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        heatmap_npy_path = hm_npy_dir / f"{stem}_heatmap.npy"
        heatmap_png_path = hm_png_dir / f"{stem}_heatmap.png"
        if not label_path.exists() or not heatmap_npy_path.exists():
            continue

        heatmap = np.load(heatmap_npy_path)
        heatmap_max = float(np.nanmax(heatmap)) if heatmap.size else 0.0
        is_event = heatmap_max > heatmap_threshold
        num_boxes, total_box_area = label_stats(label_path)
        clip_id, frame_idx = parse_stem(stem)
        samples.append(
            Sample(
                split=split,
                stem=stem,
                clip_id=clip_id,
                frame_idx=frame_idx,
                image_path=image_path,
                label_path=label_path,
                heatmap_npy_path=heatmap_npy_path,
                heatmap_png_path=heatmap_png_path if heatmap_png_path.exists() else None,
                heatmap_max=heatmap_max,
                is_event=is_event,
                num_boxes=num_boxes,
                total_box_area=total_box_area,
            )
        )
    return samples


def allocate_counts(total_by_split: dict[str, int], subset_size: int) -> dict[str, int]:
    grand_total = sum(total_by_split.values())
    splits = list(total_by_split)
    raw = {split: subset_size * total_by_split[split] / grand_total for split in splits}
    counts = {split: int(raw[split]) for split in splits}
    remaining = subset_size - sum(counts.values())
    for split in sorted(splits, key=lambda s: raw[s] - counts[s], reverse=True):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1
    return counts


def choose_farthest_in_clip(candidates: list[Sample], chosen: list[Sample]) -> Sample:
    if not chosen:
        # Start near the clip midpoint to keep subsequent picks spread on both sides.
        return candidates[len(candidates) // 2]
    chosen_idx = [s.frame_idx for s in chosen if s.frame_idx >= 0]
    if not chosen_idx:
        return candidates[len(candidates) // 2]

    def score(sample: Sample) -> tuple[int, int]:
        dist = min(abs(sample.frame_idx - idx) for idx in chosen_idx)
        return (dist, -sample.frame_idx)

    return max(candidates, key=score)


def select_temporally_sparse(samples: list[Sample], n: int, rng: random.Random) -> list[Sample]:
    if n <= 0 or not samples:
        return []

    by_clip: dict[str, list[Sample]] = {}
    for sample in samples:
        by_clip.setdefault(sample.clip_id, []).append(sample)
    for clip_samples in by_clip.values():
        clip_samples.sort(key=lambda s: (s.frame_idx, s.stem))

    selected_by_clip: dict[str, list[Sample]] = {clip: [] for clip in by_clip}
    clip_order = list(by_clip)
    rng.shuffle(clip_order)
    selected: list[Sample] = []

    # Round-robin across clips, choosing the farthest next frame within each clip.
    while len(selected) < n:
        progressed = False
        for clip_id in clip_order:
            chosen = selected_by_clip[clip_id]
            candidates = [s for s in by_clip[clip_id] if s not in chosen]
            if not candidates:
                continue
            pick = choose_farthest_in_clip(candidates, chosen)
            chosen.append(pick)
            selected.append(pick)
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break

    return selected[:n]


def write_data_yaml(out_dir: Path) -> None:
    (out_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                "names: ['fish']",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    all_samples: dict[str, list[Sample]] = {
        split: list_samples(input_dir, split, args.heatmap_threshold)
        for split in ("train", "val")
    }
    total_by_split = {split: len(samples) for split, samples in all_samples.items()}
    subset_by_split = allocate_counts(total_by_split, args.subset_size)

    selected: dict[str, list[Sample]] = {"train": [], "val": []}
    target_event_total = int(round(args.subset_size * args.event_ratio))
    available_event_total = sum(sum(s.is_event for s in samples) for samples in all_samples.values())
    event_quota_total = min(target_event_total, available_event_total)
    event_quota_by_split = allocate_counts(
        {
            split: sum(s.is_event for s in samples)
            for split, samples in all_samples.items()
        },
        event_quota_total,
    ) if event_quota_total > 0 else {"train": 0, "val": 0}

    for split, samples in all_samples.items():
        events = [s for s in samples if s.is_event]
        non_events = [s for s in samples if not s.is_event]
        rng.shuffle(events)
        non_events = sorted(non_events, key=lambda s: (s.num_boxes, s.total_box_area, s.heatmap_max, s.stem))

        take_events = min(event_quota_by_split.get(split, 0), len(events), subset_by_split[split])
        chosen = events[:take_events]
        need_more = subset_by_split[split] - len(chosen)

        if need_more > 0:
            chosen.extend(select_temporally_sparse(non_events, need_more, rng))

        selected[split] = sorted(chosen, key=lambda s: s.stem)

    for split, samples in selected.items():
        for sample in samples:
            copy_file(sample.image_path, output_dir / "images" / split / sample.image_path.name)
            copy_file(sample.label_path, output_dir / "labels" / split / sample.label_path.name)
            copy_file(sample.heatmap_npy_path, output_dir / "heatmaps_npy" / split / sample.heatmap_npy_path.name)
            if sample.heatmap_png_path is not None:
                copy_file(sample.heatmap_png_path, output_dir / "heatmaps" / split / sample.heatmap_png_path.name)

    write_data_yaml(output_dir)

    metadata_path = output_dir / "subset_manifest.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["split", "clip_id", "frame_idx", "stem", "image_name", "heatmap_max", "is_event", "num_boxes", "total_box_area"]
        )
        for split, samples in selected.items():
            for sample in samples:
                writer.writerow(
                    [
                        split,
                        sample.clip_id,
                        sample.frame_idx,
                        sample.stem,
                        sample.image_path.name,
                        f"{sample.heatmap_max:.6f}",
                        int(sample.is_event),
                        sample.num_boxes,
                        f"{sample.total_box_area:.6f}",
                    ]
                )

    total_selected = sum(len(v) for v in selected.values())
    total_events = sum(sum(s.is_event for s in v) for v in selected.values())
    print(f"[INFO] Wrote subset to: {output_dir}")
    print(f"[INFO] Total selected: {total_selected}")
    print(f"[INFO] Event frames selected (heatmap_max > {args.heatmap_threshold}): {total_events}")
    for split, samples in selected.items():
        split_events = sum(s.is_event for s in samples)
        print(f"[INFO] {split}: {len(samples)} images ({split_events} events, {len(samples) - split_events} non-events)")
    print(f"[INFO] Manifest: {metadata_path}")


if __name__ == "__main__":
    main()
