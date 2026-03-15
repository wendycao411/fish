#!/usr/bin/env python3
"""Sanity checks for heatmap fusion dataloading and transform alignment."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import LoadImagesAndLabels, create_dataloader


def build_synthetic_dataset(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "heatmaps" / "train").mkdir(parents=True, exist_ok=True)

    h, w = 96, 128
    x1, y1, x2, y2 = 20, 25, 60, 65
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[y1:y2, x1:x2] = 255
    hm = np.zeros((h, w), dtype=np.uint8)
    hm[y1:y2, x1:x2] = 255

    img_path = root / "images" / "train" / "sample0001.jpg"
    hm_path = root / "heatmaps" / "train" / "sample0001.png"
    lb_path = root / "labels" / "train" / "sample0001.txt"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(hm_path), hm)

    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    lb_path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["fish"],
        "heatmap_dir": "heatmaps",
        "heatmap_ext": ".png",
    }
    data_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    return img_path.parent


def default_hyp():
    return {
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
    }


def check_shapes(img_dir: Path, n: int = 2):
    hyp = default_hyp()
    dl_concat, _ = create_dataloader(
        str(img_dir),
        imgsz=128,
        batch_size=n,
        stride=32,
        single_cls=True,
        hyp=hyp,
        augment=False,
        workers=0,
        heatmap_fusion="concat",
        heatmap_root=str(img_dir.parent.parent / "heatmaps"),
        heatmap_ext=".png",
    )
    imgs, _, _, _ = next(iter(dl_concat))
    assert imgs.shape[1] == 4, f"Expected concat channels=4, got {tuple(imgs.shape)}"

    dl_mod, _ = create_dataloader(
        str(img_dir),
        imgsz=128,
        batch_size=n,
        stride=32,
        single_cls=True,
        hyp=hyp,
        augment=False,
        workers=0,
        heatmap_fusion="modulate",
        heatmap_root=str(img_dir.parent.parent / "heatmaps"),
        heatmap_ext=".png",
    )
    imgs, _, _, _, heatmaps = next(iter(dl_mod))
    assert imgs.shape[1] == 3, f"Expected RGB channels=3, got {tuple(imgs.shape)}"
    assert heatmaps.shape[1] == 1, f"Expected heatmap channels=1, got {tuple(heatmaps.shape)}"


def check_alignment(img_dir: Path):
    hyp = default_hyp()
    hyp["fliplr"] = 1.0  # deterministic geometric transform for alignment check
    ds = LoadImagesAndLabels(
        str(img_dir),
        img_size=128,
        batch_size=1,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=False,
        single_cls=True,
        stride=32,
        pad=0.0,
        image_weights=False,
        heatmap_fusion="modulate",
        heatmap_root=str(img_dir.parent.parent / "heatmaps"),
        heatmap_ext=".png",
    )
    img, _, _, _, heatmap = ds[0]
    rgb = img.numpy()
    hm = heatmap.numpy()
    image_mask = rgb[0] > 200
    heatmap_mask = hm[0] > 0.5
    assert image_mask.any(), "No bright image pixels found after transforms."
    assert heatmap_mask.any(), "No positive heatmap pixels found after transforms."
    yc_img, xc_img = np.argwhere(image_mask).mean(axis=0)
    yc_hm, xc_hm = np.argwhere(heatmap_mask).mean(axis=0)
    assert abs(yc_img - yc_hm) <= 2 and abs(xc_img - xc_hm) <= 2, (
        f"Heatmap/image transform misalignment: image=({yc_img:.2f},{xc_img:.2f}), "
        f"heatmap=({yc_hm:.2f},{xc_hm:.2f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2, help="number of samples for shape checks")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory(prefix="yolo_heatmap_sanity_") as tmp:
        img_dir = build_synthetic_dataset(Path(tmp))
        check_shapes(img_dir, n=args.samples)
        check_alignment(img_dir)
    print("Heatmap fusion sanity checks passed.")


if __name__ == "__main__":
    main()
