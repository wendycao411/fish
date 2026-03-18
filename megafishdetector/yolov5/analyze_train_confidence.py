#!/usr/bin/env python3
"""
Analyze training images by model confidence.

Runs the best.pt YOLOv5+heatmap-modulation model over training images,
ranks images by max-detection confidence, then plots 3-panel comparisons
(ground-truth | heatmap | prediction).

Usage (from yolov5/ directory):
    python analyze_train_confidence.py
    python analyze_train_confidence.py --video-prefix "<clip prefix>"
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths – everything is relative to the yolov5/ project root
# ---------------------------------------------------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # .../yolov5
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEIGHTS  = ROOT / "fish-heatmap-modulation/exp01_yolov5m_bs4_w011/weights/best.pt"
DATA_DIR = Path("/data/vision/beery/scratch/wendy/fish/processed"
                "/extracted_frames_200_heatmap_yolo_separate")
IMG_DIR  = DATA_DIR / "images/train"
LBL_DIR  = DATA_DIR / "labels/train"
HM_DIR   = DATA_DIR / "heatmaps_npy/train"
OUT_DIR  = ROOT / "fish-heatmap-modulation/exp01_yolov5m_bs4_w011/confidence_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ     = 1280
CONF_THRES = 0.01   # low threshold so even weak detections are returned
IOU_THRES  = 0.45
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
TOP_N     = 10
MIN_BOTTOM_CONF = 0.01
MIN_EVENT_HEATMAP_MAX = 0.01
DEFAULT_VIDEO_PREFIX = (
    "3420_FishCam01_20190920T163627.613206Z_1600x1200_"
    "awb-auto_exp-night_fr-10_q-20_sh-0_b-50_c-0_i-400_sat-0"
)

# ---------------------------------------------------------------------------
# Local imports (available after sys.path update)
# ---------------------------------------------------------------------------
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_heatmap_minmax(hm: np.ndarray) -> np.ndarray:
    hm = np.nan_to_num(hm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(hm.min()), float(hm.max())
    if hi > lo:
        return (hm - lo) / (hi - lo)
    elif hi > 0:
        return hm / hi
    return np.zeros_like(hm, dtype=np.float32)


def load_heatmap(img_stem: str) -> np.ndarray:
    hm_path = HM_DIR / f"{img_stem}_heatmap.npy"
    if not hm_path.exists():
        raise FileNotFoundError(f"Heatmap not found: {hm_path}")
    return np.load(str(hm_path)).astype(np.float32)


def load_gt_boxes(img_stem: str, img_w: int, img_h: int):
    """Return list of (x1,y1,x2,y2) in pixel coords from YOLO label file."""
    lbl_path = LBL_DIR / f"{img_stem}.txt"
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = [float(v) for v in parts[:5]]
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append((x1, y1, x2, y2))
    return boxes


def run_inference_one(model, img_path: Path, device, stride, imgsz):
    """Return (detections_xyxy_conf, orig_img_rgb, heatmap_norm)."""
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise IOError(f"Cannot read image: {img_path}")
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    h0, w0 = img0.shape[:2]

    # Load and normalize heatmap
    stem = img_path.stem
    hm_raw = load_heatmap(stem)
    hm_norm = normalize_heatmap_minmax(hm_raw)

    # Resize heatmap to match original image dims if needed
    if hm_norm.shape != (h0, w0):
        hm_norm = cv2.resize(hm_norm, (w0, h0), interpolation=cv2.INTER_LINEAR)

    # Letterbox for model
    img_lb, ratio, (dw, dh) = letterbox(img0, imgsz, stride=stride, auto=True)
    img_t = img_lb.transpose(2, 0, 1)[::-1].copy()  # BGR→RGB→CHW
    img_t = torch.from_numpy(img_t).float().to(device) / 255.0
    img_t = img_t.unsqueeze(0)  # [1,3,H,W]

    # Prepare heatmap tensor: letterbox with same params then to [1,1,H,W]
    hm_resized = cv2.resize(
        hm_norm,
        (img_lb.shape[1], img_lb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    hm_t = torch.from_numpy(hm_resized).float().to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_t, heatmap=hm_t)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]  # take raw output
    print(preds.shape)

    dets = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0]

    results = []
    if dets is not None and len(dets):
        dets_cpu = dets.clone()
        dets_cpu[:, :4] = scale_boxes(img_t.shape[2:], dets_cpu[:, :4], (h0, w0)).round()
        for *xyxy, conf, cls in dets_cpu.tolist():
            results.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(conf)))

    return results, img0_rgb, hm_norm


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_boxes_on_img(img_rgb: np.ndarray, boxes, color=(255, 0, 0),
                      label: str = None, thickness=2):
    """Draw boxes in-place on a copy of img_rgb. boxes = list of (x1,y1,x2,y2[,conf])."""
    out = img_rgb.copy()
    for b in boxes:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label is not None and len(b) > 4:
            txt = f"{label} {b[4]:.2f}"
            cv2.putText(out, txt, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def plot_triplet(img_path: Path, gt_boxes, pred_boxes, hm_norm: np.ndarray,
                 title: str, out_path: Path):
    img0 = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img0.shape[:2]

    # Panel 1: ground truth (green boxes)
    gt_img = draw_boxes_on_img(img0, gt_boxes, color=(0, 220, 0), thickness=3)

    # Panel 2: heatmap overlaid on image
    hm_resized = cv2.resize(hm_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_color = plt.cm.inferno(hm_resized)[:, :, :3]  # drop alpha, RGB [0,1]
    hm_color = (hm_color * 255).astype(np.uint8)
    alpha = 0.55
    hm_img = (alpha * hm_color + (1 - alpha) * img0).astype(np.uint8)

    # Panel 3: predictions (red boxes)
    pred_img = draw_boxes_on_img(img0, pred_boxes, color=(255, 60, 60),
                                  label="fish", thickness=3)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, panel, subtitle in zip(
        axes,
        [gt_img, hm_img, pred_img],
        ["Ground Truth", "Heatmap", "Model Prediction"],
    ):
        ax.imshow(panel)
        ax.set_title(subtitle, fontsize=13)
        ax.axis("off")

    # Legend patches
    gt_patch = mpatches.Patch(color="#00dc00", label=f"GT boxes ({len(gt_boxes)})")
    pred_patch = mpatches.Patch(
        color="#ff3c3c",
        label=f"Pred boxes ({len(pred_boxes)}) "
              f"max conf={max((b[4] for b in pred_boxes), default=0):.3f}",
    )
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center",
               ncol=2, fontsize=10)

    stem = img_path.stem
    # Truncate long stem for display
    display_stem = stem if len(stem) < 60 else "..." + stem[-57:]
    fig.suptitle(f"{title}\n{display_stem}", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze heatmap-modulated training confidence")
    parser.add_argument("--top-n", type=int, default=TOP_N, help="Number of examples to save per group")
    parser.add_argument(
        "--video-prefix",
        type=str,
        default=None,
        help="Optional clip/video prefix to export top-confidence examples for a specific video",
    )
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="When --video-prefix is set, keep only frames whose filename indicates an event frame",
    )
    parser.add_argument(
        "--min-event-heatmap-max",
        type=float,
        default=MIN_EVENT_HEATMAP_MAX,
        help="Minimum normalized heatmap max required for targeted video exports",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = select_device(DEVICE)

    print(f"Loading model from {WEIGHTS} …")
    model = attempt_load(str(WEIGHTS), device=device)
    model.eval()
    stride = int(model.stride.max())
    imgsz = check_img_size(IMGSZ, s=stride)
    print(f"  stride={stride}, imgsz={imgsz}, device={device}")

    # Collect candidate images
    if args.video_prefix:
        img_files = sorted(IMG_DIR.glob(f"{args.video_prefix}__*.jpg"))
        if args.event_only:
            img_files = [p for p in img_files if "_event_" in p.stem]
    else:
        img_files = sorted(IMG_DIR.glob("*.jpg"))
    n_total = len(img_files)
    print(f"Found {n_total} training images. Running inference …")

    scores = []  # (max_conf, img_path, pred_boxes, heatmap_max)
    for idx, img_path in enumerate(img_files):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{idx+1}/{n_total}] {img_path.name[:60]}")
        try:
            preds, _img, hm_norm = run_inference_one(model, img_path, device, stride, imgsz)
            max_conf = max((b[4] for b in preds), default=0.0)
            heatmap_max = float(hm_norm.max()) if hm_norm.size else 0.0
            scores.append((max_conf, img_path, preds, heatmap_max))
        except Exception as e:
            print(f"  WARNING: skipping {img_path.name}: {e}")

    scores.sort(key=lambda x: x[0])
    bottom_candidates = [row for row in scores if row[0] > MIN_BOTTOM_CONF]
    bottom10 = bottom_candidates[:args.top_n]
    top10 = scores[-args.top_n:][::-1]

    print(f"\nTop-10 highest confidence images (max conf):")
    for i, (mc, p, _, hm_max) in enumerate(top10):
        print(f"  {i+1:2d}. conf={mc:.4f} hm_max={hm_max:.4f}  {p.name[:70]}")

    print(f"\nBottom-10 lowest confidence images with max conf > {MIN_BOTTOM_CONF:.2f}:")
    for i, (mc, p, _, hm_max) in enumerate(bottom10):
        print(f"  {i+1:2d}. conf={mc:.4f} hm_max={hm_max:.4f}  {p.name[:70]}")

    # ---- Generate panels -------------------------------------------------
    print("\nGenerating comparison panels …")

    top_dir = OUT_DIR / "top10_highest_conf"
    bot_dir = OUT_DIR / "bottom10_lowest_conf"
    top_dir.mkdir(exist_ok=True)
    bot_dir.mkdir(exist_ok=True)

    print("\n--- Top 10 (highest confidence) ---")
    for rank, (mc, img_path, pred_boxes, hm_max) in enumerate(top10, 1):
        stem = img_path.stem
        img0 = cv2.imread(str(img_path))
        h0, w0 = img0.shape[:2]
        gt_boxes = load_gt_boxes(stem, w0, h0)
        hm_raw = load_heatmap(stem)
        hm_norm = normalize_heatmap_minmax(hm_raw)
        out_path = top_dir / f"rank{rank:02d}_conf{mc:.4f}_hm{hm_max:.4f}.jpg"
        plot_triplet(
            img_path, gt_boxes, pred_boxes, hm_norm,
            title=f"Top-{rank} Highest Confidence  (max conf={mc:.4f}, hm_max={hm_max:.4f})",
            out_path=out_path,
        )

    print(f"\n--- Bottom 10 (lowest confidence, conf > {MIN_BOTTOM_CONF:.2f}) ---")
    for rank, (mc, img_path, pred_boxes, hm_max) in enumerate(bottom10, 1):
        stem = img_path.stem
        img0 = cv2.imread(str(img_path))
        h0, w0 = img0.shape[:2]
        gt_boxes = load_gt_boxes(stem, w0, h0)
        hm_raw = load_heatmap(stem)
        hm_norm = normalize_heatmap_minmax(hm_raw)
        out_path = bot_dir / f"rank{rank:02d}_conf{mc:.4f}_hm{hm_max:.4f}.jpg"
        plot_triplet(
            img_path, gt_boxes, pred_boxes, hm_norm,
            title=f"Bottom-{rank} Lowest Confidence > {MIN_BOTTOM_CONF:.2f}  (max conf={mc:.4f}, hm_max={hm_max:.4f})",
            out_path=out_path,
        )

    if args.video_prefix:
        video_rows = [row for row in scores if row[1].stem.startswith(args.video_prefix)]
        video_rows = [row for row in video_rows if row[3] > args.min_event_heatmap_max]
        video_rows.sort(key=lambda x: x[0], reverse=True)
        video_top = video_rows[:args.top_n]

        target_dir = OUT_DIR / f"video_top_conf_{args.video_prefix[:60]}"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\nTargeted export for video '{args.video_prefix}': "
            f"{len(video_rows)} eligible frames, saving top {len(video_top)}"
        )
        for i, (mc, p, _, hm_max) in enumerate(video_top):
            print(f"  {i+1:2d}. conf={mc:.4f} hm_max={hm_max:.4f}  {p.name[:90]}")

        for rank, (mc, img_path, pred_boxes, hm_max) in enumerate(video_top, 1):
            stem = img_path.stem
            img0 = cv2.imread(str(img_path))
            h0, w0 = img0.shape[:2]
            gt_boxes = load_gt_boxes(stem, w0, h0)
            hm_raw = load_heatmap(stem)
            hm_norm = normalize_heatmap_minmax(hm_raw)
            out_path = target_dir / f"rank{rank:02d}_conf{mc:.4f}_hm{hm_max:.4f}_{stem}.jpg"
            plot_triplet(
                img_path,
                gt_boxes,
                pred_boxes,
                hm_norm,
                title=(
                    f"Video Event Top-{rank}  "
                    f"(max conf={mc:.4f}, hm_max={hm_max:.4f})"
                ),
                out_path=out_path,
            )

    # ---- Summary grid -----------------------------------------------------
    print("\nGenerating summary grids …")

    for group_name, group, group_dir in [
        ("top10_highest_conf", top10, top_dir),
        ("bottom10_lowest_conf", bottom10, bot_dir),
    ]:
        panel_paths = sorted(group_dir.glob("*.jpg"))
        if not panel_paths:
            continue
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in panel_paths]
        fig, axes = plt.subplots(args.top_n, 1, figsize=(24, 8 * args.top_n))
        if args.top_n == 1:
            axes = [axes]
        for ax, im, (mc, img_path, _, hm_max) in zip(axes, imgs, (top10 if "top" in group_name else bottom10)):
            ax.imshow(im)
            ax.set_title(f"conf={mc:.4f} hm_max={hm_max:.4f}  {img_path.stem[:80]}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        grid_path = OUT_DIR / f"{group_name}_grid.jpg"
        fig.savefig(str(grid_path), dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary grid: {grid_path}")

    print(f"\nDone! Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
