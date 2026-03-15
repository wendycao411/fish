#!/usr/bin/env python3
# Ultralytics YOLOv5 heatmap-guided ROI detection (no retraining required).

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

# Support both script execution and notebook execution (no __file__ in notebooks).
if "__file__" in globals():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent
else:
    ROOT = (Path.cwd() / "megafishdetector" / "yolov5").resolve()
    FILE = ROOT / "detect_roi_from_heatmap.py"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode


def normalize_heatmap(hm, mode="minmax"):
    # Defensive normalization so downstream ROI thresholding is stable.
    hm = np.nan_to_num(hm.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return hm
    if mode == "standardize":
        mean, std = float(hm.mean()), float(hm.std())
        if std > 1e-6:
            hm = np.clip((hm - mean) / std, -3.0, 3.0)
            hm = (hm + 3.0) / 6.0
        else:
            hm = np.zeros_like(hm, dtype=np.float32)
        return hm
    hm_min, hm_max = float(hm.min()), float(hm.max())
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    elif hm_max > 0:
        hm = hm / hm_max
    else:
        hm = np.zeros_like(hm, dtype=np.float32)
    return hm


def load_heatmap_for_frame(heatmaps, frame_idx):
    # Supports either one static map [H,W] or per-frame maps [T,H,W].
    if heatmaps.ndim == 2:
        return np.array(heatmaps, dtype=np.float32, copy=False)
    if frame_idx < 0 or frame_idx >= heatmaps.shape[0]:
        return None
    return np.array(heatmaps[frame_idx], dtype=np.float32, copy=False)


def roi_boxes_from_heatmap(hm_norm, percentile=92.0, min_area=400, pad=24, max_rois=6):
    # Convert continuous heatmap to binary mask using percentile threshold.
    h, w = hm_norm.shape
    thr = np.percentile(hm_norm, percentile)
    mask = (hm_norm >= thr).astype(np.uint8) * 255
    # Morphology reduces speckle noise before component extraction.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Connected components become candidate ROIs.
    rois = []
    for idx in range(1, n):
        x, y, bw, bh, area = stats[idx]
        if area < min_area:
            continue
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        score = float(hm_norm[y:y + bh, x:x + bw].mean()) if bw > 0 and bh > 0 else 0.0
        rois.append((x1, y1, x2, y2, score))

    # Rank ROIs by average heat value and keep top-K.
    rois.sort(key=lambda t: t[4], reverse=True)
    return rois[:max_rois]


def infer_patch(model, patch_bgr, device, imgsz, conf_thres, iou_thres, max_det, classes, agnostic_nms, half):
    # Run standard YOLOv5 detect on one cropped ROI patch.
    patch_lb = letterbox(patch_bgr, imgsz, stride=int(model.stride.max()), auto=True)[0]
    im = patch_lb.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im_t = torch.from_numpy(im).to(device)
    im_t = im_t.half() if (half and device.type != "cpu") else im_t.float()
    im_t /= 255.0
    im_t = im_t.unsqueeze(0)

    pred = model(im_t)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    if not len(pred):
        return pred

    # Map boxes from letterboxed tensor space back to original patch pixels.
    pred[:, :4] = scale_boxes(im_t.shape[2:], pred[:, :4], patch_bgr.shape).round()
    return pred


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source="",
    heatmap_npy="",
    imgsz=(1280, 1280),
    conf_thres=0.25,
    iou_thres=0.45,
    roi_iou_thres=0.5,
    max_det=1000,
    device="",
    half=False,
    classes=None,
    agnostic_nms=False,
    percentile=92.0,
    min_area=400,
    roi_pad=24,
    max_rois=6,
    fallback_full_frame=True,
    project=ROOT / "runs/detect",
    name="exp_roi_heatmap",
    exist_ok=False,
    vid_stride=1,
    min_heatmap_max=0.0,
):
    source = str(source)
    assert source, "--source is required"
    assert heatmap_npy, "--heatmap-npy is required"
    is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS + VID_FORMATS)
    assert is_file, "This script supports image/video file input."

    heatmaps = np.load(heatmap_npy, mmap_mode="r")
    if heatmaps.ndim not in (2, 3):
        raise ValueError(f"Unsupported heatmap array shape {heatmaps.shape}; expected [H,W] or [T,H,W]")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(1000)}
    imgsz = check_img_size(imgsz, s=stride)
    model.half() if (half and device.type != "cpu") else model.float()
    warm = torch.zeros((1, 3, *imgsz), device=device)
    warm = warm.half() if (half and device.type != "cpu") else warm.float()
    _ = model(warm)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, vid_stride=vid_stride)
    vid_path, vid_writer = None, None
    seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

    sample_idx = 0
    for path, _, im0, vid_cap, s in dataset:
        # Align heatmap index with current sample/frame index.
        if dataset.mode == "video":
            frame_idx = max(int(getattr(dataset, "frame", 1)) - 1, 0)  # LoadImages.frame is 1-based
        else:
            frame_idx = sample_idx
        sample_idx += 1
        hm = load_heatmap_for_frame(heatmaps, frame_idx)
        if hm is None:
            LOGGER.warning(f"Frame {frame_idx}: no heatmap; skipping frame.")
            continue
        if hm.shape != im0.shape[:2]:
            hm = cv2.resize(hm, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_LINEAR)
        hm_norm = normalize_heatmap(hm, mode="minmax")
        hm_max = float(hm_norm.max())
        if hm_max < float(min_heatmap_max):
            continue

        with dt[0]:
            # Step 1: derive heatmap-guided ROIs.
            rois = roi_boxes_from_heatmap(
                hm_norm, percentile=percentile, min_area=min_area, pad=roi_pad, max_rois=max_rois
            )
            if fallback_full_frame and not rois:
                rois = [(0, 0, im0.shape[1], im0.shape[0], 0.0)]

        all_det = []
        with dt[1]:
            # Step 2: detect inside each ROI and lift boxes back to full-frame coords.
            for x1, y1, x2, y2, _ in rois:
                patch = im0[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                det_patch = infer_patch(
                    model,
                    patch,
                    device,
                    imgsz,
                    conf_thres,
                    iou_thres,
                    max_det,
                    classes,
                    agnostic_nms,
                    half,
                )
                if len(det_patch):
                    det_patch[:, [0, 2]] += x1
                    det_patch[:, [1, 3]] += y1
                    all_det.append(det_patch)

        with dt[2]:
            # Step 3: merge ROI-level detections with global NMS.
            if all_det:
                det = torch.cat(all_det, dim=0)
                keep = torchvision.ops.nms(det[:, :4], det[:, 4], roi_iou_thres)
                det = det[keep]
                if det.shape[0] > max_det:
                    det = det[:max_det]
            else:
                det = torch.zeros((0, 6), device=device)

        seen += 1
        p = Path(path)
        frame = frame_idx
        annotator = Annotator(im0, line_width=3, example=str(names))

        if len(det):
            for c in det[:, 5].unique():
                n = int((det[:, 5] == c).sum())
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in det:
                c = int(cls)
                label = f"{names[c]} {float(conf):.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

        out = annotator.result()
        save_path = str(save_dir / p.name)
        if dataset.mode == "image":
            cv2.imwrite(save_path, out)
        else:
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else out.shape[1]
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else out.shape[0]
                save_path = str(Path(save_path).with_suffix(".mp4"))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            vid_writer.write(out)

        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}hm_max={hm_max:.3f} rois={len(rois)} "
            f"{dt[1].dt * 1e3:.1f}ms"
        )

    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    t = tuple(x.t / max(seen, 1) * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms roi, %.1fms detect, %.1fms merge per image" % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, required=True, help="RGB detector checkpoint (.pt)")
    parser.add_argument("--source", type=str, required=True, help="video/image source path")
    parser.add_argument("--heatmap-npy", type=str, required=True, help="heatmap stack path [.npy], shape [T,H,W]")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[1280], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold per ROI")
    parser.add_argument("--roi-iou-thres", type=float, default=0.5, help="global merge NMS IoU across ROIs")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per frame")
    parser.add_argument("--device", default="", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 inference")
    parser.add_argument("--classes", nargs="+", type=int, help="filter classes by id")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--percentile", type=float, default=92.0, help="heatmap percentile threshold for ROI mask")
    parser.add_argument("--min-area", type=int, default=400, help="minimum connected-component area for ROI")
    parser.add_argument("--roi-pad", type=int, default=24, help="padding pixels around each ROI")
    parser.add_argument("--max-rois", type=int, default=6, help="maximum ROIs per frame")
    parser.add_argument("--fallback-full-frame", action="store_true", help="run full-frame detect when no ROI found")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp_roi_heatmap", help="save results under project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame stride")
    parser.add_argument("--min-heatmap-max", type=float, default=0.0, help="skip frames with heatmap max below this")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
