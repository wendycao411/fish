# MegaFishDetector with modern Ultralytics YOLO

This is an additive pipeline that keeps the original YOLOv5 code intact and adds a modern Ultralytics workflow (`ultralytics` package, YOLO11/YOLO26 checkpoints).

## What changed vs original

- Original: YOLOv5 training/inference flow under `megafishdetector/yolov5/`
- New: Ultralytics YOLO flow under `megafishdetector/ultralytics_yolo/`
- New unified dataset builder that:
  - consumes per-dataset outputs from `dataset_utils` converters
  - preserves existing train/val/test splits when present
  - uses deterministic YOLOv5-style autosplit fallback (`seed=0`, `random.choices`) when split folders are absent
  - collapses all classes to single-class `fish=0`
  - verifies YOLO normalization and writes empty label files when no fish are present

Datasets are not redistributed. You must download datasets directly from their original sources and respect each source license/terms.

## Files added

- `data/megafish.yaml`
- `scripts/prepare_megafish.py`
- `scripts/train.py`
- `scripts/val.py`
- `scripts/predict_video.py`
- `requirements_ultralytics.txt`

## 1) Clone repo

```bash
git clone https://github.com/warplab/megafishdetector.git
cd megafishdetector
```

## 2) Install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ultralytics_yolo/requirements_ultralytics.txt
```

## 3) Prepare data

### Option A (recommended): use existing `dataset_utils` converters first

Run each converter as in `dataset_utils/README.md` to create processed datasets under:

- `datasets/processed/deepfish`
- `datasets/processed/fathomnet`
- `datasets/processed/viame_fishtrack`
- `datasets/processed/noaa_estuary`
- `datasets/processed/lfitw`
- `datasets/processed/aims_ozfish`

Then unify for Ultralytics:

```bash
python ultralytics_yolo/scripts/prepare_megafish.py \
  --processed-root datasets/processed \
  --output-dir datasets/megafish_ultra \
  --datasets deepfish,fathomnet,viame_fishtrack,noaa_estuary,lfitw,aims_ozfish \
  --sanity-check --sanity-samples 32
```

### Option B: ask prep script to run converters when possible

```bash
python ultralytics_yolo/scripts/prepare_megafish.py \
  --raw-root datasets/raw \
  --processed-root datasets/processed \
  --run-converters \
  --download \
  --sanity-check
```

If a dataset is missing, the script prints clear next-step instructions and continues with available datasets.

### Output format

`datasets/megafish_ultra/`:

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`

YOLO label format per line:

```text
0 x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

## 4) Train (Ultralytics YOLO)

Default model is `yolo11m.pt` (replace with `yolo26m.pt` if desired).

```bash
python ultralytics_yolo/scripts/train.py \
  --model yolo11m.pt \
  --data ultralytics_yolo/data/megafish.yaml \
  --imgsz 1280 \
  --epochs 100 \
  --batch auto \
  --device 0 \
  --project runs/ultralytics_megafish \
  --name train_yolo11m
```

## 5) Validate

```bash
python ultralytics_yolo/scripts/val.py \
  --weights runs/ultralytics_megafish/train_yolo11m/weights/best.pt \
  --data ultralytics_yolo/data/megafish.yaml \
  --imgsz 1280 \
  --batch auto \
  --device 0 \
  --split val \
  --project runs/ultralytics_megafish \
  --name val_yolo11m
```

`metrics.json` is saved inside the Ultralytics validation run dir.

## 6) Inference on video (and image folder)

Video inference:

```bash
python ultralytics_yolo/scripts/predict_video.py \
  --weights runs/ultralytics_megafish/train_yolo11m/weights/best.pt \
  --source /path/to/input.mp4 \
  --output runs/ultralytics_megafish/predict/input_annotated.mp4 \
  --csv runs/ultralytics_megafish/predict/input_detections.csv \
  --conf 0.25 --iou 0.45 --device 0
```

Image-folder inference (optional):

```bash
python ultralytics_yolo/scripts/predict_video.py \
  --weights runs/ultralytics_megafish/train_yolo11m/weights/best.pt \
  --source /path/to/images_folder \
  --output runs/ultralytics_megafish/predict/images_annotated \
  --csv runs/ultralytics_megafish/predict/images_detections.csv \
  --conf 0.25 --iou 0.45 --device 0
```

CSV columns:

- `frame_idx`
- `time_sec`
- `x1,y1,x2,y2`
- `conf`

## Notes on reproducibility

- Existing split folders from source datasets are preserved.
- Datasets without predefined split use deterministic YOLOv5-style autosplit fallback.
- Class collapse is deterministic: all classes are remapped to `0`.
- Sanity mode writes overlay images to `datasets/megafish_ultra/out/sanity/` and prints bbox min/max stats.
