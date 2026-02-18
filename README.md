# Deep Learning for Audio-Visual Identification of Underwater Sounds

## Project Overview
Many fish produce species-specific sounds that offer valuable cues for monitoring populations, yet current methods for linking sounds to species remain heavily manual. Recent advances in portable audio-video arrays (Mouy et al., 2023) have enabled sound localization and visual matching, but practical deployments often rely on single-channel recordings where traditional array-based localization is unavailable.

This project investigates whether audio-visual self-supervised learning can bridge this gap by combining mono underwater audio with synchronized 360-degree video, using hydrophone-array-derived direction-of-arrival (DOA) estimates as supervisory "teacher" signals. Our core goal is to train "student" models that, given only mono audio and video, can detect fish sounds, predict their DOA, and classify species-specific calls.

By comparing audio-only, video-only, and fused audio-visual approaches, we test whether visual motion and context provide complementary information for sound event detection and localization, particularly in noisy or murky underwater environments. This teacher-student framework allows us to generate high-quality supervision from arrays while enabling models deployable with low-cost, single-channel hydrophones and off-the-shelf 360-degree cameras.

Outcomes include:
- Benchmarks of AV models versus audio-only baselines.
- Quantitative evaluation of how video improves localization and classification accuracy.
- Visualizations such as DOA heatmaps and synchronized video overlays of predicted fish sound sources.

The broader significance lies in reducing manual annotation effort, improving detection in challenging conditions, and building a scalable framework for passive acoustic monitoring of fish communities, expanding the reach of bioacoustics as a tool for marine ecology and conservation.

## Demo
Demo video:

![Localization demo](demo.gif)

## Project Structure
```text
fish/
├── README.md
├── XAV-arrays/localization/
│   ├── batch_processing.py
│   ├── detection.py
│   ├── plot_spectrogram_detections.py
│   ├── heatmap_overlay_errors.py
│   └── large-array/               # localization config files
├── preprocessing/
│   ├── sync_audio_to_video.py
│   ├── split_synced_pairs_wav_chunks.py
│   ├── find_clips.py
│   ├── get_frames.py
│   └── mouy_dataset_stats.py
├── embedding_retrieval_scripts/
│   ├── retrieval_results.py
│   ├── sanity_checks.py
│   └── *.ipynb
└── heatmap_feature_modulation.ipynb
```

## How to Run
From repo root:

```bash
cd /Users/wendycao/fish
```

Install core Python dependencies (adjust for your environment):

```bash
pip install numpy pandas matplotlib scipy opencv-python ecosound
```

### 1) Run localization batch processing
This reads synced pairs from `synced_pairs/` and writes outputs to
`XAV-arrays/localization/out_synced_pairs/`.

```bash
python XAV-arrays/localization/batch_processing.py
```

Optional speed/debug knobs:

```bash
SYNCED_ONLY=<folder_name> MAX_FOLDERS=1 MAX_FILES_PER_FOLDER=3 python XAV-arrays/localization/batch_processing.py
```

### 2) Plot detections on spectrogram
```bash
python XAV-arrays/localization/plot_spectrogram_detections.py \
  --audio /path/to/input.wav \
  --csv /path/to/localization_results.csv \
  --out /path/to/overlay.png
```

### 3) Compute dataset summary metrics
```bash
python preprocessing/mouy_dataset_stats.py \
  --root /Users/wendycao/fish/2019-09-15_HornbyIsland_AMAR_07-HI \
  --out-dir /Users/wendycao/fish/processed/07HI_stats \
  --synced-pairs /Users/wendycao/fish/synced_pairs
```

### 4) Evaluate retrieval results
```bash
python embedding_retrieval_scripts/retrieval_results.py \
  --metadata-csv /path/to/metadata.csv \
  --audio-embeddings-npy /path/to/audio_embeddings.npy \
  --video-embeddings-npy /path/to/video_embeddings.npy \
  --output-dir /path/to/results
```

## TODO
- Migrate `megafishdetector` from YOLOv5 to current Ultralytics YOLO (YOLO26) training/inference commands.
- Reproduce fish detector training with tracked dataset versions.
- Add an end-to-end script to run preprocessing, training, evaluation, and export in one workflow.
