#!/usr/bin/env python3
"""
Create a YOLO-style dataset from taylor_islet_localization_windows frames.

Filters frames to:
1. Only include those with labels
2. Only include video IDs 1001-1594
3. Consolidates frames and labels into a clean YOLO structure
"""

import os
import shutil
import re
from collections import defaultdict

# Configuration
SOURCE_FRAMES_DIR = "/Users/wendycao/fish/data/darienne/taylor_islet_localization_windows/cvat_batches"
LABEL_SEARCH_DIRS = [
    "/Users/wendycao/fish/taylor_labels/batch1/labels/train",
    "/Users/wendycao/fish/taylor_labels/batch2/labels/train",
    "/Users/wendycao/fish/taylor_labels/batch3/labels/train",
    "/Users/wendycao/fish/taylor_labels/batch4/labels/train",
    "/Users/wendycao/fish/taylor_labels/batch_001/labels/train",
]
OUTPUT_DIR = "/Users/wendycao/fish/taylor_yolo_dataset_videos_1001_1594"
MIN_VIDEO_ID = 1001
MAX_VIDEO_ID = 1594

def extract_video_id(filename):
    """Extract video ID from filename like: 1190_FishCam03_..."""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def get_base_name_without_ext(filename):
    """Get filename without extension."""
    return os.path.splitext(filename)[0]

def main():
    print("=" * 80)
    print("Creating YOLO Dataset from Taylor Islet Localization Windows")
    print("=" * 80)
    
    # Step 1: Index all labels
    print("\n[1/4] Indexing label files...")
    label_index = {}
    total_labels = 0
    
    for label_dir in LABEL_SEARCH_DIRS:
        if not os.path.exists(label_dir):
            print(f"  ⚠ Label directory not found: {label_dir}")
            continue
        
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                base_name = get_base_name_without_ext(label_file)
                label_index[base_name] = os.path.join(label_dir, label_file)
                total_labels += 1
    
    print(f"  ✓ Found {total_labels} label files")
    
    # Step 2: Index source images and filter by label + video ID
    print(f"\n[2/4] Indexing source images and filtering (videos {MIN_VIDEO_ID}-{MAX_VIDEO_ID})...")
    frames_to_process = []
    filtered_by_video = 0
    missing_image_for_label = 0
    image_index = {}
    
    print("  Scanning source images recursively...")
    for root, _, files in os.walk(SOURCE_FRAMES_DIR):
        for frame_file in files:
            if frame_file.endswith('.jpg'):
                base_name = get_base_name_without_ext(frame_file)
                image_index[base_name] = os.path.join(root, frame_file)

    print(f"  ✓ Indexed {len(image_index)} source images")

    for base_name, label_path in label_index.items():
        video_id = extract_video_id(base_name)
        if video_id is None:
            continue

        if video_id < MIN_VIDEO_ID or video_id > MAX_VIDEO_ID:
            filtered_by_video += 1
            continue

        source_path = image_index.get(base_name)
        if source_path is None:
            missing_image_for_label += 1
            continue

        frames_to_process.append({
            'source_path': source_path,
            'base_name': base_name,
            'label_path': label_path,
            'video_id': video_id
        })

    print(f"  ✓ Filtered out {filtered_by_video} labels outside video range {MIN_VIDEO_ID}-{MAX_VIDEO_ID}")
    print(f"  ✓ Missing source image for {missing_image_for_label} labels")
    print(f"  ✓ Total frames to process: {len(frames_to_process)}")
    
    if len(frames_to_process) == 0:
        print("\n❌ No labeled frames found! Exiting.")
        return
    
    # Step 3: Create output directory structure
    print("\n[3/4] Creating output directory structure...")
    images_dir = os.path.join(OUTPUT_DIR, "images", "train")
    labels_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    print(f"  ✓ Created {OUTPUT_DIR}")
    
    # Step 4: Copy frames and labels
    print("\n[4/4] Copying frames and labels...")
    copied_count = 0
    
    for i, item in enumerate(frames_to_process, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(frames_to_process)}")
        
        # Copy frame
        dest_image = os.path.join(images_dir, item['base_name'] + '.jpg')
        shutil.copy2(item['source_path'], dest_image)
        
        # Copy label
        dest_label = os.path.join(labels_dir, item['base_name'] + '.txt')
        shutil.copy2(item['label_path'], dest_label)
        
        copied_count += 1
    
    print(f"  ✓ Copied {copied_count} image-label pairs")
    
    # Step 5: Generate data.yaml
    print("\n[5/5] Generating dataset configuration...")
    data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    data_yaml_content = """names:
  0: fish
path: .
train: train.txt
"""
    
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)
    print(f"  ✓ Created {data_yaml_path}")
    
    # Step 6: Generate train.txt
    train_txt_path = os.path.join(OUTPUT_DIR, "train.txt")
    with open(train_txt_path, 'w') as f:
        for item in sorted(frames_to_process, key=lambda x: x['base_name']):
            # Use relative path format
            rel_path = os.path.join("images", "train", item['base_name'] + ".jpg")
            f.write(rel_path + "\n")
    
    print(f"  ✓ Created {train_txt_path} with {len(frames_to_process)} entries")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("✅ YOLO Dataset Creation Complete!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total labeled frames: {len(frames_to_process)}")
    print(f"Video ID range: {MIN_VIDEO_ID}-{MAX_VIDEO_ID}")
    print(f"Image directory: {images_dir}")
    print(f"Label directory: {labels_dir}")
    print(f"Configuration: {data_yaml_path}")
    print(f"Train list: {train_txt_path}")
    
    # Statistics by video ID
    print("\nFrames per video ID (sample):")
    video_counts = defaultdict(int)
    for item in frames_to_process:
        video_counts[item['video_id']] += 1
    
    videos = sorted(video_counts.keys())
    print(f"  First video ID: {videos[0]} ({video_counts[videos[0]]} frames)")
    print(f"  Last video ID: {videos[-1]} ({video_counts[videos[-1]]} frames)")
    print(f"  Total video IDs: {len(video_counts)}")

if __name__ == "__main__":
    main()
