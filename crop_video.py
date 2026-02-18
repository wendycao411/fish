#!/usr/bin/env python3
import subprocess
from pathlib import Path

# Root directory containing all the species subfolders
ROOT = Path("/Users/wendycao/Downloads/doi_10_5061_dryad_mcvdnckcp__v20250819")

# Final crop we measured: 1920x800 starting at y=60
CROP_FILTER = "crop=1920:800:0:60"


def process_video(mp4_path: Path):
    """Crop video and extract audio for a single .mp4 file."""
    dir_path = mp4_path.parent
    base = mp4_path.stem

    cropped_video = dir_path / f"{base}_cropped.mp4"
    audio_file = dir_path / f"{base}.mp3"

    print(f"\nProcessing: {mp4_path}")

    # Skip if outputs already exist (optional—remove if you want to overwrite)
    if cropped_video.exists():
        print(f"  Skipping cropped video (already exists): {cropped_video}")
    else:
        # 1. Crop video (keeps original audio track)
        cmd_crop = [
            "ffmpeg", "-y",
            "-i", str(mp4_path),
            "-vf", CROP_FILTER,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "copy",
            str(cropped_video),
        ]
        print("  Cropping video...")
        subprocess.run(cmd_crop, check=True)
        print(f"  -> Cropped video: {cropped_video}")

    if audio_file.exists():
        print(f"  Skipping audio (already exists): {audio_file}")
    else:
        # 2. Extract audio as MP3 from original video
        cmd_audio = [
            "ffmpeg", "-y",
            "-i", str(mp4_path),
            "-vn",                   # no video
            "-acodec", "libmp3lame",
            "-q:a", "2",             # quality 2 ≈ high quality
            str(audio_file),
        ]
        print("  Extracting audio...")
        subprocess.run(cmd_audio, check=True)
        print(f"  -> Audio mp3: {audio_file}")


def main():
    mp4_files = list(ROOT.rglob("*.mp4"))
    print(f"Found {len(mp4_files)} .mp4 files under {ROOT}")

    for mp4_path in mp4_files:
        try:
            process_video(mp4_path)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR processing {mp4_path}: {e}")


if __name__ == "__main__":
    main()
