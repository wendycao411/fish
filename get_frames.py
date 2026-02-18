#!/usr/bin/env python3

from pathlib import Path
import cv2

# -------- CONFIG --------
# Root folder containing your 5-second VIDEO clips
# Example (adjust as needed):
#   /Users/wendycao/fish/fisheye/5_second_video_clips_test
CLIP_ROOT = Path("/Users/wendycao/fish/fisheye/5_second_video_clips_test")

# Root folder where you want to save the extracted FRAMES
# The script will mirror the structure of CLIP_ROOT under here.
FRAMES_ROOT = Path("/Users/wendycao/fish/fisheye/5_second_frames_test")
# ------------------------


def extract_frames_for_clip(clip_path: Path, out_dir: Path):
    """Extract all frames from a video clip into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"[!] Could not open video: {clip_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # frame_0001.jpg, frame_0002.jpg, ...
        frame_name = f"frame_{frame_idx:04d}.jpg"
        frame_path = out_dir / frame_name

        # Write as JPEG
        success = cv2.imwrite(str(frame_path), frame)
        if not success:
            print(f"[!] Failed to write frame {frame_idx} for {clip_path}")

    cap.release()
    print(f"Extracted {frame_idx} frames from {clip_path} -> {out_dir}")


def main():
    if not CLIP_ROOT.exists():
        print(f"[!] Clip root does not exist: {CLIP_ROOT}")
        return

    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

    # Find all .mp4 files under CLIP_ROOT (recursively)
    clip_files = sorted(CLIP_ROOT.rglob("*.mp4"))
    print(f"Found {len(clip_files)} clips under {CLIP_ROOT}")

    for clip in clip_files:
        # Mirror the relative path under FRAMES_ROOT
        rel = clip.relative_to(CLIP_ROOT)          # e.g. species/clipname.mp4
        out_dir = FRAMES_ROOT / rel.with_suffix("")  # drop .mp4 -> folder name

        print("-------------------------------------------------")
        print(f"Clip:   {clip}")
        print(f"Frames: {out_dir}")
        extract_frames_for_clip(clip, out_dir)

    print("\nDone extracting all frames.")


if __name__ == "__main__":
    main()
