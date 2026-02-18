#!/usr/bin/env python3

from pathlib import Path
import csv
import cv2
import re

# ----- CONFIG -----
ROOT = Path("/data/vision/beery/scratch/wendy/fisheye")
OUTPUT_CSV = Path("sound_centers_manual_multi.csv")

# matches ss:ff-ss:ff, e.g., 08:00-08:11
RANGE_PATTERN = re.compile(r"(\d{2}):(\d{2})-(\d{2}):(\d{2})")


def species_from_path(video_path: Path) -> str:
    """Species = top-level subfolder under ROOT."""
    rel = video_path.relative_to(ROOT)
    return rel.parts[0] if len(rel.parts) > 0 else "unknown"


def cropped_name_for(video_path: Path) -> str:
    """Turn X.mp4 into X_cropped.mp4"""
    return f"{video_path.stem}_cropped{video_path.suffix}"


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0  # fallback if missing


def parse_ranges(input_str: str, fps: float):
    """
    Parse one or more ranges from a string like:
      '08:00-08:11, 12:05-12:10'
    Returns list of (start_sec, end_sec, center_sec)
    """
    matches = RANGE_PATTERN.findall(input_str)
    if not matches:
        raise ValueError("No valid ranges found. Use ss:ff-ss:ff (e.g. 08:00-08:11)")

    centers = []
    for s_s, s_f, e_s, e_f in matches:
        s_s, s_f, e_s, e_f = map(int, (s_s, s_f, e_s, e_f))
        start_sec = s_s + s_f / fps
        end_sec = e_s + e_f / fps
        center = (start_sec + end_sec) / 2.0
        centers.append((start_sec, end_sec, center))

    return centers


def main():
    mp4_files = sorted(
        p for p in ROOT.rglob("*.mp4") if not p.name.endswith("_cropped.mp4")
    )

    print(f"Found {len(mp4_files)} original videos under {ROOT}\n")

    rows = []

    for vid in mp4_files:
        print("--------------------------------------------------")
        print(f"Video:   {vid.name}")
        species = species_from_path(vid)
        fps = get_fps(vid)
        cropped_name = cropped_name_for(vid)

        print(f"Species: {species}")
        print(f"FPS:     {fps:.2f}")
        print("If this video has multiple sounds, enter ALL ranges,")
        print("separated by commas or spaces, e.g.: 08:00-08:11, 12:05-12:10")
        print("Leave blank and press Enter to skip this video.")

        rng = input("Ranges (ss:ff-ss:ff ...): ").strip()
        if not rng:
            print("  -> Skipping.\n")
            continue

        try:
            centers = parse_ranges(rng, fps)
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            print("  -> Skipping this video.\n")
            continue

        for idx, (_start, _end, center_sec) in enumerate(centers, start=1):
            rows.append([cropped_name, species, idx, center_sec])
            print(f"  Event {idx}: center = {center_sec:.4f} sec")

        print()

    # Save CSV
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cropped_video_filename", "species_name", "event_index", "center_time_sec"]
        )
        writer.writerows(rows)

    print(f"\nDone! CSV saved to: {OUTPUT_CSV.resolve()}")
    print(f"Total events: {len(rows)}")


if __name__ == "__main__":
    main()
