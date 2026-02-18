#!/usr/bin/env python3

import csv
import subprocess
from pathlib import Path

# Root where your species folders + cropped videos live
ROOT = Path("/Users/wendycao/fish/fisheye")

# Your existing CSV with timestamps (test run is fine)
CSV_PATH = Path("sound_centers_manual.csv")

# Output dirs for 5-second clips
OUT_VIDEO_DIR = ROOT / "5_second_video_clips_test"
OUT_AUDIO_DIR = ROOT / "5_second_audio_clips_test"

OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def find_cropped_video(species_name: str, cropped_filename: str) -> Path | None:
    """
    Find the path to the cropped video file.
    Assumes structure like: ROOT/species_name/**/cropped_filename
    """
    species_root = ROOT / species_name
    if not species_root.exists():
        return None

    matches = list(species_root.rglob(cropped_filename))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"[!] Multiple matches for {cropped_filename} under {species_name}, using first.")
    return matches[0]


def extract_video_clip(input_path: Path, center_time: float, out_path: Path, clip_len: float = 5.0):
    """
    Use ffmpeg to extract a clip of length `clip_len` seconds centered at `center_time`.
    Video + audio together.
    """
    start_time = max(0.0, center_time - clip_len / 2.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(input_path),
        "-t", f"{clip_len:.3f}",
        "-c", "copy",        # fast copy, no re-encode
        str(out_path),
    ]

    print("  [video] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio_clip(input_path: Path, center_time: float, out_path: Path, clip_len: float = 5.0):
    """
    Use ffmpeg to extract an audio-only 5s clip centered at `center_time`.
    Outputs WAV (good for ML). Change to .mp3 if you prefer.
    """
    start_time = max(0.0, center_time - clip_len / 2.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(input_path),
        "-t", f"{clip_len:.3f}",
        "-vn",                # no video
        "-acodec", "pcm_s16le",  # WAV, 16-bit PCM
        "-ar", "44100",       # sample rate
        str(out_path),
    ]

    print("  [audio] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if not CSV_PATH.exists():
        print(f"[!] CSV file not found: {CSV_PATH}")
        return

    with CSV_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} events in {CSV_PATH}")

    for row in rows:
        cropped_filename = row["cropped_video_filename"]
        species_name = row["species_name"]
        center_time_sec = float(row["center_time_sec"])
        event_index = row.get("event_index", "1")

        print("--------------------------------------------------")
        print(f"Species:   {species_name}")
        print(f"Video:     {cropped_filename}")
        print(f"Event idx: {event_index}")
        print(f"Center:    {center_time_sec:.4f} sec")

        video_path = find_cropped_video(species_name, cropped_filename)
        if video_path is None:
            print(f"  [!] Could not find cropped video file for {cropped_filename}")
            continue

        # Build descriptive base name
        stem = video_path.stem.replace(" ", "_")
        species_clean = species_name.replace(" ", "_")
        center_str = f"{center_time_sec:.2f}".replace(".", "p")  # 8.18 -> 8p18

        base_name = f"{species_clean}__{stem}__ev{event_index}__center{center_str}s"

        out_video = OUT_VIDEO_DIR / f"{base_name}.mp4"
        out_audio = OUT_AUDIO_DIR / f"{base_name}.wav"

        try:
            extract_video_clip(video_path, center_time_sec, out_video, clip_len=5.0)
            extract_audio_clip(video_path, center_time_sec, out_audio, clip_len=5.0)
            print(f"  -> Saved video: {out_video}")
            print(f"  -> Saved audio: {out_audio}")
        except subprocess.CalledProcessError as e:
            print(f"  [!] ffmpeg failed for {video_path}: {e}")

    print("\nDone!")
    print("5-second VIDEO clips in:", OUT_VIDEO_DIR)
    print("5-second AUDIO clips in:", OUT_AUDIO_DIR)


if __name__ == "__main__":
    main()
