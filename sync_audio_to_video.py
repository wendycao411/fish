#!/usr/bin/env python3
"""Trim audio clips in synced_pairs to match video start time and duration."""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SYNCED_DIR = Path("synced_pairs")
VIDEO_DIR = Path("2019-09-15_HornbyIsland_AMAR_07-HI")

VIDEO_TS_RE = re.compile(r"_(\d{8}T\d{6}(?:\.\d+)?)Z")
AUDIO_TS_RE = re.compile(r"\.(\d{8}T\d{6})Z\.wav$")


def parse_video_start_ts(name: str) -> dt.datetime:
    match = VIDEO_TS_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse video timestamp from {name}")
    ts = match.group(1)
    if "." in ts:
        base, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{base}.{frac}"
        fmt = "%Y%m%dT%H%M%S.%f"
    else:
        fmt = "%Y%m%dT%H%M%S"
    return dt.datetime.strptime(ts, fmt).replace(tzinfo=dt.timezone.utc)


def parse_audio_start_ts(name: str) -> dt.datetime:
    match = AUDIO_TS_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse audio timestamp from {name}")
    ts = match.group(1)
    return dt.datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def get_video_duration_seconds(path: Path) -> float:
    # mdls requires Spotlight metadata; run it directly and parse raw output.
    result = subprocess.run(
        ["mdls", "-name", "kMDItemDurationSeconds", "-raw", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Unexpected mdls duration output for {path}: {value!r}") from exc


def trim_wav(path: Path, start_offset_s: float, duration_s: float) -> None:
    if start_offset_s < 0:
        raise ValueError(f"Negative start offset for {path}: {start_offset_s}")
    if duration_s <= 0:
        raise ValueError(f"Non-positive duration for {path}: {duration_s}")

    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frame_rate = params.framerate
        total_frames = params.nframes

        start_frame = int(round(start_offset_s * frame_rate))
        end_frame = int(round((start_offset_s + duration_s) * frame_rate))
        end_frame = min(end_frame, total_frames)

        if start_frame >= total_frames:
            raise ValueError(
                f"Start frame {start_frame} beyond total frames {total_frames} for {path}"
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"End frame {end_frame} not after start frame {start_frame} for {path}"
            )

        src.setpos(start_frame)
        frames_to_copy = end_frame - start_frame
        frames = src.readframes(frames_to_copy)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with wave.open(str(tmp_path), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(frames)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> int:
    if not SYNCED_DIR.exists():
        print(f"Missing synced directory: {SYNCED_DIR}", file=sys.stderr)
        return 1
    if not VIDEO_DIR.exists():
        print(f"Missing video directory: {VIDEO_DIR}", file=sys.stderr)
        return 1

    folders = sorted(p for p in SYNCED_DIR.iterdir() if p.is_dir())
    if not folders:
        print("No synced folders found.")
        return 0

    for folder in folders:
        video_name = f"{folder.name}.mp4"
        video_src = VIDEO_DIR / video_name
        if not video_src.exists():
            print(f"[skip] No matching video for {folder.name}")
            continue

        video_dst = folder / video_name
        if video_dst.is_symlink():
            video_dst.unlink()
        if not video_dst.exists():
            shutil.copy2(video_src, video_dst)
            print(f"[copy] {video_name} -> {folder}")
        else:
            print(f"[have] {video_dst.name}")

        video_start = parse_video_start_ts(video_name)
        video_duration = get_video_duration_seconds(video_src)

        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            print(f"[warn] No wav files in {folder}")
            continue

        audio_start = parse_audio_start_ts(wavs[0].name)
        offset = (video_start - audio_start).total_seconds()

        if offset < 0:
            print(
                f"[warn] Video starts before audio by {offset:.3f}s in {folder.name}; clipping from 0."
            )
            offset = 0.0

        print(
            f"[trim] {folder.name}: offset={offset:.3f}s duration={video_duration:.3f}s wavs={len(wavs)}"
        )
        for wav in wavs:
            trim_wav(wav, offset, video_duration)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
