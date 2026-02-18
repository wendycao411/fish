#!/usr/bin/env python3
"""Split all WAV files in synced_pairs into fixed-length chunks.

Writes chunks next to the originals using suffix: _chunkNNN.wav
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

SYNCED_ROOT = Path("/Users/wendycao/fish/synced_pairs")
CHUNK_SEC = 60.0


def split_wav(path: Path, chunk_sec: float) -> int:
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        sr = params.framerate
        nframes = params.nframes
        frames_per_chunk = int(round(chunk_sec * sr))
        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")

        nchunks = int(math.ceil(nframes / frames_per_chunk))

        for i in range(nchunks):
            start = i * frames_per_chunk
            length = min(frames_per_chunk, nframes - start)

            src.setpos(start)
            frames = src.readframes(length)

            out_path = path.with_name(f"{path.stem}_chunk{i:03d}.wav")
            with wave.open(str(out_path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)

    return nchunks


def main() -> None:
    wavs = sorted(SYNCED_ROOT.glob("*/*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files found under {SYNCED_ROOT}")

    total_chunks = 0
    for idx, wav_path in enumerate(wavs, start=1):
        nchunks = split_wav(wav_path, CHUNK_SEC)
        total_chunks += nchunks
        print(f"[{idx}/{len(wavs)}] {wav_path.name} -> {nchunks} chunks")

    print(f"Done. Wrote {total_chunks} chunks alongside originals.")


if __name__ == "__main__":
    main()
