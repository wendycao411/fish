#!/usr/bin/env python3
"""Plot waveform + spectrogram with detections/localizations overlaid.

This reproduces the notebook-style quick visual check from CSV outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV file.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to detections/localizations CSV.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image path.")
    parser.add_argument("--channel", type=int, default=0, help="Audio channel index if multi-channel.")
    parser.add_argument("--fmax-plot", type=float, default=4000.0, help="Upper frequency (Hz) for display.")
    parser.add_argument("--frame-sec", type=float, default=0.02, help="STFT window length (s).")
    parser.add_argument("--step-sec", type=float, default=0.01, help="STFT hop length (s).")
    parser.add_argument(
        "--time-shift-sec",
        type=float,
        default=0.0,
        help="Subtract from CSV time offsets before plotting (useful for chunked timelines).",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Optional chunk index filter when CSV includes chunk_index.",
    )
    parser.add_argument(
        "--chunk-duration-sec",
        type=float,
        default=60.0,
        help="Chunk duration for converting merged absolute time to per-chunk time.",
    )
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=0,
        help="Optionally cap number of boxes drawn (0 = no cap).",
    )
    return parser.parse_args()


def load_audio(audio_path: Path, channel: int) -> tuple[int, np.ndarray]:
    fs, audio = wavfile.read(str(audio_path))
    if audio.ndim > 1:
        if channel < 0 or channel >= audio.shape[1]:
            raise ValueError(f"Invalid channel {channel} for audio shape {audio.shape}")
        audio = audio[:, channel]
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32, copy=False)
    return int(fs), audio


def prepare_detections(
    df: pd.DataFrame,
    chunk_index: int | None,
    chunk_duration_sec: float,
    time_shift_sec: float,
) -> pd.DataFrame:
    work = df.copy()
    if chunk_index is not None and "chunk_index" in work.columns:
        work = work[work["chunk_index"] == chunk_index].copy()
        time_shift_sec += float(chunk_index) * float(chunk_duration_sec)

    required = {"time_min_offset", "time_max_offset"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"CSV missing required time columns: {sorted(missing)}")

    work["plot_t0"] = work["time_min_offset"].astype(float) - float(time_shift_sec)
    work["plot_t1"] = work["time_max_offset"].astype(float) - float(time_shift_sec)

    if "frequency_min" not in work.columns:
        work["frequency_min"] = 0.0
    if "frequency_max" not in work.columns:
        work["frequency_max"] = np.nan

    work["frequency_min"] = pd.to_numeric(work["frequency_min"], errors="coerce").fillna(0.0)
    work["frequency_max"] = pd.to_numeric(work["frequency_max"], errors="coerce")
    return work


def main() -> None:
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    fs, audio = load_audio(args.audio, args.channel)
    duration_sec = len(audio) / fs
    t_wave = np.linspace(0.0, duration_sec, len(audio), endpoint=False)

    nperseg = max(32, int(args.frame_sec * fs))
    noverlap = max(0, min(nperseg - 1, nperseg - int(args.step_sec * fs)))
    f, t_spec, sxx = spectrogram(
        audio,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
    )
    sxx_db = 20.0 * np.log10(sxx + 1e-12)
    vmin = np.percentile(sxx_db, 5)
    vmax = np.percentile(sxx_db, 99)

    df = pd.read_csv(args.csv)
    det = prepare_detections(df, args.chunk_index, args.chunk_duration_sec, args.time_shift_sec)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    ax1.plot(t_wave, audio, linewidth=0.7, color="#1f1f1f")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Waveform: {args.audio.name}")

    im = ax2.pcolormesh(t_spec, f, sxx_db, shading="auto", vmin=vmin, vmax=vmax)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, min(args.fmax_plot, fs / 2.0))
    ax2.set_title("Spectrogram with detections/localizations")
    fig.colorbar(im, ax=ax2, label="Amplitude (dB)")

    n_boxes_drawn = 0
    for _, row in det.iterrows():
        t0 = float(row["plot_t0"])
        t1 = float(row["plot_t1"])
        if t1 <= 0 or t0 >= duration_sec:
            continue
        t0 = max(0.0, t0)
        t1 = min(duration_sec, t1)

        f0 = float(row.get("frequency_min", 0.0))
        f1 = row.get("frequency_max", np.nan)
        if pd.isna(f1):
            f1 = min(args.fmax_plot, fs / 2.0)
        f0 = max(0.0, min(float(f0), fs / 2.0))
        f1 = max(f0 + 1e-6, min(float(f1), fs / 2.0))

        ax1.axvspan(t0, t1, color="#d62728", alpha=0.15)
        rect = patches.Rectangle(
            (t0, f0),
            max(1e-6, t1 - t0),
            max(1e-6, f1 - f0),
            linewidth=1.2,
            edgecolor="#d62728",
            facecolor="none",
            alpha=0.95,
        )
        ax2.add_patch(rect)
        n_boxes_drawn += 1

        if args.max_boxes > 0 and n_boxes_drawn >= args.max_boxes:
            break

    ax2.text(
        0.01,
        0.98,
        f"Boxes drawn: {n_boxes_drawn}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 4},
    )

    plt.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
