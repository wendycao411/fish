"""Render localization heatmaps over video using error bars (anisotropic blobs).

This variant uses localization error bars to control the spread/shape of each event,
instead of adding point-mass circles and blurring with a fixed sigma.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import soundfile as sf


DEFAULT_OUT_ROOT = Path("/Users/wendycao/fish/XAV-arrays/localization/out_synced_pairs")
DEFAULT_SYNCED_ROOT = Path("/Users/wendycao/fish/synced_pairs")


def load_homography(h_path: Path) -> np.ndarray:
    H = np.load(h_path)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape} from {h_path}")
    return H.astype(np.float32)


def parse_synced_folder_meta(folder_name: str) -> tuple[int, str] | None:
    match = re.match(r"^(?P<id>\d+)_?(?P<camera>FishCam\d+)_", folder_name)
    if not match:
        return None
    return int(match.group("id")), match.group("camera")


def parse_chunk_index(path: Path) -> int | None:
    match = re.search(r"_chunk(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def merge_chunk_localizations(
    csv_paths: list[Path],
    out_csv: Path,
    chunk_duration_sec: float,
) -> Path:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        chunk_idx = parse_chunk_index(csv_path)
        if chunk_idx is None:
            print(f"[skip] Could not parse chunk index from {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        offset = float(chunk_idx) * float(chunk_duration_sec)
        if "time_min_offset" in df.columns:
            df["time_min_offset"] = df["time_min_offset"] + offset
        if "time_max_offset" in df.columns:
            df["time_max_offset"] = df["time_max_offset"] + offset
        df["chunk_index"] = chunk_idx
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No chunk CSVs could be merged")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[merge] wrote {len(merged)} rows to {out_csv}")
    return out_csv


def project_xy(x: float, y: float, H: np.ndarray) -> tuple[int, int]:
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(pt, H)[0, 0]
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def heat_to_colormap(heat: np.ndarray) -> np.ndarray:
    h = heat.astype(np.float32)
    h = h / (h.max() + 1e-8)
    heat_u8 = np.clip(255 * h, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR


def overlay_heat(frame_bgr: np.ndarray, heat_smooth: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_color = heat_to_colormap(heat_smooth)
    if heat_color.shape[:2] != frame_bgr.shape[:2]:
        heat_color = cv2.resize(heat_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return cv2.addWeighted(frame_bgr, 1 - alpha, heat_color, alpha, 0)


def _pos_float(v: object) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    if not np.isfinite(f) or f <= 0:
        return None
    return f


def error_sigma_px(
    x: float,
    y: float,
    H: np.ndarray,
    x_err_span: object = None,
    y_err_span: object = None,
    x_err_low: object = None,
    x_err_high: object = None,
    y_err_low: object = None,
    y_err_high: object = None,
    base_sigma_px: float = 20.0,
    clamp_min_px: float = 5.0,
    clamp_max_px: float = 120.0,
) -> tuple[float, float]:
    """Estimate pixel sigmas from world-coordinate error bars via local finite differences."""
    xs = _pos_float(x_err_span)
    ys = _pos_float(y_err_span)

    if xs is None:
        xl = _pos_float(x_err_low)
        xh = _pos_float(x_err_high)
        if xl is not None and xh is not None:
            xs = max(xl, xh)
    if ys is None:
        yl = _pos_float(y_err_low)
        yh = _pos_float(y_err_high)
        if yl is not None and yh is not None:
            ys = max(yl, yh)

    if xs is None and ys is None:
        return float(base_sigma_px), float(base_sigma_px)

    u0, v0 = project_xy(x, y, H)

    def px_delta(dx: float, dy: float) -> tuple[float, float]:
        u1, v1 = project_xy(x + dx, y + dy, H)
        return abs(u1 - u0), abs(v1 - v0)

    du = dv = 0.0
    if xs is not None:
        du_x, dv_x = px_delta(xs, 0.0)
        du = max(du, du_x)
        dv = max(dv, dv_x)
    if ys is not None:
        du_y, dv_y = px_delta(0.0, ys)
        du = max(du, du_y)
        dv = max(dv, dv_y)

    du = float(np.clip(max(1.0, du), clamp_min_px, clamp_max_px))
    dv = float(np.clip(max(1.0, dv), clamp_min_px, clamp_max_px))
    return du, dv


def add_anisotropic_gaussian(
    heat: np.ndarray,
    u: int,
    v: int,
    w: float,
    sigma_u: float,
    sigma_v: float,
    cutoff_sigma: float = 3.0,
) -> None:
    """Add an anisotropic Gaussian bump centered at (u,v) into heat in-place."""
    Hh, W = heat.shape
    ru = int(max(2, round(cutoff_sigma * sigma_u)))
    rv = int(max(2, round(cutoff_sigma * sigma_v)))

    u0 = max(0, u - ru)
    u1 = min(W - 1, u + ru)
    v0 = max(0, v - rv)
    v1 = min(Hh - 1, v + rv)
    if u0 >= u1 or v0 >= v1:
        return

    xs = np.arange(u0, u1 + 1, dtype=np.float32)
    ys = np.arange(v0, v1 + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    dx = (X - float(u)) / float(max(1e-6, sigma_u))
    dy = (Y - float(v)) / float(max(1e-6, sigma_v))
    g = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)

    heat[v0 : v1 + 1, u0 : u1 + 1] += float(w) * g


def build_event_table(
    df: pd.DataFrame,
    video_w: int,
    video_h: int,
    fps: float,
    H: np.ndarray,
    time_offset_sec: float = 0.0,
) -> pd.DataFrame:
    """Return a DataFrame with columns: frame_idx, u, v, w and error bars."""
    required = {"time_min_offset"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    x_col = "x" if "x" in df.columns else ("x_m" if "x_m" in df.columns else None)
    y_col = "y" if "y" in df.columns else ("y_m" if "y_m" in df.columns else None)
    if x_col is None or y_col is None:
        raise KeyError(f"Could not find x/y columns in CSV. Columns: {list(df.columns)}")

    rows: list[tuple[int, int, int, float, float, float, float, float, float, float]] = []
    has_tmax = "time_max_offset" in df.columns

    for _, r in df.iterrows():
        t0 = float(r["time_min_offset"])
        t1 = float(r["time_max_offset"]) if has_tmax else t0
        frame_idx = int(round((t0 + float(time_offset_sec)) * fps))

        x_val = r.get(x_col, np.nan)
        y_val = r.get(y_col, np.nan)
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue

        u, v = project_xy(float(x_val), float(y_val), H)
        if not (0 <= u < video_w and 0 <= v < video_h):
            continue

        # Uniform weight; errors control spread.
        w = 1.0

        rows.append(
            (
                frame_idx,
                u,
                v,
                w,
                float(t0),
                float(t1),
                float(r.get("x_err_span", np.nan)),
                float(r.get("y_err_span", np.nan)),
                float(r.get("x_err_low", np.nan)),
                float(r.get("x_err_high", np.nan)),
            )
        )

    events = pd.DataFrame(
        rows,
        columns=[
            "frame_idx",
            "u",
            "v",
            "w",
            "t0",
            "t1",
            "x_err_span",
            "y_err_span",
            "x_err_low",
            "x_err_high",
        ],
    )

    # Attach any available y error bounds for sigma estimation.
    if "y_err_low" in df.columns:
        events["y_err_low"] = df["y_err_low"].to_numpy()[: len(events)]
    else:
        events["y_err_low"] = np.nan
    if "y_err_high" in df.columns:
        events["y_err_high"] = df["y_err_high"].to_numpy()[: len(events)]
    else:
        events["y_err_high"] = np.nan

    # Also keep world coordinates for local sigma estimation.
    events["x_world"] = df[x_col].to_numpy()[: len(events)]
    events["y_world"] = df[y_col].to_numpy()[: len(events)]

    return events


def save_debug_projection_frame(video_path: Path, events: pd.DataFrame, out_png: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if len(events) > 0:
        target_frame = int(events["frame_idx"].min())
    else:
        target_frame = n_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {target_frame}")

    subset = events[events["frame_idx"] == int(target_frame)]
    for _, r in subset.iterrows():
        cv2.circle(frame, (int(r["u"]), int(r["v"])), 10, (0, 255, 0), -1)

    cv2.putText(
        frame,
        f"frame={target_frame} events={len(subset)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)
    print("Saved:", out_png)


def make_time_varying_heat_overlay_video_errors(
    video_path: Path,
    localizations_csv: Path,
    H: np.ndarray,
    out_video_path: Path,
    mode: str = "trail",  # "trail" | "cumulative"
    window_sec: float = 1.5,
    sigma_px: float = 20.0,
    alpha: float = 0.6,
    decay_sec: float = 0.0,
    hard_cut: bool = True,
    time_offset_sec: float = 0.0,
    max_frames: int | None = None,
    debug: bool = True,
) -> None:
    df = pd.read_csv(localizations_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = n_frames_total if max_frames is None else min(n_frames_total, int(max_frames))
    print(f"[video] fps={fps:.2f}, size={W}x{Hh}, frames={n_frames} (total={n_frames_total})")

    events = build_event_table(df, video_w=W, video_h=Hh, fps=fps, H=H, time_offset_sec=time_offset_sec)

    print("[events] usable events:", len(events))
    if len(events) > 0:
        print("[events] frame_idx min/max:", int(events["frame_idx"].min()), int(events["frame_idx"].max()))
        print("[events] u min/max:", int(events["u"].min()), int(events["u"].max()))
        print("[events] v min/max:", int(events["v"].min()), int(events["v"].max()))

    if debug:
        debug_png = out_video_path.with_suffix("").with_name(out_video_path.stem + "_debug.png")
        save_debug_projection_frame(video_path, events, out_png=debug_png)

    events_by_frame = {int(k): g for k, g in events.groupby("frame_idx")} if len(events) > 0 else {}

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, Hh))

    heat = np.zeros((Hh, W), dtype=np.float32)

    window_frames = int(round(window_sec * fps))
    decay = 1.0
    if decay_sec and decay_sec > 0:
        decay = float(np.exp(-1.0 / (decay_sec * fps)))
    frame_adds: list[list[tuple[int, int, float]] | None] = [None] * n_frames

    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        adds: list[tuple[int, int, float]] = []

        def _add_rows(rows: pd.DataFrame, adds_list: list[tuple[int, int, float]]) -> None:
            for _, r in rows.iterrows():
                u, v, w = int(r["u"]), int(r["v"]), float(r["w"])
                if not np.isfinite(w) or w <= 0:
                    continue

                sigma_u, sigma_v = error_sigma_px(
                    x=float(r.get("x_world", np.nan)),
                    y=float(r.get("y_world", np.nan)),
                    H=H,
                    x_err_span=r.get("x_err_span", np.nan),
                    y_err_span=r.get("y_err_span", np.nan),
                    x_err_low=r.get("x_err_low", np.nan),
                    x_err_high=r.get("x_err_high", np.nan),
                    y_err_low=r.get("y_err_low", np.nan),
                    y_err_high=r.get("y_err_high", np.nan),
                    base_sigma_px=sigma_px,
                )

                add_anisotropic_gaussian(heat, u, v, w, sigma_u=sigma_u, sigma_v=sigma_v)
                adds_list.append((v, u, w))

        if mode == "trail" and hard_cut:
            heat.fill(0)
            start = max(0, frame_idx - window_frames + 1)
            for f in range(start, frame_idx + 1):
                g = events_by_frame.get(int(f))
                if g is not None:
                    _add_rows(g, adds)
        else:
            g = events_by_frame.get(int(frame_idx))
            if g is not None:
                _add_rows(g, adds)

        frame_adds[frame_idx] = adds

        if mode == "trail" and not hard_cut and frame_idx - window_frames >= 0:
            old = frame_adds[frame_idx - window_frames]
            if old:
                for v, u, w in old:
                    heat[v, u] -= w
                np.maximum(heat, 0, out=heat)

        if decay < 1.0:
            heat *= decay

        if debug and frame_idx % 50 == 0:
            print(
                f"[render] frame {frame_idx}/{n_frames} | adds={len(adds)} | heat.max={float(heat.max()):.6g}"
            )

        heat_smooth = cv2.GaussianBlur(
            heat,
            ksize=(0, 0),
            sigmaX=float(sigma_px),
            sigmaY=float(sigma_px),
        )

        out = overlay_heat(frame, heat_smooth, alpha=alpha)

        cv2.putText(
            out,
            f"errors  t={frame_idx/fps:.2f}s  adds={len(adds)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(out)
        frame_idx += 1

    cap.release()
    writer.release()
    print("Wrote:", out_video_path)


def mux_audio_from_source_video(rendered_video_path: Path, source_video_path: Path) -> None:
    """Attach audio from source_video_path to rendered_video_path in-place via ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("[warn] ffmpeg not found; skipping audio mux.")
        return

    tmp_out = rendered_video_path.with_name(rendered_video_path.stem + "_tmp_audio.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(rendered_video_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(tmp_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[warn] ffmpeg audio mux failed; keeping silent overlay video.")
        if result.stderr:
            print(result.stderr.strip().splitlines()[-1])
        if tmp_out.exists():
            tmp_out.unlink()
        return

    tmp_out.replace(rendered_video_path)
    print(f"[audio] Muxed source audio into: {rendered_video_path}")


@dataclass(frozen=True)
class SyncedJob:
    folder: Path
    video_path: Path
    csv_path: Path
    out_video_path: Path


def find_single_video(folder: Path) -> Path:
    videos = sorted(folder.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 found in {folder}")
    if len(videos) > 1:
        print(f"[warn] Multiple videos found in {folder}; using {videos[0].name}")
    return videos[0]


def build_jobs_for_synced_folder(
    synced_folder: Path,
    out_root: Path,
    use_merged: bool,
    merged_csv_name: str,
    combine_chunks: bool,
    chunk_duration_sec: float,
) -> list[SyncedJob]:
    video_path = find_single_video(synced_folder)

    csv_dir = out_root / synced_folder.name
    if not csv_dir.exists():
        raise FileNotFoundError(f"Expected localization outputs in {csv_dir}")

    overlays_dir = csv_dir / "overlays_errors"
    jobs: list[SyncedJob] = []
    if use_merged:
        csv_path = csv_dir / merged_csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected merged CSV in {csv_path}")
        out_video_path = overlays_dir / f"{synced_folder.name}_heat_errors.mp4"
        jobs.append(SyncedJob(folder=synced_folder, video_path=video_path, csv_path=csv_path, out_video_path=out_video_path))
        return jobs

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No localization CSVs found in {csv_dir}")

    if combine_chunks:
        merged_csv = overlays_dir / "localizations_merged.csv"
        csv_path = merge_chunk_localizations(
            csv_paths=csv_paths,
            out_csv=merged_csv,
            chunk_duration_sec=chunk_duration_sec,
        )
        out_video_path = overlays_dir / f"{synced_folder.name}_heat_errors.mp4"
        jobs.append(SyncedJob(folder=synced_folder, video_path=video_path, csv_path=csv_path, out_video_path=out_video_path))
        return jobs

    for csv_path in csv_paths:
        out_video_path = overlays_dir / f"{csv_path.stem}_heat_errors.mp4"
        jobs.append(SyncedJob(folder=synced_folder, video_path=video_path, csv_path=csv_path, out_video_path=out_video_path))
    return jobs


def iter_synced_jobs(
    synced_root: Path,
    out_root: Path,
    synced_only: str | None,
    max_folders: int,
    min_folder_id: int,
    camera_name: str | None,
    use_merged: bool,
    merged_csv_name: str,
    combine_chunks: bool,
    chunk_duration_sec: float,
) -> Iterable[SyncedJob]:
    folders = sorted(p for p in synced_root.iterdir() if p.is_dir())
    if synced_only:
        folders = [p for p in folders if p.name == synced_only]
        if not folders:
            raise FileNotFoundError(f"SYNCED_ONLY not found under {synced_root}: {synced_only}")
    if max_folders > 0:
        folders = folders[:max_folders]

    for folder in folders:
        if min_folder_id > 0 or camera_name:
            meta = parse_synced_folder_meta(folder.name)
            if meta is None:
                print(f"[skip] {folder.name}: could not parse folder id/camera")
                continue
            folder_id, folder_camera = meta
            if min_folder_id > 0 and folder_id < min_folder_id:
                continue
            if camera_name and folder_camera != camera_name:
                continue
        try:
            jobs = build_jobs_for_synced_folder(
                folder,
                out_root=out_root,
                use_merged=use_merged,
                merged_csv_name=merged_csv_name,
                combine_chunks=combine_chunks,
                chunk_duration_sec=chunk_duration_sec,
            )
        except FileNotFoundError as exc:
            print(f"[skip] {folder.name}: {exc}")
            continue
        for job in jobs:
            yield job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--homography-npy", type=Path, required=True, help="Path to 3x3 homography .npy file")

    # Direct single-job mode
    parser.add_argument("--video", type=Path, help="Path to video (.mp4)")
    parser.add_argument("--localizations-csv", type=Path, help="Path to localization CSV")
    parser.add_argument("--out-video", type=Path, help="Path to output overlay video")

    # Synced pipeline mode
    parser.add_argument("--synced-root", type=Path, default=DEFAULT_SYNCED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--synced-only", type=str, default="", help="Process only this synced folder name")
    parser.add_argument("--max-folders", type=int, default=0, help="Limit number of synced folders")
    parser.add_argument(
        "--min-folder-id",
        type=int,
        default=0,
        help="Process only synced folders with numeric prefix >= this value (e.g., 2729)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="",
        help="Process only folders with this camera name (e.g., FishCam01)",
    )
    parser.add_argument(
        "--use-merged",
        action="store_true",
        help="Use a pre-merged CSV in each folder (e.g., localizations_merged_filtered.csv)",
    )
    parser.add_argument(
        "--merged-csv-name",
        type=str,
        default="localizations_merged_filtered.csv",
        help="Merged CSV name used with --use-merged",
    )
    parser.add_argument(
        "--combine-chunks",
        action="store_true",
        help="Merge per-chunk localization CSVs into a single timeline for the full video",
    )
    parser.add_argument(
        "--chunk-duration-sec",
        type=float,
        default=60.0,
        help="Duration of each audio chunk in seconds (used when --combine-chunks)",
    )

    # Rendering options
    parser.add_argument("--mode", choices=["trail", "cumulative"], default="trail")
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--sigma-px", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument(
        "--decay-sec",
        type=float,
        default=0.0,
        help="Exponential decay time constant for heatmap fade-out (seconds). 0 disables.",
    )
    parser.add_argument(
        "--hard-cut",
        action="store_true",
        default=True,
        help="Rebuild heat each frame from the trailing window (hard cutoff at window-sec).",
    )
    parser.add_argument("--time-offset-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for faster debugging")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug prints and debug frame PNG")
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Attach audio from the source video to rendered overlay output (requires ffmpeg).",
    )

    return parser.parse_args()


def run_single_job(args: argparse.Namespace, H: np.ndarray) -> None:
    if not (args.video and args.localizations_csv and args.out_video):
        raise SystemExit("Single-job mode requires --video, --localizations-csv, and --out-video")

    make_time_varying_heat_overlay_video_errors(
        video_path=args.video,
        localizations_csv=args.localizations_csv,
        H=H,
        out_video_path=args.out_video,
        mode=args.mode,
        window_sec=args.window_sec,
        sigma_px=args.sigma_px,
        alpha=args.alpha,
        decay_sec=args.decay_sec,
        hard_cut=args.hard_cut,
        time_offset_sec=args.time_offset_sec,
        max_frames=None if args.max_frames <= 0 else args.max_frames,
        debug=not args.no_debug,
    )
    if args.with_audio:
        mux_audio_from_source_video(rendered_video_path=args.out_video, source_video_path=args.video)


def run_synced_pipeline(args: argparse.Namespace, H: np.ndarray) -> None:
    jobs = list(
        iter_synced_jobs(
            synced_root=args.synced_root,
            out_root=args.out_root,
            synced_only=args.synced_only.strip() or None,
            max_folders=args.max_folders,
            min_folder_id=args.min_folder_id,
            camera_name=args.camera.strip() or None,
            use_merged=args.use_merged,
            merged_csv_name=args.merged_csv_name,
            combine_chunks=args.combine_chunks,
            chunk_duration_sec=args.chunk_duration_sec,
        )
    )
    print(f"Found {len(jobs)} overlay jobs.")

    for idx, job in enumerate(jobs, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(jobs)}] {job.folder.name} :: {job.csv_path.name}")
        if job.out_video_path.exists():
            print(f"[skip] Already exists: {job.out_video_path}")
            continue
        make_time_varying_heat_overlay_video_errors(
            video_path=job.video_path,
            localizations_csv=job.csv_path,
            H=H,
            out_video_path=job.out_video_path,
            mode=args.mode,
            window_sec=args.window_sec,
            sigma_px=args.sigma_px,
            alpha=args.alpha,
            decay_sec=args.decay_sec,
            hard_cut=args.hard_cut,
            time_offset_sec=args.time_offset_sec,
            max_frames=None if args.max_frames <= 0 else args.max_frames,
            debug=not args.no_debug,
        )
        if args.with_audio:
            mux_audio_from_source_video(
                rendered_video_path=job.out_video_path,
                source_video_path=job.video_path,
            )


def main() -> None:
    args = parse_args()
    H = load_homography(args.homography_npy)

    if args.video or args.localizations_csv or args.out_video:
        run_single_job(args, H)
    else:
        run_synced_pipeline(args, H)


if __name__ == "__main__":
    main()
