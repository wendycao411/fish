"""Combine per-chunk localization CSVs into a single timeline.

Also filters out high-uncertainty rows (matches notebook thresholds).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_chunk_index(path: Path) -> int | None:
    match = re.search(r"_chunk(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def load_and_offset(csv_path: Path, chunk_duration_sec: float) -> pd.DataFrame | None:
    chunk_idx = parse_chunk_index(csv_path)
    if chunk_idx is None:
        print(f"[skip] Could not parse chunk index: {csv_path.name}")
        return None
    df = pd.read_csv(csv_path)
    offset = float(chunk_idx) * float(chunk_duration_sec)
    if "time_min_offset" in df.columns:
        df["time_min_offset"] = df["time_min_offset"] + offset
    if "time_max_offset" in df.columns:
        df["time_max_offset"] = df["time_max_offset"] + offset
    df["chunk_index"] = chunk_idx
    return df


def filter_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    return df.query("x_err_span < 1 and y_err_span < 1 and z_err_span < 1.5")


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [c for c in ("audio_file_name", "time_min_offset", "time_max_offset") if c in df.columns]
    if not key_cols:
        return df
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def select_unique_chunk_csvs(csv_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Keep one CSV per chunk index to avoid duplicated hydrophone outputs."""
    by_chunk: dict[int, Path] = {}
    duplicates: list[Path] = []
    kept: list[Path] = []
    for path in sorted(csv_paths):
        chunk_idx = parse_chunk_index(path)
        if chunk_idx is None:
            continue
        if chunk_idx not in by_chunk:
            by_chunk[chunk_idx] = path
            kept.append(path)
        else:
            duplicates.append(path)
            print(
                f"[warn] duplicate chunk CSV in {path.parent.name}: "
                f"keeping {by_chunk[chunk_idx].name}, skipping {path.name}"
            )
    return kept, duplicates


def combine_folder(
    folder: Path,
    chunk_duration_sec: float,
    output_name: str,
    clean_chunks: bool,
    clean_netcdf: bool,
) -> Path | None:
    csv_paths = sorted(folder.glob("*_chunk*.csv"))
    if not csv_paths:
        print(f"[skip] No CSVs in {folder}")
        return None

    csv_paths, duplicate_chunk_csvs = select_unique_chunk_csvs(csv_paths)
    if not csv_paths:
        print(f"[skip] No chunk CSVs with parseable chunk index in {folder}")
        return None

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = load_and_offset(csv_path, chunk_duration_sec)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        print(f"[skip] No chunk CSVs could be merged in {folder}")
        return None

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = filter_uncertainty(merged)
    n_before_dedupe = len(merged)
    merged = dedupe_rows(merged)
    if len(merged) != n_before_dedupe:
        print(f"[dedupe] removed {n_before_dedupe - len(merged)} duplicate merged rows in {folder}")

    out_path = folder / output_name
    merged.to_csv(out_path, index=False)
    print(f"[write] {out_path} rows={len(merged)}")

    if clean_chunks or clean_netcdf:
        removed = 0
        if clean_chunks:
            for csv_path in csv_paths + duplicate_chunk_csvs:
                if csv_path == out_path:
                    continue
                if "_chunk" not in csv_path.stem:
                    continue
                csv_path.unlink()
                removed += 1
        if clean_netcdf:
            nc_paths = sorted(folder.glob("*_chunk*.wav.nc"))
            for nc_path in nc_paths:
                nc_path.unlink()
                removed += 1
        ds_store = folder / ".DS_Store"
        if ds_store.exists():
            ds_store.unlink()
            removed += 1
        print(f"[clean] removed {removed} chunk artifacts in {folder}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True, help="out_synced_pairs root")
    parser.add_argument("--synced-only", type=str, default="", help="Process only this folder name")
    parser.add_argument("--chunk-duration-sec", type=float, default=60.0)
    parser.add_argument("--output-name", type=str, default="localizations_merged_filtered.csv")
    parser.add_argument("--clean-chunks", action="store_true", help="Delete per-chunk CSVs after merge")
    parser.add_argument(
        "--clean-nc",
        action="store_true",
        help="Delete per-chunk NetCDF files (e.g., *_chunk*.wav.nc) after merge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synced_only:
        folder = args.out_root / args.synced_only
        if not folder.exists():
            raise FileNotFoundError(folder)
        combine_folder(
            folder,
            args.chunk_duration_sec,
            args.output_name,
            args.clean_chunks,
            args.clean_nc,
        )
        return

    folders = sorted(p for p in args.out_root.iterdir() if p.is_dir())
    for folder in folders:
        combine_folder(
            folder,
            args.chunk_duration_sec,
            args.output_name,
            args.clean_chunks,
            args.clean_nc,
        )


if __name__ == "__main__":
    main()
