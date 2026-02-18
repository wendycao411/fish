"""
Batch localization over the synced_pairs folder structure.

This version:
1. Iterates over each subfolder in synced_pairs.
2. Points all hydrophones' data_path to the current subfolder.
3. Trims/aligns nothing (assumes audio already aligned to video).
4. Writes results to per-folder output directories.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import ecosound
import ecosound.core.tools
import numpy as np
import pandas as pd
from ecosound.core.metadata import DeploymentInfo

import detection
import localization
import tools


# #############################################################################
# Input parameters #############################################################

# Root directory that contains one folder per video/audio pair.
SYNCED_ROOT = Path("/Users/wendycao/fish/synced_pairs")

# Where localization outputs should be written.
OUT_ROOT = Path("/Users/wendycao/fish/XAV-arrays/localization/out_synced_pairs")

# Optional speed knobs via environment variables:
# 1. SYNCED_ONLY: process just one folder name under SYNCED_ROOT.
# 2. MAX_FOLDERS: cap how many folders to process.
# 3. MAX_FILES_PER_FOLDER: cap wavs per folder.
# 4. GRID_SPACING_M: override grid spacing (larger is faster).
SYNCED_ONLY = os.environ.get("SYNCED_ONLY", "").strip()
MAX_FOLDERS = int(os.environ.get("MAX_FOLDERS", "0"))
MAX_FILES_PER_FOLDER = int(os.environ.get("MAX_FILES_PER_FOLDER", "0"))
GRID_SPACING_OVERRIDE = float(os.environ.get("GRID_SPACING_M", "0"))
TDOA_STD_MAX = float(os.environ.get("TDOA_STD_MAX", "2e-4"))
APPLY_BOUNDS_FILTER = os.environ.get("APPLY_BOUNDS_FILTER", "0").strip() in {
    "1",
    "true",
    "True",
}
DISABLE_DASK = os.environ.get("DISABLE_DASK", "1").strip() not in {"0", "false", "False"}

# Use the local configuration files checked into this repo.
CONFIG_ROOT = Path("/Users/wendycao/fish/XAV-arrays/localization/large-array")

deployment_info_file = CONFIG_ROOT / "deployment_info.csv"
hydrophones_config_file = CONFIG_ROOT / "hydrophones_config_07-HI.csv"
detection_config_file = CONFIG_ROOT / "detection_config_large_array.yaml"
localization_config_file = CONFIG_ROOT / "localization_config_large_array.yaml"

# #############################################################################
# #############################################################################


def list_synced_folders(root: Path) -> list[Path]:
    """Return synced subfolders that contain wav files."""
    folders: list[Path] = []
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if SYNCED_ONLY:
        candidates = [p for p in candidates if p.name == SYNCED_ONLY]
        if not candidates:
            print(f"[warn] SYNCED_ONLY folder not found under {root}: {SYNCED_ONLY}")
    if MAX_FOLDERS > 0:
        candidates = candidates[:MAX_FOLDERS]

    for folder in candidates:
        wavs = list(folder.glob("*.wav"))
        if wavs:
            folders.append(folder)
        else:
            print(f"[skip] No wav files in {folder}")
    return folders


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_chunk_index_from_name(path: str | Path) -> int | None:
    name = Path(path).stem
    match = re.search(r"_chunk(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def dedupe_files_by_chunk(files: list[str]) -> list[str]:
    """Keep one file per chunk index to avoid duplicate hydrophone processing."""
    chosen: dict[int, str] = {}
    no_chunk: list[str] = []
    for file_path in sorted(files):
        idx = parse_chunk_index_from_name(file_path)
        if idx is None:
            no_chunk.append(file_path)
            continue
        if idx not in chosen:
            chosen[idx] = file_path
        else:
            print(
                "[warn] Duplicate chunk index found; keeping first and skipping:",
                Path(file_path).name,
            )
    deduped = [chosen[i] for i in sorted(chosen.keys())]
    return deduped + no_chunk


def canonical_gridsearch_config(raw_cfg: dict) -> dict:
    """Normalize GRIDSEARCH settings to stable numeric values."""
    return {
        "x_limits_m": [float(raw_cfg["x_limits_m"][0]), float(raw_cfg["x_limits_m"][1])],
        "y_limits_m": [float(raw_cfg["y_limits_m"][0]), float(raw_cfg["y_limits_m"][1])],
        "z_limits_m": [float(raw_cfg["z_limits_m"][0]), float(raw_cfg["z_limits_m"][1])],
        "spacing_m": float(raw_cfg["spacing_m"]),
    }


def count_localizations(localizations) -> int:
    data = getattr(localizations, "data", None)
    if data is None:
        return len(localizations)
    return len(data)


def main() -> None:
    if not SYNCED_ROOT.exists():
        raise FileNotFoundError(f"SYNCED_ROOT not found: {SYNCED_ROOT}")

    ensure_output_dir(OUT_ROOT)

    # Load deployment metadata
    deployment = DeploymentInfo()
    deployment.read(str(deployment_info_file))

    # Load hydrophone configuration
    hydrophones_config = pd.read_csv(
        hydrophones_config_file,
        skipinitialspace=True,
        dtype={"name": str, "file_name_root": str},
    )

    # Load configs
    detection_config = ecosound.core.tools.read_yaml(str(detection_config_file))
    localization_config = ecosound.core.tools.read_yaml(str(localization_config_file))

    # Dask can be memory-hungry and may get the process killed on laptops.
    # Default to disabling it unless explicitly re-enabled.
    if DISABLE_DASK:
        if "SPECTROGRAM" in detection_config:
            detection_config["SPECTROGRAM"]["use_dask"] = False
            detection_config["SPECTROGRAM"]["dask_chunks"] = 1
        for section in ("DENOISER", "DETECTOR"):
            if section in detection_config:
                detection_config[section]["use_dask"] = False
                # ecosound still touches dask_chunks even when use_dask=False
                detection_config[section]["dask_chunks"] = [1, 1]
        print("[warn] DISABLE_DASK=1 -> forcing use_dask=False in detection config")

    # Load GRIDSEARCH bounds for post-localization plausibility filtering.
    gridsearch_cfg = localization_config.get("GRIDSEARCH")
    if not gridsearch_cfg:
        # Fallback defaults if GRIDSEARCH is absent from YAML.
        gridsearch_cfg = {
            "x_limits_m": [-1.5, 1.5],
            "y_limits_m": [-1.5, 1.5],
            "z_limits_m": [-1.0, 1.5],
            "spacing_m": 0.02,
        }
        print("[warn] GRIDSEARCH missing from localization config; using defaults:", gridsearch_cfg)
    gridsearch_cfg = canonical_gridsearch_config(gridsearch_cfg)
    if GRID_SPACING_OVERRIDE > 0:
        gridsearch_cfg["spacing_m"] = GRID_SPACING_OVERRIDE
        print(f"[warn] Overriding grid spacing via GRID_SPACING_M={GRID_SPACING_OVERRIDE}")
    print("[info] GRIDSEARCH config in use:", gridsearch_cfg)

    # Precompute max TDOA window based on array geometry
    sound_speed_mps = localization_config.get("ENVIRONMENT", {}).get(
        "sound_speed_mps", 1484.0
    )
    hydrophones_dist_matrix = tools.calc_hydrophones_distances(hydrophones_config)
    tdoa_max_sec = float(np.max(hydrophones_dist_matrix) / sound_speed_mps)

    synced_folders = list_synced_folders(SYNCED_ROOT)
    print(f"Found {len(synced_folders)} synced folders with wav files.")

    for folder_idx, folder in enumerate(synced_folders, start=1):
        print("\n" + "=" * 80)
        print(f"[{folder_idx}/{len(synced_folders)}] Folder: {folder}")

        # Point all hydrophones to the current synced folder.
        hydrophones_config_current = hydrophones_config.copy()
        hydrophones_config_current["data_path"] = str(folder)

        # Output directory per synced folder.
        out_dir = OUT_ROOT / folder.name
        ensure_output_dir(out_dir)

        # Find all wav files in this synced folder.
        files = ecosound.core.tools.list_files(
            str(folder),
            ".wav",
            recursive=False,
            case_sensitive=True,
        )
        # Only process chunked wavs (e.g., *_chunk000.wav) to reduce memory load.
        files = [f for f in files if "_chunk" in os.path.basename(f)]
        # Only process the reference channel wavs to avoid duplicate localization CSVs.
        ref_channel = localization_config["TDOA"]["ref_channel"]
        ref_root = hydrophones_config_current.loc[
            hydrophones_config_current["array_channel"] == ref_channel,
            "file_name_root",
        ]
        if len(ref_root) == 1:
            ref_prefix = str(ref_root.iloc[0])
            files = [f for f in files if os.path.basename(f).startswith(ref_prefix)]
            files = dedupe_files_by_chunk(files)
        else:
            print(
                "[warn] Could not uniquely resolve ref_channel file_name_root; "
                "processing all chunked wavs."
            )

        nfiles = len(files)
        if nfiles == 0:
            print(f"[skip] No chunked wav files found in {folder}")
            continue
        if MAX_FILES_PER_FOLDER > 0:
            files = files[:MAX_FILES_PER_FOLDER]
            nfiles = len(files)
            print(f"[info] Limiting to {nfiles} wav files in this folder.")

        for idx, in_file in enumerate(files, start=1):
            in_name = os.path.split(in_file)[1]
            out_file = out_dir / in_name

            print(f"{idx}/{nfiles} {in_name}")

            if (out_file.with_suffix(out_file.suffix + ".nc")).exists():
                print("File already processed")
                continue

            # Look up data files for all channels
            audio_files = tools.find_audio_files(in_file, hydrophones_config_current)
            audio_files["path"] = [p.replace("\\", "/") for p in audio_files["path"]]

            # Run detector on selected channel
            print("Detection in progress...")
            detections = detection.run_detector(
                audio_files["path"][detection_config["AUDIO"]["channel"]],
                audio_files["channel"][detection_config["AUDIO"]["channel"]],
                detection_config,
                deployment_file=str(deployment_info_file),
            )
            print(f"-> {len(detections)} detections found.")

            # Optional guard: drop the first detection that fails stacking.
            for det_i, det in detections.data.reset_index(drop=True).iterrows():
                try:
                    _ = tools.stack_waveforms(audio_files, det, tdoa_max_sec)
                except Exception as exc:  # pragma: no cover - debug safety net
                    print(
                        "stack_waveforms failed on detection index",
                        det_i,
                        "error:",
                        exc,
                    )
                    print("Detection row:", det.to_dict())
                    detections.data = (
                        detections.data.drop(detections.data.index[det_i])
                        .reset_index(drop=True)
                    )
                    print("Dropped offending detection", det_i)
                    break

            # Perform localization using linearized inversion
            print("Localization (LinearizedInversion)")
            localizations = localization.LinearizedInversion.run_localization(
                audio_files,
                detections,
                str(deployment_info_file),
                detection_config,
                hydrophones_config_current,
                localization_config,
                verbose=False,
            )

            # Filter localization results for physical plausibility and fit quality.
            n_before_filters = count_localizations(localizations)
            print(f"[info] Localizations before filtering: {n_before_filters}")

            if APPLY_BOUNDS_FILTER:
                x_limits = gridsearch_cfg["x_limits_m"]
                y_limits = gridsearch_cfg["y_limits_m"]
                z_limits = gridsearch_cfg["z_limits_m"]
                loc_df = localizations.data
                bounds_mask = (
                    (loc_df["x"] > x_limits[0])
                    & (loc_df["x"] < x_limits[1])
                    & (loc_df["y"] > y_limits[0])
                    & (loc_df["y"] < y_limits[1])
                    & (loc_df["z"] > z_limits[0])
                    & (loc_df["z"] < z_limits[1])
                )
                localizations.data = loc_df.loc[bounds_mask].reset_index(drop=True)
                print(f"[info] After bounds filter: {count_localizations(localizations)}")
            else:
                print("[info] Skipping bounds filter (APPLY_BOUNDS_FILTER=0)")

            localizations.filter(
                "x_err_span < 1 & y_err_span < 1 & z_err_span < 1.5",
                inplace=True,
            )
            print(f"[info] After error-span filter: {count_localizations(localizations)}")

            if "tdoa_errors_std" in localizations.data.columns:
                localizations.filter(f"tdoa_errors_std < {TDOA_STD_MAX}", inplace=True)
                print(
                    f"[info] After tdoa_errors_std<{TDOA_STD_MAX} filter: "
                    f"{count_localizations(localizations)}"
                )
            else:
                print("[info] Skipping tdoa_errors_std filter (column not available)")

            # Save results as csv and netcdf file
            print("Saving results...")
            localizations.to_csv(str(out_file) + ".csv")
            if "iterations_logs" in localizations.data.columns:
                localizations.data.drop(columns=["iterations_logs"], inplace=True)
            localizations.to_netcdf(str(out_file) + ".nc")
            print(" ")

    print("Processing complete!")


if __name__ == "__main__":
    main()
