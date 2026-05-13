#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("/Users/wendycao/fish/must_relabel")
DRY_RUN = False  # change to False after checking output

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


def main():
    total_would_delete = 0
    total_deleted = 0
    total_missing = 0
    folders_checked = 0
    csvs_missing = 0

    for video_dir in sorted(ROOT.iterdir()):
        if not video_dir.is_dir():
            continue

        folders_checked += 1
        csv_path = video_dir / "frame_actions.csv"

        if not csv_path.exists():
            print(f"[missing csv] {video_dir}")
            csvs_missing += 1
            continue

        df = pd.read_csv(csv_path)

        if "action" not in df.columns or "frame_index" not in df.columns:
            print(f"[bad csv columns] {csv_path}")
            print(f"  columns: {list(df.columns)}")
            continue

        skipped = df[df["action"].astype(str).str.lower().str.strip() == "skipped"]

        for frame_index in skipped["frame_index"]:
            if pd.isna(frame_index):
                continue

            frame_index = int(frame_index)

            # Your files look like frame_001096.jpg
            possible_paths = [
                video_dir / f"frame_{frame_index:06d}{ext}"
                for ext in IMAGE_EXTS
            ]

            existing_paths = [p for p in possible_paths if p.exists()]

            if not existing_paths:
                total_missing += 1
                continue

            for frame_path in existing_paths:
                if DRY_RUN:
                    print(f"[would delete] {frame_path}")
                    total_would_delete += 1
                else:
                    frame_path.unlink()
                    print(f"[deleted] {frame_path}")
                    total_deleted += 1

    print("\nDone.")
    print(f"Folders checked: {folders_checked}")
    print(f"Missing CSVs: {csvs_missing}")
    print(f"Skipped rows whose frame file was not found: {total_missing}")

    if DRY_RUN:
        print(f"Files that would be deleted: {total_would_delete}")
        print("This was a dry run. Set DRY_RUN = False to actually delete files.")
    else:
        print(f"Files deleted: {total_deleted}")


if __name__ == "__main__":
    main()