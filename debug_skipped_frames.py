from pathlib import Path
import pandas as pd

ROOT = Path("/Users/wendycao/fish/must_relabel")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

for video_dir in sorted(ROOT.iterdir()):
    if not video_dir.is_dir():
        continue

    csv_path = video_dir / "frame_actions.csv"
    df = pd.read_csv(csv_path)

    skipped = df[df["action"].astype(str).str.lower().str.strip() == "skipped"]
    if skipped.empty:
        continue

    image_files = [
        p for p in video_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]

    print("\nVIDEO:", video_dir.name)
    print("Skipped frame indices:", skipped["frame_index"].head(10).tolist())
    print("Example image files:")
    for p in image_files[:20]:
        print(" ", p.name)

    break