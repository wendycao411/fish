import zipfile
import os
from pathlib import Path

base_dir = Path("/Users/wendycao/Downloads/doi_10_5061_dryad_mcvdnckcp__v20250819")

for zip_path in base_dir.rglob("*.zip"):
    extract_dir = zip_path.parent / zip_path.stem
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Unzipped {zip_path} -> {extract_dir}")
