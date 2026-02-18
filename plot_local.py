#!/usr/bin/env python3
"""
Plot 3D localizations with hydrophone positions.

Usage:
  python plot_localizations_3d.py \
    --csv /path/to/localizations_merged_filtered.csv \
    --out out_3d.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


HYDROPHONES = [
    ("H0", -0.858, -0.86, -0.671),
    ("H1", -0.858, -0.86,  0.479),
    ("H2",  0.858, -0.86, -0.671),
    ("H3",  0.028,  0.00, -0.002),
    ("H4", -0.858,  0.86, -0.671),
    ("H5",  0.858,  0.86,  0.671),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to localizations CSV")
    ap.add_argument("--out", default=None, help="Optional output image path (png)")
    ap.add_argument("--title", default="3D Localizations + Hydrophones")
    ap.add_argument("--max_points", type=int, default=None, help="Optionally subsample for speed")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)

    # Basic column checks
    for c in ("x", "y", "z"):
        if c not in df.columns:
            raise ValueError(f"CSV is missing required column '{c}'. Columns: {list(df.columns)}")

    # Optional subsample
    if args.max_points is not None and len(df) > args.max_points:
        df = df.sample(args.max_points, random_state=0).reset_index(drop=True)

    xs, ys, zs = df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    # Color by chunk_index if present, otherwise plot all one color (default matplotlib)
    cvals = df["chunk_index"].to_numpy() if "chunk_index" in df.columns else None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if cvals is None:
        ax.scatter(xs, ys, zs, s=8, alpha=0.7)
    else:
        sc = ax.scatter(xs, ys, zs, s=8, alpha=0.7, c=cvals)
        cb = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
        cb.set_label("chunk_index")

    # Plot hydrophones
    hx = [h[1] for h in HYDROPHONES]
    hy = [h[2] for h in HYDROPHONES]
    hz = [h[3] for h in HYDROPHONES]
    ax.scatter(hx, hy, hz, s=80)  # bigger markers for hydrophones

    for name, x, y, z in HYDROPHONES:
        ax.text(x, y, z, f" {name}", fontsize=9)

    ax.set_title(args.title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Equal-ish aspect so geometry doesn't look skewed
    # (Matplotlib 3D doesn't truly do equal aspect, but this helps.)
    xlim = (min(xs.min(), min(hx)), max(xs.max(), max(hx)))
    ylim = (min(ys.min(), min(hy)), max(ys.max(), max(hy)))
    zlim = (min(zs.min(), min(hz)), max(zs.max(), max(hz)))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

