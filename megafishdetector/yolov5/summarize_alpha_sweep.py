#!/usr/bin/env python3
"""Summarize a YOLOv5 heatmap-alpha sweep into combined plots and a CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ALPHA_RE = re.compile(r"alpha([0-9.]+)_fixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YOLOv5 alpha sweep runs")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("/data/vision/beery/scratch/wendy/fish/megafishdetector/yolov5/fish-heatmap-modulation-subset250-alpha-sweep"),
        help="Directory containing one subdir per alpha sweep run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs (default: <project-dir>/summary)",
    )
    return parser.parse_args()


def alpha_from_name(name: str) -> float | None:
    m = ALPHA_RE.search(name)
    if not m:
        return None
    return float(m.group(1))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = (args.output_dir or (project_dir / "summary")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []
    for run_dir in sorted(project_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        alpha = alpha_from_name(run_dir.name)
        results_csv = run_dir / "results.csv"
        if alpha is None or not results_csv.exists():
            continue

        df = normalize_columns(pd.read_csv(results_csv))
        df["alpha"] = alpha
        df["run_name"] = run_dir.name
        histories.append(df)

        best_idx = df["metrics/mAP_0.5:0.95"].idxmax()
        best = df.loc[best_idx]
        rows.append(
            {
                "run_name": run_dir.name,
                "alpha": alpha,
                "best_epoch": int(best["epoch"]),
                "best_precision": float(best["metrics/precision"]),
                "best_recall": float(best["metrics/recall"]),
                "best_map50": float(best["metrics/mAP_0.5"]),
                "best_map5095": float(best["metrics/mAP_0.5:0.95"]),
                "final_box_loss": float(df.iloc[-1]["val/box_loss"]),
                "final_obj_loss": float(df.iloc[-1]["val/obj_loss"]),
            }
        )

    if not rows:
        raise RuntimeError(f"No alpha-sweep runs with results.csv found under {project_dir}")

    summary_df = pd.DataFrame(rows).sort_values("alpha")
    summary_csv = output_dir / "alpha_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Plot alpha vs summary metrics.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_specs = [
        ("best_precision", "Best Precision"),
        ("best_recall", "Best Recall"),
        ("best_map50", "Best mAP@0.5"),
        ("best_map5095", "Best mAP@0.5:0.95"),
    ]
    for ax, (col, title) in zip(axes.flat, metric_specs):
        ax.plot(summary_df["alpha"], summary_df[col], marker="o")
        for _, row in summary_df.iterrows():
            ax.annotate(f"{row[col]:.3f}", (row["alpha"], row[col]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xlabel("Heatmap alpha")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "alpha_vs_metrics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Overlay training histories for a few key metrics.
    hist_df = pd.concat(histories, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    history_specs = [
        ("metrics/precision", "Precision"),
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("val/box_loss", "Val Box Loss"),
    ]
    for ax, (col, title) in zip(axes.flat, history_specs):
        for alpha, group in hist_df.groupby("alpha"):
            group = group.sort_values("epoch")
            ax.plot(group["epoch"], group[col], label=f"alpha={alpha:g}")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "alpha_histories.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Summary CSV: {summary_csv}")
    print(f"[INFO] Summary metric plot: {output_dir / 'alpha_vs_metrics.png'}")
    print(f"[INFO] History plot: {output_dir / 'alpha_histories.png'}")


if __name__ == "__main__":
    main()
