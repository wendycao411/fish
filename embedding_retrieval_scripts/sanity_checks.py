#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import csv

np.seterr(all="ignore")


def load_embeddings(path: Path) -> np.ndarray:
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings at {path}, got shape {emb.shape}")
    return emb


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x_centered = x - x.mean(axis=0, keepdims=True)
    # Compute top-2 principal components via SVD
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    return x_centered @ vt[:2].T


def topk_indices(sim_row: np.ndarray, k: int, exclude_index: int | None = None) -> np.ndarray:
    if exclude_index is not None and 0 <= exclude_index < sim_row.shape[0]:
        sim_row = sim_row.copy()
        sim_row[exclude_index] = -np.inf
    idx = np.argsort(-sim_row)
    return idx[:k]


def make_plots(
    audio_emb: np.ndarray,
    video_emb: np.ndarray,
    audio_meta: list[dict],
    video_meta: list[dict],
    out_dir: Path,
    k: int,
    n_queries: int,
    seed: int,
) -> Tuple[list[dict], list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_norm = l2_normalize(audio_emb)
    video_norm = l2_normalize(video_emb)

    # PCA on combined embeddings so both modalities share the same 2D space
    combined = np.vstack([audio_emb, video_emb])
    combined_2d = pca_2d(combined)
    audio_2d = combined_2d[: len(audio_emb)]
    video_2d = combined_2d[len(audio_emb) :]

    rng = np.random.default_rng(seed)
    audio_query_idx = rng.choice(len(audio_emb), size=min(n_queries, len(audio_emb)), replace=False)
    video_query_idx = rng.choice(len(video_emb), size=min(n_queries, len(video_emb)), replace=False)

    audio_rows: list[dict] = []
    video_rows: list[dict] = []

    # Defer matplotlib import until needed, force non-interactive backend
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Audio -> Video
    sim_a2v = audio_norm @ video_norm.T
    for qi in audio_query_idx:
        topk = topk_indices(sim_a2v[qi], k=k, exclude_index=None)
        audio_rows.append(
            {
                "query_index": int(qi),
                "query_clip_path": audio_meta[qi].get("clip_path", ""),
                "topk_indices": topk.tolist(),
                "topk_clip_paths": [video_meta[j].get("clip_path", "") for j in topk],
                "topk_similarities": sim_a2v[qi, topk].tolist(),
            }
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(audio_2d[:, 0], audio_2d[:, 1], s=10, c="#c7c7c7", alpha=0.5, label="Audio")
        ax.scatter(video_2d[:, 0], video_2d[:, 1], s=10, c="#a6cee3", alpha=0.5, label="Video")
        ax.scatter(
            audio_2d[qi, 0],
            audio_2d[qi, 1],
            s=90,
            c="#e31a1c",
            marker="*",
            label="Audio query",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(
            video_2d[topk, 0],
            video_2d[topk, 1],
            s=60,
            c="#1f78b4",
            label="Top-K video",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_title(f"Audio→Video top-{k} (query {qi})")
        ax.legend(loc="best", frameon=False)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(out_dir / f"audio_to_video_query_{qi}.png", dpi=200)
        plt.close(fig)

    # Video -> Audio
    sim_v2a = video_norm @ audio_norm.T
    for qi in video_query_idx:
        topk = topk_indices(sim_v2a[qi], k=k, exclude_index=None)
        video_rows.append(
            {
                "query_index": int(qi),
                "query_clip_path": video_meta[qi].get("clip_path", ""),
                "topk_indices": topk.tolist(),
                "topk_clip_paths": [audio_meta[j].get("clip_path", "") for j in topk],
                "topk_similarities": sim_v2a[qi, topk].tolist(),
            }
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(audio_2d[:, 0], audio_2d[:, 1], s=10, c="#c7c7c7", alpha=0.5, label="Audio")
        ax.scatter(video_2d[:, 0], video_2d[:, 1], s=10, c="#a6cee3", alpha=0.5, label="Video")
        ax.scatter(
            video_2d[qi, 0],
            video_2d[qi, 1],
            s=90,
            c="#33a02c",
            marker="*",
            label="Video query",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(
            audio_2d[topk, 0],
            audio_2d[topk, 1],
            s=60,
            c="#ff7f00",
            label="Top-K audio",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_title(f"Video→Audio top-{k} (query {qi})")
        ax.legend(loc="best", frameon=False)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(out_dir / f"video_to_audio_query_{qi}.png", dpi=200)
        plt.close(fig)

    audio_csv = out_dir / "audio_to_video_topk.csv"
    video_csv = out_dir / "video_to_audio_topk.csv"

    if audio_rows:
        with audio_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audio_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audio_rows)

    if video_rows:
        with video_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
            writer.writeheader()
            writer.writerows(video_rows)

    return audio_rows, video_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity checks for retrievals and embedding plots.")
    parser.add_argument("--audio-emb", type=Path, required=True)
    parser.add_argument("--video-emb", type=Path, required=True)
    parser.add_argument("--audio-meta", type=Path, required=True)
    parser.add_argument("--video-meta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_emb = load_embeddings(args.audio_emb)
    video_emb = load_embeddings(args.video_emb)

    def load_meta(path: Path) -> list[dict]:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    audio_meta = load_meta(args.audio_meta)
    video_meta = load_meta(args.video_meta)

    # Basic count output
    print(f"Audio embeddings: {audio_emb.shape}")
    print(f"Video embeddings: {video_emb.shape}")

    make_plots(
        audio_emb=audio_emb,
        video_emb=video_emb,
        audio_meta=audio_meta,
        video_meta=video_meta,
        out_dir=args.out_dir,
        k=args.k,
        n_queries=args.n_queries,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
