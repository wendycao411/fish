#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def species_from_clip_path(clip_path: str) -> str:
    return Path(clip_path).parent.name


def resolve_species_column(metadata: pd.DataFrame) -> pd.Series:
    if "species" in metadata.columns:
        raw_species = metadata["species"].astype(str).str.strip()
        fallback = metadata["clip_path"].astype(str).map(species_from_clip_path)
        return raw_species.where(raw_species.ne("") & raw_species.ne("nan"), fallback)
    return metadata["clip_path"].astype(str).map(species_from_clip_path)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def train_test_split_indices(
    n_samples: int, test_size: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(round(n_samples * test_size))
    n_test = max(1, min(n_samples - 1, n_test))
    test_indices = np.sort(indices[:n_test])
    train_indices = np.sort(indices[n_test:])
    return train_indices, test_indices


def evaluate_within_modal_species(
    query_embeddings: np.ndarray,
    db_embeddings: np.ndarray,
    query_species: np.ndarray,
    db_species: np.ndarray,
    query_clip_paths: np.ndarray,
    db_clip_paths: np.ndarray,
    exclude_self: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    query_norm = l2_normalize(query_embeddings)
    db_norm = l2_normalize(db_embeddings)
    similarities = query_norm @ db_norm.T

    same_arrays = (
        exclude_self
        and query_embeddings.shape[0] == db_embeddings.shape[0]
        and np.array_equal(query_clip_paths, db_clip_paths)
    )
    if same_arrays:
        np.fill_diagonal(similarities, -np.inf)

    n_queries = similarities.shape[0]
    n_db = similarities.shape[1]
    ranks = np.full(n_queries, n_db + 1, dtype=np.int64)
    best_positive_cosine = np.full(n_queries, np.nan, dtype=np.float64)

    for i in range(n_queries):
        positives = db_species == query_species[i]
        if same_arrays:
            positives[i] = False

        if not positives.any():
            continue

        row = similarities[i]
        best_positive_cosine[i] = np.max(row[positives])

        sorted_idx = np.argsort(-row)
        positive_positions = np.where(positives[sorted_idx])[0]
        if positive_positions.size > 0:
            ranks[i] = int(positive_positions[0] + 1)

    metrics = {
        "num_queries": float(n_queries),
        "num_db": float(n_db),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
    }

    ranks_df = pd.DataFrame(
        {
            "clip_path": query_clip_paths,
            "species": query_species,
            "rank": ranks,
            "best_positive_cosine": best_positive_cosine,
        }
    )
    return metrics, ranks_df


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"{name}: queries={int(metrics['num_queries'])} db={int(metrics['num_db'])} "
        f"R@1={metrics['recall_at_1']:.4f} R@5={metrics['recall_at_5']:.4f} "
        f"median_rank={metrics['median_rank']:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval using species labels for within-modal retrieval."
    )
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--audio-embeddings-npy", type=str, required=True)
    parser.add_argument("--video-embeddings-npy", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="species_within_modal",
        choices=["species_within_modal", "cross_modal"],
        help="Evaluation task. Default: species_within_modal.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    if "clip_path" not in metadata.columns:
        raise ValueError("metadata CSV must contain a 'clip_path' column.")
    metadata = metadata.copy()
    metadata["species"] = resolve_species_column(metadata)

    audio_embeddings = np.load(args.audio_embeddings_npy)
    video_embeddings = np.load(args.video_embeddings_npy)

    if audio_embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"audio rows ({audio_embeddings.shape[0]}) != metadata rows ({len(metadata)})."
        )
    if video_embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"video rows ({video_embeddings.shape[0]}) != metadata rows ({len(metadata)})."
        )

    train_idx, test_idx = train_test_split_indices(
        n_samples=len(metadata), test_size=args.test_size, random_seed=args.random_seed
    )
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    clip_paths = metadata["clip_path"].astype(str).to_numpy()
    species = metadata["species"].astype(str).to_numpy()

    if args.task == "species_within_modal":
        splits: List[Tuple[str, np.ndarray]] = [("test", test_idx), ("train", train_idx)]
        for split_name, split_indices in splits:
            split_paths = clip_paths[split_indices]
            split_species = species[split_indices]

            audio_metrics, audio_ranks = evaluate_within_modal_species(
                query_embeddings=audio_embeddings[split_indices],
                db_embeddings=audio_embeddings[split_indices],
                query_species=split_species,
                db_species=split_species,
                query_clip_paths=split_paths,
                db_clip_paths=split_paths,
                exclude_self=True,
            )
            audio_csv = output_dir / f"audio_to_audio_ranks_{split_name}.csv"
            audio_ranks.to_csv(audio_csv, index=False)
            print_metrics(f"audio_to_audio_{split_name}", audio_metrics)

            video_metrics, video_ranks = evaluate_within_modal_species(
                query_embeddings=video_embeddings[split_indices],
                db_embeddings=video_embeddings[split_indices],
                query_species=split_species,
                db_species=split_species,
                query_clip_paths=split_paths,
                db_clip_paths=split_paths,
                exclude_self=True,
            )
            video_csv = output_dir / f"video_to_video_ranks_{split_name}.csv"
            video_ranks.to_csv(video_csv, index=False)
            print_metrics(f"video_to_video_{split_name}", video_metrics)
    else:
        print(
            "Task 'cross_modal' is not run in this species-within-modal workflow. "
            "Use --task species_within_modal for audio->audio and video->video retrieval."
        )


if __name__ == "__main__":
    main()
