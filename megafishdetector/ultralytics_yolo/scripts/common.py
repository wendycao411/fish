#!/usr/bin/env python3
"""Shared helpers for Ultralytics MegaFish scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    # scripts/ -> ultralytics_yolo/ -> megafishdetector/
    return Path(__file__).resolve().parents[2]


def default_data_yaml() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "megafish.yaml"


def default_runs_root() -> Path:
    return repo_root() / "runs" / "ultralytics_megafish"


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def to_serializable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
