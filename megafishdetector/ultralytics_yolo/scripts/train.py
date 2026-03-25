#!/usr/bin/env python3
"""Train Ultralytics YOLO on MegaFish unified dataset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from common import default_data_yaml, default_runs_root, ensure_dir, save_json, to_serializable

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def parse_batch(value: str) -> int | float:
    try:
        numeric = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("batch must be numeric, e.g. 16, 0.7, or -1") from e

    if numeric.is_integer():
        return int(numeric)
    return numeric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO for MegaFish")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Model checkpoint (e.g., yolo11m.pt or yolo26m.pt)")
    parser.add_argument("--data", type=Path, default=default_data_yaml(), help="Data YAML path")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=parse_batch, default=-1, help="Batch size as int/float, or -1 for autobatch")
    parser.add_argument("--device", type=str, default="0", help="Device id or 'cpu'")
    parser.add_argument("--project", type=Path, default=default_runs_root())
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from the most recent checkpoint in project/name")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every N epochs (<1 disables snapshots)")
    parser.add_argument("--wandb-project", type=str, default="cfdd-yolo11", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb-tags", nargs="*", default=(), help="Optional Weights & Biases tags")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_serializable(payload), sort_keys=True) + "\n")


def existing_checkpoint_candidates(run_dir: Path, explicit: Path | None = None) -> list[Path]:
    weights_dir = run_dir / "weights"
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    add(explicit)
    add(weights_dir / "last.pt")

    epoch_ckpts = sorted(
        (p.resolve() for p in weights_dir.glob("epoch*.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in epoch_ckpts:
        add(path)

    add(weights_dir / "best.pt")
    return candidates


def pick_resume_checkpoint(run_dir: Path, explicit: Path | None = None) -> Path | None:
    candidates = existing_checkpoint_candidates(run_dir, explicit=explicit)
    return candidates[0] if candidates else None


def build_training_callbacks(args: argparse.Namespace, run_dir: Path, resume_ckpt: Path | None):
    state_path = run_dir / "training_state.json"
    history_path = run_dir / "epoch_history.jsonl"
    wandb_id_path = run_dir / "wandb_run_id.txt"
    wandb_state: dict[str, Any] = {"run": None}
    progress_state: dict[str, Any] = {"best_epoch": None}

    def checkpoint_state(trainer) -> dict[str, Any]:
        current_epoch = getattr(trainer, "epoch", -1) + 1
        start_epoch = getattr(trainer, "start_epoch", 0)
        epochs_target = getattr(trainer, "epochs", args.epochs)
        trainer_best_epoch = getattr(getattr(trainer, "stopper", None), "best_epoch", None)
        if trainer_best_epoch is not None:
            trainer_best_epoch += 1
        best_epoch = progress_state["best_epoch"] if progress_state["best_epoch"] is not None else trainer_best_epoch
        metrics = getattr(trainer, "metrics", None) or {}
        return {
            "resume_requested": bool(args.resume or args.resume_checkpoint),
            "resume_checkpoint": str(resume_ckpt) if resume_ckpt else None,
            "last_checkpoint": str(trainer.last.resolve()) if trainer.last.exists() else None,
            "best_checkpoint": str(trainer.best.resolve()) if trainer.best.exists() else None,
            "epochs_target": epochs_target,
            "start_epoch": start_epoch,
            "current_epoch": current_epoch,
            "epochs_completed": max(current_epoch, start_epoch),
            "best_epoch": best_epoch,
            "best_fitness": getattr(trainer, "best_fitness", None),
            "fitness": getattr(trainer, "fitness", None),
            "metrics": metrics,
            "updated_at": utc_now(),
        }

    def init_wandb(trainer) -> None:
        if wandb is None:
            return
        run_id = wandb_id_path.read_text(encoding="utf-8").strip() if wandb_id_path.exists() else wandb.util.generate_id()
        ensure_dir(wandb_id_path.parent)
        wandb_id_path.write_text(run_id + "\n", encoding="utf-8")
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            id=run_id,
            resume="allow",
            dir=str(run_dir),
            config={k: to_serializable(v) for k, v in vars(args).items()},
            tags=list(args.wandb_tags),
        )
        if wandb.run is None:
            try:
                wandb_state["run"] = wandb.init(**init_kwargs)
            except Exception as exc:
                if "No API key configured" not in str(exc):
                    raise
                print(
                    "[WARN] W&B API key not configured on this node; falling back to offline logging. "
                    "Run `wandb login` or export WANDB_API_KEY for online sync."
                )
                wandb_state["run"] = wandb.init(mode="offline", **init_kwargs)
        else:
            wandb_state["run"] = wandb.run

    def on_pretrain_routine_start(trainer) -> None:
        ensure_dir(run_dir)
        init_wandb(trainer)
        save_json(
            state_path,
            {
                "status": "initializing",
                "run_dir": str(run_dir.resolve()),
                "run_name": args.name,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_id": wandb_state["run"].id if wandb_state["run"] else None,
                **checkpoint_state(trainer),
            },
        )

    def on_train_start(trainer) -> None:
        save_json(
            state_path,
            {
                "status": "training",
                "run_dir": str(run_dir.resolve()),
                "run_name": args.name,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_id": wandb_state["run"].id if wandb_state["run"] else None,
                **checkpoint_state(trainer),
            },
        )

    def on_fit_epoch_end(trainer) -> None:
        epoch = trainer.epoch + 1
        train_metrics = trainer.label_loss_items(trainer.tloss, prefix="train") if getattr(trainer, "tloss", None) is not None else {}
        log_payload = {
            "epoch": epoch,
            "epochs_target": trainer.epochs,
            "progress": epoch / trainer.epochs if trainer.epochs else None,
            "best_fitness": getattr(trainer, "best_fitness", None),
            **train_metrics,
            **getattr(trainer, "lr", {}),
            **(getattr(trainer, "metrics", None) or {}),
        }
        if getattr(trainer, "fitness", None) is not None and getattr(trainer, "best_fitness", None) == trainer.fitness:
            progress_state["best_epoch"] = epoch
        append_jsonl(
            history_path,
            {
                "timestamp": utc_now(),
                **log_payload,
            },
        )
        save_json(
            state_path,
            {
                "status": "training",
                "run_dir": str(run_dir.resolve()),
                "run_name": args.name,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_id": wandb_state["run"].id if wandb_state["run"] else None,
                **checkpoint_state(trainer),
            },
        )
        if wandb_state["run"] is not None:
            wandb_state["run"].log(log_payload, step=epoch)

    def on_train_end(trainer) -> None:
        save_json(
            state_path,
            {
                "status": "completed",
                "run_dir": str(run_dir.resolve()),
                "run_name": args.name,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_run_id": wandb_state["run"].id if wandb_state["run"] else None,
                **checkpoint_state(trainer),
                "completed_at": utc_now(),
            },
        )
        if wandb_state["run"] is not None:
            wandb_state["run"].finish()
            wandb_state["run"] = None

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.project / args.name
    resume_ckpt = pick_resume_checkpoint(run_dir, explicit=args.resume_checkpoint) if (args.resume or args.resume_checkpoint) else None

    train_kwargs = dict(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        save_period=args.save_period,
        exist_ok=True,
    )

    if resume_ckpt is not None:
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        train_kwargs["resume"] = str(resume_ckpt)
    else:
        if args.resume or args.resume_checkpoint:
            print(f"[WARN] Resume requested but no checkpoint found under {run_dir / 'weights'}; starting fresh")
        model = YOLO(args.model)

    callbacks = build_training_callbacks(args=args, run_dir=run_dir, resume_ckpt=resume_ckpt)
    for event, callback in callbacks.items():
        model.add_callback(event, callback)

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
