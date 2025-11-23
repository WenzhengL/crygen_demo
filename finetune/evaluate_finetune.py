#!/usr/bin/env python3
"""Evaluate a finetuned MatterGen checkpoint against the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from omegaconf import open_dict
from pytorch_lightning.callbacks import LearningRateMonitor

from mattergen.diffusion.run import maybe_instantiate
from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
from common import constants as C
from finetune.local import FinetuneOptions, build_overrides, compose_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate finetuned MatterGen checkpoints.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=C.DEFAULT_FINETUNE_DATA_ROOT,
        help=(
            "Path containing train/ and val/ caches. "
            "参考：微调时使用的目录 ../iter_0/data。"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=C.DEFAULT_FINETUNE_MODEL_DIR,
        help=(
            "Path to the baseline checkpoint used for finetuning;"
            " 默认读取 crygen_demo/model/last.ckpt。"
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=C.DEFAULT_MODEL_CHECKPOINT,
        help=(
            "Finetuned Lightning checkpoint to evaluate."
            " 参考值：微调后 best.ckpt 位于 checkpoints/ 目录。"
        ),
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of accelerator devices for validation (default: 1).",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Lightning accelerator choice (default: auto).",
    )
    parser.add_argument(
        "--precision",
        choices=["32", "16"],
        default="32",
        help="Evaluation precision (default: 32-bit).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Validation batch size override (default: 16，与微调脚本一致可比较)。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4).",
    )
    return parser.parse_args()


def _prepare_options(args: argparse.Namespace) -> FinetuneOptions:
    data_root = args.data_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    return FinetuneOptions(
        data_root=data_root,
        model_dir=model_dir,
        output_dir=PROJECT_ROOT,
        max_epochs=1,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy="auto",
        precision=args.precision,
        resume=None,
        use_wandb=False,
    )


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        candidate = (PROJECT_ROOT / path).resolve()
        if candidate.exists():
            resolved = candidate
        else:
            resolved = (SCRIPT_DIR / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Checkpoint {resolved} 不存在，请检查 --ckpt 参数或确保微调结果保存在 checkpoints/ 下。"
        )
    return resolved


def _format_metrics(title: str, metrics: list[Dict[str, float]]) -> None:
    """Pretty-print Lightning validation metrics."""
    if not metrics:
        print(f"{title}: no metrics returned")
        return
    print(title)
    for key, value in metrics[0].items():
        print(f"  {key:>20s}: {value:.6f}")


def main() -> None:
    args = parse_args()
    ckpt_path = _resolve_checkpoint(args.ckpt)
    finetune_options = _prepare_options(args)

    overrides = build_overrides(finetune_options)
    cfg = compose_config(overrides)

    trainer = maybe_instantiate(cfg.trainer)
    datamodule = maybe_instantiate(cfg.data_module)

    if not trainer.logger:
        trainer.callbacks = [
            cb for cb in trainer.callbacks if not isinstance(cb, LearningRateMonitor)
        ]

    pl_module, lightning_cfg = init_adapter_lightningmodule_from_pretrained(
        cfg.adapter, cfg.lightning_module
    )
    with open_dict(cfg):
        cfg.lightning_module = lightning_cfg

    print("==> 基线模型 (未微调) 验证")
    base_metrics = trainer.validate(model=pl_module, datamodule=datamodule)
    _format_metrics("Baseline metrics", base_metrics)

    print(f"==> 微调模型验证: {ckpt_path}")
    finetuned_metrics = trainer.validate(
        model=pl_module, datamodule=datamodule, ckpt_path=str(ckpt_path)
    )
    _format_metrics("Finetuned metrics", finetuned_metrics)

    if base_metrics and finetuned_metrics:
        deltas = {
            key: finetuned_metrics[0][key] - base_metrics[0].get(key, 0.0)
            for key in finetuned_metrics[0]
        }
        print("==> 指标变化 (finetuned - baseline)")
        for key, delta in deltas.items():
            print(f"  {key:>20s}: {delta:+.6f}")


if __name__ == "__main__":
    main()
