"""Local finetuning entry point for MatterGen.

This module is imported by remote submission code only to access the
FinetuneOptions dataclass. To avoid requiring heavy training dependencies
for submission, most deep imports are deferred until `run_finetune`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:  # minimal torch usage for accelerator resolution; fallback to CPU if missing
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _TorchStub:  # noqa: D401
        @staticmethod
        def cuda():  # mimic torch.cuda
            class _C:
                @staticmethod
                def is_available():
                    return False
            return _C()

        @staticmethod
        def set_float32_matmul_precision(*_args, **_kwargs):
            return None

    torch = _TorchStub()  # type: ignore

# Heavy imports (Hydra, Lightning, OmegaConf, mattergen) are deferred to run_finetune/compose_config.

from common import cli as cli_helpers


@dataclass(slots=True)
class FinetuneOptions:
    data_root: Path
    model_dir: Path
    output_dir: Path
    max_epochs: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    devices: int
    accelerator: str
    strategy: str
    precision: str
    resume: Path | None
    use_wandb: bool

    @property
    def effective_accelerator(self) -> str:
        if self.accelerator != "auto":
            return self.accelerator
        return "gpu" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 MatterGen 微调任务")
    cli_helpers.add_finetune_data_args(parser)
    cli_helpers.add_finetune_model_args(parser)
    cli_helpers.add_finetune_training_args(parser)
    return parser


def parse_args(argv: Iterable[str] | None = None) -> FinetuneOptions:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return FinetuneOptions(
        data_root=Path(args.data_root),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        resume=Path(args.resume) if args.resume else None,
        use_wandb=args.use_wandb,
    )


def build_overrides(options: FinetuneOptions) -> List[str]:
    data_root = options.data_root.resolve()
    model_dir = options.model_dir.resolve()

    overrides = [
        f"data_module.root_dir={data_root.as_posix()}",
        f"data_module.train_dataset.cache_path={(data_root / 'train').as_posix()}",
        f"data_module.val_dataset.cache_path={(data_root / 'val').as_posix()}",
        "data_module.test_dataset=null",
        f"adapter.model_path={model_dir.as_posix()}",
        "adapter.pretrained_name=null",
        f"trainer.max_epochs={options.max_epochs}",
        f"data_module.max_epochs={options.max_epochs}",
        f"trainer.devices={options.devices}",
        f"trainer.accelerator={options.effective_accelerator}",
        f"trainer.precision={options.precision}",
        f"trainer.strategy={options.strategy}",
        f"data_module.batch_size.train={options.train_batch_size}",
        f"data_module.batch_size.val={options.val_batch_size}",
        f"data_module.batch_size.test={options.val_batch_size}",
        f"data_module.num_workers.train={options.num_workers}",
        f"data_module.num_workers.val={options.num_workers}",
        f"data_module.num_workers.test={max(0, options.num_workers // 2)}",
    ]
    if not options.use_wandb:
        overrides.append("trainer.logger=false")
    return overrides


def compose_config(overrides: List[str]):
    try:
        from hydra import compose, initialize_config_dir  # type: ignore
        from hydra.core.global_hydra import GlobalHydra  # type: ignore
        from mattergen.common.utils.globals import MODELS_PROJECT_ROOT  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "Hydra 未安装。请在需要本地训练/微调时先执行: pip install hydra-core omegaconf"
        ) from e
    config_dir = MODELS_PROJECT_ROOT / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), job_name="mattergen_finetune", version_base="1.1"):
        cfg = compose(config_name="finetune", overrides=overrides)
    return cfg


def run_finetune(options: FinetuneOptions) -> None:
    # Import heavy dependencies only when actually performing training.
    from omegaconf import OmegaConf, open_dict  # type: ignore
    from pytorch_lightning import LightningDataModule, Trainer  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
    from pytorch_lightning.cli import SaveConfigCallback  # type: ignore
    from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate  # type: ignore
    from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained  # type: ignore
    from mattergen.common.utils.globals import MODELS_PROJECT_ROOT  # type: ignore
    output_dir_abs = options.output_dir.expanduser().resolve()
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(output_dir_abs)

    train_cache = options.data_root / "train"
    val_cache = options.data_root / "val"
    if not train_cache.exists() or not val_cache.exists():
        raise FileNotFoundError(f"期望在 {options.data_root} 下找到 train/ 与 val/ 缓存目录")
    if not options.model_dir.exists():
        raise FileNotFoundError(f"预训练模型目录 {options.model_dir} 不存在")

    overrides = build_overrides(options)
    cfg = compose_config(overrides)

    torch.set_float32_matmul_precision("high")

    trainer: Trainer = maybe_instantiate(cfg.trainer, Trainer)
    datamodule: LightningDataModule = maybe_instantiate(cfg.data_module, LightningDataModule)

    if not trainer.logger:
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, LearningRateMonitor)]

    pl_module, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(cfg.adapter, cfg.lightning_module)

    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer.callbacks.append(SaveConfigCallback(parser=SimpleParser(), config=config_as_dict, overwrite=True))
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    # Ensure checkpoints are saved (best + last) similar to original mattergen trainer default.
    ckpt_dir = output_dir_abs / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    has_ckpt = any(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
    if not has_ckpt:
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="last",  # last epoch ckpt
                save_last=True,
                every_n_epochs=1,
                save_top_k=0,
            )
        )

    trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=str(options.resume) if options.resume else None)

    # Always emit a final checkpoint to a deterministic location.
    try:
        final_ckpt = ckpt_dir / "last.ckpt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_ckpt))
    except Exception:
        pass


def main(argv: Iterable[str] | None = None) -> None:
    options = parse_args(argv)
    run_finetune(options)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
