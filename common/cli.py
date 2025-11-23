"""Command-line parsing helpers shared across modules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from . import constants as C


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def add_generation_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-dir",
        type=_path,
        default=C.DEFAULT_GENERATION_RESULTS_DIR,
        help="生成结果输出目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        type=_path,
        default=C.DEFAULT_GENERATION_MODEL_DIR,
        help="MatterGen 预训练模型目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="单批次生成结构数量 (默认: 4，或配合 --num-gen 自动计算)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="生成批次数 (默认: 2，或配合 --num-gen 自动计算)",
    )
    parser.add_argument(
        "--num-gen",
        type=int,
        default=None,
        help="目标生成结构总数；设置后将自动调整 batch_size 与 num_batches",
    )


def add_remote_common_args(parser: argparse.ArgumentParser, *, job_name_default: str) -> None:
    parser.add_argument("--dp-email", default="liaowenzheng@nimte.ac.cn", help="Bohrium 账户邮箱")
    parser.add_argument("--dp-password", default="947436424948wz.", help="Bohrium 账户密码")
    parser.add_argument("--program-id", type=int, default=C.DEFAULT_PROGRAM_ID, help="Bohrium program_id")
    parser.add_argument("--job-name", default=job_name_default, help="远程作业名称")
    parser.add_argument("--scass-type", default=C.DEFAULT_SCASS_TYPE, help="算力配置")
    parser.add_argument("--platform", default=C.DEFAULT_PLATFORM, help="Bohrium 平台")
    parser.add_argument("--image-name", default=C.DEFAULT_REMOTE_IMAGE, help="容器镜像")
    parser.add_argument("--group-size", type=int, default=1, help="dpdispatcher group_size")
    parser.add_argument("--wait", action="store_true", help="阻塞等待远程任务完成")


def add_finetune_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=_path,
        default=C.DEFAULT_FINETUNE_DATA_ROOT,
        help="本地数据缓存目录",
    )
    parser.add_argument(
        "--data-root-remote",
        default=None,
        help="远端已有的数据目录 (设置后跳过上传)",
    )
    parser.add_argument(
        "--remote-data-dir-name",
        default="finetune_data",
        help="远端展开数据目录名 (相对路径)",
    )


def add_finetune_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-dir",
        type=_path,
        default=C.DEFAULT_FINETUNE_MODEL_DIR,
        help="本地预训练模型目录",
    )
    parser.add_argument(
        "--model-dir-remote",
        default=None,
        help="远端已有模型目录 (设置后跳过上传)",
    )
    parser.add_argument(
        "--remote-model-dir-name",
        default="finetune_model",
        help="远端展开模型目录名 (相对路径)",
    )


def add_finetune_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=_path, default=C.DEFAULT_FINETUNE_OUTPUT_DIR, help="Lightning 输出目录")
    parser.add_argument("--max-epochs", type=int, default=50, help="最大训练轮数")
    parser.add_argument("--train-batch-size", type=int, default=32, help="训练 batch size")
    parser.add_argument("--val-batch-size", type=int, default=32, help="验证 batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--devices", type=int, default=1, help="训练使用的设备数量")
    parser.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto", help="Lightning accelerator")
    parser.add_argument("--strategy", default="auto", help="Lightning strategy")
    parser.add_argument("--precision", choices=["32", "16"], default="32", help="训练精度")
    parser.add_argument("--resume", type=_path, default=None, help="断点恢复 checkpoint")
    parser.add_argument("--use-wandb", action="store_true", help="启用 WandB")


def merge_known_args(parser: argparse.ArgumentParser, argv: Iterable[str] | None = None) -> argparse.Namespace:
    args, _ = parser.parse_known_args(argv)
    return args
