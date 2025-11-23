"""Remote submission logic for finetuning."""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile

SUPPORT_ARCHIVE_NAME = "support.tar.gz"
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from dpdispatcher import Task

from common import cli as cli_helpers
from common import constants as C
from common import remote as remote_helpers
from common import staging
from finetune.local import FinetuneOptions

LOG = logging.getLogger(__name__)

RUN_SCRIPT = "run_finetune.py"
OUTPUT_DIR_NAME = "outputs"
LOG_DIR_NAME = "local_log"

DEFAULT_RUN_SCRIPT = C.TEST_ROOT / "finetune" / RUN_SCRIPT


@dataclass(slots=True)
class FinetuneSubmitOptions:
    training: FinetuneOptions
    data_root_remote: str
    model_dir_remote: str
    model_remote_path: str
    remote_config: remote_helpers.RemoteConfig
    wait: bool
    run_script: Path
    stage_data: bool
    stage_model: bool
    resume_path: Optional[Path]
    remote_data_dir_name: str
    remote_model_dir_name: str


def _path(value: str | None) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="提交 MatterGen 微调任务到 Bohrium")
    cli_helpers.add_remote_common_args(parser, job_name_default="mattergen_finetune")
    cli_helpers.add_finetune_data_args(parser)
    cli_helpers.add_finetune_model_args(parser)
    cli_helpers.add_finetune_training_args(parser)
    parser.add_argument(
        "--run-script",
        type=lambda v: Path(v).expanduser().resolve(),
        default=DEFAULT_RUN_SCRIPT,
        help="需要提交到远端执行的 run_finetune.py 路径",
    )
    parser.add_argument(
        "--include-backup",
        action="store_true",
        help="保留远端原始产出备份 (提交完成后不整理)",
    )
    return parser


def _resolve_remote_dir(arg_value: Optional[str], default_relative: str, *, stage: bool) -> str:
    if arg_value:
        return arg_value
    if stage:
        return f"./{default_relative}"
    raise SystemExit("未提供远端路径且本地目录不存在，无法继续。")


def _join_remote_path(base: str, name: str) -> str:
    if base.endswith("/"):
        return f"{base}{name}"
    return f"{base}/{name}"


def parse_args(argv: Iterable[str] | None = None) -> FinetuneSubmitOptions:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    resume_path = _path(args.resume)

    stage_data = data_root.exists() and args.data_root_remote is None
    stage_model = model_dir.exists() and args.model_dir_remote is None
    model_is_file = model_dir.is_file()

    remote_data_root = _resolve_remote_dir(args.data_root_remote, args.remote_data_dir_name, stage=stage_data)
    remote_model_dir = _resolve_remote_dir(args.model_dir_remote, args.remote_model_dir_name, stage=stage_model)

    if stage_data and remote_data_root.startswith("/"):
        raise SystemExit("当需要上传数据时，远端数据目录必须是相对路径 (例: ./finetune_data)")
    if stage_model and remote_model_dir.startswith("/"):
        raise SystemExit("当需要上传模型时，远端模型目录必须是相对路径 (例: ./finetune_model)")

    if stage_model and model_is_file:
        model_remote_path = _join_remote_path(remote_model_dir.rstrip("/"), model_dir.name)
    elif args.model_dir_remote is not None:
        model_remote_path = args.model_dir_remote
    else:
        model_remote_path = remote_model_dir

    training_opts = FinetuneOptions(
        data_root=data_root,
        model_dir=model_dir,
        output_dir=Path(OUTPUT_DIR_NAME),
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        resume=resume_path,
        use_wandb=args.use_wandb,
    )

    config = remote_helpers.RemoteConfig(
        email=args.dp_email,
        password=args.dp_password,
        program_id=args.program_id,
        job_name=args.job_name,
        scass_type=args.scass_type,
        platform=args.platform,
        image_name=args.image_name,
        group_size=args.group_size,
    )

    return FinetuneSubmitOptions(
        training=training_opts,
        data_root_remote=remote_data_root,
        model_dir_remote=remote_model_dir,
    model_remote_path=model_remote_path,
        remote_config=config,
        wait=args.wait,
        run_script=args.run_script,
        stage_data=stage_data,
        stage_model=stage_model,
        resume_path=resume_path,
        remote_data_dir_name=args.remote_data_dir_name,
        remote_model_dir_name=args.remote_model_dir_name,
    )


def _support_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if not parts:
        return tarinfo
    if "__pycache__" in parts:
        return None
    if parts[0] == "finetune" and len(parts) > 1 and parts[1].startswith("job_"):
        return None
    return tarinfo


def _stage_support_archive(job_root: Path) -> Path:
    archive_path = job_root / SUPPORT_ARCHIVE_NAME
    with tarfile.open(archive_path, "w:gz") as tar:
        for name in ("common", "finetune"):
            src = C.TEST_ROOT / name
            if not src.exists():
                continue
            tar.add(src, arcname=name, filter=_support_filter)
    return archive_path


def _archive_name(stem: str) -> str:
    return f"{stem}.tar.gz"


def _build_python_command(options: FinetuneSubmitOptions) -> str:
    parts = [
        f"python {RUN_SCRIPT}",
        f"--data-root {options.data_root_remote}",
        f"--model-dir {options.model_remote_path}",
        f"--output-dir {OUTPUT_DIR_NAME}",
        f"--max-epochs {options.training.max_epochs}",
        f"--train-batch-size {options.training.train_batch_size}",
        f"--val-batch-size {options.training.val_batch_size}",
        f"--num-workers {options.training.num_workers}",
        f"--devices {options.training.devices}",
        f"--accelerator {options.training.accelerator}",
        f"--strategy {options.training.strategy}",
        f"--precision {options.training.precision}",
    ]
    if options.resume_path:
        parts.append(f"--resume {options.resume_path.name}")
    if options.training.use_wandb:
        parts.append("--use-wandb")
    return " ".join(parts)


def submit_finetune_job(options: FinetuneSubmitOptions) -> None:
    if not options.remote_config.email or not options.remote_config.password:
        raise SystemExit("需要提供 Bohrium 账户邮箱和密码，可通过 --dp-email/--dp-password 或环境变量设置。")

    job_root = staging.new_job_root(C.TEST_ROOT / "finetune")
    output_dir = job_root / OUTPUT_DIR_NAME
    log_dir = job_root / LOG_DIR_NAME
    staging.ensure_dir(output_dir)
    staging.ensure_dir(log_dir)

    staged_script = staging.stage_script(options.run_script, job_root)

    forward_files: List[str] = [staged_script.name]
    support_archive = _stage_support_archive(job_root)
    forward_files.append(support_archive.name)
    command_parts = ["source /root/dev/mattergen/.venv/bin/activate", "pip install --no-cache-dir dpdispatcher"]
    command_parts.append(f"tar -xf {support_archive.name}")

    if options.stage_data:
        archive_name = _archive_name(options.remote_data_dir_name)
        archive_path = staging.stage_directory_to_archive(options.training.data_root, job_root, options.remote_data_dir_name)
        desired = job_root / archive_name
        if archive_path.name != archive_name:
            archive_path.rename(desired)
        forward_files.append(archive_name)
        command_parts.append(
            f"rm -rf {options.data_root_remote} && mkdir -p {options.data_root_remote} && tar -xf {archive_name} -C {options.data_root_remote}"
        )
    if options.stage_model:
        archive_name = _archive_name(options.remote_model_dir_name)
        model_source = options.training.model_dir
        if model_source.is_dir():
            archive_path = staging.stage_directory_to_archive(model_source, job_root, options.remote_model_dir_name)
        else:
            payload_dir = job_root / f"{options.remote_model_dir_name}_payload"
            if payload_dir.exists():
                shutil.rmtree(payload_dir)
            staging.ensure_dir(payload_dir)
            staging.stage_single_file(model_source, payload_dir)
            archive_path = staging.stage_directory_to_archive(payload_dir, job_root, options.remote_model_dir_name)
            shutil.rmtree(payload_dir, ignore_errors=True)
        desired = job_root / archive_name
        if archive_path.name != archive_name:
            archive_path.rename(desired)
        forward_files.append(archive_name)
        command_parts.append(
            f"rm -rf {options.model_dir_remote} && mkdir -p {options.model_dir_remote} && tar -xf {archive_name} -C {options.model_dir_remote}"
        )

    if options.resume_path:
        staged_resume = staging.stage_single_file(options.resume_path, job_root)
        forward_files.append(staged_resume.name)

    command_parts.append(_build_python_command(options))
    command = " && ".join(command_parts)

    backward_files = [
        OUTPUT_DIR_NAME,
        f"{OUTPUT_DIR_NAME}/**",
        "log",
        "err",
        "lightning_logs",
        "lightning_logs/**",
    ]

    task = Task(
        command=command,
        task_work_path="./",
        forward_files=forward_files,
        backward_files=backward_files,
    )

    submission = remote_helpers.build_submission(job_root, [task], options.remote_config)
    LOG.info("提交微调任务，job_root=%s", job_root)
    remote_helpers.run_submission(submission, options.wait)

    if options.wait:
        staging.relocate_artifacts(job_root, {staged_script.name, OUTPUT_DIR_NAME, LOG_DIR_NAME}, log_dir)
        LOG.info("微调任务完成，产出位于: %s", output_dir)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    options = parse_args(argv)
    submit_finetune_job(options)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
