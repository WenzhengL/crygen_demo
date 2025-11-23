"""远程提交 fine_gen 流水线 (微调+评估+生成) 到 Bohrium.

流程:
1. 打包支持代码(common/ finetune/ generation/ fine_gen/)。
2. 可选上传数据与基线模型目录 (若本地存在且未指定远端路径)。
3. 在远端环境激活虚拟环境、安装 dpdispatcher、解压资源并执行 run_fine_gen.py。
4. 回传 pipeline_run/ 全部产物以及 log/ err。
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from dpdispatcher import Task

from common import constants as C
from common import remote as remote_helpers
from common import staging

LOG = logging.getLogger(__name__)

RUN_SCRIPT = "run_fine_gen.py"
SUPPORT_ARCHIVE_NAME = "support.tar.gz"
OUTPUT_DIR_NAME = "pipeline_run"
LOG_DIR_NAME = "local_log"

DEFAULT_RUN_SCRIPT = Path(__file__).resolve().parent / RUN_SCRIPT


@dataclass(slots=True)
class FineGenSubmitOptions:
    data_root: Path
    base_model_dir: Path
    remote_data_root: str
    remote_model_dir: str
    pipeline_root: Path
    max_epochs: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    devices: int
    accelerator: str
    precision: str
    num_gen: int
    use_wandb: bool
    remote_config: remote_helpers.RemoteConfig
    wait: bool
    run_script: Path
    stage_data: bool
    stage_model: bool
    remote_data_dir_name: str
    remote_model_dir_name: str


def _path(value: str | None) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="提交 Fine-Gen 全流程到 Bohrium")
    # 手动添加远程运行所需通用参数（与其它 remote 脚本保持一致语义）。
    p.add_argument("--dp-email", type=str, default="", help="Bohrium 登录邮箱")
    p.add_argument("--dp-password", type=str, default="", help="Bohrium 登录密码")
    p.add_argument("--program-id", type=int, default=C.DEFAULT_PROGRAM_ID, help="Bohrium Program ID")
    p.add_argument("--job-name", type=str, default="fine_gen_pipeline", help="作业名称")
    p.add_argument("--scass-type", type=str, default=C.DEFAULT_SCASS_TYPE, help="计算资源规格")
    p.add_argument("--platform", type=str, default=C.DEFAULT_PLATFORM, help="平台标识 (ali 等)")
    p.add_argument("--image-name", type=str, default=C.DEFAULT_REMOTE_IMAGE, help="容器镜像名称")
    p.add_argument("--group-size", type=int, default=1, help="节点数量")
    p.add_argument("--data-root", type=lambda v: Path(v).expanduser().resolve(), default=C.DEFAULT_FINETUNE_DATA_ROOT,
                   help="本地数据根目录，包含 train/ 与 val/ 子目录")
    p.add_argument("--data-root-remote", type=str, default=None, help="远端数据目录(存在则不打包上传)")
    p.add_argument("--remote-data-dir-name", type=str, default="finetune_data", help="上传数据在远端的解压目录名")
    p.add_argument("--base-model-dir", type=lambda v: Path(v).expanduser().resolve(), default=C.DEFAULT_FINETUNE_MODEL_DIR,
                   help="基线模型目录 (mattergen_base)")
    p.add_argument("--model-dir-remote", type=str, default=None, help="远端基线模型目录(存在则不上传)")
    p.add_argument("--remote-model-dir-name", type=str, default="model", help="上传模型在远端的解压目录名")
    p.add_argument("--pipeline-root", type=lambda v: Path(v).expanduser().resolve(), default=Path("./") / OUTPUT_DIR_NAME,
                   help="远端流水线根目录 (默认: ./pipeline_run)")
    p.add_argument("--max-epochs", type=int, default=1, help="微调 epoch 数")
    p.add_argument("--train-batch-size", type=int, default=32, help="训练 batch 大小")
    p.add_argument("--val-batch-size", type=int, default=32, help="验证 batch 大小")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--devices", type=int, default=1, help="加速器设备数量")
    p.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto", help="Lightning accelerator")
    p.add_argument("--precision", choices=["32", "16"], default="32", help="计算精度")
    p.add_argument("--num-gen", type=int, default=8, help="生成结构数量")
    p.add_argument("--use-wandb", action="store_true", help="启用 W&B 日志")
    p.add_argument("--run-script", type=lambda v: Path(v).expanduser().resolve(), default=DEFAULT_RUN_SCRIPT,
                   help="本地运行脚本路径 run_fine_gen.py")
    p.add_argument("--wait", action="store_true", help="提交后等待完成并回传产物")
    return p


def parse_args(argv: Iterable[str] | None = None) -> FineGenSubmitOptions:
    p = build_parser()
    args = p.parse_args(list(argv) if argv is not None else None)

    data_root = args.data_root
    model_dir = args.base_model_dir

    stage_data = data_root.exists() and args.data_root_remote is None
    stage_model = model_dir.exists() and args.model_dir_remote is None

    if stage_data and str(args.remote_data_dir_name).startswith("/"):
        raise SystemExit("上传数据时 remote_data_dir_name 必须为相对路径")
    if stage_model and str(args.remote_model_dir_name).startswith("/"):
        raise SystemExit("上传模型时 remote_model_dir_name 必须为相对路径")

    remote_data_root = args.data_root_remote or f"./{args.remote_data_dir_name}"
    remote_model_dir = args.model_dir_remote or f"./{args.remote_model_dir_name}"

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

    return FineGenSubmitOptions(
        data_root=data_root,
        base_model_dir=model_dir,
        remote_data_root=remote_data_root,
        remote_model_dir=remote_model_dir,
        pipeline_root=args.pipeline_root,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        num_gen=args.num_gen,
        use_wandb=args.use_wandb,
        remote_config=config,
        wait=args.wait,
        run_script=args.run_script,
        stage_data=stage_data,
        stage_model=stage_model,
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
    if parts[0] == "fine_gen" and len(parts) > 1 and parts[1].startswith("job_"):
        return None
    return tarinfo


def _stage_support(job_root: Path) -> Path:
    archive = job_root / SUPPORT_ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as tf:
        # Stage demo support code
        for name in ("common", "finetune", "generation", "fine_gen"):
            src = C.TEST_ROOT / name
            if src.exists():
                tf.add(src, arcname=name, filter=_support_filter)
        # Stage mattergen source (exclude heavy checkpoint directory; keep data-release for evaluation)
        mg_root = C.MATTERGEN_ROOT
        if mg_root.exists():
            def _mg_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
                parts = Path(tarinfo.name).parts
                if not parts:
                    return tarinfo
                # Skip checkpoints and datasets (very large),
                # but KEEP data-release so evaluation can access
                # reference_MP2020correction.gz and mp_20.zip.
                if len(parts) > 1 and parts[1] in {"checkpoints", "datasets"}:
                    return None
                if "__pycache__" in parts:
                    return None
                return tarinfo
            tf.add(mg_root, arcname="mattergen", filter=_mg_filter)
    return archive


def _archive_name(stem: str) -> str:
    return f"{stem}.tar.gz"


def submit_pipeline(options: FineGenSubmitOptions) -> None:
    if not options.remote_config.email or not options.remote_config.password:
        raise SystemExit("需要提供 Bohrium 账户 --dp-email / --dp-password")

    job_root = staging.new_job_root(C.TEST_ROOT / "fine_gen")
    output_dir = job_root / OUTPUT_DIR_NAME
    log_dir = job_root / LOG_DIR_NAME
    staging.ensure_dir(output_dir)
    staging.ensure_dir(log_dir)

    staged_script = staging.stage_script(options.run_script, job_root)
    support_archive = _stage_support(job_root)

    forward_files: List[str] = [staged_script.name, support_archive.name]
    command_parts: List[str] = [
        "source /root/dev/mattergen/.venv/bin/activate",
        "pip install --no-cache-dir dpdispatcher",
        f"tar -xf {support_archive.name}",
        # Install mattergen so that mattergen-generate CLI becomes available
        "pip install --no-cache-dir ./mattergen",
    ]

    if options.stage_data:
        archive_name = _archive_name(options.remote_data_dir_name)
        data_archive = staging.stage_directory_to_archive(options.data_root, job_root, options.remote_data_dir_name)
        desired = job_root / archive_name
        if data_archive.name != archive_name:
            data_archive.rename(desired)
        forward_files.append(archive_name)
        command_parts.append(
            f"rm -rf {options.remote_data_root} && mkdir -p {options.remote_data_root} && tar -xf {archive_name} -C {options.remote_data_root}"
        )

    if options.stage_model:
        archive_name = _archive_name(options.remote_model_dir_name)
        model_archive = staging.stage_directory_to_archive(options.base_model_dir, job_root, options.remote_model_dir_name)
        desired = job_root / archive_name
        if model_archive.name != archive_name:
            model_archive.rename(desired)
        forward_files.append(archive_name)
        command_parts.append(
            f"rm -rf {options.remote_model_dir} && mkdir -p {options.remote_model_dir} && tar -xf {archive_name} -C {options.remote_model_dir}"
        )

    python_cmd = [
        "python",
        staged_script.name,
        f"--data-root {options.remote_data_root}",
        f"--base-model-dir {options.remote_model_dir}",
        f"--pipeline-root {OUTPUT_DIR_NAME}",
        f"--max-epochs {options.max_epochs}",
        f"--train-batch-size {options.train_batch_size}",
        f"--val-batch-size {options.val_batch_size}",
        f"--num-workers {options.num_workers}",
        f"--devices {options.devices}",
        f"--accelerator {options.accelerator}",
        f"--precision {options.precision}",
        f"--num-gen {options.num_gen}",
    ]
    if options.use_wandb:
        python_cmd.append("--use-wandb")

    command_parts.append(" ".join(python_cmd))
    command = " && ".join(command_parts)

    backward_files = [
        OUTPUT_DIR_NAME,
        f"{OUTPUT_DIR_NAME}/**",
        "log",
        "err",
        LOG_DIR_NAME,
        f"{LOG_DIR_NAME}/**",
    ]

    task = Task(
        command=command,
        task_work_path="./",
        forward_files=forward_files,
        backward_files=backward_files,
    )

    submission = remote_helpers.build_submission(job_root, [task], options.remote_config)
    LOG.info("提交 fine_gen 流水线任务 job_root=%s", job_root)
    remote_helpers.run_submission(submission, options.wait)

    if options.wait:
        staging.relocate_artifacts(job_root, {staged_script.name, OUTPUT_DIR_NAME, LOG_DIR_NAME}, log_dir)
        LOG.info("流水线完成，产出位于: %s", output_dir)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    options = parse_args(argv)
    submit_pipeline(options)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
