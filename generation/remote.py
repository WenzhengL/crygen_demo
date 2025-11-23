"""Remote submission helper for generation tasks."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dpdispatcher import Task

from common import cli as cli_helpers
from common import constants as C
from common import remote as remote_helpers
from common import staging

LOG = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "output"
LOG_DIR_NAME = "local_log"

DEFAULT_GENERATE_SCRIPT = C.TEST_ROOT / "run_generation.py"
DEFAULT_PULL_SCRIPT = C.TEST_ROOT / "generation" / "pull_results.py"


@dataclass(slots=True)
class GenerationSubmitOptions:
    generate_script: Path
    model_dir: Path
    remote_results_dir: str
    remote_model_dir: str
    batch_size: int | None
    num_batches: int | None
    num_gen: int | None
    wait: bool
    remote_config: remote_helpers.RemoteConfig


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="提交 MatterGen 生成任务到 Bohrium")
    cli_helpers.add_remote_common_args(parser, job_name_default="mattergen_generate")
    parser.add_argument(
        "--generate-script",
        type=_path,
        default=DEFAULT_GENERATE_SCRIPT,
        help="提交到远端执行的本地脚本 (默认: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        type=_path,
        default=C.DEFAULT_GENERATION_MODEL_DIR,
        help="本地 MatterGen 模型目录 (默认: %(default)s)",
    )
    parser.add_argument(
        "--remote-results-dir",
        default=OUTPUT_DIR_NAME,
        help="远程生成输出目录 (默认: %(default)s)",
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
        help="目标生成结构总数；设置后自动调整 batch_size 与 num_batches",
    )
    parser.add_argument(
        "--remote-model-dir",
        default=C.REMOTE_DEFAULT_MODEL_DIR,
        help="远端 MatterGen 模型目录",
    )
    return parser


def parse_args(argv: Iterable[str] | None = None) -> GenerationSubmitOptions:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
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

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    if args.num_batches is not None and args.num_batches <= 0:
        parser.error("--num-batches 必须为正整数")
    if args.num_gen is not None and args.num_gen <= 0:
        parser.error("--num-gen 必须为正整数")

    return GenerationSubmitOptions(
        generate_script=Path(args.generate_script),
        model_dir=Path(args.model_dir),
        remote_results_dir=args.remote_results_dir,
        remote_model_dir=args.remote_model_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_gen=args.num_gen,
        wait=args.wait,
        remote_config=config,
    )


def submit_generation_job(options: GenerationSubmitOptions) -> None:
    if not options.remote_config.email or not options.remote_config.password:
        raise SystemExit("需要提供 Bohrium 账户邮箱和密码，可通过 --dp-email/--dp-password 或环境变量设置。")

    job_root = staging.new_job_root(C.TEST_ROOT)
    output_dir = job_root / OUTPUT_DIR_NAME
    log_dir = job_root / LOG_DIR_NAME
    staging.ensure_dir(output_dir)
    staging.ensure_dir(log_dir)

    staged_script = staging.stage_script(options.generate_script, job_root)
    staging.stage_script(DEFAULT_PULL_SCRIPT, job_root)
    forward_files = [staged_script.name]

    if not options.model_dir.exists():
        raise FileNotFoundError(f"模型目录 {options.model_dir} 不存在")
    remote_model_path = Path(options.remote_model_dir)
    if remote_model_path.is_absolute():
        raise SystemExit("--remote-model-dir 必须是相对路径，例如 ./model")
    remote_parts = [part for part in remote_model_path.parts if part not in {"."}]
    if not remote_parts:
        remote_parts = ["model"]
    relative_target = "/".join(remote_parts)
    staging.stage_directory(options.model_dir, job_root, dst_name=relative_target)
    if remote_parts[0] not in forward_files:
        forward_files.append(remote_parts[0])

    python_cmd = [
        "python",
        staged_script.name,
        "--results-dir",
        options.remote_results_dir,
        "--model-dir",
        options.remote_model_dir,
    ]
    if options.batch_size is not None:
        python_cmd.extend(["--batch-size", str(options.batch_size)])
    if options.num_batches is not None:
        python_cmd.extend(["--num-batches", str(options.num_batches)])
    if options.num_gen is not None:
        python_cmd.extend(["--num-gen", str(options.num_gen)])

    command = (
        "source /root/dev/mattergen/.venv/bin/activate && "
        "pip install --no-cache-dir dpdispatcher && "
        f"{shlex.join(python_cmd)}"
    )

    task = Task(
        command=command,
        task_work_path="./",
        forward_files=forward_files,
        backward_files=[
            OUTPUT_DIR_NAME,
            f"{OUTPUT_DIR_NAME}/**",
            "log",
            "err",
            "lightning_logs",
            "lightning_logs/**",
        ],
    )
    submission = remote_helpers.build_submission(job_root, [task], options.remote_config)
    LOG.info("提交生成任务，job_root=%s", job_root)
    metadata = {
        "job_name": options.remote_config.job_name,
        "scass_type": options.remote_config.scass_type,
        "platform": options.remote_config.platform,
        "image_name": options.remote_config.image_name,
        "remote_results_dir": str(options.remote_results_dir),
        "remote_model_dir": str(options.remote_model_dir),
        "generate_script": staged_script.name,
        "batch_size": options.batch_size,
        "num_batches": options.num_batches,
        "num_gen": options.num_gen,
    }
    (job_root / "job_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    remote_helpers.run_submission(submission, options.wait)

    if options.wait:
        staging.relocate_artifacts(job_root, {staged_script.name, OUTPUT_DIR_NAME, LOG_DIR_NAME}, log_dir)
        LOG.info("生成任务完成，产出位于: %s", output_dir)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    options = parse_args(argv)
    submit_generation_job(options)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
