#!/usr/bin/env python3
"""Download artifacts for a finished MatterGen generation run."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dpdispatcher import Task

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from common import cli as cli_helpers
from common import constants as C
from common import remote as remote_helpers

LOG = logging.getLogger(__name__)

DEFAULT_REMOTE_RESULTS_DIR = "output"
DEFAULT_GENERATE_SCRIPT = "run_generation.py"
JOB_METADATA_FILE = "job_config.json"


@dataclass(slots=True)
class GenerationPullOptions:
    job_dir: Path
    generate_script: str
    remote_results_dir: str
    remote_model_path: str
    remote_config: remote_helpers.RemoteConfig
    clean: bool


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _load_job_metadata(job_dir: Path) -> dict:
    metadata_path = job_dir / JOB_METADATA_FILE
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        LOG.warning("Failed to parse %s: %s", metadata_path, exc)
        return {}


def build_parser(metadata: dict) -> argparse.ArgumentParser:
    job_name_default = metadata.get("job_name", "mattergen_generate")
    parser = argparse.ArgumentParser(description="拉取已完成的远程生成任务产出")
    cli_helpers.add_remote_common_args(parser, job_name_default=job_name_default)

    parser.add_argument(
        "--job-dir",
        type=_path,
        default=Path(__file__).resolve().parent,
        help="submit_gen.py 创建的 job_XXXX 目录 (默认: 当前脚本所在目录)",
    )
    parser.add_argument(
        "--generate-script",
        default=metadata.get("generate_script", DEFAULT_GENERATE_SCRIPT),
        help="远端执行的脚本名称 (默认: %(default)s)",
    )
    parser.add_argument(
        "--remote-results-dir",
        default=metadata.get("remote_results_dir", DEFAULT_REMOTE_RESULTS_DIR),
        help="远端产出目录，与提交时 --remote-results-dir 保持一致",
    )
    parser.add_argument(
        "--remote-model-path",
        default=metadata.get("remote_model_dir", C.REMOTE_DEFAULT_MODEL_DIR),
        help="远端模型路径，需与提交时一致",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="下载完成后清理远端工作目录",
    )
    return parser


def parse_args(argv: Iterable[str] | None = None) -> GenerationPullOptions:
    metadata = _load_job_metadata(Path(__file__).resolve().parent)
    parser = build_parser(metadata)
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

    return GenerationPullOptions(
        job_dir=Path(args.job_dir),
        generate_script=args.generate_script,
        remote_results_dir=args.remote_results_dir,
        remote_model_path=args.remote_model_path,
        remote_config=config,
        clean=args.clean,
    )


def _build_download_task(options: GenerationPullOptions) -> Task:
    command = (
        "source /root/dev/mattergen/.venv/bin/activate && "
        f"python {options.generate_script} --results-dir {options.remote_results_dir} "
        f"--model-dir {options.remote_model_path}"
    )
    return Task(
        command=command,
        task_work_path="./",
        forward_files=[options.generate_script],
        backward_files=[
            options.remote_results_dir,
            f"{options.remote_results_dir}/**",
            "log",
            "err",
        ],
    )


def pull_generation_results(options: GenerationPullOptions) -> None:
    if not options.job_dir.exists():
        raise FileNotFoundError(f"未找到 job 目录 {options.job_dir}")

    submission = remote_helpers.build_submission(options.job_dir, [_build_download_task(options)], options.remote_config)
    submission.run_submission(exit_on_submit=False, clean=options.clean)


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    options = parse_args(argv)
    pull_generation_results(options)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
