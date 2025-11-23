"""远程评估 MatterGen 生成结构的脚本。

功能：
- 将 `generation/evaluate.py` 与 dpdispatcher/bohrium 集成，在远程容器内
  对 `generated_crystals.extxyz` 做结构评估，回传 metrics。

典型用法（本地）：
    python crygen_demo/generation/submit_eval_gen.py \
        --wait \
        --dp-email ... --dp-password '...' --program-id 29496 \
        --results-dir crygen_demo/fine_gen/job_xxx/pipeline_run/generation/results
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dpdispatcher import Task

from common import constants as C
from common import remote as remote_helpers
from common import staging

LOG = logging.getLogger(__name__)

RUN_MODULE = "generation.evaluate"
SUPPORT_ARCHIVE_NAME = "support_eval.tar.gz"
LOG_DIR_NAME = "local_log"


@dataclass(slots=True)
class EvalGenSubmitOptions:
    results_dir: Path
    output_dir: Path | None
    relax: bool
    use_remote_relax: bool
    remote_config: remote_helpers.RemoteConfig
    wait: bool


def _path(v: str) -> Path:
    return Path(v).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="提交生成结构评估任务到 Bohrium")
    # dpdispatcher / 远程通用参数
    p.add_argument("--dp-email", type=str, default="", help="Bohrium 登录邮箱")
    p.add_argument("--dp-password", type=str, default="", help="Bohrium 登录密码")
    p.add_argument("--program-id", type=int, default=C.DEFAULT_PROGRAM_ID, help="Program ID")
    p.add_argument("--job-name", type=str, default="eval_generated_structures", help="作业名称")
    p.add_argument("--scass-type", type=str, default=C.DEFAULT_SCASS_TYPE, help="计算资源规格")
    p.add_argument("--platform", type=str, default=C.DEFAULT_PLATFORM, help="平台标识")
    p.add_argument("--image-name", type=str, default=C.DEFAULT_REMOTE_IMAGE, help="容器镜像名称")
    p.add_argument("--group-size", type=int, default=1, help="节点数量")

    # 评估相关
    p.add_argument(
        "--results-dir",
        type=_path,
        required=True,
        help="本地包含 generated_crystals.extxyz 的目录 (会被打包上传)",
    )
    p.add_argument(
        "--output-dir",
        type=_path,
        default=None,
        help="本地保存 metrics 的目录 (默认与 results-dir 同级)",
    )
    p.add_argument("--relax", action="store_true", help="远程 relax 结构后再评估")
    p.add_argument(
        "--no-relax", action="store_true", help="禁用 relax (覆盖 --relax)"
    )
    p.add_argument("--wait", action="store_true", help="等待作业完成并拉回结果")
    return p


def parse_args(argv: Iterable[str] | None = None) -> EvalGenSubmitOptions:
    p = build_parser()
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.dp_email or not args.dp_password:
        raise SystemExit("需要提供 --dp-email 和 --dp-password 用于 Bohrium 登录")

    if args.no_relax:
        args.relax = False

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

    return EvalGenSubmitOptions(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        relax=args.relax,
        use_remote_relax=True,
        remote_config=config,
        wait=args.wait,
    )


def _stage_support(job_root: Path) -> Path:
    """打包 crygen_demo 支持代码 + mattergen 源码，用于远程评估。"""
    archive = job_root / SUPPORT_ARCHIVE_NAME
    import tarfile

    with tarfile.open(archive, "w:gz") as tf:
        for name in ("common", "generation"):
            src = C.TEST_ROOT / name
            if src.exists():
                tf.add(src, arcname=name)

        mg_root = C.MATTERGEN_ROOT
        if mg_root.exists():
            def _mg_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
                parts = Path(ti.name).parts
                if not parts:
                    return ti
                if len(parts) > 1 and parts[1] in {"checkpoints", "data-release", "datasets"}:
                    return None
                if "__pycache__" in parts:
                    return None
                return ti

            tf.add(mg_root, arcname="mattergen", filter=_mg_filter)
    return archive


def submit_eval(options: EvalGenSubmitOptions) -> None:
    job_root = staging.new_job_root(C.TEST_ROOT / "generation")
    log_dir = job_root / LOG_DIR_NAME
    staging.ensure_dir(log_dir)

    support_archive = _stage_support(job_root)

    # 打包 results 目录
    results_rel = "results"
    results_archive = staging.stage_directory_to_archive(options.results_dir, job_root, results_rel)

    forward_files: List[str] = [support_archive.name, results_archive.name]

    command_parts: List[str] = [
        "source /root/dev/mattergen/.venv/bin/activate",
        "pip install --no-cache-dir dpdispatcher",
        "tar -xf " + support_archive.name,
        "pip install --no-cache-dir ./mattergen",
        f"mkdir -p {results_rel} && tar -xf {results_archive.name} -C {results_rel}",
    ]

    # 构造远程 python 命令
    eval_cmd = [
        "python",
        "-m",
        RUN_MODULE,
        f"--results-dir {results_rel}",
        "--save-structures" if options.relax else "",
    ]
    if not options.relax:
        eval_cmd.append("--no-relax")

    command_parts.append(" ".join(c for c in eval_cmd if c))
    command = " && ".join(command_parts)

    backward_files = [
        results_rel,
        f"{results_rel}/**",
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
    LOG.info("提交生成结构评估任务 job_root=%s", job_root)
    remote_helpers.run_submission(submission, options.wait)

    if options.wait:
        # 回迁结果目录
        keep = {results_rel, LOG_DIR_NAME}
        staging.relocate_artifacts(job_root, keep, log_dir)
        LOG.info("评估完成，本地结果位于: %s", job_root / results_rel)


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    opts = parse_args(argv)
    submit_eval(opts)


if __name__ == "__main__":  # pragma: no cover
    import sys
    main(sys.argv[1:])
