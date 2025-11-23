"""Shared dpdispatcher helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
LOG = logging.getLogger(__name__)


from dpdispatcher import Machine, Resources, Submission, Task


@dataclass(slots=True)
class RemoteConfig:
    email: str
    password: str
    program_id: int
    job_name: str
    scass_type: str
    platform: str
    image_name: str
    group_size: int = 1
    keep_backup: bool = True


def build_machine(config: RemoteConfig) -> Machine:
    machine_dict = {
        "batch_type": "Bohrium",
        "context_type": "BohriumContext",
        "local_root": "./",
        "remote_profile": {
            "email": config.email,
            "password": config.password,
            "program_id": config.program_id,
            "keep_backup": config.keep_backup,
            "input_data": {
                "job_type": "container",
                "grouped": True,
                "job_name": config.job_name,
                "scass_type": config.scass_type,
                "platform": config.platform,
                "image_name": config.image_name,
            },
        },
    }
    return Machine.load_from_dict(machine_dict)


def build_resources(config: RemoteConfig) -> Resources:
    return Resources.load_from_dict({"group_size": config.group_size})


def build_submission(
    work_base: Path,
    task_list: Sequence[Task],
    config: RemoteConfig,
    *,
    forward_common_files: Iterable[str] | None = None,
    backward_common_files: Iterable[str] | None = None,
) -> Submission:
    submission = Submission(
        work_base=str(work_base),
        machine=build_machine(config),
        resources=build_resources(config),
        task_list=list(task_list),
        forward_common_files=list(forward_common_files or ()),
        backward_common_files=list(backward_common_files or ()),
    )
    return submission


def run_submission(submission: Submission, wait: bool) -> None:
    exit_on_submit = not wait
    try:
        submission.run_submission(exit_on_submit=exit_on_submit)
    except FileNotFoundError as exc:
        LOG.warning(
            "dpdispatcher 未能找到 out.zip (%s)。通常表示远端仍在打包结果，"
            "可稍后使用 pull_results.py 或 Bohrium 网站获取产出。",
            exc,
        )
        if exit_on_submit:
            raise
