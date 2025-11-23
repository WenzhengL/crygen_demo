"""Utilities for preparing dpdispatcher job folders."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def new_job_root(base_dir: Path) -> Path:
    job_id = uuid.uuid4().hex[:12]
    job_root = base_dir / f"job_{job_id}"
    ensure_dir(job_root)
    return job_root


def stage_script(script_path: Path, job_root: Path) -> Path:
    staged_script = job_root / script_path.name
    if script_path.resolve() != staged_script.resolve():
        shutil.copy2(script_path, staged_script)
    return staged_script


def relocate_artifacts(job_root: Path, keep_names: Iterable[str], log_dir: Path) -> None:
    for item in job_root.iterdir():
        if item.name in keep_names:
            continue
        destination = log_dir / item.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(item), str(destination))


def stage_directory_to_archive(src: Path, job_root: Path, archive_stem: str) -> Path:
    ensure_dir(job_root)
    archive_base = job_root / archive_stem
    archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=str(src))
    return Path(archive_path)


def stage_single_file(src: Path, job_root: Path, dst_name: str | None = None) -> Path:
    dst = job_root / (dst_name or src.name)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def stage_directory(src: Path, job_root: Path, dst_name: str | None = None) -> Path:
    dst = job_root / (dst_name or src.name)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
