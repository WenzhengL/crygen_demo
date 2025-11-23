#!/usr/bin/env python3
"""Collect remote finetuning artifacts produced by submit_finetune.py."""

from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List, Set

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"

# Files we care about in each remote job result archive
INTEREST_PREFIXES: tuple[str, ...] = ("outputs", "lightning_logs", "log", "err")


def _ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _copy_path(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        logging.info("[dry-run] copy %s -> %s", src, dst)
        return
    if src.is_dir():
        for child in src.rglob("*"):
            rel = child.relative_to(src)
            target = dst / rel
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _collect_existing_paths(job_dir: Path) -> List[Path]:
    found: List[Path] = []
    for prefix in INTEREST_PREFIXES:
        candidate = job_dir / prefix
        if candidate.exists():
            found.append(candidate)
    return found


def _archive_base_name(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".zip"):
        return name[:-4]
    return path.stem


def _extract_relevant_zip(zip_path: Path, force: bool) -> List[Path]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            interesting: List[str] = []
            for entry in members:
                normalized = entry.strip("/")
                if not normalized:
                    continue
                top_level = normalized.split("/", 1)[0]
                if top_level in INTEREST_PREFIXES:
                    interesting.append(entry)
            if not interesting:
                return []
            target = zip_path.parent / f"{_archive_base_name(zip_path)}_unpacked"
            if target.exists():
                if not force:
                    logging.info("Existing unpacked directory %s, skip %s", target, zip_path.name)
                    return [target / prefix for prefix in INTEREST_PREFIXES if (target / prefix).exists()]
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            for entry in interesting:
                zf.extract(entry, path=target)
            return [target / prefix for prefix in INTEREST_PREFIXES if (target / prefix).exists()]
    except zipfile.BadZipFile:
        logging.warning("Bad zip archive: %s", zip_path)
    return []


def _extract_relevant_tar(archive_path: Path, force: bool) -> List[Path]:
    base = _archive_base_name(archive_path)
    if base not in INTEREST_PREFIXES:
        return []
    target = archive_path.parent / f"{base}_unpacked"
    if target.exists():
        if not force:
            logging.info("Existing unpacked directory %s, skip %s", target, archive_path.name)
            return [target]
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target)
        return [target]
    except (tarfile.TarError, OSError) as err:
        logging.warning("Failed to extract %s: %s", archive_path, err)
        return []


def _iter_archives(job_dir: Path, include_backup: bool) -> Iterator[Path]:
    patterns = ["*.zip", "*.tar", "*.tar.gz", "*.tgz"]
    for pattern in patterns:
        yield from job_dir.glob(pattern)
    if include_backup:
        backup_dir = job_dir / "backup"
        if backup_dir.is_dir():
            for pattern in patterns:
                yield from backup_dir.glob(pattern)


def _find_checkpoints(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(path.rglob("*.ckpt"))


def _collect_job(job_dir: Path, dest_root: Path, force_extract: bool, include_backup: bool, dry_run: bool) -> None:
    logging.info("Processing job directory: %s", job_dir.name)
    existing = _collect_existing_paths(job_dir)

    extracted_paths: Set[Path] = set()
    for archive in _iter_archives(job_dir, include_backup):
        if archive.name in {"finetune_data.tar.gz", "finetune_model.tar.gz"}:
            continue
        if archive.suffix == ".zip":
            extracted_paths.update(_extract_relevant_zip(archive, force_extract))
        else:
            extracted_paths.update(_extract_relevant_tar(archive, force_extract))

    candidate_paths: Set[Path] = set(existing)
    for path in extracted_paths:
        if path.is_dir():
            for prefix in INTEREST_PREFIXES:
                nested = path / prefix
                if nested.exists():
                    candidate_paths.add(nested)
        else:
            candidate_paths.add(path)

    if not candidate_paths:
        logging.warning("No outputs/lightning_logs/log/err found in %s", job_dir.name)
        return

    dest_job_dir = dest_root / job_dir.name
    if not dry_run:
        _ensure_dir(dest_job_dir)

    for path in sorted(candidate_paths):
        target = dest_job_dir / path.name
        _copy_path(path, target, dry_run)

    checkpoints = []
    for path in candidate_paths:
        if path.is_dir():
            checkpoints.extend(_find_checkpoints(path))
        elif path.suffix == ".ckpt":
            checkpoints.append(path)

    if checkpoints:
        rel_paths = [ckpt.relative_to(job_dir) for ckpt in checkpoints]
        logging.info("Found %d checkpoints: %s", len(checkpoints), ", ".join(map(str, rel_paths)))
    else:
        logging.info("No checkpoints located.")


def _resolve_job_dirs(job_ids: list[str], include_all: bool) -> List[Path]:
    job_dirs: List[Path] = []
    if include_all:
        job_dirs.extend(sorted(p for p in SCRIPT_DIR.glob("job_*") if p.is_dir()))
    for job_id in job_ids:
        if job_id.startswith("job_"):
            candidate = SCRIPT_DIR / job_id
        else:
            candidate = SCRIPT_DIR / f"job_{job_id}"
        if not candidate.is_dir():
            logging.warning("Job directory not found: %s", candidate)
            continue
        job_dirs.append(candidate)
    seen: Set[Path] = set()
    unique: List[Path] = []
    for path in job_dirs:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect artifacts from Bohrium finetune jobs")
    parser.add_argument("job_ids", nargs="*", help="job directories to collect (job_xxx or hash)")
    parser.add_argument("--all", action="store_true", help="process all job_* directories in cwd")
    parser.add_argument("--dest", type=Path, default=DEFAULT_RESULTS_ROOT, help="destination directory")
    parser.add_argument("--force-extract", action="store_true", help="re-extract archives even if unpacked folder exists")
    parser.add_argument("--include-backup", action="store_true", help="also scan job_xxx/backup")
    parser.add_argument("--dry-run", action="store_true", help="print operations without copying")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    job_dirs = _resolve_job_dirs(args.job_ids, args.all)
    if not job_dirs:
        raise SystemExit("No job directories to process. Specify job ids or use --all.")

    dest_root = args.dest.expanduser().resolve()
    if not args.dry_run:
        _ensure_dir(dest_root)

    for job_dir in job_dirs:
        _collect_job(job_dir, dest_root, args.force_extract, args.include_backup, args.dry_run)


if __name__ == "__main__":
    main()
