"""Local MatterGen generation runner."""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from common import cli as cli_helpers

DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_BATCHES = 2


@dataclass(slots=True)
class GenerationOptions:
    results_dir: Path
    model_dir: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    num_batches: int = DEFAULT_NUM_BATCHES
    num_gen: int | None = None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "GenerationOptions":
        batch_size = ns.batch_size
        num_batches = ns.num_batches
        num_gen = getattr(ns, "num_gen", None)

        if batch_size is not None and batch_size <= 0:
            raise SystemExit("--batch-size 必须为正整数")
        if num_batches is not None and num_batches <= 0:
            raise SystemExit("--num-batches 必须为正整数")
        if num_gen is not None and num_gen <= 0:
            raise SystemExit("--num-gen 必须为正整数")

        resolved_batch_size, resolved_num_batches = _resolve_generation_plan(batch_size, num_batches, num_gen)

        return cls(
            results_dir=Path(ns.results_dir).expanduser().resolve(),
            model_dir=Path(ns.model_dir).expanduser().resolve(),
            batch_size=resolved_batch_size,
            num_batches=resolved_num_batches,
            num_gen=num_gen,
        )

    @property
    def total_structures(self) -> int:
        return self.batch_size * self.num_batches


def _resolve_generation_plan(
    batch_size: int | None, num_batches: int | None, total: int | None
) -> tuple[int, int]:
    bs = batch_size if batch_size else DEFAULT_BATCH_SIZE
    nb = num_batches if num_batches else DEFAULT_NUM_BATCHES

    if total:
        if batch_size and num_batches:
            bs = batch_size
            nb = max(num_batches, math.ceil(total / bs))
        elif batch_size:
            bs = batch_size
            nb = math.ceil(total / bs)
        elif num_batches:
            nb = num_batches
            bs = math.ceil(total / nb)
        else:
            bs = min(DEFAULT_BATCH_SIZE, total)
            nb = math.ceil(total / bs)

        bs = max(1, min(bs, total))
        nb = max(1, math.ceil(total / bs))

    return bs, nb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 MatterGen 生成测试")
    cli_helpers.add_generation_local_args(parser)
    return parser


def parse_args(argv: Iterable[str] | None = None) -> GenerationOptions:
    parser = build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)
    return GenerationOptions.from_namespace(ns)


def run_generation(options: GenerationOptions) -> None:
    options.results_dir.mkdir(parents=True, exist_ok=True)
    total_target = options.num_gen
    total_actual = options.total_structures
    if total_target is not None and total_actual != total_target:
        print(
            f"[info] 请求生成 {total_target} 个结构，实际生成 {total_actual} 个"
            f" (batch_size={options.batch_size}, num_batches={options.num_batches})"
        )

    cmd = [
        "mattergen-generate",
        str(options.results_dir),
        f"--model_path={options.model_dir}",
        f"--batch_size={options.batch_size}",
        f"--num_batches={options.num_batches}",
    ]
    subprocess.run(cmd, check=True)


def main(argv: Iterable[str] | None = None) -> None:
    options = parse_args(argv)
    run_generation(options)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
