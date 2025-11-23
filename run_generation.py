#!/usr/bin/env python3
"""Entry point for local or remote MatterGen generation runs."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:  # Prefer the shared module when available (local development).
    from generation.local import main as generation_main  # type: ignore
except ModuleNotFoundError:  # Remote environment fallback.

    DEFAULT_BATCH_SIZE = 4
    DEFAULT_NUM_BATCHES = 2

    def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="运行 MatterGen 生成任务")
        parser.add_argument("--results-dir", type=str, required=True, help="生成结果输出目录")
        parser.add_argument("--model-dir", type=str, required=True, help="MatterGen 预训练模型目录")
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
        return parser.parse_args(list(argv) if argv is not None else None)

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

    def generation_main(argv: Iterable[str] | None = None) -> None:
        args = _parse_args(argv)
        if args.batch_size is not None and args.batch_size <= 0:
            raise SystemExit("--batch-size 必须为正整数")
        if args.num_batches is not None and args.num_batches <= 0:
            raise SystemExit("--num-batches 必须为正整数")
        if args.num_gen is not None and args.num_gen <= 0:
            raise SystemExit("--num-gen 必须为正整数")

        batch_size, num_batches = _resolve_generation_plan(args.batch_size, args.num_batches, args.num_gen)

        results_dir = Path(args.results_dir).expanduser().resolve()
        model_dir = Path(args.model_dir).expanduser().resolve()

        results_dir.mkdir(parents=True, exist_ok=True)
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录 {model_dir} 不存在")

        total_target = args.num_gen
        total_actual = batch_size * num_batches
        if total_target is not None and total_actual != total_target:
            print(
                f"[info] 请求生成 {total_target} 个结构，实际生成 {total_actual} 个"
                f" (batch_size={batch_size}, num_batches={num_batches})"
            )

        cmd = [
            "mattergen-generate",
            str(results_dir),
            f"--model_path={model_dir}",
            f"--batch_size={batch_size}",
            f"--num_batches={num_batches}",
        ]
        subprocess.run(cmd, check=True)


def main(argv: Iterable[str] | None = None) -> None:
    generation_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
