"""CLI 包装：提交生成结构评估任务到 Bohrium。

用法示例：

    python crygen_demo/generation/submit_eval_gen.py \
        --wait \
        --dp-email your@email \
        --dp-password '***' \
        --program-id 29496 \
        --results-dir crygen_demo/fine_gen/job_227d3ee4dd81/pipeline_run/generation/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def _add_project_root_to_path() -> None:
    # 将 crygen_demo 根目录和其父目录都加入 sys.path，
    # 以便通过 "from generation import remote_eval" 导入本地包。
    demo_root = Path(__file__).resolve().parents[1]
    project_root = demo_root.parent
    for p in (demo_root, project_root):
        if str(p) not in sys.path:
            sys.path.append(str(p))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit evaluation of generated structures to Bohrium")
    # 把所有参数原样传给 remote_eval.parse_args
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to generation.remote_eval")
    return p


def main(argv: Iterable[str] | None = None) -> None:
    _add_project_root_to_path()
    from generation import remote_eval

    parser = _build_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)

    # argparse.REMAINDER 会包含一个可能的 "--"，去掉它
    forwarded = ns.args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    remote_eval.main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
