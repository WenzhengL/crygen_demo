#!/usr/bin/env python3
"""CLI wrapper to submit the fine_gen end-to-end pipeline remotely and optionally wait for completion.

Adds parent directory to sys.path so that `fine_gen` package can be imported when executed from repository root.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT = SCRIPT_PATH.parent.parent  # crygen_demo
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from fine_gen.remote import main as remote_main  # type: ignore


def main(argv: list[str] | None = None) -> None:
    remote_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
