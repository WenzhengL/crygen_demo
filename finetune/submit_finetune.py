#!/usr/bin/env python3
"""Wrapper CLI delegating to ``finetune.remote``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from finetune.remote import main as remote_main


def main(argv: Iterable[str] | None = None) -> None:
    remote_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
