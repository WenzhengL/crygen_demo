#!/usr/bin/env python3
"""Entry point for local or remote MatterGen finetuning runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
POSSIBLE_ROOTS = (SCRIPT_DIR, SCRIPT_DIR.parent)
for root in POSSIBLE_ROOTS:
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))

from finetune.local import main as finetune_main


def main(argv: Iterable[str] | None = None) -> None:
    finetune_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
