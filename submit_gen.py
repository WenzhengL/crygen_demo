#!/usr/bin/env python3
"""Wrapper CLI delegating to ``generation.remote``."""

from __future__ import annotations

import sys
from typing import Iterable

from generation.remote import main as remote_main


def main(argv: Iterable[str] | None = None) -> None:
    remote_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
