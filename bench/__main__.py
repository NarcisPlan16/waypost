"""Entry point for ``python -m bench``."""

from __future__ import annotations

import sys

from bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
