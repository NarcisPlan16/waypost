"""Entry point for ``python -m waypost``.

The console script from ``[project.scripts]`` is the usual way in. This
module exists because it is *not* always available: ``bin/waypost.js`` falls
back to ``python -m waypost`` when no ``waypost`` executable is on PATH, and
that spelling only works if this file is here. Without it the interpreter
answers "No module named waypost.__main__" and exits 1 -- which the CLI
otherwise uses to mean "nothing matched".
"""

from __future__ import annotations

import sys

from waypost.cli import main

if __name__ == "__main__":
    sys.exit(main())
