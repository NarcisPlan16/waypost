"""File discovery for the indexer.

Responsibilities (Sprint 1):
    - Walk a repository root, honouring ``.gitignore`` (including nested
      ignore files) via ``pathspec``.
    - Skip binaries by sniffing the first 8 KB for a null byte.
    - Hard-skip vendored and generated trees: ``node_modules/``, ``.venv/``,
      ``venv/``, ``dist/``, ``build/``, ``vendor/``, ``.git/``,
      ``__pycache__/``, ``*.min.js`` and lockfiles.
    - Skip files larger than 1 MB, and stop at 50,000 files with a warning
      rather than hanging on a pathological repository.
"""
