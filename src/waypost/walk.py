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

Nested-``.gitignore`` semantics: each directory's own ``.gitignore`` is
loaded as an independent ``pathspec`` (so within-file negation, e.g.
``!kept.log`` after ``*.log``, works exactly like real git). A path is
ignored if *any* applicable ancestor directory's spec matches it, checked
with the path made relative to that directory (which is what lets a
slash-free pattern like ``build`` match at any depth beneath the directory
that declares it). One corner of real git behaviour is intentionally not
replicated: a deeper ``.gitignore`` cannot un-ignore something a shallower
one already ignored (cross-file negation override). That is a rare pattern
in practice and full support would require re-implementing git's own
precedence resolution; documenting the gap here is cheaper than chasing it
for a v0.1 indexer.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)

# Skip a file once it exceeds this size -- large generated/data files are
# rarely useful to a symbol index and are expensive to parse.
MAX_FILE_BYTES = 1_000_000  # 1 MB

# Stop walking once this many files have been yielded. A hard ceiling so a
# pathological repository (a huge vendored tree that slipped past
# .gitignore, a build output nobody committed rules for) can't turn indexing
# into an unbounded scan.
MAX_FILES = 50_000

# How much of a file to sniff for a null byte when deciding "binary".
_SNIFF_BYTES = 8192

_GITIGNORE_FILENAME = ".gitignore"

# Directories that are always skipped, regardless of what .gitignore says --
# either because they're never source (vendored deps, VCS metadata) or
# because indexing them would be pure noise (bytecode caches, build output).
HARD_SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        "vendor",
        ".git",
        "__pycache__",
    }
)

# Filenames/patterns skipped everywhere, regardless of .gitignore. Generated
# minified bundles and dependency lockfiles are large, not hand-written, and
# not useful to a symbol map.
_HARD_SKIP_NAMES = pathspec.PathSpec.from_lines(
    "gitignore",
    [
        "*.min.js",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "npm-shrinkwrap.json",
        "poetry.lock",
        "Pipfile.lock",
        "uv.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "go.sum",
        "*.lock",
    ],
)

_SpecStack = tuple[tuple[Path, "pathspec.PathSpec"], ...]


@dataclass
class _WalkState:
    max_file_bytes: int
    max_files: int
    count: int = 0
    truncated: bool = False


def iter_files(
    root: Path | str,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_files: int = MAX_FILES,
) -> Iterator[Path]:
    """Yield source file paths under ``root``, relative to ``root``.

    Applies, in order: hard-skip of vendored/generated directories and
    lockfile-like names, ``.gitignore`` rules (including nested
    ``.gitignore`` files), a size cutoff, and a binary sniff. Stops after
    ``max_files`` files have been yielded and logs a warning rather than
    walking a pathological tree indefinitely.

    Paths are yielded in a deterministic (name-sorted, depth-first) order so
    repeated runs over an unchanged tree produce identical output -- index
    diffs stay meaningful.
    """
    root = Path(root).resolve()
    state = _WalkState(max_file_bytes=max_file_bytes, max_files=max_files)

    root_spec = _load_gitignore(root)
    specs: _SpecStack = ((root, root_spec),) if root_spec is not None else ()

    yield from _walk_dir(root, root, specs, state)

    if state.truncated:
        logger.warning(
            "waypost: reached the %d file limit while walking %s; "
            "stopping early, results are incomplete",
            state.max_files,
            root,
        )


def _walk_dir(
    dir_path: Path,
    root: Path,
    specs: _SpecStack,
    state: _WalkState,
) -> Iterator[Path]:
    try:
        entries = sorted(os.scandir(dir_path), key=lambda e: e.name)
    except OSError as exc:
        logger.debug("waypost: cannot list %s: %s", dir_path, exc)
        return

    for entry in entries:
        if state.truncated:
            return

        if entry.is_symlink():
            continue

        entry_path = Path(entry.path)

        if entry.is_dir(follow_symlinks=False):
            if entry.name in HARD_SKIP_DIRS:
                continue
            if _is_ignored(entry_path, specs, is_dir=True):
                continue
            child_spec = _load_gitignore(entry_path)
            child_specs = specs + ((entry_path, child_spec),) if child_spec is not None else specs
            yield from _walk_dir(entry_path, root, child_specs, state)
            continue

        if not entry.is_file(follow_symlinks=False):
            continue  # sockets, FIFOs, device files -- not source.

        if _HARD_SKIP_NAMES.match_file(entry.name):
            continue
        if _is_ignored(entry_path, specs, is_dir=False):
            continue

        try:
            size = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        if size > state.max_file_bytes:
            continue

        if _looks_binary(entry_path):
            continue

        if state.count >= state.max_files:
            state.truncated = True
            return

        state.count += 1
        yield entry_path.relative_to(root)


def _load_gitignore(dir_path: Path) -> pathspec.PathSpec | None:
    """Load ``dir_path``'s own ``.gitignore`` as an independent spec, or None."""
    gitignore_path = dir_path / _GITIGNORE_FILENAME
    try:
        if not gitignore_path.is_file():
            return None
        text = gitignore_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.debug("waypost: cannot read %s: %s", gitignore_path, exc)
        return None
    return pathspec.PathSpec.from_lines("gitignore", text.splitlines())


def _is_ignored(path: Path, specs: _SpecStack, *, is_dir: bool) -> bool:
    """True if any applicable ancestor .gitignore spec matches ``path``.

    Each spec is checked against ``path`` relative to *its own* directory --
    which is what makes a bare pattern like ``build`` (no slash) match at
    any depth beneath the directory that declares it, matching git's own
    rule for the shape of the pattern.
    """
    ignored = False
    for spec_dir, spec in specs:
        rel = path.relative_to(spec_dir).as_posix()
        if is_dir:
            rel += "/"
        if spec.match_file(rel):
            ignored = True
    return ignored


def _looks_binary(path: Path) -> bool:
    """Sniff the first 8 KB for a null byte -- cheap, standard binary heuristic."""
    try:
        with path.open("rb") as f:
            chunk = f.read(_SNIFF_BYTES)
    except OSError:
        return True  # unreadable: treat as not worth indexing, not a crash.
    return b"\0" in chunk
