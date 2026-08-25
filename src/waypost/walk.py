"""File discovery for the indexer.

Responsibilities (Sprint 1):
    - Walk a repository root, honouring ``.gitignore`` (including nested
      ignore files) via ``pathspec``.
    - Skip binaries by sniffing the first 8 KB for a null byte.
    - Hard-skip vendored and generated trees: ``node_modules/``, ``.venv/``,
      ``venv/``, ``dist/``, ``build/``, ``vendor/``, ``.git/``,
      ``__pycache__/``, ``*.min.js`` and known lockfiles.
    - Skip files larger than 1 MB, and stop at 50,000 *yielded* files or
      500,000 *examined* entries, whichever comes first, with a warning
      rather than walking a pathological repository to the end.

Two caps, not one: the file cap bounds the output, but on its own it does
not bound the work -- a tree of a million binaries yields nothing and would
still be scanned and sniffed in full. The entry cap is what actually stops
the walk.

Symlinks: symlinked *directories* are followed, with multiple links to the same
real directory deduplicated (the guard is the resolved path, so link cycles terminate). Skipping
them would silently lose whole subtrees in repositories that link a source
directory into place. Symlinked *files* are skipped -- they either duplicate
a file already walked or point outside the tree, and an index should
describe the target, not the link.

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
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)

# Skip a file once it exceeds this size -- large generated/data files are
# rarely useful to a symbol index and are expensive to parse.
MAX_FILE_BYTES = 1_000_000  # 1 MB

# Stop yielding once this many files have been returned.
MAX_FILES = 50_000

# Stop walking once this many directory entries have been examined,
# whether or not they were yielded. This is the cap that bounds the work.
MAX_ENTRIES = 500_000

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
# not useful to a symbol map. Listed explicitly rather than as a `*.lock`
# glob: the glob made every entry below it dead code and swept up
# hand-written files that happen to end in .lock.
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
        "flake.lock",
        "mix.lock",
        "Podfile.lock",
        "go.sum",
    ],
)

_SpecStack = tuple[tuple[Path, "pathspec.PathSpec"], ...]


@dataclass
class _WalkState:
    max_file_bytes: int
    max_files: int
    max_entries: int
    count: int = 0
    examined: int = 0
    stop_reason: str | None = None
    visited_dirs: set[Path] = field(default_factory=set)


def iter_files(
    root: Path | str,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_files: int = MAX_FILES,
    max_entries: int = MAX_ENTRIES,
) -> Iterator[Path]:
    """Yield source file paths under ``root``, relative to ``root``.

    Applies, in order: hard-skip of vendored/generated directories and
    lockfile-like names, ``.gitignore`` rules (including nested
    ``.gitignore`` files), a size cutoff, and a binary sniff. Stops once
    ``max_files`` files have been yielded or ``max_entries`` directory
    entries have been examined, logging a warning either way rather than
    walking a pathological tree to the end.

    Symlinked directories are followed once per real directory; symlinked
    files are skipped. See the module docstring for the reasoning.

    Paths are yielded in a deterministic (name-sorted, depth-first) order so
    repeated runs over an unchanged tree produce identical output -- index
    diffs stay meaningful.
    """
    root = Path(root).resolve()
    state = _WalkState(
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        max_entries=max_entries,
    )
    state.visited_dirs.add(root)

    root_spec = _load_gitignore(root)
    specs: _SpecStack = ((root, root_spec),) if root_spec is not None else ()

    yield from _walk_dir(root, root, specs, state)

    if state.stop_reason == "files":
        logger.warning(
            "waypost: reached the %d file limit while walking %s; "
            "stopping early, results are incomplete",
            state.max_files,
            root,
        )
    elif state.stop_reason == "entries":
        logger.warning(
            "waypost: reached the %d entry limit while walking %s; "
            "stopping early, results are incomplete",
            state.max_entries,
            root,
        )


def _walk_dir(
    dir_path: Path,
    root: Path,
    specs: _SpecStack,
    state: _WalkState,
) -> Iterator[Path]:
    try:
        with os.scandir(dir_path) as scan:
            entries = sorted(scan, key=lambda e: e.name)
    except OSError as exc:
        logger.debug("waypost: cannot list %s: %s", dir_path, exc)
        return

    for entry in entries:
        if state.stop_reason:
            return

        state.examined += 1
        if state.examined > state.max_entries:
            state.stop_reason = "entries"
            return

        entry_path = Path(entry.path)
        is_link = entry.is_symlink()

        if entry.is_dir(follow_symlinks=True):
            if entry.name in HARD_SKIP_DIRS:
                continue
            if _is_ignored(entry_path, specs, is_dir=True):
                continue
            if is_link:
                # Follow it, but only the first time this real directory is
                # reached -- that is what makes link cycles terminate.
                try:
                    real = entry_path.resolve()
                except OSError:
                    continue
                if real in state.visited_dirs:
                    continue
                state.visited_dirs.add(real)
            child_spec = _load_gitignore(entry_path)
            child_specs = specs + ((entry_path, child_spec),) if child_spec is not None else specs
            yield from _walk_dir(entry_path, root, child_specs, state)
            continue

        if is_link:
            continue  # symlinked file -- see the module docstring.

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
            state.stop_reason = "files"
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
