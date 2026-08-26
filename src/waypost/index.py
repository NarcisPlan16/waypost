"""Index construction, persistence and incremental refresh.

Responsibilities (Sprint 2):
    - Serialise the schema (version 1) to ``.waypost/index.json``, and
      validate it on read.
    - Refresh incrementally: compare each file's content SHA against the
      stored one, reparse only what changed, drop entries for deleted files,
      and recompute ranks (cheap enough to always do in full).
    - Treat an unknown or older ``schema`` version, or corrupt JSON, as a
      trigger for a silent full rebuild -- never a crash, and never a stale
      wrong answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from waypost.parse import ParsedFile, Reference, Symbol, parse_file
from waypost.rank import compute_ranks
from waypost.walk import iter_files

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes. `load` refuses anything else and
# triggers a full rebuild rather than guessing how to migrate it.
SCHEMA_VERSION = 1

# Relative to a repo root -- kept in one place so `index`, CLI commands and
# tests all agree on where the index lives.
INDEX_DIR = ".waypost"
INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class FileEntry:
    """One indexed file: its parse result, content hash and rank."""

    sha256: str
    rank: float
    parsed: ParsedFile


@dataclass
class Index:
    """The whole on-disk index: schema version, root and every file entry."""

    root: str
    files: dict[str, FileEntry] = field(default_factory=dict)
    schema: int = SCHEMA_VERSION


def default_index_path(root: Path | str) -> Path:
    """Where ``build``/``refresh`` persist by default: ``<root>/.waypost/index.json``."""
    return Path(root) / INDEX_DIR / INDEX_FILENAME


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(
    root: Path | str,
    *,
    rank_strategy: str = "simple",
    personalize: Iterable[str] | None = None,
) -> Index:
    """Full index build: walk, parse every file, rank the result.

    Files ``walk.iter_files`` yields but ``parse.parse_file`` can't handle
    (unsupported extension, unreadable, unparseable) are silently absent --
    both modules already log why, so nothing is re-logged here.

    ``personalize`` is forwarded to :func:`waypost.rank.compute_ranks` and
    only means anything under ``rank_strategy="pagerank"``; it carries the
    CLI's ``--focus`` paths.
    """
    root = Path(root)
    parsed_by_path: dict[str, ParsedFile] = {}
    shas: dict[str, str] = {}

    for rel_path in iter_files(root):
        abs_path = root / rel_path
        try:
            data = abs_path.read_bytes()
        except OSError as exc:
            logger.debug("waypost: cannot read %s: %s", abs_path, exc)
            continue

        parsed = parse_file(rel_path, root=root)
        if parsed is None:
            continue

        parsed_by_path[parsed.path] = parsed
        shas[parsed.path] = _sha256_bytes(data)

    ranks = compute_ranks(parsed_by_path, strategy=rank_strategy, personalize=personalize)

    files = {
        path: FileEntry(sha256=shas[path], rank=ranks.get(path, 0.0), parsed=parsed)
        for path, parsed in parsed_by_path.items()
    }
    return Index(root=root.as_posix(), files=files, schema=SCHEMA_VERSION)


def refresh(
    root: Path | str,
    existing: Index | None,
    *,
    rank_strategy: str = "simple",
    personalize: Iterable[str] | None = None,
) -> Index:
    """Incremental refresh: reparse only files whose content SHA changed.

    A file present before but no longer yielded by the walk (deleted,
    newly gitignored, now oversized) is dropped. Ranks are always
    recomputed in full over the resulting file set -- cheap enough that
    incrementalising it isn't worth the bug surface.

    ``existing=None`` (no prior index, or one that failed to load) makes
    this equivalent to :func:`build`.
    """
    if existing is None:
        return build(root, rank_strategy=rank_strategy, personalize=personalize)

    root = Path(root)
    parsed_by_path: dict[str, ParsedFile] = {}
    shas: dict[str, str] = {}

    for rel_path in iter_files(root):
        abs_path = root / rel_path
        try:
            data = abs_path.read_bytes()
        except OSError as exc:
            logger.debug("waypost: cannot read %s: %s", abs_path, exc)
            continue

        rel_posix = rel_path.as_posix()
        sha = _sha256_bytes(data)
        prior = existing.files.get(rel_posix)

        if prior is not None and prior.sha256 == sha:
            parsed_by_path[rel_posix] = prior.parsed
            shas[rel_posix] = sha
            continue

        parsed = parse_file(rel_path, root=root)
        if parsed is None:
            continue
        parsed_by_path[parsed.path] = parsed
        shas[parsed.path] = sha

    ranks = compute_ranks(parsed_by_path, strategy=rank_strategy, personalize=personalize)

    files = {
        path: FileEntry(sha256=shas[path], rank=ranks.get(path, 0.0), parsed=parsed)
        for path, parsed in parsed_by_path.items()
    }
    return Index(root=root.as_posix(), files=files, schema=SCHEMA_VERSION)


def _symbol_to_dict(sym: Symbol) -> dict[str, Any]:
    return asdict(sym)


def _reference_to_dict(ref: Reference) -> dict[str, Any]:
    return asdict(ref)


def to_dict(index: Index) -> dict[str, Any]:
    """Serialise ``index`` to a plain JSON-able dict."""
    return {
        "schema": index.schema,
        "root": index.root,
        "files": {
            path: {
                "sha256": entry.sha256,
                "rank": entry.rank,
                "language": entry.parsed.language,
                "loc": entry.parsed.loc,
                "truncated": entry.parsed.truncated,
                "defs": [_symbol_to_dict(s) for s in entry.parsed.defs],
                "refs": [_reference_to_dict(r) for r in entry.parsed.refs],
            }
            for path, entry in index.files.items()
        },
    }


def _dict_to_index(data: dict[str, Any]) -> Index:
    files: dict[str, FileEntry] = {}
    for path, raw in data["files"].items():
        defs = tuple(Symbol(**d) for d in raw["defs"])
        refs = tuple(Reference(**r) for r in raw["refs"])
        parsed = ParsedFile(
            path=path,
            language=raw["language"],
            loc=raw["loc"],
            defs=defs,
            refs=refs,
            truncated=raw["truncated"],
        )
        files[path] = FileEntry(sha256=raw["sha256"], rank=raw["rank"], parsed=parsed)
    return Index(root=data["root"], files=files, schema=data["schema"])


def save(index: Index, path: Path | str | None = None) -> Path:
    """Write ``index`` to ``path`` (default ``<root>/.waypost/index.json``)."""
    target = Path(path) if path is not None else default_index_path(index.root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_dict(index), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load(path: Path | str) -> Index | None:
    """Read an index from ``path``.

    Returns ``None`` -- never raises -- for a missing file, corrupt JSON, a
    shape that doesn't match what this version expects, or a ``schema``
    that isn't exactly :data:`SCHEMA_VERSION`. Every one of those is a
    trigger for a silent full rebuild by the caller, not a crash.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("waypost: cannot read index at %s: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("waypost: index at %s is corrupt (%s); rebuilding", path, exc)
        return None

    if not isinstance(data, dict) or data.get("schema") != SCHEMA_VERSION:
        logger.info(
            "waypost: index at %s has schema %r, expected %d; rebuilding",
            path,
            data.get("schema") if isinstance(data, dict) else None,
            SCHEMA_VERSION,
        )
        return None

    try:
        return _dict_to_index(data)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("waypost: index at %s is malformed (%s); rebuilding", path, exc)
        return None
