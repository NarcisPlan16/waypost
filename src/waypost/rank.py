"""Importance ranking over the reference graph.

Responsibilities (Sprint 2):
    - Default strategy, cheap and explainable::

          rank = inbound_reference_count
                 + 0.5 * is_exported
                 - 0.3 * is_test

    - Optional ``--rank pagerank``: PageRank over the graph where file A
      points at file B when A references a symbol defined in B, with the
      personalisation vector biased toward ``--focus`` paths or files in the
      current ``git diff``.

    Ship the simple strategy as the default. Whether PageRank becomes the
    default in v0.2 is a question for the Sprint 6 benchmark, not a matter
    of taste.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from waypost.parse import ParsedFile

# File-path shapes treated as tests: a leading/embedded ``tests`` or
# ``test`` directory, or a ``test_``/``_test``/``.test``/``.spec`` name --
# covers pytest, Jest and Mocha conventions without a config file to read.
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?)/|(^|/)(test_[^/]+|[^/]+_test\.[^./]+|[^/]+\.(test|spec)\.[^./]+)$"
)

DEFAULT_DAMPING = 0.85
DEFAULT_ITERATIONS = 100
DEFAULT_TOLERANCE = 1.0e-8


def is_test_path(path: str) -> bool:
    """True if ``path`` looks like a test file, by name or directory."""
    return _TEST_PATH_RE.search(path) is not None


def _defining_files(files: Mapping[str, ParsedFile]) -> dict[str, list[str]]:
    """Map a defined symbol name to the file(s) it is defined in.

    Both the qualified name (``Client.request``) and its last segment
    (``request``) are indexed, since references are unqualified (a call
    site says ``request(...)``, not ``Client.request(...)``).
    """
    index: dict[str, list[str]] = {}
    for path, parsed in files.items():
        for sym in parsed.defs:
            index.setdefault(sym.name, []).append(path)
            tail = sym.name.rsplit(".", 1)[-1]
            if tail != sym.name:
                index.setdefault(tail, []).append(path)
    return index


def inbound_reference_counts(files: Mapping[str, ParsedFile]) -> dict[str, int]:
    """Count, per file, how many references from *other* files resolve to it.

    Resolution is by name, not by full scope analysis -- ``parse.py``
    doesn't hand us that -- so a common name defined in several files
    credits all of them. Self-references (a file calling its own symbols)
    are not counted; they say nothing about a file's importance to the rest
    of the repo.
    """
    defining = _defining_files(files)
    counts: dict[str, int] = dict.fromkeys(files, 0)
    for path, parsed in files.items():
        for ref in parsed.refs:
            for target in defining.get(ref.name, ()):
                if target != path:
                    counts[target] += 1
    return counts


def _export_fraction(parsed: ParsedFile) -> float:
    if not parsed.defs:
        return 0.0
    exported = sum(1 for s in parsed.defs if s.exported)
    return exported / len(parsed.defs)


def simple_rank(files: Mapping[str, ParsedFile]) -> dict[str, float]:
    """The default strategy: cheap, explainable, no graph traversal.

    ``rank = inbound_reference_count + 0.5 * is_exported - 0.3 * is_test``

    ``is_exported`` is the fraction of the file's own definitions that are
    exported (0 for a file with none, up to 1 for a file that is all public
    surface); ``is_test`` is a 0/1 flag from :func:`is_test_path`.
    """
    inbound = inbound_reference_counts(files)
    scores: dict[str, float] = {}
    for path, parsed in files.items():
        score = inbound[path] + 0.5 * _export_fraction(parsed)
        if is_test_path(path):
            score -= 0.3
        scores[path] = score
    return scores


def _reference_graph(files: Mapping[str, ParsedFile]) -> dict[str, set[str]]:
    """file -> set of files it references (edges for PageRank)."""
    defining = _defining_files(files)
    graph: dict[str, set[str]] = {path: set() for path in files}
    for path, parsed in files.items():
        for ref in parsed.refs:
            for target in defining.get(ref.name, ()):
                if target != path:
                    graph[path].add(target)
    return graph


def pagerank_rank(
    files: Mapping[str, ParsedFile],
    *,
    personalize: Iterable[str] | None = None,
    damping: float = DEFAULT_DAMPING,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float]:
    """PageRank over the file reference graph (A -> B when A references a
    symbol B defines), personalised toward ``personalize`` paths if given
    (e.g. ``--focus`` paths or the files touched in the current git diff).

    Plain power iteration -- no extra dependency for a graph this small, and
    the fixed iteration/tolerance caps keep it bounded even on a large repo.
    """
    paths = list(files)
    n = len(paths)
    if n == 0:
        return {}

    graph = _reference_graph(files)
    out_degree = {path: len(targets) for path, targets in graph.items()}

    focus = {p for p in personalize if p in files} if personalize else set()
    if focus:
        teleport = {p: (1.0 / len(focus) if p in focus else 0.0) for p in paths}
    else:
        teleport = dict.fromkeys(paths, 1.0 / n)

    scores = dict.fromkeys(paths, 1.0 / n)

    for _ in range(iterations):
        new_scores = dict.fromkeys(paths, 0.0)
        dangling_mass = sum(scores[p] for p in paths if out_degree[p] == 0)

        for path in paths:
            new_scores[path] += (1 - damping) * teleport[path]
            new_scores[path] += damping * dangling_mass * teleport[path]

        for path, targets in graph.items():
            if not targets:
                continue
            share = damping * scores[path] / len(targets)
            for target in targets:
                new_scores[target] += share

        delta = sum(abs(new_scores[p] - scores[p]) for p in paths)
        scores = new_scores
        if delta < tolerance:
            break

    return scores


def compute_ranks(
    files: Mapping[str, ParsedFile],
    *,
    strategy: str = "simple",
    personalize: Iterable[str] | None = None,
) -> dict[str, float]:
    """Entry point: per-file importance scores under ``strategy``.

    ``"simple"`` (the default) is cheap and explainable; ``"pagerank"`` is
    the graph-based alternative gated behind ``--rank pagerank``.
    """
    if strategy == "simple":
        return simple_rank(files)
    if strategy == "pagerank":
        return pagerank_rank(files, personalize=personalize)
    raise ValueError(f"unknown rank strategy: {strategy!r}")
