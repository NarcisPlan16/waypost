"""Importance ranking over the reference graph.

Responsibilities (Sprint 2):
    - Default strategy, cheap and explainable::

          rank = inbound_reference_credit
                 + 0.5 * is_exported
                 - 0.3 * is_test

      where a reference to a name that N files define is worth ``1/N`` to
      each of them, not 1 to each -- see `_resolved_reference_weights`.

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

# Every strategy `compute_ranks` accepts. The CLI's `--rank` choices come
# from here, and `index.load` validates a stored strategy against it, so
# there is one list rather than three that can drift apart.
STRATEGIES = ("simple", "pagerank")

DEFAULT_STRATEGY = "simple"


def is_test_path(path: str) -> bool:
    """True if ``path`` looks like a test file, by name or directory."""
    return _TEST_PATH_RE.search(path) is not None


def normalise_path(path: str) -> str:
    """A user-supplied path in the form index keys use: posix, no ``./``.

    Only a literal leading ``./`` is removed -- stripping the characters
    ``.`` and ``/`` individually would turn ``.github/workflows`` into
    ``github/workflows`` and match nothing.
    """
    cleaned = path.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def matches_focus(path: str, prefixes: Iterable[str]) -> bool:
    """True if ``path`` is one of ``prefixes`` or lives under one of them.

    ``--focus`` means the same thing everywhere it appears: a path prefix,
    so a directory works. This is the single definition of that, shared by
    :func:`pagerank_rank`'s personalisation and ``render``'s map ordering --
    they disagreed once, and ``index --focus src/api`` silently personalised
    toward nothing because it matched no file exactly.
    """
    for prefix in prefixes:
        normalised = normalise_path(prefix).rstrip("/")
        if not normalised:
            continue
        if path == normalised or path.startswith(normalised + "/"):
            return True
    return False


def _defining_files(files: Mapping[str, ParsedFile]) -> dict[str, list[str]]:
    """Map a defined symbol name to the file(s) it is defined in.

    Both the qualified name (``Client.request``) and its last segment
    (``request``) are indexed, since references are unqualified (a call
    site says ``request(...)``, not ``Client.request(...)``).
    """
    index: dict[str, list[str]] = {}
    for path, parsed in files.items():
        for sym in parsed.defs:
            for name in {sym.name, sym.name.rsplit(".", 1)[-1]}:
                bucket = index.setdefault(name, [])
                # A file that defines `Client.get` and `Server.get` defines the
                # name `get` once, not twice. Listing it twice would both
                # double its credit and stop the 1/N split summing to 1.
                if not bucket or bucket[-1] != path:
                    bucket.append(path)
    return index


def _resolved_reference_weights(
    files: Mapping[str, ParsedFile],
) -> dict[str, dict[str, float]]:
    """Per source file, the credit each *other* file earns from its refs.

    Resolution is by name, not by full scope analysis -- ``parse.py``
    doesn't hand us that -- so a name defined in N files is genuinely
    ambiguous. One reference is therefore worth one point in total, split
    ``1/N`` between the candidates, rather than a full point to each.

    Paying all N in full is not a harmless over-count, it inverts the
    ranking: measured on Flask, ``tests/test_views.py`` was the single
    highest-ranked file in the repository because it defines a method called
    ``get``, a name 15 files define, and ``__init__`` (36 definers) paid
    every one of them for every reference in the tree. Splitting the credit
    leaves uniquely-named symbols at full weight and puts ``scaffold.py``
    and ``app.py`` on top, with no ``is_test`` special case involved.

    Self-references (a file calling its own symbols) earn nothing; they say
    nothing about a file's importance to the rest of the repo. They still
    count toward N -- the name is that ambiguous whoever asks.
    """
    defining = _defining_files(files)
    weights: dict[str, dict[str, float]] = {path: {} for path in files}
    for path, parsed in files.items():
        row = weights[path]
        for ref in parsed.refs:
            targets = defining.get(ref.name)
            if not targets:
                continue
            share = 1.0 / len(targets)
            for target in targets:
                if target != path:
                    row[target] = row.get(target, 0.0) + share
    return weights


def inbound_reference_counts(files: Mapping[str, ParsedFile]) -> dict[str, float]:
    """Per file, the credit from references in *other* files that resolve to it.

    Fractional, not integral: see :func:`_resolved_reference_weights` for why
    an ambiguous name splits its point rather than paying each candidate.
    """
    counts: dict[str, float] = dict.fromkeys(files, 0.0)
    for row in _resolved_reference_weights(files).values():
        for target, weight in row.items():
            counts[target] += weight
    return counts


def _export_fraction(parsed: ParsedFile) -> float:
    if not parsed.defs:
        return 0.0
    exported = sum(1 for s in parsed.defs if s.exported)
    return exported / len(parsed.defs)


def simple_rank(files: Mapping[str, ParsedFile]) -> dict[str, float]:
    """The default strategy: cheap, explainable, no graph traversal.

    ``rank = inbound_reference_credit + 0.5 * is_exported - 0.3 * is_test``

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


def _reference_graph(files: Mapping[str, ParsedFile]) -> dict[str, dict[str, float]]:
    """file -> {file it references: edge weight} (edges for PageRank).

    Weighted by the same ``1/N`` split as
    :func:`_resolved_reference_weights`, and for the same reason: an
    unweighted set of edges lets one reference to an ambiguous name build N
    equal edges, so a file spends as much of its rank on the 14 unrelated
    files that happen to define ``get`` as on the one it actually meant.
    """
    return _resolved_reference_weights(files)


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
    out_weight = {path: sum(targets.values()) for path, targets in graph.items()}

    # Prefix matching, not equality: `--focus src/api` names a directory far
    # more often than it names a file, and an exact-match-only focus set is
    # empty in that case -- personalisation silently doing nothing.
    prefixes = list(personalize) if personalize else []
    focus = {p for p in paths if matches_focus(p, prefixes)} if prefixes else set()
    if focus:
        teleport = {p: (1.0 / len(focus) if p in focus else 0.0) for p in paths}
    else:
        teleport = dict.fromkeys(paths, 1.0 / n)

    scores = dict.fromkeys(paths, 1.0 / n)

    for _ in range(iterations):
        new_scores = dict.fromkeys(paths, 0.0)
        dangling_mass = sum(scores[p] for p in paths if not graph[p])

        for path in paths:
            new_scores[path] += (1 - damping) * teleport[path]
            new_scores[path] += damping * dangling_mass * teleport[path]

        for path, targets in graph.items():
            total = out_weight[path]
            if not total:
                continue
            mass = damping * scores[path] / total
            for target, weight in targets.items():
                new_scores[target] += mass * weight

        delta = sum(abs(new_scores[p] - scores[p]) for p in paths)
        scores = new_scores
        if delta < tolerance:
            break

    return scores


def compute_ranks(
    files: Mapping[str, ParsedFile],
    *,
    strategy: str = DEFAULT_STRATEGY,
    personalize: Iterable[str] | None = None,
) -> dict[str, float]:
    """Entry point: per-file importance scores under ``strategy``.

    ``"simple"`` (the default) is cheap and explainable; ``"pagerank"`` is
    the graph-based alternative gated behind ``--rank pagerank``. ``strategy``
    must be one of :data:`STRATEGIES`.
    """
    if strategy == "simple":
        return simple_rank(files)
    if strategy == "pagerank":
        return pagerank_rank(files, personalize=personalize)
    raise ValueError(f"unknown rank strategy: {strategy!r}")
