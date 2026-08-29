"""Budget-aware rendering of index queries into compact text.

Responsibilities (Sprint 3):
    - ``map``: fill a token budget in rank order -- highest-ranked files
      first, and within a file, classes and exported functions before
      private helpers. Truncate at a symbol boundary, never mid-symbol.
    - ``find`` / ``show`` / ``refs`` / ``outline`` renderers.

The rule that makes this tool worth using: every default output is
token-frugal, and budgets are *measured* with the tokenizer rather than
estimated. ``show`` prints only a symbol's own span, never the surrounding
file. If any command can dump a whole file into an agent's context, the tool
is worse than grep and the benchmark will say so.

Output shape for ``map`` -- indentation and ``:line`` suffixes carry the
structure, so no tokens are spent on JSON punctuation::

    src/client.py (412 loc)
      class Client(BaseClient) :22 - HTTP client with retry and pooling.
        def request(self, method, url, *, timeout=None) -> Response :58

Design notes worth keeping:

**One symbol per line is what makes the budget honest.** Truncation happens
in :func:`waypost.tokens.fit_lines`, which cuts on line boundaries; because
a line is exactly one symbol (or one file header), a cut can never land
mid-signature. Every renderer here builds a list of lines and hands it to
that function -- none of them do their own arithmetic.

**``show`` is budgeted too, and that is not paranoia.** A symbol's own span
is usually small, but a file that is one 900-line class has a span the size
of the file, which is precisely the "worse than grep" failure. The budget is
the backstop; the span is the intent.

**Ranking decides order, never membership.** Nothing is filtered out for
being low-ranked -- it is simply further down, and the budget decides where
the line falls. That keeps ``map --budget`` monotone: raising the budget
only ever adds output, it never reshuffles what was already there.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from waypost.index import Index
from waypost.parse import Symbol
from waypost.rank import matches_focus, normalise_path
from waypost.tokens import TRUNCATION_MARKER, Tokenizer, count, fit_lines, get_tokenizer

# Defaults chosen to be useful in an agent's first turn without being a
# noticeable share of its context. `map` is the one an agent calls blind, so
# it gets the tightest default.
DEFAULT_MAP_BUDGET = 2000
DEFAULT_SHOW_BUDGET = 800
DEFAULT_LIST_BUDGET = 1200

# `find` caps hits before the budget does, so a pattern matching half the
# repo produces a short answer rather than a truncated flood.
DEFAULT_FIND_LIMIT = 40

# Kinds that hold other symbols. They sort first within a file: a class name
# tells an agent where to look; a module-level constant rarely does.
CONTAINER_KINDS = frozenset({"class", "interface", "enum", "module", "type", "struct"})

# Kinds that are somewhere to put a breakpoint. These are what an agent is
# usually looking for, so they outrank public data.
CALLABLE_KINDS = frozenset({"function", "method"})

# Characters that make a `find` pattern a glob rather than a substring.
_GLOB_CHARS = set("*?[")

_INDENT = "  "

# How many top-ranked files `select_map_entries` formats on its first pass
# before deciding it needs more candidates. See the note there.
_INITIAL_FILE_WINDOW = 64


@dataclass(frozen=True)
class MapEntry:
    """One file's contribution to a ``map``: the file, and what fit of it."""

    path: str
    loc: int
    rank: float
    symbols: tuple[tuple[Symbol, int], ...]  # (symbol, nesting depth)


@dataclass(frozen=True)
class Hit:
    """One symbol match, with the file it came from."""

    path: str
    rank: float
    symbol: Symbol


# --------------------------------------------------------------------------
# Ordering and line construction
# --------------------------------------------------------------------------


def _symbol_priority(symbol: Symbol) -> int:
    """Sort key within a file: containers, exported callables, then the rest.

    Constants sort below functions even when public. ``MAX_RETRIES = 3``
    tells an agent nothing about where to make a change, and a module full
    of them would otherwise crowd out the functions in a small budget --
    which is exactly the failure ``map`` exists to avoid.
    """
    if symbol.kind in CONTAINER_KINDS:
        return 0
    if symbol.exported:
        return 1 if symbol.kind in CALLABLE_KINDS else 2
    return 3


def order_symbols(defs: Sequence[Symbol]) -> list[tuple[Symbol, int]]:
    """Order a file's symbols for display, as ``(symbol, depth)`` pairs.

    Nesting comes from the qualified name: ``Client.request`` is a child of
    ``Client`` when the file also defines ``Client``. Top-level symbols are
    ordered by :func:`_symbol_priority` then by line; children always follow
    their parent in line order, because within a class the source order is
    the one a reader already has a model of.

    A symbol whose apparent parent is *not* defined in the same file (a
    method on a class declared elsewhere, say) is treated as top-level
    rather than dropped -- being unable to place it is not a reason to hide
    it.
    """
    by_name = {s.name: s for s in defs}
    children: dict[str, list[Symbol]] = {}
    roots: list[Symbol] = []

    for symbol in defs:
        parent_name = symbol.name.rsplit(".", 1)[0] if "." in symbol.name else ""
        if parent_name and parent_name in by_name:
            children.setdefault(parent_name, []).append(symbol)
        else:
            roots.append(symbol)

    ordered: list[tuple[Symbol, int]] = []

    def emit(symbol: Symbol, depth: int) -> None:
        ordered.append((symbol, depth))
        for child in sorted(children.get(symbol.name, ()), key=lambda s: (s.line, s.name)):
            emit(child, depth + 1)

    for root in sorted(roots, key=lambda s: (_symbol_priority(s), s.line, s.name)):
        emit(root, 0)

    return ordered


def _file_header(path: str, loc: int) -> str:
    return f"{path} ({loc} loc)"


def _symbol_line(symbol: Symbol, depth: int) -> str:
    """``  class Client(BaseClient) :22 - one-line doc.``"""
    line = f"{_INDENT * (depth + 1)}{symbol.signature} :{symbol.line}"
    if symbol.doc:
        line += f" - {symbol.doc}"
    return line


def _hit_line(hit: Hit) -> str:
    """``src/client.py:58 def request(...) -> Response - doc``

    The qualified name is printed only when it adds something the signature
    does not, i.e. when the symbol is nested inside another.
    """
    parts = [f"{hit.path}:{hit.symbol.line}"]
    if "." in hit.symbol.name:
        parts.append(f"{hit.symbol.name}:")
    parts.append(hit.symbol.signature)
    line = " ".join(parts)
    if hit.symbol.doc:
        line += f" - {hit.symbol.doc}"
    return line


def _render(lines: Sequence[str], budget: int, tokenizer: Tokenizer | None) -> str:
    """Apply the budget and join. Every renderer ends here."""
    return "\n".join(fit_lines(lines, budget, tokenizer=tokenizer))


# --------------------------------------------------------------------------
# map
# --------------------------------------------------------------------------


def _ranked_paths(index: Index, focus: Iterable[str] | None = None) -> list[str]:
    """Index paths, most important first.

    ``focus`` paths (prefix match, so a directory works) sort ahead of
    everything else while keeping their relative rank order -- that is what
    ``map --focus src/api`` means: not a filter, a promotion. The prefix
    rule itself is :func:`waypost.rank.matches_focus`, shared with
    pagerank's personalisation so the flag means one thing.
    """
    prefixes = tuple(focus) if focus else ()

    return sorted(
        index.files,
        key=lambda p: (not matches_focus(p, prefixes), -index.files[p].rank, p),
    )


# One selectable symbol: which file it belongs to, its position in that
# file's display order, and its nesting depth.
_Unit = tuple[str, int, int]  # (path, position within the file's ordering, depth)


def _map_units(
    plans: Sequence[tuple[str, Sequence[tuple[Symbol, int]]]],
) -> list[_Unit]:
    """Order every candidate symbol by when it earns its place in a map.

    Headline symbols -- top-level classes, interfaces and exported
    functions, for *every* file in rank order -- come before any file's
    private helpers. That is what makes a small budget produce a map of the
    repository rather than a very thorough listing of its single
    highest-ranked file. Detail then fills whatever is left, still in rank
    order, so a generous budget deepens the top files first.
    """
    headline: list[_Unit] = []
    detail: list[_Unit] = []
    for path, ordered in plans:
        for position, (symbol, depth) in enumerate(ordered):
            bucket = headline if depth == 0 and _symbol_priority(symbol) <= 1 else detail
            bucket.append((path, position, depth))
    return headline + detail


def _compose(
    index: Index,
    plans: Sequence[tuple[str, Sequence[tuple[Symbol, int]]]],
    units: Sequence[_Unit],
) -> tuple[list[MapEntry], list[str]]:
    """Group ``units`` back into per-file blocks, and render their lines.

    A file with no selected symbol is dropped entirely -- a bare header
    spends tokens to say a file exists, which ``stats`` already said.
    """
    picked: dict[str, set[int]] = {}
    for path, position, _depth in units:
        picked.setdefault(path, set()).add(position)

    entries: list[MapEntry] = []
    lines: list[str] = []
    for path, ordered in plans:
        positions = picked.get(path)
        if not positions:
            continue
        chosen = tuple(pair for position, pair in enumerate(ordered) if position in positions)
        file_entry = index.files[path]
        entries.append(
            MapEntry(
                path=path,
                loc=file_entry.parsed.loc,
                rank=file_entry.rank,
                symbols=chosen,
            )
        )
        lines.append(_file_header(path, file_entry.parsed.loc))
        lines.extend(_symbol_line(symbol, depth) for symbol, depth in chosen)

    return entries, lines


def select_map_entries(
    index: Index,
    *,
    budget: int = DEFAULT_MAP_BUDGET,
    focus: Iterable[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> tuple[list[MapEntry], bool]:
    """Choose what fits in ``budget``, in rank order. Returns (entries, truncated).

    Selection is a binary search for the largest prefix of :func:`_map_units`
    whose rendering fits. Adding a unit can only add lines, so the token
    count is monotone in the prefix length and the search is exact -- and
    because a unit is a whole symbol, the cut is always on a symbol
    boundary. Eleven tokenizer calls decide a 2,000-token map, rather than
    one per candidate line.

    The candidate window grows rather than starting at the theoretical
    maximum. A unit costs at least one token, so no more than ``budget``
    files could ever appear -- but ordering 2,000 files' symbols to render
    60 lines of them is most of the work for none of the output. Starting
    small and quadrupling until the budget runs out before the candidates
    do costs a few extra passes over what is already formatted, and turns
    the 5,000-file case from ~90 ms into single digits. The answer is
    identical either way: the window only ever stops growing once the
    budget, not the window, is what ended the selection.

    Selection counts the truncation marker whenever it is going to be
    printed. Leaving it out and letting the final :func:`fit_lines` guard
    absorb it looked equivalent and was not: the guard would cut a symbol
    line to make room, and a file whose only selected symbol was the one cut
    became a bare header -- a filename spending tokens to say nothing.
    """
    tokenizer = tokenizer or get_tokenizer()
    cap = max(budget, 1)

    ranked = _ranked_paths(index, focus)
    window = min(_INITIAL_FILE_WINDOW, cap)

    while True:
        paths = ranked[:window]
        plans = [(path, order_symbols(index.files[path].parsed.defs)) for path in paths]

        units = _map_units(plans)
        # More candidates exist than this pass is looking at, so whatever it
        # selects is a truncation of the repository even if all of it fits.
        incomplete = len(units) > cap or len(paths) < len(ranked)
        units = units[:cap]

        def fits(lines: list[str], truncating: bool) -> bool:
            if truncating:
                lines = [*lines, TRUNCATION_MARKER]
            return count("\n".join(lines), tokenizer=tokenizer) <= budget

        entries, lines = _compose(index, plans, units)
        if fits(lines, incomplete):
            if not incomplete:
                return entries, False
            if window < min(len(ranked), cap):
                window = min(window * 4, cap)
                continue
            return entries, True

        # Truncation is now certain, so every candidate carries the marker
        # and the count is monotone in the prefix length again.
        low, high = 0, len(units)
        while low < high:
            mid = (low + high + 1) // 2
            _entries, mid_lines = _compose(index, plans, units[:mid])
            if fits(mid_lines, True):
                low = mid
            else:
                high = mid - 1

        entries, _lines = _compose(index, plans, units[:low])
        return entries, True


def render_map(
    index: Index,
    *,
    budget: int = DEFAULT_MAP_BUDGET,
    focus: Iterable[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> str:
    """The repo map: ranked files, each with the symbols that fit."""
    tokenizer = tokenizer or get_tokenizer()
    entries, truncated = select_map_entries(index, budget=budget, focus=focus, tokenizer=tokenizer)

    lines: list[str] = []
    for entry in entries:
        lines.append(_file_header(entry.path, entry.loc))
        lines.extend(_symbol_line(symbol, depth) for symbol, depth in entry.symbols)
    if truncated:
        lines.append(TRUNCATION_MARKER)

    # Already selected to fit; this is the guard that makes it a guarantee.
    return _render(lines, budget, tokenizer)


def map_data(
    index: Index,
    *,
    budget: int = DEFAULT_MAP_BUDGET,
    focus: Iterable[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> dict:
    """``--json`` form of :func:`render_map`, over the same selection."""
    tokenizer = tokenizer or get_tokenizer()
    entries, truncated = select_map_entries(index, budget=budget, focus=focus, tokenizer=tokenizer)
    return {
        "budget": budget,
        "tokenizer": tokenizer.name,
        "truncated": truncated,
        "files": [
            {
                "path": entry.path,
                "loc": entry.loc,
                "rank": round(entry.rank, 4),
                "symbols": [_symbol_dict(s, depth) for s, depth in entry.symbols],
            }
            for entry in entries
        ],
    }


def _symbol_dict(symbol: Symbol, depth: int | None = None) -> dict:
    data = {
        "name": symbol.name,
        "kind": symbol.kind,
        "line": symbol.line,
        "end_line": symbol.end_line,
        "signature": symbol.signature,
        "doc": symbol.doc,
        "exported": symbol.exported,
    }
    if depth is not None:
        data["depth"] = depth
    return data


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


def _is_exact(name: str, needle: str) -> bool:
    """True if ``name``, or its last segment, *is* ``needle`` (lowercased)."""
    lowered = name.lower()
    return lowered == needle or lowered.rsplit(".", 1)[-1] == needle


def partition_hits(hits: Sequence[Hit], pattern: str) -> tuple[list[Hit], list[Hit]]:
    """Split hits into the ones that *are* ``pattern`` and the ones that merely contain it.

    ``find AppContext`` on Flask matches 22 definitions, of which one is the
    class asked for; the other 21 are ``do_teardown_appcontext``,
    ``with_appcontext``, the class's own methods and seven test functions.
    Printing all 22 cost 586 tokens to answer "where is this class", against
    12 for the ``grep`` an agent would otherwise run -- so a command meant to
    undercut grep instead cost 49x more, which is the whole product promise
    inverted.

    A glob pattern has no exact tier by construction (``*_client`` is equal
    to no symbol's name), which is right: a glob is a discovery query and
    every match is the answer.
    """
    needle = pattern.lower()
    exact = [h for h in hits if _is_exact(h.symbol.name, needle)]
    partial = [h for h in hits if not _is_exact(h.symbol.name, needle)]
    return exact, partial


def _partial_summary(partial: Sequence[Hit], *, capped: bool) -> str:
    """The one line that stands in for the partial matches not printed.

    ``capped`` means the search itself stopped at ``--limit``, so the count
    is a floor rather than a total and must not be printed as if it were one.
    """
    files = len({h.path for h in partial})
    n = len(partial)
    return (
        f"+ {'at least ' if capped else ''}{n} other symbol{'' if n == 1 else 's'}"
        f" whose name contains this, in {files} file{'' if files == 1 else 's'}"
        f" (--all to list)"
    )


def search(index: Index, pattern: str, *, limit: int = DEFAULT_FIND_LIMIT) -> list[Hit]:
    """Symbols matching ``pattern``, best-ranked file first.

    A pattern containing ``*``, ``?`` or ``[`` is a glob against both the
    qualified name and its last segment; anything else is a
    case-insensitive substring. Exact matches sort ahead of partial ones, so
    ``find Client`` leads with ``Client`` even in a repo full of
    ``ClientPool``, ``ClientError`` and ``_build_client``.
    """
    is_glob = any(ch in _GLOB_CHARS for ch in pattern)
    needle = pattern.lower()

    def matches(name: str) -> bool:
        tail = name.rsplit(".", 1)[-1]
        if is_glob:
            return fnmatch.fnmatch(name.lower(), needle) or fnmatch.fnmatch(tail.lower(), needle)
        return needle in name.lower()

    hits = [
        Hit(path=path, rank=entry.rank, symbol=symbol)
        for path, entry in index.files.items()
        for symbol in entry.parsed.defs
        if matches(symbol.name)
    ]
    hits.sort(
        key=lambda h: (0 if _is_exact(h.symbol.name, needle) else 1, -h.rank, h.path, h.symbol.line)
    )
    return hits[:limit]


def render_find(
    index: Index,
    pattern: str,
    *,
    limit: int = DEFAULT_FIND_LIMIT,
    budget: int = DEFAULT_LIST_BUDGET,
    tokenizer: Tokenizer | None = None,
    hits: Sequence[Hit] | None = None,
    all_matches: bool = False,
) -> str:
    """``hits`` lets a caller that already searched pass the result in.

    The CLI needs the hits anyway, to pick an exit code. Without this it
    searched every symbol in the index twice per invocation.

    By default only the exact-name tier is printed, with the partial matches
    summarised in one line -- see :func:`partition_hits`. ``all_matches``
    (the CLI's ``--all``) prints every hit, which is the old behaviour.
    """
    hits = search(index, pattern, limit=limit) if hits is None else hits
    if not hits:
        return f"no symbol matching {pattern!r} in {len(index.files)} indexed files"
    exact, partial = partition_hits(hits, pattern)
    # With no exact tier there is nothing to lead with, and suppressing the
    # partials would answer a discovery query with a bare count.
    if all_matches or not exact:
        lines = [_hit_line(hit) for hit in hits]
    else:
        lines = [_hit_line(hit) for hit in exact]
        if partial:
            lines.append(_partial_summary(partial, capped=len(hits) >= limit))
    return _render(lines, budget, tokenizer)


def find_data(
    index: Index,
    pattern: str,
    *,
    limit: int = DEFAULT_FIND_LIMIT,
    hits: Sequence[Hit] | None = None,
    all_matches: bool = False,
) -> dict:
    hits = search(index, pattern, limit=limit) if hits is None else hits
    exact, partial = partition_hits(hits, pattern)
    shown = hits if (all_matches or not exact) else exact
    return {
        "pattern": pattern,
        "count": len(shown),
        # What the text output summarises in its trailing line, so a JSON
        # consumer can tell "one definition" from "one of 22 matches".
        "partial_omitted": 0 if shown is hits else len(partial),
        "hits": [
            {"path": h.path, "rank": round(h.rank, 4), **_symbol_dict(h.symbol)} for h in shown
        ],
    }


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------


def definitions(index: Index, name: str) -> list[Hit]:
    """Symbols whose qualified name, or its last segment, is exactly ``name``."""
    lowered = name.lower()
    hits = [
        Hit(path=path, rank=entry.rank, symbol=symbol)
        for path, entry in index.files.items()
        for symbol in entry.parsed.defs
        if symbol.name.lower() == lowered or symbol.name.rsplit(".", 1)[-1].lower() == lowered
    ]
    hits.sort(key=lambda h: (-h.rank, h.path, h.symbol.line))
    return hits


def read_span(root: Path | str, hit: Hit) -> list[str]:
    """The symbol's own source lines -- never the surrounding file.

    An unreadable or since-shortened file yields a one-line explanation
    rather than an exception: an index can be stale, and a stale index is a
    reason to say so, not to crash.
    """
    path = Path(root) / hit.path
    try:
        source = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"    <cannot read {hit.path}: {exc}>"]

    start = hit.symbol.line
    end = min(hit.symbol.end_line, len(source))
    if start > len(source):
        return [f"    <{hit.path} is shorter than the index expects; re-run `waypost index`>"]

    width = len(str(end))
    return [f"{n:>{width}}| {source[n - 1]}" for n in range(start, end + 1)]


def render_show(
    index: Index,
    name: str,
    *,
    root: Path | str | None = None,
    budget: int = DEFAULT_SHOW_BUDGET,
    max_matches: int = 3,
    tokenizer: Tokenizer | None = None,
    hits: Sequence[Hit] | None = None,
) -> str:
    """Print the source span of ``name``, and nothing around it.

    Several files can define the same name -- ``parse.py`` resolves by name,
    not by scope. Rather than guess, this shows the best-ranked few and says
    how many others there were.

    ``hits`` accepts an already-computed :func:`definitions` result, so a
    caller that needed it for its own reasons doesn't pay for it twice.
    """
    hits = definitions(index, name) if hits is None else hits
    if not hits:
        near = search(index, name, limit=5)
        if near:
            suggestions = ", ".join(sorted({h.symbol.name for h in near}))
            return f"no symbol named {name!r}; did you mean: {suggestions}"
        return f"no symbol named {name!r} in {len(index.files)} indexed files"

    root = root if root is not None else index.root
    shown = hits[:max_matches]

    lines: list[str] = []
    for hit in shown:
        symbol = hit.symbol
        lines.append(f"{hit.path}:{symbol.line}-{symbol.end_line} {symbol.kind} {symbol.name}")
        lines.extend(read_span(root, hit))
    if len(hits) > len(shown):
        lines.append(f"({len(hits) - len(shown)} further definition(s) of {name!r} not shown)")

    return _render(lines, budget, tokenizer)


def show_data(
    index: Index,
    name: str,
    *,
    root: Path | str | None = None,
    max_matches: int = 3,
    hits: Sequence[Hit] | None = None,
) -> dict:
    hits = definitions(index, name) if hits is None else hits
    root = root if root is not None else index.root
    return {
        "name": name,
        "count": len(hits),
        "definitions": [
            {
                "path": h.path,
                "rank": round(h.rank, 4),
                **_symbol_dict(h.symbol),
                "source": "\n".join(read_span(root, h)),
            }
            for h in hits[:max_matches]
        ],
    }


# --------------------------------------------------------------------------
# refs
# --------------------------------------------------------------------------


# One referring file: its path, its rank, and the (line, kind) of each site.
Referrer = tuple[str, float, list[tuple[int, str]]]


def referencing_files(index: Index, name: str) -> list[Referrer]:
    """Files referencing ``name``, ranked, with the (line, kind) of each site.

    Matching is by the reference's own name, which is unqualified at a call
    site -- ``request(...)`` never says ``Client.request`` -- so the last
    segment of ``name`` is what is compared. Same resolution rule as
    ``rank.py`` uses to build the graph, deliberately: ``refs`` should show
    the edges the ranking actually counted.
    """
    target = name.rsplit(".", 1)[-1].lower()
    out: list[Referrer] = []
    for path, entry in index.files.items():
        sites = sorted(
            {(ref.line, ref.kind) for ref in entry.parsed.refs if ref.name.lower() == target}
        )
        if sites:
            out.append((path, entry.rank, sites))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def render_refs(
    index: Index,
    name: str,
    *,
    budget: int = DEFAULT_LIST_BUDGET,
    tokenizer: Tokenizer | None = None,
    defined: Sequence[Hit] | None = None,
    referrers: Sequence[Referrer] | None = None,
) -> str:
    """``defined``/``referrers`` accept already-computed results; see
    :func:`render_find`."""
    defined = definitions(index, name) if defined is None else defined
    referrers = referencing_files(index, name) if referrers is None else referrers

    if not defined and not referrers:
        return f"no definitions or references to {name!r} in {len(index.files)} indexed files"

    lines: list[str] = []
    if defined:
        lines.append(f"{name} defined in:")
        lines += [
            f"{_INDENT}{hit.path}:{hit.symbol.line} {hit.symbol.signature}" for hit in defined
        ]
    if referrers:
        total = sum(len(sites) for _, _, sites in referrers)
        lines.append(f"{name} referenced {total}x from {len(referrers)} file(s):")
        for path, _rank, sites in referrers:
            where = ",".join(str(line) for line, _ in sites)
            kinds = ",".join(sorted({kind for _, kind in sites}))
            lines.append(f"{_INDENT}{path}:{where} ({kinds})")
    else:
        lines.append(f"{name} is referenced from no indexed file")

    return _render(lines, budget, tokenizer)


def refs_data(
    index: Index,
    name: str,
    *,
    defined: Sequence[Hit] | None = None,
    referrers: Sequence[Referrer] | None = None,
) -> dict:
    defined = definitions(index, name) if defined is None else defined
    referrers = referencing_files(index, name) if referrers is None else referrers
    return {
        "name": name,
        "definitions": [
            {"path": h.path, "rank": round(h.rank, 4), **_symbol_dict(h.symbol)} for h in defined
        ],
        "references": [
            {
                "path": path,
                "rank": round(rank, 4),
                "sites": [{"line": line, "kind": kind} for line, kind in sites],
            }
            for path, rank, sites in referrers
        ],
    }


# --------------------------------------------------------------------------
# outline
# --------------------------------------------------------------------------


def resolve_path(index: Index, path: str) -> str | None:
    """Accept an exact index path, or an unambiguous suffix of one.

    Agents pass what they have -- ``client.py`` as often as
    ``src/waypost/client.py`` -- and refusing the short form for no reason
    costs a round trip.
    """
    wanted = normalise_path(path)
    if wanted in index.files:
        return wanted
    candidates = [p for p in index.files if p == wanted or p.endswith("/" + wanted)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def render_outline(
    index: Index,
    path: str,
    *,
    budget: int = DEFAULT_LIST_BUDGET,
    tokenizer: Tokenizer | None = None,
) -> str:
    resolved = resolve_path(index, path)
    if resolved is None:
        matches = [p for p in index.files if p.endswith("/" + normalise_path(path))]
        if matches:
            listed = "\n".join(f"{_INDENT}{p}" for p in sorted(matches))
            return f"{path!r} is ambiguous; did you mean:\n{listed}"
        return f"{path!r} is not in the index; run `waypost index` if it is new"

    entry = index.files[resolved]
    parsed = entry.parsed
    lines = [f"{_file_header(resolved, parsed.loc)} {parsed.language} rank={entry.rank:.2f}"]
    if parsed.truncated:
        lines.append(f"{_INDENT}(file was truncated during parsing; symbols may be incomplete)")
    if not parsed.defs:
        lines.append(f"{_INDENT}(no symbols extracted)")
    lines.extend(_symbol_line(symbol, depth) for symbol, depth in order_symbols(parsed.defs))
    return _render(lines, budget, tokenizer)


def outline_data(index: Index, path: str) -> dict:
    resolved = resolve_path(index, path)
    if resolved is None:
        return {"path": path, "found": False, "symbols": []}
    entry = index.files[resolved]
    return {
        "path": resolved,
        "found": True,
        "language": entry.parsed.language,
        "loc": entry.parsed.loc,
        "rank": round(entry.rank, 4),
        "truncated": entry.parsed.truncated,
        "symbols": [_symbol_dict(s, depth) for s, depth in order_symbols(entry.parsed.defs)],
    }


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def stats_data(index: Index, *, index_path: Path | str | None = None) -> dict:
    languages: dict[str, int] = {}
    kinds: dict[str, int] = {}
    symbol_count = 0
    loc_total = 0

    for entry in index.files.values():
        languages[entry.parsed.language] = languages.get(entry.parsed.language, 0) + 1
        loc_total += entry.parsed.loc
        for symbol in entry.parsed.defs:
            symbol_count += 1
            kinds[symbol.kind] = kinds.get(symbol.kind, 0) + 1

    size_bytes: int | None = None
    if index_path is not None:
        try:
            size_bytes = Path(index_path).stat().st_size
        except OSError:
            size_bytes = None

    top = sorted(index.files.items(), key=lambda kv: (-kv[1].rank, kv[0]))[:10]

    return {
        "root": index.root,
        "schema": index.schema,
        "tokenizer": get_tokenizer().name,
        "files": len(index.files),
        "symbols": symbol_count,
        "loc": loc_total,
        "index_bytes": size_bytes,
        "languages": dict(sorted(languages.items(), key=lambda kv: (-kv[1], kv[0]))),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_files": [{"path": p, "rank": round(e.rank, 4)} for p, e in top],
    }


def render_stats(
    index: Index,
    *,
    index_path: Path | str | None = None,
    budget: int = DEFAULT_LIST_BUDGET,
    tokenizer: Tokenizer | None = None,
) -> str:
    data = stats_data(index, index_path=index_path)
    size = "unknown" if data["index_bytes"] is None else f"{data['index_bytes'] / 1024:.0f} KiB"

    lines = [
        f"{data['files']} files, {data['symbols']} symbols, {data['loc']} loc",
        f"schema {data['schema']} - index {size} - tokenizer {data['tokenizer']}",
        "languages: " + ", ".join(f"{k} {v}" for k, v in data["languages"].items()),
        "kinds: " + ", ".join(f"{k} {v}" for k, v in data["kinds"].items()),
        "top files by rank:",
    ]
    lines += [f"{_INDENT}{f['path']} {f['rank']}" for f in data["top_files"]]
    return _render(lines, budget, tokenizer)
