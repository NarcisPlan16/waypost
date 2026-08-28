"""Zero-cost context probe: what does *looking something up* cost, in tokens?

This is not the benchmark. The benchmark (``bench run``) measures whether an
agent solves tasks with fewer tokens, and it costs money because it calls a
model. This module answers the narrower question that needs no model at all:

    when an agent wants to know about a symbol, how many tokens does
    ``waypost show`` hand it, versus reading the file that defines it?

That is an *input-side* comparison, and it is deliberately generous to
waypost in one direction and harsh in the other. Generous: a real agent that
greps first also pays for the grep output and for the files it opened and
discarded, none of which is counted here. Harsh: reading a whole file gives
the agent surrounding context that ``show`` withholds, and sometimes that
context is what actually solves the task. So treat the ratio as an *upper
bound on input-side savings*, never as the measured result. Only the paid
benchmark can tell you whether the withheld context mattered.

The probe is repo-wide and takes no per-task choices: it walks every symbol
in the index rather than a hand-picked list, precisely so the numbers cannot
be steered by choosing flattering queries.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waypost import index as index_mod
from waypost import render, tokens

# Same defaults the CLI ships, so the probe measures the tool as it is
# actually used rather than a tuned configuration.
DEFAULT_MAP_BUDGET = render.DEFAULT_MAP_BUDGET
DEFAULT_SHOW_BUDGET = render.DEFAULT_SHOW_BUDGET


@dataclass(frozen=True)
class SymbolProbe:
    """One symbol: the cost of `show`-ing it against the cost of its file."""

    name: str
    path: str
    show_tokens: int
    file_tokens: int

    @property
    def saved(self) -> int:
        return self.file_tokens - self.show_tokens

    @property
    def ratio(self) -> float:
        """Fraction of the file-read cost avoided; negative if `show` is worse."""
        if self.file_tokens <= 0:
            return 0.0
        return self.saved / self.file_tokens


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Median plus the surrounding quartiles, for a possibly tiny sample."""
    ordered = sorted(values)
    if not ordered:
        return (0.0, 0.0, 0.0)
    if len(ordered) < 4:
        median = statistics.median(ordered)
        return (ordered[0], median, ordered[-1])
    quarts = statistics.quantiles(ordered, n=4, method="inclusive")
    return (quarts[0], quarts[1], quarts[2])


def read_file_texts(root: Path, index: index_mod.Index) -> dict[str, str]:
    """Source of every indexed file, read once and reused by both probes."""
    texts: dict[str, str] = {}
    for path in index.files:
        try:
            texts[path] = (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            # An unreadable file cannot be charged to either side; drop it
            # from the comparison rather than scoring it as free.
            continue
    return texts


def definitions_by_name(index: index_mod.Index) -> dict[str, list[render.Hit]]:
    """Every name `show` can resolve, mapped to the hits it would return.

    :func:`render.definitions` rescans every symbol in the index per lookup,
    which is fine for one CLI call and quadratic for a probe that looks up
    every symbol there is -- on a large repository that is the difference
    between minutes and hours. This builds the same answer once, keyed by
    both the qualified name and its last segment, matching the resolution
    rule and the ranking order that :func:`render.definitions` uses.
    """
    by_name: dict[str, list[render.Hit]] = {}
    for path, entry in index.files.items():
        for symbol in entry.parsed.defs:
            hit = render.Hit(path=path, rank=entry.rank, symbol=symbol)
            lowered = symbol.name.lower()
            # A name and its last segment coincide for unqualified symbols;
            # the same hit must not be listed twice for the same key.
            for key in {lowered, lowered.rsplit(".", 1)[-1]}:
                by_name.setdefault(key, []).append(hit)

    for hits in by_name.values():
        hits.sort(key=lambda h: (-h.rank, h.path, h.symbol.line))
    return by_name


def probe_symbols(
    root: Path,
    index: index_mod.Index,
    *,
    budget: int = DEFAULT_SHOW_BUDGET,
    tokenizer: tokens.Tokenizer | None = None,
    file_texts: dict[str, str] | None = None,
) -> list[SymbolProbe]:
    """Every indexed symbol, scored as `show` output against its whole file.

    ``show`` is invoked by name, exactly as an agent would: the cost of a name
    that several files define includes the extra matches, because the agent
    pays for those too.
    """
    tokenizer = tokenizer or tokens.get_tokenizer()
    texts = read_file_texts(root, index) if file_texts is None else file_texts

    file_costs = {path: tokens.count(text, tokenizer=tokenizer) for path, text in texts.items()}
    resolved = definitions_by_name(index)
    # One probe per distinct name: `show` resolves all of a name's definitions
    # in a single call, so charging for it once per definition would invent
    # savings nobody makes.
    seen: set[str] = set()
    probes: list[SymbolProbe] = []

    for path, entry in index.files.items():
        if path not in file_costs:
            continue
        for symbol in entry.parsed.defs:
            if symbol.name in seen:
                continue
            seen.add(symbol.name)
            shown = render.render_show(
                index,
                symbol.name,
                root=root,
                budget=budget,
                tokenizer=tokenizer,
                hits=resolved.get(symbol.name.lower(), []),
            )
            probes.append(
                SymbolProbe(
                    name=symbol.name,
                    path=path,
                    show_tokens=tokens.count(shown, tokenizer=tokenizer),
                    file_tokens=file_costs[path],
                )
            )
    return probes


def probe_orientation(
    index: index_mod.Index,
    *,
    budget: int = DEFAULT_MAP_BUDGET,
    tokenizer: tokens.Tokenizer | None = None,
    file_texts: dict[str, str],
) -> dict[str, int]:
    """The other half: cost of `map` against cost of reading the repository.

    "Read the whole repository" is not what a competent agent does, so this
    number is an orientation ceiling, not a baseline anyone should claim to
    beat. It is here because the ceiling is what the map is bounded against,
    and a bounded map that grew with the repository would be a bug.
    """
    tokenizer = tokenizer or tokens.get_tokenizer()
    rendered = render.render_map(index, budget=budget, tokenizer=tokenizer)
    return {
        "map_tokens": tokens.count(rendered, tokenizer=tokenizer),
        "repo_tokens": sum(tokens.count(t, tokenizer=tokenizer) for t in file_texts.values()),
        "files": len(index.files),
    }


def build_probe(
    root: Path,
    *,
    map_budget: int = DEFAULT_MAP_BUDGET,
    show_budget: int = DEFAULT_SHOW_BUDGET,
    label: str | None = None,
) -> dict[str, Any]:
    """Run both probes against an already-indexed repository."""
    index_path = index_mod.default_index_path(root)
    index = index_mod.load(index_path)
    if index is None:
        raise FileNotFoundError(
            f"no waypost index at {index_path}; run `waypost index` in {root} first"
        )

    tokenizer = tokens.get_tokenizer()
    texts = read_file_texts(root, index)

    symbols = probe_symbols(root, index, budget=show_budget, tokenizer=tokenizer, file_texts=texts)
    orientation = probe_orientation(index, budget=map_budget, tokenizer=tokenizer, file_texts=texts)

    ratios = [p.ratio for p in symbols]
    low, median, high = _quartiles(ratios)
    show_total = sum(p.show_tokens for p in symbols)
    file_total = sum(p.file_tokens for p in symbols)

    return {
        "label": label or root.name,
        "root": str(root),
        "tokenizer": type(tokenizer).__name__,
        "map_budget": map_budget,
        "show_budget": show_budget,
        "orientation": orientation,
        "symbols": {
            "count": len(symbols),
            "show_tokens_total": show_total,
            "file_tokens_total": file_total,
            "pooled_reduction": ((file_total - show_total) / file_total if file_total else 0.0),
            "median_reduction": median,
            "iqr": [low, high],
            "worse_than_file": sum(1 for p in symbols if p.saved <= 0),
            "median_show_tokens": (
                statistics.median([p.show_tokens for p in symbols]) if symbols else 0
            ),
            "median_file_tokens": (
                statistics.median([p.file_tokens for p in symbols]) if symbols else 0
            ),
        },
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_probe(report: dict[str, Any]) -> str:
    """Human-readable probe output, hedged where the number needs hedging."""
    sym = report["symbols"]
    orient = report["orientation"]
    return "\n".join(
        [
            f"context probe: {report['label']}",
            f"  tokenizer {report['tokenizer']}, "
            f"map budget {report['map_budget']}, show budget {report['show_budget']}",
            "",
            f"  symbol lookup ({sym['count']} distinct symbols)",
            f"    median `show`                    {sym['median_show_tokens']:.0f} tokens",
            f"    median enclosing file            {sym['median_file_tokens']:.0f} tokens",
            f"    median reduction                 {_pct(sym['median_reduction'])}"
            f"  (IQR {_pct(sym['iqr'][0])} .. {_pct(sym['iqr'][1])})",
            f"    pooled reduction                 {_pct(sym['pooled_reduction'])}",
            f"    symbols where `show` costs more  {sym['worse_than_file']}",
            "",
            f"  orientation ({orient['files']} indexed files)",
            f"    `map` output                     {orient['map_tokens']} tokens",
            f"    whole repository                 {orient['repo_tokens']} tokens",
            "",
            "  Input-side upper bound, not a benchmark result: it charges the",
            "  baseline for one whole file and nothing for the search that found",
            "  it, and it does not test whether the withheld context was needed.",
            "  Only `bench run` answers that, and it calls a model.",
        ]
    )


CAVEAT = (
    "Input-side upper bound, not a benchmark result: the baseline is charged\n"
    "for one whole file and nothing for the search that found it, and nothing\n"
    "here tests whether the withheld context was needed. Only `bench run`\n"
    "answers that, and it calls a model."
)


def render_comparison(reports: list[dict[str, Any]]) -> str:
    """Several repositories side by side.

    One repository proves nothing: a single flattering result could just be
    that repository's file sizes. The point of the table is whether the
    reduction and the map's bounded size hold as the repositories change
    language and grow by orders of magnitude.
    """
    header = (
        f"{'repo':<12}{'files':>7}{'symbols':>9}{'med show':>10}"
        f"{'med file':>10}{'median':>9}{'pooled':>9}{'map':>8}{'repo tok':>11}"
    )
    rows = [header, "-" * len(header)]
    for report in reports:
        sym = report["symbols"]
        orient = report["orientation"]
        rows.append(
            f"{report['label']:<12}{orient['files']:>7}{sym['count']:>9}"
            f"{sym['median_show_tokens']:>10.0f}{sym['median_file_tokens']:>10.0f}"
            f"{_pct(sym['median_reduction']):>9}{_pct(sym['pooled_reduction']):>9}"
            f"{orient['map_tokens']:>8}{orient['repo_tokens']:>11}"
        )
    rows.extend(["", CAVEAT])
    return "\n".join(rows)
