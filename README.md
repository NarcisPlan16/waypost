# Waypost

A zero-LLM-call repo indexer that gives AI coding agents (Claude Code, Cowork,
and others) a token-frugal map of a codebase, instead of them grepping and
re-reading whole files.

**Status: Sprint 1 (discovery + symbol extraction).** `walk.py` finds the
files; `parse.py` extracts symbols from Python, JavaScript, TypeScript and
TSX. There is no index, no ranking and no command output yet — those are
Sprints 2 and 3. See the roadmap doc for the full sprint plan.

### What `parse.py` extracts, and what it deliberately doesn't

Per definition: a qualified name (`Client.Pool.acquire`), a kind, a line span,
a one-line signature with the body cut off, a one-line doc summary, and
whether it is exported. Per file: outbound references — calls, type mentions,
and **imports**, which no bundled tags query covers and which carry most of
the signal about which file depends on which. An import or re-export records
the name the *other* file defines, never the local alias: `import { Logger as
Log }` contributes `Logger`, because `Log` is bound nowhere else and would
match nothing in the graph.

Not extracted, on purpose: constructors (reachable through their class),
class fields and property signatures, enum members, bindings scoped to a
block rather than to the module (`if (…) { const FALLBACK = … }`), and the
keys of an object literal passed straight into a call. Each one costs a line
in every map output and leads nowhere.
`tests/fixtures/parse/expected.json` lists them per fixture with the reason,
and a test asserts they stay out.

Symbol recall is measured, not asserted: `tests/test_parse.py` computes it
against a hand-written inventory of four 30-plus-symbol fixtures and fails
below 95%. Extraction is not a straight pass-through of tree-sitter's bundled
tags queries — three of their rules are dead against current grammars, and
`parse.py`'s module docstring records which, how that was measured, and what
this module does instead.

## Why

Coding agents burn a large share of their input tokens re-discovering a
codebase's structure every session: grepping for symbols, reading whole files
to find one function, re-establishing context that a five-line signature list
would have given them for a fraction of the cost. `waypost` builds that map
once, offline, deterministically — no LLM calls, no network — and serves it
through a CLI, an npm wrapper, and a Claude Skill.

## Install (not yet published)

```bash
uv sync --extra dev
```

## Usage

```bash
waypost --version
```

Subcommands (`index`, `map`, `find`, `show`, `refs`, `outline`, `stats`) are
planned but not yet implemented — each lands in its own sprint.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy src/
uv run pytest
```

## Non-goals for v0.1

LLM-generated summaries, an MCP server, diff/output compression, file-length
linting, any write capability, and languages beyond Python and TS/JS are all
deliberately deferred — see the roadmap doc.

## Benchmarks

Nothing here yet. Measured token-reduction results land here once the
Sprint 6 benchmark runs -- per-category reductions with confidence
intervals, the pinned repo commits and model, run count, total cost, and
a one-command reproduction. Until then, this project makes no
performance claims.

## License

MIT
