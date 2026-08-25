# Waypost

A zero-LLM-call repo indexer that gives AI coding agents (Claude Code, Cowork,
and others) a token-frugal map of a codebase, instead of them grepping and
re-reading whole files.

**Status: Sprint 0 (Foundations).** Only scaffolding exists — no indexing,
parsing, or query logic yet. See the roadmap doc for the full sprint plan.

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
