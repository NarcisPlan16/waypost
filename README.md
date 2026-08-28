# Waypost

A zero-LLM-call repo indexer that gives AI coding agents (Claude Code, Cowork,
and others) a token-frugal map of a codebase, instead of them grepping and
re-reading whole files.

**Status: pre-alpha, no released version.** `walk.py` finds the files;
`parse.py` extracts symbols from Python, JavaScript, TypeScript and TSX;
`index.py` builds and persists a schema-versioned index with incremental
refresh by content SHA; `rank.py` scores files; and `tokens.py` /
`render.py` turn all of that into budgeted output behind the seven CLI
subcommands. The tool is usable end to end, and now has an npm wrapper
(`bin/waypost.js`) and a Claude Skill (`SKILL.md`) on top of the CLI. What
it does not yet have — the part that decides whether any of it was worth
building — is the benchmark.

### Token budgets are measured, not estimated

`map --budget 2000` emits at most 2,000 tokens as counted by the configured
tokenizer, and the cut always lands on a whole symbol. Selection is a binary
search over an ordered list of symbols for the largest prefix whose
*rendered* output fits, so the guarantee is a property of one function
(`tokens.fit_lines`) rather than of each renderer's arithmetic — counting
content and a truncation notice separately, then adding them, is off by a
token or two, which for the one function whose job is not exceeding a number
is the whole ballgame. `tests/test_render.py` asserts the ceiling holds
across budgets from 0 to 5,000 under both tokenizers.

A small budget buys a map of the repository rather than a very thorough
listing of its single highest-ranked file: top-level classes and exported
functions for *every* file in rank order come first, and private helpers
fill whatever is left. Raising the budget only ever adds lines, never
reshuffles them.

The default tokenizer is tiktoken's `cl100k_base`. That is not Claude's
tokenizer, and nothing published is — what a budget guarantees is a hard,
reproducible ceiling under a *named* tokenizer, which every output that
quotes a number reports. `tiktoken` also downloads its BPE table on first
use; since this package promises no network, a machine that cannot load it
falls back to a deterministic offline tokenizer that over-counts rather than
risking an under-count. `WAYPOST_TOKENIZER=heuristic` forces that path.

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

Once published, `npx waypost <command>` saves the separate `pip install` on
any machine that already has `uv`, `pipx`, or a Python holding `waypost` —
which is not the same as needing nothing at all. `bin/waypost.js` is a thin
resolver, not a bundled runtime: it tries `waypost` (already on PATH), `uvx
waypost`, `pipx run waypost`, then `python3 -m waypost` / `python -m
waypost`, probes each with `--version`, and uses the first one that actually
answers. A launcher that exists but cannot run waypost — the Microsoft Store
`python3` alias is the common one — is skipped rather than allowed to
swallow the command. If none answer it exits `2` and names what it tried. It
makes no LLM calls and adds no logic beyond that resolution; the indexer it
hands off to is exactly the one described in this README.

## Claude Skill

`SKILL.md` at the repo root tells an agent when to reach for `waypost`
(getting oriented, finding a symbol, reading one function, tracing refs,
outlining a file) and when not to (small repos, non-Python/JS/TS files,
edits that need the real file content). It is shipped in the npm package
too, so `node_modules/waypost/SKILL.md` is a copy source as well.

Claude Code discovers skills by directory, so install it as one:

```bash
mkdir -p ~/.claude/skills/waypost          # or .claude/skills/waypost, per project
cp SKILL.md ~/.claude/skills/waypost/SKILL.md
```

## Usage

```bash
waypost index                     # build .waypost/index.json (once, then on demand)
waypost index --rank pagerank --focus src/api   # and remember that from now on

waypost map                       # ranked symbol map, 2000 tokens by default
waypost map --budget 800 --focus src/api
waypost find "*_client"           # locate symbols by name, substring or glob
waypost show Client.request       # that symbol's own source span, nothing around it
waypost refs build_client         # where it is defined, and every file that uses it
waypost outline src/client.py     # every symbol in one file
waypost stats                     # what the index holds
```

Every command takes `--root`, `--json`, `--budget` and `--measure` (which
reports the token count of what it just printed, on stderr). Query commands
read the stored index rather than re-walking the repository — that is what
keeps them fast — so pass `--refresh` after changing files, or re-run
`waypost index`. A missing index is built automatically on first use.

An index remembers how it was ranked. `--rank` and `--focus` are settings of
the index, not of a single command, so a repository indexed with `--rank
pagerank --focus src/api` keeps those scores through every `--refresh` and
every plain `waypost index`; changing them takes another explicit `--rank` or
`--focus`. `--focus` is a path prefix everywhere it appears — a directory
works, and means the same thing to pagerank's personalisation as it does to
`map`'s ordering.

Exit codes are meant to be branched on: `0` success, `1` nothing matched,
`2` a real error — including a `--root` that is not a directory, which every
command checks before it does anything.

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
deliberately deferred.

## Benchmarks

**No numbers yet, and none are implied.** The harness that will produce
them lives in `bench/`; it has not been run against the API at scale, so
this project still makes no performance claims.

What the harness does: it runs the same coding tasks twice against a
pinned checkout of a real repository -- once with an agent that has only
`bash`, `read_file`, `grep` and `edit_file`, and once with the same agent
plus waypost -- and compares the input tokens each spent. It drives the
Anthropic API directly rather than going through any coding agent, whose
context management and caching would make the difference unattributable.
Prompt caching is off in both arms and every response is checked for it,
each run gets a fresh git worktree, indexing happens outside the measured
window, and the number of times the agent actually called waypost is
recorded -- if that is near zero, the skill wording is the thing to fix,
not the indexer.

```bash
uv sync --extra dev --extra bench
python -m bench tasks                              # validate the task suite
python -m bench run --dry-run --repo flask         # walk the matrix, no API calls
python -m bench run --repo flask --trials 1        # asks before spending
python -m bench report bench/results/<file>.jsonl
```

The report always breaks results out per category, including the control
category, where the task prompt already names the file. A large "saving"
there would mean the harness is biased rather than that the tool works,
and that check is printed next to the headline, not buried.

Two things are still stubs: the judge grader for explanation tasks (it
reports runs as ungraded rather than guessing), and the `tests` grader,
which works but needs a repository environment with the dependencies
installed -- the seed tasks grade by diff instead.

## License

MIT
