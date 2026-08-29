---
name: waypost
description: Use before grepping or opening files to understand a codebase's structure. Gives a token-budgeted map of symbols (functions, classes, methods) across a repo, or the exact source span of one named symbol, without reading whole files. Zero LLM calls, offline, deterministic. Covers Python, JavaScript, TypeScript and TSX.
---

# waypost

`waypost` indexes a repository with tree-sitter and serves a symbol-level
view of it through a CLI. It answers "what's in this codebase and where" far
cheaper than grep-and-read, because it never returns a whole file — only
signatures, one-line doc summaries, and, for `show`, one symbol's complete
source. `show` gives you the real code: after it you do not need to open the
file.

## When to use it

- **Getting oriented in an unfamiliar repo or directory.** Run `waypost map`
  instead of opening several files to see what's there.
- **Reading one function or class — the common case.** `waypost show <name>`
  returns that symbol's whole source and nothing around it. It takes a bare
  name (`waypost show wsgi_app`), so you do not need the qualified
  `Class.method` form, and you do not need to locate it first. Go straight
  here whenever you want to *see* code: one call, no follow-up read.
- **Finding where a name is defined, when the location is the answer.**
  `waypost find <pattern>` instead of grepping for a definition. Leads with
  the exact-name definitions; `--all` adds the partial-name matches. `find`
  returns locations, not bodies — if what you actually wanted was the code,
  you wanted `show`, and reaching for `find` first costs you a round trip.
- **Finding callers/usages of a symbol.** `waypost refs <symbol>` instead of
  a repo-wide grep for its name.
- **Seeing everything defined in one file.** `waypost outline <path>`
  instead of reading the file to build a mental table of contents.

## When NOT to use it

- **You are about to edit, diff or write.** Read the file: an edit has to
  match surrounding bytes `show` does not give you, and the index can lag
  uncommitted changes (`--refresh` fixes staleness, not the missing context).
- **The file isn't Python, JS, TS, or TSX.** waypost parses only those, so
  `find`/`show` finding nothing elsewhere is expected, not a bug.
- **The repo is tiny (a handful of files).** Just read them.

## Commands

```bash
waypost index                     # build/refresh .waypost/index.json (once, then on demand)
waypost map --budget 2000         # ranked symbol map across the repo, budgeted in tokens
waypost map --focus src/api       # same map, with these paths sorted first
waypost show wsgi_app             # one symbol's full source, nothing around it (bare name is fine)
waypost find Client               # only where that name is defined (--all: partial matches too)
waypost refs build_client         # where a symbol is defined, and every file that calls it
waypost outline src/client.py     # every symbol in one file
waypost stats                     # what the index currently holds
```

All commands take `--root <path>` (default cwd), `--json`, and `--measure`
(real token count to stderr). The six query commands also take `--budget <n>`
and `--refresh` (re-index changed files first — use it if an answer looks
stale). A missing index is built on first use. Exit `1` means nothing matched;
`2` is a real failure.

Every output respects a measured token budget, so no command can blow a
context window. No network and no LLM calls anywhere: pure static analysis.
