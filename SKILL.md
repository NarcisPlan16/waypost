---
name: waypost
description: Use before grepping or opening files to understand a codebase's structure. Gives a token-budgeted map of symbols (functions, classes, methods) across a repo, or the exact source span of one named symbol, without reading whole files. Zero LLM calls, offline, deterministic. Covers Python, JavaScript, TypeScript and TSX.
---

# waypost

`waypost` indexes a repository with tree-sitter and serves a symbol-level
view of it through a CLI. It answers "what's in this codebase and where" far
cheaper than grep-and-read, because it never returns a whole file — only
signatures, one-line doc summaries, and (for `show`) a single symbol's own
span.

## When to use it

- **Getting oriented in an unfamiliar repo or directory.** Run `waypost map`
  instead of opening several files to see what's there.
- **Finding a symbol by name.** `waypost find <pattern>` (substring or glob)
  instead of grepping for a definition.
- **Reading one function or class.** `waypost show <symbol>` returns just
  that symbol's source, not the surrounding file.
- **Finding callers/usages of a symbol.** `waypost refs <symbol>` instead of
  a repo-wide grep for its name.
- **Seeing everything defined in one file.** `waypost outline <path>`
  instead of reading the file to build a mental table of contents.

## When NOT to use it

- **You already know the exact file and need to edit it.** Read the file
  directly — waypost's outputs are signatures, not full bodies, and editing
  needs the real source anyway.
- **The symbol isn't Python, JS, TS, or TSX.** waypost only parses those
  languages; it will not find symbols in other files, and `find`/`show`
  reporting nothing is expected there, not a bug.
- **You need the literal current file content for a diff or a write.** The
  index can be stale relative to uncommitted changes; pass `--refresh` first
  if freshness matters, or just read the file.
- **The repo is tiny (a handful of files).** The overhead of indexing isn't
  worth it — just read the files.

## Commands

```bash
waypost index                     # build/refresh .waypost/index.json (once, then on demand)
waypost map --budget 2000         # ranked symbol map across the repo, budgeted in tokens
waypost map --focus src/api       # same map, with these paths sorted first
waypost find "*_client"           # locate symbols by name (substring or glob)
waypost show Client.request       # one symbol's own source span, nothing around it
waypost refs build_client         # where a symbol is defined, and every file that calls it
waypost outline src/client.py     # every symbol in one file
waypost stats                     # what the index currently holds
```

Every command accepts `--root <path>` (defaults to cwd), `--json` for
machine-readable output, and `--measure` to print the output's real token
count to stderr. The six query commands additionally accept `--budget <n>`
to cap token output and `--refresh` to re-index changed files before
answering (`index` takes neither: it always writes, and it is what `--refresh`
calls). A missing index is built automatically on first use. Exit code `1`
means nothing matched (not an error); `2` means a real failure (bad root,
bad flag).

## Notes

- Every output respects a measured token budget (via the configured
  tokenizer) — it will never dump enough content to blow a context window.
- No network access and no LLM calls happen anywhere in this tool. It is
  pure static analysis.
- If a query command's answer looks stale (a symbol you just added is
  missing), re-run with `--refresh` before concluding it doesn't exist.
