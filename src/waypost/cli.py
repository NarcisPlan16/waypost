"""waypost CLI entry point.

Sprint 3 wires the real subcommands: ``index`` builds and persists, and
``map | find | show | refs | outline | stats`` query what it persisted.

Three decisions this module makes, all of them visible to an agent:

**Query commands do not re-walk the repository.** They read
``.waypost/index.json`` as it stands. Refreshing costs a full walk and a
SHA per file, which is the wrong price to pay on every ``find`` and would
blow the "``map --budget 2000`` in under 100 ms" target outright. The index
is therefore as fresh as the last ``waypost index`` -- pass ``--refresh`` to
any query command to update it first. A *missing* index is different from a
stale one: that is built automatically, once, with a note on stderr, because
failing a first-ever ``waypost map`` on a setup step nobody was told about
is a bad first impression.

**Not-found is exit 1, not an error.** ``find``, ``show``, ``refs`` and
``outline`` exit 1 when they match nothing, with the explanation on stdout.
An agent can branch on the code; a human can read the line. Genuine failures
(no such root, a bad ``--rank``) exit 2 with the message on stderr.

**Everything goes to stdout except diagnostics.** Notes about rebuilding,
and the ``--measure`` token count, go to stderr, so piping ``waypost map``
into a file or a prompt never picks up commentary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from waypost import __version__, render
from waypost import index as index_mod
from waypost.tokens import count, get_tokenizer

# Kept for the help text and for tests that assert the surface; every one of
# these now has a real implementation.
PLANNED_COMMANDS = ["index", "map", "find", "show", "refs", "outline", "stats"]

RANK_STRATEGIES = ("simple", "pagerank")

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_ERROR = 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root to index or query (default: the current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of the token-frugal text form",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="print the token count of the output, and the tokenizer used, to stderr",
    )


def _add_refresh(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-index changed files before answering (a full walk; not the default)",
    )


def _add_budget(parser: argparse.ArgumentParser, default: int) -> None:
    parser.add_argument(
        "--budget",
        type=int,
        default=default,
        help=f"hard token ceiling on the output, measured not estimated (default: {default})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="waypost",
        description=(
            "Zero-LLM-call repo indexer that gives coding agents a token-frugal map of a codebase."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"waypost {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    p_index = subparsers.add_parser("index", help="build or refresh .waypost/index.json")
    _add_common(p_index)
    p_index.add_argument(
        "--rank",
        choices=RANK_STRATEGIES,
        default="simple",
        help="ranking strategy (default: simple)",
    )
    p_index.add_argument(
        "--focus",
        action="append",
        default=[],
        metavar="PATH",
        help="bias pagerank personalisation toward this path; repeatable",
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="full rebuild, ignoring any existing index",
    )

    p_map = subparsers.add_parser("map", help="ranked symbol map of the repository")
    _add_common(p_map)
    _add_refresh(p_map)
    _add_budget(p_map, render.DEFAULT_MAP_BUDGET)
    p_map.add_argument(
        "--focus",
        action="append",
        default=[],
        metavar="PATH",
        help="sort these paths first (a prefix, so a directory works); repeatable",
    )

    p_find = subparsers.add_parser("find", help="locate symbols by name, substring or glob")
    _add_common(p_find)
    _add_refresh(p_find)
    _add_budget(p_find, render.DEFAULT_LIST_BUDGET)
    p_find.add_argument("pattern", help="symbol name, substring, or glob (*, ?, [])")
    p_find.add_argument(
        "--limit",
        type=int,
        default=render.DEFAULT_FIND_LIMIT,
        help=f"maximum hits (default: {render.DEFAULT_FIND_LIMIT})",
    )

    p_show = subparsers.add_parser("show", help="print one symbol's own source span")
    _add_common(p_show)
    _add_refresh(p_show)
    _add_budget(p_show, render.DEFAULT_SHOW_BUDGET)
    p_show.add_argument("name", help="qualified name (Client.request) or bare name (request)")
    p_show.add_argument(
        "--max-matches",
        type=int,
        default=3,
        help="how many same-named definitions to print (default: 3)",
    )

    p_refs = subparsers.add_parser("refs", help="where a symbol is defined and referenced")
    _add_common(p_refs)
    _add_refresh(p_refs)
    _add_budget(p_refs, render.DEFAULT_LIST_BUDGET)
    p_refs.add_argument("name", help="symbol name to trace")

    p_outline = subparsers.add_parser("outline", help="every symbol in one file")
    _add_common(p_outline)
    _add_refresh(p_outline)
    _add_budget(p_outline, render.DEFAULT_LIST_BUDGET)
    p_outline.add_argument("path", help="indexed path, or an unambiguous suffix of one")

    p_stats = subparsers.add_parser("stats", help="index size, languages, kinds, top files")
    _add_common(p_stats)
    _add_refresh(p_stats)
    _add_budget(p_stats, render.DEFAULT_LIST_BUDGET)

    return parser


def _emit(text: str, args: argparse.Namespace) -> None:
    """Print output, then the measurement if asked for."""
    print(text)
    if getattr(args, "measure", False):
        tokenizer = get_tokenizer()
        print(
            f"[{count(text, tokenizer=tokenizer)} tokens, tokenizer {tokenizer.name}]",
            file=sys.stderr,
        )


def _emit_json(data: object, args: argparse.Namespace) -> None:
    _emit(json.dumps(data, indent=2, sort_keys=True), args)


def _acquire_index(args: argparse.Namespace) -> tuple[index_mod.Index, Path]:
    """Load the index for a query command, building or refreshing if asked.

    Returns the index and the path it lives at (``stats`` reports the file
    size, and a caller that rebuilt should not have to guess where it went).
    """
    root: Path = args.root
    path = index_mod.default_index_path(root)
    existing = index_mod.load(path)

    if existing is not None and not getattr(args, "refresh", False):
        return existing, path

    if existing is None:
        print(
            f"waypost: no usable index at {path}; building one (this happens once)",
            file=sys.stderr,
        )
    updated = index_mod.refresh(root, existing)
    index_mod.save(updated, path)
    return updated, path


def _cmd_index(args: argparse.Namespace) -> int:
    root: Path = args.root
    if not root.is_dir():
        print(f"waypost: {root} is not a directory", file=sys.stderr)
        return EXIT_ERROR

    path = index_mod.default_index_path(root)
    existing = None if args.force else index_mod.load(path)
    focus = args.focus or None

    updated = index_mod.refresh(root, existing, rank_strategy=args.rank, personalize=focus)
    index_mod.save(updated, path)

    data = render.stats_data(updated, index_path=path)
    if args.json:
        _emit_json({**data, "index_path": str(path)}, args)
        return EXIT_OK

    mode = "rebuilt" if existing is None else "refreshed"
    written = data["index_bytes"]
    size = "unknown size" if written is None else f"{written / 1024:.0f} KiB"
    _emit(
        f"{mode} {path}: {data['files']} files, {data['symbols']} symbols, "
        f"{data['loc']} loc, {size}, rank={args.rank}",
        args,
    )
    return EXIT_OK


def _cmd_map(args: argparse.Namespace) -> int:
    idx, _ = _acquire_index(args)
    focus = args.focus or None
    if args.json:
        _emit_json(render.map_data(idx, budget=args.budget, focus=focus), args)
    else:
        _emit(render.render_map(idx, budget=args.budget, focus=focus), args)
    return EXIT_OK


def _cmd_find(args: argparse.Namespace) -> int:
    idx, _ = _acquire_index(args)
    hits = render.search(idx, args.pattern, limit=args.limit)
    if args.json:
        _emit_json(render.find_data(idx, args.pattern, limit=args.limit), args)
    else:
        _emit(render.render_find(idx, args.pattern, limit=args.limit, budget=args.budget), args)
    return EXIT_OK if hits else EXIT_NOT_FOUND


def _cmd_show(args: argparse.Namespace) -> int:
    idx, _ = _acquire_index(args)
    found = bool(render.definitions(idx, args.name))
    if args.json:
        _emit_json(render.show_data(idx, args.name, max_matches=args.max_matches), args)
    else:
        _emit(
            render.render_show(idx, args.name, budget=args.budget, max_matches=args.max_matches),
            args,
        )
    return EXIT_OK if found else EXIT_NOT_FOUND


def _cmd_refs(args: argparse.Namespace) -> int:
    idx, _ = _acquire_index(args)
    found = bool(render.definitions(idx, args.name)) or bool(
        render.referencing_files(idx, args.name)
    )
    if args.json:
        _emit_json(render.refs_data(idx, args.name), args)
    else:
        _emit(render.render_refs(idx, args.name, budget=args.budget), args)
    return EXIT_OK if found else EXIT_NOT_FOUND


def _cmd_outline(args: argparse.Namespace) -> int:
    idx, _ = _acquire_index(args)
    found = render.resolve_path(idx, args.path) is not None
    if args.json:
        _emit_json(render.outline_data(idx, args.path), args)
    else:
        _emit(render.render_outline(idx, args.path, budget=args.budget), args)
    return EXIT_OK if found else EXIT_NOT_FOUND


def _cmd_stats(args: argparse.Namespace) -> int:
    idx, path = _acquire_index(args)
    if args.json:
        _emit_json(render.stats_data(idx, index_path=path), args)
    else:
        _emit(render.render_stats(idx, index_path=path, budget=args.budget), args)
    return EXIT_OK


_HANDLERS = {
    "index": _cmd_index,
    "map": _cmd_map,
    "find": _cmd_find,
    "show": _cmd_show,
    "refs": _cmd_refs,
    "outline": _cmd_outline,
    "stats": _cmd_stats,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    handler = _HANDLERS.get(args.command)
    if handler is None:  # pragma: no cover -- argparse rejects unknown commands first.
        parser.print_help()
        return EXIT_ERROR

    try:
        return handler(args)
    except OSError as exc:
        print(f"waypost: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
