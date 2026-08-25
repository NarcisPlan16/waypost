"""waystone CLI entry point.

Sprint 0 scope only: argument parsing scaffold + `--version`.
Real subcommands (index | map | find | show | refs | outline | stats)
land in later sprints.
"""

from __future__ import annotations

import argparse
import sys

from waystone import __version__

# Subcommands that will exist once their sprints land. Listed now so the
# help text and error messages are honest about scope without requiring
# real implementations yet.
PLANNED_COMMANDS = ["index", "map", "find", "show", "refs", "outline", "stats"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="waystone",
        description=(
            "Zero-LLM-call repo indexer that gives coding agents a "
            "token-frugal map of a codebase."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"waystone {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    for name in PLANNED_COMMANDS:
        subparsers.add_parser(
            name, help=f"(not yet implemented — lands in a later sprint)"
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command in PLANNED_COMMANDS:
        print(
            f"waystone {args.command}: not implemented yet (Sprint 0 is scaffolding only)",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
