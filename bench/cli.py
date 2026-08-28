"""`python -m bench` -- run a batch, or report on one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.client import build_client
from bench.context import (
    DEFAULT_MAP_BUDGET,
    DEFAULT_SHOW_BUDGET,
    build_probe,
    render_comparison,
    render_probe,
)
from bench.loop import DEFAULT_EFFORT, DEFAULT_MODEL
from bench.report import build_report, load_records, render_text
from bench.runner import Runner, build_matrix, execute_batch, rough_cost
from bench.tasks import TaskError, load_tasks
from bench.worktree import RepoSpec, build_index, load_repos, run_worktree
from waypost import index as index_mod

EXIT_OK = 0
EXIT_ABORTED = 1
EXIT_ERROR = 2

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent / "tasks"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_PROBE_REPOS = Path(__file__).resolve().parent / "probe-repos.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="waypost benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute a batch of runs")
    p_run.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_DIR, help="task directory")
    p_run.add_argument("--repos", type=Path, default=None, help="pinned repository table")
    p_run.add_argument("--repo", default=None, help="only run tasks for this repo")
    p_run.add_argument(
        "--category",
        action="append",
        default=None,
        metavar="A-E",
        help="only run this category; repeatable",
    )
    p_run.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="ID",
        help="only run this task id; repeatable",
    )
    p_run.add_argument("--trials", type=int, default=1, help="trials per task per arm")
    p_run.add_argument("--seed", type=int, default=1, help="seed for the task order")
    p_run.add_argument("--model", default=DEFAULT_MODEL, help="model under test")
    p_run.add_argument("--effort", default=DEFAULT_EFFORT, help="output_config effort level")
    p_run.add_argument("--out", type=Path, default=None, help="result file (JSONL)")
    p_run.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="clone cache")
    p_run.add_argument("--work-dir", type=Path, default=None, help="where worktrees are created")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="walk the whole matrix with a stub client; never calls the API",
    )
    p_run.add_argument("--resume", action="store_true", help="skip runs already in --out")
    p_run.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    p_run.set_defaults(handler=_cmd_run)

    p_report = sub.add_parser("report", help="summarise a result file")
    p_report.add_argument("results", type=Path, help="JSONL file written by `bench run`")
    p_report.add_argument("--seed", type=int, default=1, help="seed for the bootstrap")
    p_report.add_argument("--json", action="store_true", help="emit the report as JSON")
    p_report.set_defaults(handler=_cmd_report)

    p_tasks = sub.add_parser("tasks", help="validate and list the task suite")
    p_tasks.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_DIR, help="task directory")
    p_tasks.add_argument("--repo", default=None, help="only list tasks for this repo")
    p_tasks.add_argument(
        "--category",
        action="append",
        default=None,
        metavar="A-E",
        help="only list this category; repeatable",
    )
    p_tasks.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="ID",
        help="only list this task id; repeatable",
    )
    p_tasks.set_defaults(handler=_cmd_tasks)

    p_context = sub.add_parser(
        "context",
        help="token cost of `show`/`map` against reading files -- calls no model, spends nothing",
    )
    src = p_context.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", default=None, help="a repo from the probe table")
    src.add_argument("--root", type=Path, default=None, help="any local checkout instead")
    src.add_argument("--all", action="store_true", help="every repo in the table, side by side")
    p_context.add_argument(
        "--repos",
        type=Path,
        default=DEFAULT_PROBE_REPOS,
        help="probe repository table (defaults to bench/probe-repos.json)",
    )
    p_context.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="clone cache")
    p_context.add_argument("--work-dir", type=Path, default=None, help="where worktrees go")
    p_context.add_argument("--map-budget", type=int, default=DEFAULT_MAP_BUDGET)
    p_context.add_argument("--show-budget", type=int, default=DEFAULT_SHOW_BUDGET)
    p_context.add_argument("--json", action="store_true", help="emit the probe as JSON")
    p_context.set_defaults(handler=_cmd_context)

    return parser


def _cmd_tasks(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks, args.repo, args.category, args.task)
    for task in tasks:
        print(f"{task.id:<24}{task.repo:<12}{task.category:<4}{task.grade_kind}")
    print(f"\n{len(tasks)} task(s), all valid")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks, args.repo, args.category, args.task)
    repos = load_repos(args.repos)

    unknown = sorted({task.repo for task in tasks} - set(repos))
    if unknown:
        print(f"bench: tasks reference unknown repo(s) {unknown}", file=sys.stderr)
        return EXIT_ERROR

    matrix = build_matrix(tasks, args.trials, args.seed)
    # A filtered batch gets its own default file. Sharing the unfiltered name
    # would let `bench report` average a two-task smoke together with a full
    # matrix and call the mixture a result.
    stem = f"{args.model}-seed{args.seed}"
    if args.repo or args.category or args.task:
        stem += "-partial"
    results_path = args.out or DEFAULT_RESULTS_DIR / f"{stem}.jsonl"

    if not args.dry_run:
        estimate = rough_cost(args.model, len(matrix))
        shown = "unpriced model" if estimate is None else f"roughly ${estimate:.2f}"
        print(
            f"{len(matrix)} runs on {args.model}: {shown} (a rough order of magnitude, not a quote)"
        )
        if not args.yes and not _confirm():
            print("bench: aborted before spending anything")
            return EXIT_ABORTED

    runner = Runner(
        client=build_client(dry_run=args.dry_run),
        tasks={task.id: task for task in tasks},
        repos=repos,
        results_path=results_path,
        cache_dir=args.cache_dir,
        model=args.model,
        effort=args.effort,
        seed=args.seed,
        work_dir=args.work_dir,
    )

    written = execute_batch(runner, matrix, resume=args.resume)
    print(f"\n{written} run(s) written to {results_path}")
    return EXIT_OK


def _probe_pinned(args: argparse.Namespace, spec: RepoSpec) -> dict:
    work_dir = args.work_dir or DEFAULT_CACHE_DIR / "probe"
    # The same pinned, freshly indexed worktree the paid runs use, so the
    # probe describes the tree the benchmark would measure.
    with run_worktree(spec, args.cache_dir, work_dir) as root:
        return build_probe(
            root, map_budget=args.map_budget, show_budget=args.show_budget, label=spec.name
        )


def _cmd_context(args: argparse.Namespace) -> int:
    reports: list[dict]
    if args.root is not None:
        root = args.root.resolve()
        if index_mod.load(index_mod.default_index_path(root)) is None:
            build_index(root)
        reports = [
            build_probe(
                root, map_budget=args.map_budget, show_budget=args.show_budget, label=root.name
            )
        ]
    elif args.all:
        repos = load_repos(args.repos)
        reports = []
        for name, spec in repos.items():
            print(f"probing {name} ...", file=sys.stderr)
            reports.append(_probe_pinned(args, spec))
    else:
        repos = load_repos(args.repos)
        chosen = repos.get(args.repo)
        if chosen is None:
            print(f"bench: unknown repo {args.repo!r}", file=sys.stderr)
            return EXIT_ERROR
        reports = [_probe_pinned(args, chosen)]

    if args.json:
        print(json.dumps(reports if args.all else reports[0], indent=2, sort_keys=True))
    elif len(reports) > 1:
        print(render_comparison(reports))
    else:
        print(render_probe(reports[0]))
    return EXIT_OK


def _confirm() -> bool:
    if not sys.stdin.isatty():
        print("bench: not a terminal; pass --yes to run without confirmation", file=sys.stderr)
        return False
    return input("proceed? [y/N] ").strip().lower() in {"y", "yes"}


def _cmd_report(args: argparse.Namespace) -> int:
    report = build_report(load_records(args.results), seed=args.seed)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (TaskError, ValueError, RuntimeError, OSError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return EXIT_ERROR
