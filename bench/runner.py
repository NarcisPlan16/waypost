"""Batch execution: the run matrix, its ordering, and the result file.

Three properties matter more than anything else in this file:

- **Order is randomised from a seed, and the two arms of a task are run
  adjacently.** API-side drift over a multi-hour batch then hits both arms
  almost equally, and the batch is reproducible.
- **Every finished run is appended and flushed immediately.** A batch that
  dies at run 137 loses nothing; ``--resume`` picks it up.
- **A failed run is recorded as a failure**, never dropped. Dropping failures
  silently favours whichever arm fails more.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import __version__ as bench_version
from bench.arms import ARMS, ArmSpec, build_arm
from bench.client import estimate_cost
from bench.grade import grade
from bench.loop import DEFAULT_EFFORT, DEFAULT_MODEL, TURN_CAP, CacheLeakError, run_task
from bench.tasks import Task
from bench.tools import Executor
from bench.worktree import RepoSpec, run_worktree

# Deliberately rough, and labelled as such wherever it is printed. It exists to
# stop someone starting a 200-run batch believing it costs cents; it is not a
# measurement and nothing in the results depends on it.
ROUGH_INPUT_TOKENS_PER_RUN = 150_000
ROUGH_OUTPUT_TOKENS_PER_RUN = 8_000


@dataclass(frozen=True)
class RunKey:
    """Identifies one cell of the matrix, for resume."""

    task_id: str
    arm: str
    trial: int

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.task_id, self.arm, self.trial)


def build_matrix(tasks: list[Task], trials: int, seed: int) -> list[RunKey]:
    """Task order shuffled from *seed*; the two arms of a task stay adjacent.

    Which arm goes first alternates, so a systematic first-position effect --
    a warm filesystem cache, say -- cannot land on one arm every time.
    """
    pairs = [(task, trial) for task in tasks for trial in range(1, trials + 1)]
    random.Random(seed).shuffle(pairs)

    matrix: list[RunKey] = []
    for position, (task, trial) in enumerate(pairs):
        order = ARMS if position % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            matrix.append(RunKey(task_id=task.id, arm=arm, trial=trial))
    return matrix


def completed_keys(results_path: Path) -> set[tuple[str, str, int]]:
    """Keys already present in a result file, for ``--resume``."""
    if not results_path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        done.add((record["task_id"], record["arm"], int(record["trial"])))
    return done


def rough_cost(model: str, runs: int) -> float | None:
    return estimate_cost(
        model,
        runs * ROUGH_INPUT_TOKENS_PER_RUN,
        runs * ROUGH_OUTPUT_TOKENS_PER_RUN,
    )


def environment_record(model: str, effort: str, seed: int) -> dict[str, Any]:
    """What the batch was, recorded next to what it measured."""
    return {
        "bench_version": bench_version,
        "model": model,
        "effort": effort,
        "seed": seed,
        "turn_cap": TURN_CAP,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


@dataclass
class Runner:
    """Executes a matrix and appends one JSON line per run."""

    client: Any
    tasks: dict[str, Task]
    repos: dict[str, RepoSpec]
    results_path: Path
    cache_dir: Path
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    seed: int = 1
    work_dir: Path | None = None
    skill_path: Path | None = None

    def __post_init__(self) -> None:
        self._arms: dict[str, ArmSpec] = {name: build_arm(name, self.skill_path) for name in ARMS}

    def run(self, matrix: list[RunKey], resume: bool = False) -> Iterator[dict[str, Any]]:
        """Run each key in order, yielding the record written for it."""
        done = completed_keys(self.results_path) if resume else set()
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        with self.results_path.open("a", encoding="utf-8") as sink:
            for key in matrix:
                if key.as_tuple() in done:
                    continue
                record = self._run_one(key)
                sink.write(json.dumps(record, sort_keys=True) + "\n")
                sink.flush()
                yield record

    def _run_one(self, key: RunKey) -> dict[str, Any]:
        task = self.tasks[key.task_id]
        repo = self.repos[task.repo]
        arm = self._arms[key.arm]

        base = {
            "task_id": task.id,
            "arm": key.arm,
            "trial": key.trial,
            "repo": task.repo,
            "repo_sha": repo.sha,
            "category": task.category,
            "grade_kind": task.grade_kind,
            "started_at": time.time(),
            **environment_record(self.model, self.effort, self.seed),
            **arm.as_record(),
        }

        work_dir = self.work_dir or Path(tempfile.gettempdir()) / "waypost-bench"
        with run_worktree(repo, self.cache_dir, work_dir, setup=task.setup) as tree:
            execute = Executor(tree)
            # The clock and the token counter start here: the clone, the
            # worktree and the index are all outside the measurement.
            result = run_task(
                self.client,
                system=arm.system,
                tools=arm.tools,
                prompt=task.prompt,
                execute=execute,
                model=self.model,
                effort=self.effort,
            )
            verdict = grade(task, result, tree, repo.test_command)

        return {
            **base,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "turns": result.turns,
            "wall_clock_s": round(result.wall_clock_s, 3),
            "tool_calls": dict(result.tool_calls),
            "waypost_calls": result.tool_calls.get("waypost", 0),
            "stop_reason": result.stop_reason,
            "failure_reason": result.failure_reason,
            "final_text": result.final_text,
            **verdict.as_record(),
        }


def execute_batch(runner: Runner, matrix: list[RunKey], resume: bool = False) -> int:
    """Run a matrix, reporting progress; returns the number of runs written."""
    written = 0
    total = len(matrix)
    try:
        for record in runner.run(matrix, resume=resume):
            written += 1
            status = record["failure_reason"] or record["success"]
            print(
                f"[{written}/{total}] {record['task_id']} {record['arm']}"
                f" trial {record['trial']}: {record['input_tokens']} input tokens,"
                f" {record['turns']} turns, waypost x{record['waypost_calls']},"
                f" result {status}",
                flush=True,
            )
    except CacheLeakError as exc:
        print(f"bench: aborting batch -- {exc}", file=sys.stderr)
        raise
    return written
