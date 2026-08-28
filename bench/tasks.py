"""Benchmark task definitions, loaded from JSON and validated up front.

A malformed task file must fail at startup, not ninety minutes into a paid
batch. The loader is therefore strict: unknown keys, unknown categories and
unknown grader kinds are all errors, and every grader's own required fields
are checked here rather than at grading time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Categories from the roadmap. E is the control: the task prompt already names
# the file, so a large reduction there means the harness is biased, not that
# the tool works.
CATEGORIES = {
    "A": "localization",
    "B": "cross-file edit",
    "C": "bug fix",
    "D": "trace/explain",
    "E": "control",
}

GRADER_KINDS = {"localization", "tests", "diff", "judge"}

# Required and optional keys per grader kind. Anything else in a ``grade``
# block is a typo, and a typo that silently grades nothing is worse than a
# crash.
_GRADE_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "localization": ({"expect_files"}, set()),
    "tests": (set(), {"command"}),
    "diff": ({"expect_files"}, {"expect_contains", "expect_absent"}),
    "judge": ({"rubric"}, set()),
}

_TASK_KEYS = {"id", "repo", "category", "prompt", "grade", "setup"}

_SETUP_KEYS = {"path", "find", "replace"}


class TaskError(ValueError):
    """A task file is malformed."""


@dataclass(frozen=True)
class Task:
    """One benchmark task, run identically in both arms."""

    id: str
    repo: str
    category: str
    prompt: str
    grade: dict[str, Any]
    setup: tuple[dict[str, str], ...] = ()
    source: Path | None = None

    @property
    def grade_kind(self) -> str:
        return str(self.grade["kind"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskError(message)


def parse_task(data: Any, source: Path | None = None) -> Task:
    """Validate one decoded task object and return it as a :class:`Task`."""
    where = f" in {source}" if source is not None else ""
    _require(isinstance(data, dict), f"task must be a JSON object{where}")

    unknown = sorted(set(data) - _TASK_KEYS)
    _require(not unknown, f"unknown task key(s) {unknown}{where}")
    missing = sorted(_TASK_KEYS - {"setup"} - set(data))
    _require(not missing, f"missing task key(s) {missing}{where}")

    for key in ("id", "repo", "category", "prompt"):
        value = data[key]
        _require(
            isinstance(value, str) and value.strip() != "",
            f"{key} must be a non-empty string{where}",
        )

    category = data["category"]
    _require(category in CATEGORIES, f"unknown category {category!r}{where}")

    grade = data["grade"]
    _require(isinstance(grade, dict), f"grade must be an object{where}")
    _require("kind" in grade, f"grade.kind is required{where}")
    kind = grade["kind"]
    _require(kind in GRADER_KINDS, f"unknown grade.kind {kind!r}{where}")

    required, optional = _GRADE_FIELDS[str(kind)]
    keys = set(grade) - {"kind"}
    missing_grade = sorted(required - keys)
    _require(not missing_grade, f"grade.{kind} needs {missing_grade}{where}")
    extra_grade = sorted(keys - required - optional)
    _require(not extra_grade, f"unknown grade key(s) {extra_grade} for kind {kind!r}{where}")

    setup = data.get("setup", [])
    _require(isinstance(setup, list), f"setup must be an array{where}")
    edits: list[dict[str, str]] = []
    for entry in setup:
        _require(isinstance(entry, dict), f"each setup entry must be an object{where}")
        bad = sorted(set(entry) ^ _SETUP_KEYS)
        _require(
            not bad, f"setup entry needs exactly {sorted(_SETUP_KEYS)}, differs by {bad}{where}"
        )
        _require(
            all(isinstance(entry[key], str) for key in _SETUP_KEYS),
            f"setup entry values must be strings{where}",
        )
        edits.append({key: entry[key] for key in _SETUP_KEYS})

    return Task(
        id=data["id"],
        repo=data["repo"],
        category=category,
        prompt=data["prompt"],
        grade=dict(grade),
        setup=tuple(edits),
        source=source,
    )


def load_task(path: Path) -> Task:
    """Read and validate a single task file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskError(f"{path} is not valid JSON: {exc}") from exc
    return parse_task(data, source=path)


def load_tasks(directory: Path, repo: str | None = None) -> list[Task]:
    """Load every ``*.json`` task under *directory*, sorted by id.

    Sorted so the run matrix is deterministic before the seeded shuffle; the
    shuffle is what randomises order, not filesystem iteration order.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise TaskError(f"no task files found in {directory}")

    tasks = [load_task(path) for path in paths]

    seen: dict[str, Path | None] = {}
    for task in tasks:
        if task.id in seen:
            raise TaskError(f"duplicate task id {task.id!r} ({seen[task.id]} and {task.source})")
        seen[task.id] = task.source

    if repo is not None:
        tasks = [task for task in tasks if task.repo == repo]
        if not tasks:
            raise TaskError(f"no tasks for repo {repo!r} in {directory}")

    return sorted(tasks, key=lambda t: t.id)
