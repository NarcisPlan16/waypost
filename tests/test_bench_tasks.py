from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.tasks import CATEGORIES, TaskError, load_tasks, parse_task

VALID = {
    "id": "demo-A-01",
    "repo": "demo",
    "category": "A",
    "prompt": "Find the thing.",
    "grade": {"kind": "localization", "expect_files": ["src/thing.py"]},
}


def write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parses_a_valid_task():
    task = parse_task(VALID)
    assert task.id == "demo-A-01"
    assert task.grade_kind == "localization"
    assert task.setup == ()


def test_rejects_unknown_category():
    with pytest.raises(TaskError, match="unknown category"):
        parse_task({**VALID, "category": "Z"})


def test_rejects_unknown_grade_kind():
    with pytest.raises(TaskError, match="unknown grade.kind"):
        parse_task({**VALID, "grade": {"kind": "vibes"}})


def test_rejects_grade_missing_its_required_field():
    with pytest.raises(TaskError, match="expect_files"):
        parse_task({**VALID, "grade": {"kind": "localization"}})


def test_rejects_a_typo_in_a_grade_key():
    # The whole point of validating up front: a silently ignored `expect_file`
    # would grade nothing and report a success.
    with pytest.raises(TaskError, match="unknown grade key"):
        parse_task(
            {**VALID, "grade": {"kind": "localization", "expect_files": ["a"], "expect_file": "b"}}
        )


def test_rejects_unknown_top_level_key():
    with pytest.raises(TaskError, match="unknown task key"):
        parse_task({**VALID, "notes": "hello"})


def test_setup_entries_are_validated():
    good = {**VALID, "setup": [{"path": "a.py", "find": "x", "replace": "y"}]}
    assert parse_task(good).setup == ({"path": "a.py", "find": "x", "replace": "y"},)

    with pytest.raises(TaskError, match="setup entry"):
        parse_task({**VALID, "setup": [{"path": "a.py", "find": "x"}]})


def test_load_tasks_rejects_duplicate_ids(tmp_path):
    write(tmp_path, "one", VALID)
    write(tmp_path, "two", VALID)
    with pytest.raises(TaskError, match="duplicate task id"):
        load_tasks(tmp_path)


def test_load_tasks_filters_by_repo(tmp_path):
    write(tmp_path, "a", VALID)
    write(tmp_path, "b", {**VALID, "id": "other-A-01", "repo": "other"})
    assert [t.id for t in load_tasks(tmp_path, repo="other")] == ["other-A-01"]

    with pytest.raises(TaskError, match="no tasks for repo"):
        load_tasks(tmp_path, repo="missing")


def test_load_tasks_errors_on_an_empty_directory(tmp_path):
    with pytest.raises(TaskError, match="no task files"):
        load_tasks(tmp_path)


def test_the_shipped_seed_suite_is_valid_and_covers_the_categories():
    tasks = load_tasks(Path("bench/tasks"))
    assert len(tasks) == 8
    assert {t.repo for t in tasks} == {"flask", "hono"}
    # Every category present in the seeds is a real one, and the control is
    # there: without E there is nothing to detect a biased harness with.
    assert {t.category for t in tasks} <= set(CATEGORIES)
    assert "E" in {t.category for t in tasks}
