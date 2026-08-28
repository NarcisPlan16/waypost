from __future__ import annotations

import json
from collections import Counter
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


def test_the_shipped_suite_is_valid_and_balanced_across_repos_and_categories():
    tasks = load_tasks(Path("bench/tasks"))
    # The roadmap's matrix: 20 tasks, 2 repos x 5 categories x 2 each. The
    # balance is the assertion that matters -- per-category reductions are
    # only comparable if every cell is the same size, and the control cell
    # has to be as well stocked as the rest to detect a biased harness.
    assert len(tasks) == 20
    assert {t.repo for t in tasks} == {"flask", "hono"}
    assert {t.category for t in tasks} == set(CATEGORIES)

    cells = Counter((t.repo, t.category) for t in tasks)
    assert set(cells.values()) == {2}, cells


def test_load_tasks_filters_by_category(tmp_path):
    write(tmp_path, "a", VALID)
    write(
        tmp_path,
        "d",
        {
            **VALID,
            "id": "demo-D-01",
            "category": "D",
            "grade": {"kind": "rubric", "expect_all": [["x"]]},
        },
    )
    assert [t.id for t in load_tasks(tmp_path, categories=["D"])] == ["demo-D-01"]
    # Lowercase is what a hurried command line actually types.
    assert [t.id for t in load_tasks(tmp_path, categories=["d"])] == ["demo-D-01"]
    assert len(load_tasks(tmp_path, categories=["A", "D"])) == 2

    with pytest.raises(TaskError, match="unknown categor"):
        load_tasks(tmp_path, categories=["Z"])
    with pytest.raises(TaskError, match="no tasks in categor"):
        load_tasks(tmp_path, categories=["B"])


def test_load_tasks_filters_by_id(tmp_path):
    write(tmp_path, "a", VALID)
    write(tmp_path, "b", {**VALID, "id": "demo-A-02"})
    assert [t.id for t in load_tasks(tmp_path, ids=["demo-A-02"])] == ["demo-A-02"]

    # A mistyped id must not silently run a different, more expensive batch.
    with pytest.raises(TaskError, match="no such task id"):
        load_tasks(tmp_path, ids=["demo-A-99"])


def test_load_tasks_filters_compose(tmp_path):
    write(tmp_path, "a", VALID)
    write(tmp_path, "b", {**VALID, "id": "other-A-01", "repo": "other"})
    with pytest.raises(TaskError, match="no such task id"):
        load_tasks(tmp_path, repo="demo", ids=["other-A-01"])


def test_bench_tasks_cli_accepts_the_filters(capsys):
    from bench.cli import main

    assert main(["tasks", "--repo", "flask", "--category", "D"]) == 0
    out = capsys.readouterr().out
    assert "flask-D-01" in out and "flask-A-01" not in out
    assert "2 task(s)" in out
