from __future__ import annotations

import subprocess

import pytest

from bench.grade import grade
from bench.loop import LoopResult
from bench.tasks import parse_task
from bench.worktree import apply_setup

BASE = {"id": "t-01", "repo": "demo", "category": "A", "prompt": "p"}


def make_task(**overrides):
    return parse_task({**BASE, **overrides})


def git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "config.py").write_text("KEY = 1\n", encoding="utf-8")
    git(["init", "-q"], tmp_path)
    git(["add", "-A"], tmp_path)
    git(["-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "init"], tmp_path)
    return tmp_path


def finished(text: str = "") -> LoopResult:
    return LoopResult(turns=1, final_text=text)


def test_localization_accepts_the_path_anywhere_in_the_answer():
    task = make_task(grade={"kind": "localization", "expect_files": ["src/config.py"]})
    verdict = grade(task, finished("I think it lives in src/config.py."), None)  # type: ignore[arg-type]
    assert verdict.success is True


def test_localization_fails_when_the_path_is_absent():
    task = make_task(grade={"kind": "localization", "expect_files": ["src/config.py"]})
    assert grade(task, finished("no idea"), None).success is False  # type: ignore[arg-type]


def test_a_run_that_did_not_complete_fails_regardless_of_what_it_said():
    task = make_task(grade={"kind": "localization", "expect_files": ["src/config.py"]})
    capped = LoopResult(turns=40, final_text="src/config.py", failure_reason="turn_cap")
    verdict = grade(task, capped, None)  # type: ignore[arg-type]
    assert verdict.success is False
    assert "turn_cap" in verdict.detail


RUBRIC = {
    "kind": "rubric",
    "expect_files": ["src/config.py"],
    "expect_all": [["wsgi_app"], ["url_map", "url map"]],
}


def test_rubric_passes_when_every_group_is_hit():
    task = make_task(category="D", grade=RUBRIC)
    answer = "src/config.py defines wsgi_app, which consults the url map."
    verdict = grade(task, finished(answer), None)  # type: ignore[arg-type]
    assert verdict.success is True


def test_rubric_needs_all_groups_not_just_one():
    task = make_task(category="D", grade=RUBRIC)
    answer = "src/config.py defines wsgi_app."
    verdict = grade(task, finished(answer), None)  # type: ignore[arg-type]
    assert verdict.success is False
    assert "url_map" in verdict.detail


def test_rubric_group_is_satisfied_by_any_alternative():
    task = make_task(category="D", grade=RUBRIC)
    # "url map" rather than "url_map": alternatives exist so that a correct
    # answer is not failed for prose phrasing.
    answer = "src/config.py defines wsgi_app and matches against the URL map."
    assert grade(task, finished(answer), None).success is True  # type: ignore[arg-type]


def test_rubric_tolerates_a_line_wrap_inside_an_anchor():
    task = make_task(category="D", grade={"kind": "rubric", "expect_all": [["request context"]]})
    answer = "it pushes a request" + chr(10) + "   context onto the stack"
    assert grade(task, finished(answer), None).success is True  # type: ignore[arg-type]


def test_rubric_still_requires_the_expected_files():
    task = make_task(category="D", grade=RUBRIC)
    answer = "wsgi_app consults the url_map, but I will not say where."
    verdict = grade(task, finished(answer), None)  # type: ignore[arg-type]
    assert verdict.success is False
    assert "src/config.py" in verdict.detail


def test_rubric_never_returns_ungraded():
    """Category D is the reason the report has an ungraded column at all.

    It is now graded, and that must not silently regress: an ungraded run
    costs API spend and yields no verdict.
    """
    task = make_task(category="D", grade=RUBRIC)
    assert grade(task, finished("nothing relevant"), None).success is False  # type: ignore[arg-type]


def test_diff_grader_checks_files_and_fragments(repo):
    task = make_task(
        grade={
            "kind": "diff",
            "expect_files": ["src/config.py"],
            "expect_contains": ["KEY = 2"],
            "expect_absent": ["KEY = 3"],
        }
    )
    assert grade(task, finished(), repo).success is False  # nothing changed yet

    (repo / "src" / "config.py").write_text("KEY = 2\n", encoding="utf-8")
    assert grade(task, finished(), repo).success is True

    (repo / "src" / "config.py").write_text("KEY = 3\n", encoding="utf-8")
    assert grade(task, finished(), repo).success is False


def test_tests_grader_uses_the_repo_command(repo):
    passing = make_task(category="C", grade={"kind": "tests"})
    assert grade(passing, finished(), repo, 'python -c "raise SystemExit(0)"').success is True
    assert grade(passing, finished(), repo, 'python -c "raise SystemExit(1)"').success is False


def test_tests_grader_refuses_to_run_with_no_command(repo):
    task = make_task(category="C", grade={"kind": "tests"})
    with pytest.raises(ValueError, match="no command was configured"):
        grade(task, finished(), repo, None)


def test_injected_setup_is_committed_so_a_correct_fix_shows_up_as_a_diff(repo):
    # The failure this guards: with the injection left uncommitted, restoring
    # the original text produces an empty diff and grades as a failure.
    apply_setup(repo, [{"path": "src/config.py", "find": "KEY = 1", "replace": "KEY = 999"}])
    task = make_task(
        category="C",
        grade={"kind": "diff", "expect_files": ["src/config.py"], "expect_contains": ["KEY = 1"]},
    )
    (repo / "src" / "config.py").write_text("KEY = 1\n", encoding="utf-8")
    assert grade(task, finished(), repo).success is True


def test_expect_absent_ignores_the_removed_line_it_is_naming(repo):
    # The bug this guards, which every category C task tripped: the injected
    # text is what a *correct* fix deletes, so it appears on the diff's `-`
    # line. Matching against the whole diff graded every correct fix a
    # failure, and category C would have read as 0% success in both arms.
    apply_setup(repo, [{"path": "src/config.py", "find": "KEY = 1", "replace": "KEY = 999"}])
    task = make_task(
        category="C",
        grade={
            "kind": "diff",
            "expect_files": ["src/config.py"],
            "expect_contains": ["KEY = 1"],
            "expect_absent": ["KEY = 999"],
        },
    )
    (repo / "src" / "config.py").write_text("KEY = 1\n", encoding="utf-8")
    assert grade(task, finished(), repo).success is True


def test_expect_contains_is_not_satisfied_by_a_removed_line(repo):
    # The mirror image: deleting a line that happens to contain the fragment
    # is not the same as writing it, and must not pass.
    task = make_task(
        grade={"kind": "diff", "expect_files": ["src/config.py"], "expect_contains": ["KEY = 1"]},
    )
    (repo / "src" / "config.py").write_text("OTHER = 2\n", encoding="utf-8")
    assert grade(task, finished(), repo).success is False


def test_setup_refuses_an_ambiguous_or_stale_match(repo):
    with pytest.raises(RuntimeError, match="found 0"):
        apply_setup(repo, [{"path": "src/config.py", "find": "MOVED", "replace": "x"}])
