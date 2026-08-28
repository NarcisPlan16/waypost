"""Graders.

Every category now grades automatically and offline. Category D --
trace/explain -- was the hold-out: judging prose properly wants a second
model, blind to the arm, and until that exists a ``judge`` stub returned
``None`` so a verbose arm could never be scored as a correct one. But an
ungraded category is not free. It is a fifth of the matrix producing API
spend and no verdict, and the paid run is the scarce resource here, so D is
graded by *anchors* instead: a set of groups, each listing alternative
phrasings, all of which the answer must hit.

That is a weaker grader than a judge and the weakness has a direction worth
stating. An anchor rubric rewards naming things, and the waypost arm is
handed symbol names by the index, so it can in principle satisfy an anchor it
did not understand. Two things keep that honest: the anchors are chosen to
name behaviour that only appears inside a function body rather than in any
signature the map prints, and category E exists precisely to expose a harness
that flatters the tool. Read a D result as "did the run find and describe the
right machinery", never as "was the explanation good".
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.loop import LoopResult
from bench.tasks import Task
from bench.tools import TOOL_TIMEOUT_S, child_env

GRADE_TIMEOUT_S = 600


@dataclass(frozen=True)
class Verdict:
    """The outcome of grading one run.

    ``success`` is ``None`` when the task is not automatically gradeable; that
    is distinct from ``False`` and must stay distinct all the way into the
    report.
    """

    success: bool | None
    detail: str

    def as_record(self) -> dict[str, Any]:
        return {"success": self.success, "grade_detail": self.detail}


def _normalise(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def grade(
    task: Task, result: LoopResult, worktree: Path, test_command: str | None = None
) -> Verdict:
    """Grade one completed run.

    A run that never finished is a failure regardless of what it said: a
    turn-capped run that happened to mention the right file on turn 40 did not
    do the task.
    """
    if not result.completed:
        return Verdict(False, f"run did not complete: {result.failure_reason}")

    kind = task.grade_kind
    if kind == "localization":
        return _grade_localization(task, result)
    if kind == "diff":
        return _grade_diff(task, worktree)
    if kind == "tests":
        return _grade_tests(task, worktree, test_command)
    if kind == "rubric":
        return _grade_rubric(task, result)
    raise ValueError(f"unknown grade kind {kind!r}")


def _grade_localization(task: Task, result: LoopResult) -> Verdict:
    """Did the final answer name the expected file(s)?

    Matched against the whole final message rather than only the ``FILES:``
    block. The prompt asks for that block, but grading an answer wrong because
    the model formatted it differently would measure formatting compliance,
    not localization.
    """
    expected = [_normalise(path) for path in task.grade["expect_files"]]
    # Only the separator is normalised: lstrip-ing the whole message the way a
    # single path is normalised would corrupt it.
    haystack = result.final_text.replace("\\", "/")
    found = [path for path in expected if path in haystack]
    missing = sorted(set(expected) - set(found))
    if missing:
        return Verdict(False, f"final answer did not name {missing}")
    return Verdict(True, f"named all of {expected}")


def _grade_rubric(task: Task, result: LoopResult) -> Verdict:
    """Did the final answer hit every anchor group?

    Matching is case-insensitive and whitespace-collapsed so that a line wrap
    between two words of an anchor does not fail a correct answer; it is
    otherwise literal. ``expect_files`` is checked with the same
    path-separator normalisation the other graders use.
    """
    haystack = _collapse(result.final_text)

    expected_files = [_normalise(path) for path in task.grade.get("expect_files", [])]
    missing_files = [path for path in expected_files if path.lower() not in haystack]
    if missing_files:
        return Verdict(False, f"final answer did not name {sorted(missing_files)}")

    missed = [
        group
        for group in task.grade["expect_all"]
        if not any(_collapse(alt) in haystack for alt in group)
    ]
    if missed:
        # The first alternative is the canonical phrasing; reporting the whole
        # group would make the failure line unreadable for a wide rubric.
        return Verdict(False, f"final answer missed {[group[0] for group in missed]}")
    return Verdict(True, f"hit all {len(task.grade['expect_all'])} anchor group(s)")


def _collapse(text: str) -> str:
    """Lowercase, normalise path separators, and collapse runs of whitespace."""
    return " ".join(text.replace("\\", "/").lower().split())


def _git_diff(worktree: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "diff", *args],
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_S,
        env=child_env(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git diff failed in {worktree}: {completed.stderr.strip()}")
    return completed.stdout


def _added_lines(diff_text: str) -> str:
    """Only the lines the run *added*, with their ``+`` markers stripped.

    Matching fragments against the whole diff looks equivalent and is not.
    A category C task injects a bug and asks for it to be removed, so the
    buggy text is exactly what appears on the ``-`` lines of a **correct**
    fix -- and an ``expect_absent`` fragment naming it would then fail every
    correct run. The question both assertions are really asking is what the
    fixed code says, and that is the added lines.
    """
    return "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _grade_diff(task: Task, worktree: Path) -> Verdict:
    changed = {_normalise(line) for line in _git_diff(worktree, "--name-only").splitlines() if line}
    expected = {_normalise(path) for path in task.grade["expect_files"]}
    missing = sorted(expected - changed)
    if missing:
        return Verdict(False, f"expected edits to {missing}; changed {sorted(changed)}")

    added = _added_lines(_git_diff(worktree))
    for fragment in task.grade.get("expect_contains", []):
        if fragment not in added:
            return Verdict(False, f"no added line contains {fragment!r}")
    for fragment in task.grade.get("expect_absent", []):
        if fragment in added:
            return Verdict(False, f"an added line still contains {fragment!r}")
    return Verdict(True, f"edited {sorted(expected)} as expected")


def _grade_tests(task: Task, worktree: Path, test_command: str | None) -> Verdict:
    """Run the repository's test suite in the worktree; exit 0 is success."""
    command = task.grade.get("command") or test_command
    if not command:
        raise ValueError(f"task {task.id} uses the tests grader but no command was configured")

    completed = subprocess.run(
        command,
        shell=True,
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=GRADE_TIMEOUT_S,
        env=child_env(),
    )
    if completed.returncode == 0:
        return Verdict(True, "test command exited 0")
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-20:]
    return Verdict(
        False, "test command exited {}: {}".format(completed.returncode, " / ".join(tail))
    )
