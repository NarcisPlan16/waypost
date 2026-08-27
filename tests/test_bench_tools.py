from __future__ import annotations

import shutil
import sys

import pytest

from bench.tools import (
    BASELINE_TOOLS,
    MAX_TOOL_OUTPUT_BYTES,
    Executor,
    child_env,
    tools_for_arm,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def build_client():\n    return 1\n\n\ndef main():\n    return build_client()\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    return tmp_path


def test_treatment_adds_exactly_one_tool():
    baseline = tools_for_arm("baseline")
    treatment = tools_for_arm("treatment")

    assert [t["name"] for t in baseline] == [t["name"] for t in BASELINE_TOOLS]
    # One tool, not seven: seven schemas would be a fixed per-turn tax on the
    # treatment arm that has nothing to do with whether the tool helps.
    assert len(treatment) == len(baseline) + 1
    assert treatment[-1]["name"] == "waypost"


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        tools_for_arm("placebo")


def test_read_file_returns_numbered_lines_and_honours_a_range(repo):
    execute = Executor(repo)
    whole = execute("read_file", {"path": "src/app.py"})
    assert whole.content.startswith("1\tdef build_client():")

    ranged = execute("read_file", {"path": "src/app.py", "offset": 5, "limit": 1})
    assert ranged.content == "5\tdef main():"


def test_read_file_refuses_to_escape_the_repository(repo):
    outcome = Executor(repo)("read_file", {"path": "../secrets.txt"})
    assert outcome.is_error
    assert "escapes the repository root" in outcome.content


def test_edit_file_requires_a_unique_match(repo):
    execute = Executor(repo)

    missing = execute("edit_file", {"path": "src/app.py", "old_string": "nope", "new_string": "x"})
    assert missing.is_error

    ambiguous = execute(
        "edit_file", {"path": "src/app.py", "old_string": "build_client", "new_string": "x"}
    )
    assert ambiguous.is_error and "appears 2 times" in ambiguous.content

    applied = execute(
        "edit_file", {"path": "src/app.py", "old_string": "return 1", "new_string": "return 2"}
    )
    assert not applied.is_error
    assert "return 2" in (repo / "src" / "app.py").read_text(encoding="utf-8")


def test_output_is_truncated_at_the_shared_cap(repo):
    (repo / "big.txt").write_text("x" * (MAX_TOOL_OUTPUT_BYTES * 2), encoding="utf-8")
    outcome = Executor(repo)("read_file", {"path": "big.txt"})
    assert "[truncated," in outcome.content
    assert len(outcome.content.encode("utf-8")) < MAX_TOOL_OUTPUT_BYTES * 1.1


def test_unknown_tool_is_an_error_result_not_an_exception(repo):
    outcome = Executor(repo)("teleport", {})
    assert outcome.is_error


def test_calls_are_counted_per_tool(repo):
    execute = Executor(repo)
    execute("read_file", {"path": "README.md"})
    execute("read_file", {"path": "README.md"})
    assert execute.calls["read_file"] == 2


@pytest.mark.skipif(shutil.which("grep") is None, reason="grep is not on PATH")
def test_grep_reports_no_matches_rather_than_failing(repo):
    execute = Executor(repo)
    hit = execute("grep", {"pattern": "build_client"})
    assert not hit.is_error and "src/app.py" in hit.content

    miss = execute("grep", {"pattern": "zzz-not-here"})
    assert not miss.is_error and miss.content == "[no matches]"


def test_credentials_are_stripped_from_tool_subprocesses(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    # Nothing a benchmark task runs should be able to spend money, and a task
    # that finds a key is a task that can cheat.
    assert "ANTHROPIC_API_KEY" not in child_env()


def test_waypost_tool_rejects_a_command_outside_the_allowed_set(repo):
    outcome = Executor(repo)("waypost", {"command": "rm"})
    assert outcome.is_error


def test_waypost_tool_runs_the_real_cli(repo):
    execute = Executor(repo, waypost_cmd=[sys.executable, "-m", "waypost"])
    outcome = execute("waypost", {"command": "find", "args": ["build_client"]})
    assert not outcome.is_error
    assert "build_client" in outcome.content
