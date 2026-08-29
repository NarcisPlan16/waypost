from __future__ import annotations

import sys
import time

import pytest

from bench import tools
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


# No skipif here, deliberately. This test used to be skipped whenever `grep`
# was missing from PATH -- which is exactly the host where the baseline arm's
# only search tool was silently returning an error for every call, and the
# benchmark was crediting the difference to waypost. The tool no longer shells
# out, so there is nothing left to skip on.
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


def test_grep_never_depends_on_an_external_binary(repo, monkeypatch):
    # Emptying PATH would have broken the old shell-out entirely.
    monkeypatch.setenv("PATH", "")
    outcome = Executor(repo)("grep", {"pattern": "build_client"})
    assert not outcome.is_error
    assert "src/app.py" in outcome.content


def test_grep_numbers_lines_and_filters_by_glob_and_path(repo):
    execute = Executor(repo)
    hit = execute("grep", {"pattern": "build_client", "glob": "*.py"})
    assert hit.content.startswith("./src/app.py:1:")

    scoped = execute("grep", {"pattern": "build_client", "path": "src"})
    assert "src/app.py" in scoped.content

    assert execute("grep", {"pattern": "build_client", "glob": "*.md"}).content == "[no matches]"


def test_grep_rejects_a_bad_pattern_instead_of_raising(repo):
    outcome = Executor(repo)("grep", {"pattern": "def ("})
    assert outcome.is_error and "regular expression" in outcome.content


def test_grep_skips_the_generated_index_and_binary_files(repo):
    (repo / ".waypost").mkdir(exist_ok=True)
    (repo / ".waypost" / "index.json").write_text('"name": "build_client"', encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"build_client" + bytes(3))

    outcome = Executor(repo)("grep", {"pattern": "build_client"})
    # The index exists in both arms; a baseline grep matching inside the
    # tool-under-test's own output would be measuring the benchmark.
    assert ".waypost" not in outcome.content
    assert "blob.bin" not in outcome.content


def test_bash_runs_a_posix_shell_when_the_host_has_one(repo):
    execute = Executor(repo)
    if execute.shell is None:  # pragma: no cover - host without any POSIX shell
        pytest.skip("no POSIX shell on this host")
    # cmd.exe would fail on both of these, and only the baseline arm pays for
    # that, so the failure reads as a saving for waypost.
    assert execute("bash", {"command": "echo hi | cat"}).content.strip() == "hi"
    assert not execute("bash", {"command": "ls src"}).is_error


def test_bash_is_given_no_stdin(repo):
    # A command that reads stdin would otherwise inherit the harness's and
    # block forever; nothing in a benchmark task should be interactive.
    execute = Executor(repo)
    if execute.shell is None:  # pragma: no cover - host without any POSIX shell
        pytest.skip("no POSIX shell on this host")
    assert not execute("bash", {"command": "cat"}).is_error


def test_bash_timeout_kills_grandchildren_not_just_the_shell(repo, monkeypatch):
    # `Popen.kill` only kills the shell. A surviving grandchild keeps the
    # stdout pipe open, so the drain never returns and the timeout is fiction:
    # a real `find /` ran on past its 120s cap for twelve minutes.
    execute = Executor(repo)
    if execute.shell is None:  # pragma: no cover - host without any POSIX shell
        pytest.skip("no POSIX shell on this host")
    monkeypatch.setattr(tools, "TOOL_TIMEOUT_S", 2)

    started = time.monotonic()
    outcome = execute("bash", {"command": "sleep 60 | cat"})
    elapsed = time.monotonic() - started

    assert outcome.is_error and "timed out" in outcome.content
    # Generous, but far below the 60s the grandchild wanted: if the tree
    # survived, the drain would hold this open until KILL_DRAIN_S expired.
    assert elapsed < 2 + tools.KILL_DRAIN_S
