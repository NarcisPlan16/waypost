from __future__ import annotations

import json
import subprocess
import sys

import pytest

from waypost import __version__
from waypost.cli import EXIT_ERROR, EXIT_NOT_FOUND, EXIT_OK, build_parser, main
from waypost.index import default_index_path
from waypost.tokens import count

CLIENT_PY = '''
"""Transport layer."""

DEFAULT_TIMEOUT = 30


class Client:
    """HTTP client with retry and pooling."""

    def request(self, method, url):
        """Send one request."""
        return (method, url)


def build_client():
    """Construct a Client with sensible defaults."""
    return Client()
'''

APP_PY = """
from .client import build_client


def main():
    return build_client().request("GET", "/")
"""


@pytest.fixture
def repo(tmp_path):
    pkg = tmp_path / "src" / "app"
    pkg.mkdir(parents=True)
    (pkg / "client.py").write_text(CLIENT_PY, encoding="utf-8")
    (pkg / "app.py").write_text(APP_PY, encoding="utf-8")
    return tmp_path


@pytest.fixture
def indexed(repo, capsys):
    assert main(["index", "--root", str(repo)]) == EXIT_OK
    capsys.readouterr()
    return repo


def run(capsys, *argv):
    """Invoke the CLI and return (exit code, stdout, stderr)."""
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------
# Sprint 0 surface, still true
# --------------------------------------------------------------------------


def test_version_flag_exits_zero_and_prints_version(capsys):
    exit_code = None
    try:
        main(["--version"])
    except SystemExit as e:
        exit_code = e.code
    assert exit_code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_no_args_prints_help_and_exits_zero(capsys):
    exit_code = main([])
    assert exit_code == EXIT_OK
    out = capsys.readouterr().out
    assert "waypost" in out


def test_parser_builds_without_error():
    parser = build_parser()
    assert parser.prog == "waypost"


def test_cli_entrypoint_via_subprocess():
    # Exercises the console_script installed by pyproject.toml, i.e. the
    # actual `waypost --version` invocation named in the Sprint 0 exit
    # condition, not just the importable main().
    result = subprocess.run(
        [sys.executable, "-m", "waypost.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


def test_index_writes_the_index_and_reports_what_it_found(repo, capsys):
    code, out, _err = run(capsys, "index", "--root", str(repo))
    assert code == EXIT_OK
    assert default_index_path(repo).is_file()
    assert "2 files" in out
    assert "rank=simple" in out


def test_index_refreshes_in_place_on_a_second_run(indexed, capsys):
    code, out, _err = run(capsys, "index", "--root", str(indexed))
    assert code == EXIT_OK
    assert "refreshed" in out


def test_index_force_rebuilds(indexed, capsys):
    code, out, _err = run(capsys, "index", "--root", str(indexed), "--force")
    assert code == EXIT_OK
    assert "rebuilt" in out


def test_index_accepts_pagerank_with_focus(repo, capsys):
    code, out, _err = run(
        capsys, "index", "--root", str(repo), "--rank", "pagerank", "--focus", "src/app/app.py"
    )
    assert code == EXIT_OK
    assert "rank=pagerank" in out


def test_index_rejects_an_unknown_rank_strategy(repo):
    with pytest.raises(SystemExit) as excinfo:
        main(["index", "--root", str(repo), "--rank", "nonsense"])
    assert excinfo.value.code == 2


def test_index_reports_a_missing_root(tmp_path, capsys):
    code, _out, err = run(capsys, "index", "--root", str(tmp_path / "nope"))
    assert code == EXIT_ERROR
    assert "not a directory" in err


@pytest.mark.parametrize("command", ["map", "find", "show", "refs", "outline", "stats"])
def test_query_commands_report_a_missing_root_instead_of_inventing_one(tmp_path, capsys, command):
    """A mistyped `--root` is an error, not an empty repository.

    Only `index` checked, so `waypost map --root typo` walked nothing,
    reported success, and left a stray `.waypost/index.json` inside a
    directory it had just created.
    """
    missing = tmp_path / "nope"
    argv = [command, "--root", str(missing)]
    if command in {"find", "show", "refs", "outline"}:
        argv.append("whatever")

    code, _out, err = run(capsys, *argv)

    assert code == EXIT_ERROR
    assert "not a directory" in err
    assert not missing.exists()


def test_index_remembers_its_rank_strategy_across_a_plain_reindex(repo, capsys):
    run(capsys, "index", "--root", str(repo), "--rank", "pagerank", "--focus", "src/app")

    code, out, _err = run(capsys, "index", "--root", str(repo))

    assert code == EXIT_OK
    assert "rank=pagerank" in out
    assert "focus=src/app" in out


def test_index_rank_can_still_be_changed_explicitly(repo, capsys):
    run(capsys, "index", "--root", str(repo), "--rank", "pagerank")

    code, out, _err = run(capsys, "index", "--root", str(repo), "--rank", "simple")

    assert code == EXIT_OK
    assert "rank=simple" in out


def test_refreshing_a_query_does_not_re_rank_the_repository(repo, capsys):
    """`map --refresh` updates files; it does not redefine what "important" means.

    Refresh used to fall back to the default strategy, so one `--refresh`
    silently replaced a pagerank index with a simple-ranked one.
    """
    run(capsys, "index", "--root", str(repo), "--rank", "pagerank", "--focus", "src/app")
    before = json.loads(default_index_path(repo).read_text(encoding="utf-8"))

    run(capsys, "map", "--root", str(repo), "--refresh")
    after = json.loads(default_index_path(repo).read_text(encoding="utf-8"))

    assert after["rank_strategy"] == "pagerank"
    assert after["focus"] == ["src/app"]
    assert {p: e["rank"] for p, e in after["files"].items()} == {
        p: e["rank"] for p, e in before["files"].items()
    }


# --------------------------------------------------------------------------
# query commands
# --------------------------------------------------------------------------


def test_map_renders_ranked_symbols(indexed, capsys):
    code, out, _err = run(capsys, "map", "--root", str(indexed))
    assert code == EXIT_OK
    assert "class Client" in out
    assert "def build_client" in out


@pytest.mark.parametrize("budget", ["10", "50", "200"])
def test_map_honours_the_budget_it_was_given(indexed, capsys, budget):
    _code, out, _err = run(capsys, "map", "--root", str(indexed), "--budget", budget)
    assert count(out.rstrip("\n")) <= int(budget)


def test_measure_reports_the_count_on_stderr_not_stdout(indexed, capsys):
    _code, out, err = run(capsys, "map", "--root", str(indexed), "--measure")
    assert "tokens, tokenizer" in err
    assert "tokens, tokenizer" not in out


def test_a_missing_index_is_built_once_with_a_note_on_stderr(repo, capsys):
    code, out, err = run(capsys, "map", "--root", str(repo))
    assert code == EXIT_OK
    assert "building one" in err
    assert "class Client" in out
    assert default_index_path(repo).is_file()

    # And the note does not repeat now that the index exists.
    _code, _out, err = run(capsys, "map", "--root", str(repo))
    assert "building one" not in err


def test_query_commands_do_not_reindex_unless_asked(indexed, capsys):
    (indexed / "src" / "app" / "late.py").write_text("def late():\n    pass\n", encoding="utf-8")

    code, out, _err = run(capsys, "find", "--root", str(indexed), "late")
    assert code == EXIT_NOT_FOUND
    assert "def late" not in out

    _code, out, _err = run(capsys, "find", "--root", str(indexed), "late", "--refresh")
    assert "def late" in out


def test_find_exits_one_when_nothing_matches(indexed, capsys):
    code, out, _err = run(capsys, "find", "--root", str(indexed), "no_such_thing")
    assert code == EXIT_NOT_FOUND
    assert "no symbol matching" in out


def test_find_exits_zero_on_a_hit(indexed, capsys):
    code, out, _err = run(capsys, "find", "--root", str(indexed), "build_client")
    assert code == EXIT_OK
    assert "src/app/client.py:" in out


def test_show_prints_only_the_symbols_span(indexed, capsys):
    code, out, _err = run(capsys, "show", "--root", str(indexed), "build_client")
    assert code == EXIT_OK
    assert "def build_client" in out
    assert "class Client" not in out
    assert "DEFAULT_TIMEOUT" not in out


def test_show_exits_one_for_an_unknown_symbol(indexed, capsys):
    code, out, _err = run(capsys, "show", "--root", str(indexed), "not_a_symbol")
    assert code == EXIT_NOT_FOUND
    assert "no symbol named" in out


def test_refs_traces_a_symbol_across_files(indexed, capsys):
    code, out, _err = run(capsys, "refs", "--root", str(indexed), "build_client")
    assert code == EXIT_OK
    assert "src/app/client.py:" in out
    assert "src/app/app.py:" in out


def test_refs_exits_one_when_the_name_is_unknown(indexed, capsys):
    code, _out, _err = run(capsys, "refs", "--root", str(indexed), "nothing_here")
    assert code == EXIT_NOT_FOUND


def test_outline_accepts_a_suffix_and_lists_one_file(indexed, capsys):
    code, out, _err = run(capsys, "outline", "--root", str(indexed), "client.py")
    assert code == EXIT_OK
    assert "class Client" in out
    assert "app.py" not in out


def test_outline_exits_one_for_an_unknown_path(indexed, capsys):
    code, out, _err = run(capsys, "outline", "--root", str(indexed), "src/app/ghost.py")
    assert code == EXIT_NOT_FOUND
    assert "not in the index" in out


def test_stats_reports_the_index_on_disk(indexed, capsys):
    code, out, _err = run(capsys, "stats", "--root", str(indexed))
    assert code == EXIT_OK
    assert "2 files" in out
    assert "KiB" in out


# --------------------------------------------------------------------------
# --json
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "check"),
    [
        (["map"], lambda d: d["files"][0]["symbols"]),
        (["find", "build_client"], lambda d: d["hits"][0]["path"] == "src/app/client.py"),
        (["show", "build_client"], lambda d: "def build_client" in d["definitions"][0]["source"]),
        (["refs", "build_client"], lambda d: d["references"]),
        (["outline", "client.py"], lambda d: d["found"] is True),
        (["stats"], lambda d: d["files"] == 2),
    ],
)
def test_json_output_is_valid_json_with_the_expected_shape(indexed, capsys, argv, check):
    _code, out, _err = run(capsys, *argv, "--root", str(indexed), "--json")
    data = json.loads(out)
    assert check(data)


def test_json_and_text_agree_on_the_exit_code(indexed, capsys):
    text_code, _out, _err = run(capsys, "find", "--root", str(indexed), "absent_symbol")
    json_code, _out, _err = run(capsys, "find", "--root", str(indexed), "absent_symbol", "--json")
    assert text_code == json_code == EXIT_NOT_FOUND
