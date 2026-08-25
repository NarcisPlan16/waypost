import subprocess
import sys

from tacit import __version__
from tacit.cli import build_parser, main


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
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tacit" in out


def test_planned_command_reports_not_implemented(capsys):
    exit_code = main(["index"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not implemented" in err


def test_parser_builds_without_error():
    parser = build_parser()
    assert parser.prog == "tacit"


def test_cli_entrypoint_via_subprocess():
    # Exercises the console_script installed by pyproject.toml, i.e. the
    # actual `tacit --version` invocation named in the Sprint 0 exit
    # condition, not just the importable main().
    result = subprocess.run(
        [sys.executable, "-m", "tacit.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
