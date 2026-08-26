"""Tests for the npm wrapper and the ``python -m waypost`` entry point it leans on.

Not part of the Python test suite's normal concerns, but the wrapper is the
whole point of Sprint 4a, and a silent breakage there would only surface as
"npx waypost doesn't work" on someone else's machine.

CI runs these under ``uv run``, which puts the real ``waypost`` console
script on PATH -- so the happy path is always candidate #1 there and the
fallbacks never get exercised. The fall-through test below therefore builds
its own PATH from scratch instead of trusting the ambient one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import waypost

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = REPO_ROOT / "bin" / "waypost.js"
PACKAGE_JSON = REPO_ROOT / "package.json"

node = shutil.which("node")
requires_node = pytest.mark.skipif(node is None, reason="node not on PATH")


def _run_wrapper(*args: str, path: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if path is not None:
        env["PATH"] = path
    assert node is not None
    return subprocess.run(
        [node, str(WRAPPER), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def _python_only_path(tmp_path: Path) -> str:
    """A PATH holding exactly one launcher: `python`, onto this interpreter.

    Deterministic on purpose. The ambient PATH has `waypost` on it under
    ``uv run`` and not under a bare ``python -m pytest``, so a test that
    trusts it asserts something different on every machine.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if sys.platform == "win32":
        _shim(bin_dir, "python", f'"{sys.executable}" %*')
    else:
        _shim(bin_dir, "python", f'exec "{sys.executable}" "$@"')
    return str(bin_dir)


def _shim(directory: Path, name: str, body: str) -> None:
    """Write a launcher named ``name`` that a PATH lookup will find.

    On Windows this is a ``.cmd`` file, which doubles as coverage for the
    wrapper's COMSPEC path: Node cannot spawn ``.cmd``/``.bat`` directly.
    """
    if sys.platform == "win32":
        (directory / f"{name}.cmd").write_text(f"@echo off\r\n{body}\r\n", encoding="utf-8")
    else:
        script = directory / name
        script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        script.chmod(0o755)


def test_module_entry_point_runs() -> None:
    """``python -m waypost`` must work: it is the wrapper's last-resort backend."""
    result = subprocess.run(
        [sys.executable, "-m", "waypost", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert waypost.__version__ in result.stdout


@requires_node
def test_wrapper_execs_waypost_and_forwards_exit_codes(tmp_path: Path) -> None:
    """Whichever candidate answers, argv and exit code pass straight through."""
    path = _python_only_path(tmp_path)

    version = _run_wrapper("--version", path=path)
    assert version.returncode == 0, version.stderr
    assert waypost.__version__ in version.stdout

    # `find` with no hits is exit 1 and not an error; the wrapper must not
    # flatten that into 0, nor into its own exit 2.
    missing = _run_wrapper(
        "find", "zzz_definitely_no_such_symbol", "--root", str(REPO_ROOT), path=path
    )
    assert missing.returncode == 1, missing.stderr

    # A genuine CLI failure is exit 2, and must not be confused with the
    # wrapper's own "no backend" exit 2 -- the message tells them apart.
    bad_root = _run_wrapper("stats", "--root", str(tmp_path / "nope"), path=path)
    assert bad_root.returncode == 2
    assert "is not a directory" in bad_root.stderr


@requires_node
def test_wrapper_execs_a_waypost_on_path(tmp_path: Path) -> None:
    """Candidate #1: an installed console script is used directly."""
    installed = shutil.which("waypost", path=str(Path(sys.executable).parent))
    if installed is None:
        pytest.skip("no waypost console script alongside this interpreter")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shutil.copy2(installed, bin_dir / Path(installed).name)

    result = _run_wrapper("--version", path=str(bin_dir))
    assert result.returncode == 0, result.stderr
    assert waypost.__version__ in result.stdout


@requires_node
def test_wrapper_skips_a_launcher_that_cannot_run_waypost(tmp_path: Path) -> None:
    """A resolvable-but-broken candidate must not swallow the invocation.

    This is the Microsoft Store `python3` alias in miniature: a file that a
    PATH lookup finds happily and that then exits nonzero without running
    Python at all. The wrapper has to notice and move on to `python`, which
    here is a shim onto the interpreter running these tests.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _shim(bin_dir, "python3", "exit 9009")
    if sys.platform == "win32":
        _shim(bin_dir, "python", f'"{sys.executable}" %*')
    else:
        _shim(bin_dir, "python", f'exec "{sys.executable}" "$@"')

    result = _run_wrapper("--version", path=str(bin_dir))
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert waypost.__version__ in result.stdout


@requires_node
def test_wrapper_reports_no_launcher_cleanly(tmp_path: Path) -> None:
    """With nothing resolvable on PATH at all: exit 2 and a clear stderr message."""
    result = _run_wrapper("--version", path=str(tmp_path))
    assert result.returncode == 2
    assert "no Python launcher found" in result.stderr


@requires_node
def test_wrapper_reports_a_dead_launcher_cleanly(tmp_path: Path) -> None:
    """Launchers present but none of them able to run waypost: exit 2, and say which."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _shim(bin_dir, "python3", "exit 9009")

    result = _run_wrapper("--version", path=str(bin_dir))
    assert result.returncode == 2
    assert "none of them could run waypost" in result.stderr
    assert "python3 -m waypost" in result.stderr


def test_package_json_bin_points_at_the_wrapper() -> None:
    manifest = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    assert (REPO_ROOT / manifest["bin"]["waypost"]).is_file()
    for shipped in manifest["files"]:
        assert (REPO_ROOT / shipped).exists(), f"package.json ships a missing file: {shipped}"


def test_npm_version_tracks_the_python_version() -> None:
    """package.json is a second version string; keep it from drifting.

    ``pyproject.toml`` single-sources the version from
    ``src/waypost/__init__.py``, but npm cannot read that and will not accept
    a PEP 440 spelling. So the manifest carries the semver equivalent
    (``0.1.0.dev0`` -> ``0.1.0-dev.0``) and this asserts they still agree.
    """
    manifest = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    base, sep, prerelease = manifest["version"].partition("-")
    expected = base if not sep else f"{base}.{prerelease.replace('.', '')}"
    assert expected == waypost.__version__
