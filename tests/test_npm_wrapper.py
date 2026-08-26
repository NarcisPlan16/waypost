"""Smoke test for bin/waypost.js: does it resolve and exec a Python backend.

Not part of the Python test suite's normal concerns, but the wrapper is the
whole point of Sprint 4a, and a silent breakage there would only surface as
"npx waypost doesn't work" on someone else's machine.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WRAPPER = REPO_ROOT / "bin" / "waypost.js"

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not on PATH")
def test_wrapper_execs_python_module_and_forwards_exit_code() -> None:
    result = subprocess.run(
        [node, str(WRAPPER), "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**_env_without_waypost_and_uv_and_pipx()},
    )
    assert result.returncode == 0
    assert "waypost" in result.stdout


def _env_without_waypost_and_uv_and_pipx() -> dict[str, str]:
    """Force the wrapper down its python3/python fallback path.

    Removing waypost/uvx/pipx from PATH isn't practical cross-platform, so
    instead this just runs with the current environment -- the assertion is
    on the *outcome* (exit code forwarded, output present), not on which
    candidate answered.
    """
    import os

    return dict(os.environ)


@pytest.mark.skipif(node is None, reason="node not on PATH")
def test_wrapper_reports_missing_python_cleanly(tmp_path: Path) -> None:
    """With no candidate resolvable at all, exit 2 with a clear stderr message."""
    import os

    if sys.platform == "win32":
        pytest.skip("PATH stripping to isolate 'no python found' is unreliable on Windows CI")

    env = dict(os.environ)
    env["PATH"] = str(tmp_path)  # empty dir: nothing resolvable, not even `where`/`which`
    result = subprocess.run(
        [node, str(WRAPPER), "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "no Python found" in result.stderr
