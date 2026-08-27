"""Repository checkouts: one clean, pinned worktree per run.

Runs must not see each other's edits, so every run gets a detached worktree at
the pinned SHA and it is removed afterwards. The clone itself is cached, so
the network is touched once per repository, never per run.

The index is built here, before the run starts, for **both** arms. Two reasons
it must be both: the timing must not differ between arms for a reason that is
not the tool, and an index built only in the treatment arm would also be an
index built at a different point in each worktree's life.

Config is JSON rather than TOML on purpose -- ``tomllib`` is 3.11+, and this
package has to import on 3.10 with only the dev extra installed, which is what
CI does.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bench.tools import child_env, default_waypost_cmd

CLONE_TIMEOUT_S = 900
GIT_TIMEOUT_S = 300
INDEX_TIMEOUT_S = 600


@dataclass(frozen=True)
class RepoSpec:
    """One benchmark repository, pinned so a reader gets the identical tree."""

    name: str
    url: str
    sha: str
    test_command: str | None = None

    @classmethod
    def from_dict(cls, name: str, data: dict[str, object]) -> RepoSpec:
        for key in ("url", "sha"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"repo {name!r} needs a non-empty {key!r}")
        test_command = data.get("test_command")
        return cls(
            name=name,
            url=str(data["url"]),
            sha=str(data["sha"]),
            test_command=str(test_command) if isinstance(test_command, str) else None,
        )


def default_repos_path() -> Path:
    return Path(__file__).resolve().parent / "repos.json"


def load_repos(path: Path | None = None) -> dict[str, RepoSpec]:
    """Load the pinned repository table."""
    path = path or default_repos_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object of repo name -> spec")
    return {name: RepoSpec.from_dict(name, spec) for name, spec in data.items()}


def _git(args: list[str], cwd: Path | None = None, timeout: int = GIT_TIMEOUT_S) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(),
    )
    if completed.returncode != 0:
        where = f" in {cwd}" if cwd else ""
        raise RuntimeError(f"git {' '.join(args)} failed{where}: {completed.stderr.strip()}")
    return completed.stdout


def ensure_clone(spec: RepoSpec, cache_dir: Path) -> Path:
    """Clone the repository once into the cache, and make sure the SHA is there."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    clone = cache_dir / spec.name
    if not (clone / ".git").exists():
        _git(["clone", spec.url, str(clone)], timeout=CLONE_TIMEOUT_S)

    try:
        _git(["cat-file", "-e", f"{spec.sha}^{{commit}}"], cwd=clone)
    except RuntimeError:
        # The pin may be newer than the cached clone.
        _git(["fetch", "--all", "--tags"], cwd=clone, timeout=CLONE_TIMEOUT_S)
        _git(["cat-file", "-e", f"{spec.sha}^{{commit}}"], cwd=clone)
    return clone


def apply_setup(root: Path, edits: Sequence[Mapping[str, str]]) -> None:
    """Apply a task's setup edits, then commit them.

    Category C needs a bug to fix, and the pinned trees do not contain one, so
    a task may inject it. The edits are **committed** before the run starts,
    for a reason that is easy to get wrong: the diff grader compares against
    ``HEAD``, so an uncommitted injection would make a correct fix -- which
    restores the original text -- show up as an empty diff and be graded a
    failure.
    """
    for edit in edits:
        target = (root / edit["path"]).resolve()
        if root.resolve() not in target.parents:
            raise ValueError(f"setup path {edit['path']!r} escapes the worktree")
        text = target.read_text(encoding="utf-8")
        occurrences = text.count(edit["find"])
        if occurrences != 1:
            raise RuntimeError(
                f"setup for {edit['path']} expected exactly one match for "
                f"{edit['find']!r}, found {occurrences}; the pin has moved"
            )
        target.write_text(text.replace(edit["find"], edit["replace"], 1), encoding="utf-8")

    if edits:
        _git(["add", "-A"], cwd=root)
        _git(
            [
                "-c",
                "user.name=bench",
                "-c",
                "user.email=bench@localhost",
                "commit",
                "-q",
                "-m",
                "bench: inject task setup",
            ],
            cwd=root,
        )


def build_index(root: Path, waypost_cmd: list[str] | None = None) -> None:
    """Build the waypost index in *root*, before the run is timed."""
    argv = [*(waypost_cmd or default_waypost_cmd()), "index"]
    completed = subprocess.run(
        argv,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=INDEX_TIMEOUT_S,
        env=child_env(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"waypost index failed in {root}: {completed.stderr.strip()}")


@contextlib.contextmanager
def run_worktree(
    spec: RepoSpec,
    cache_dir: Path,
    work_dir: Path,
    setup: Sequence[Mapping[str, str]] = (),
    waypost_cmd: list[str] | None = None,
) -> Iterator[Path]:
    """Yield a fresh worktree at the pinned SHA, indexed and ready.

    Removed on the way out, including after a failure: a leaked worktree makes
    the next run's ``git diff`` grading meaningless.
    """
    clone = ensure_clone(spec, cache_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{spec.name}-{spec.sha[:8]}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    _git(["worktree", "add", "--detach", str(target), spec.sha], cwd=clone)
    try:
        # Order matters: the index must describe the tree the model will see,
        # injected bug included.
        apply_setup(target, setup)
        build_index(target, waypost_cmd)
        yield target
    finally:
        with contextlib.suppress(RuntimeError):
            _git(["worktree", "remove", "--force", str(target)], cwd=clone)
        shutil.rmtree(target, ignore_errors=True)
