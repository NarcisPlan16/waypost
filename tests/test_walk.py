import logging
import os
from pathlib import Path

import pytest

from waypost.walk import HARD_SKIP_DIRS, iter_files


def _touch(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _names(root: Path, **kwargs) -> set[str]:
    return {p.as_posix() for p in iter_files(root, **kwargs)}


def test_yields_plain_files_relative_to_root(tmp_path):
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "sub" / "b.py")

    result = _names(tmp_path)

    assert result == {"a.py", "sub/b.py"}


def test_deterministic_order(tmp_path):
    _touch(tmp_path / "z.py")
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "m" / "b.py")

    first = list(iter_files(tmp_path))
    second = list(iter_files(tmp_path))

    assert first == second
    assert first == sorted(first)


def test_honours_root_gitignore(tmp_path):
    _touch(tmp_path / "keep.py")
    _touch(tmp_path / "skip.py")
    _touch(tmp_path / ".gitignore", "skip.py\n")

    result = _names(tmp_path)

    assert result == {"keep.py", ".gitignore"}


def test_gitignore_pattern_matches_at_any_depth(tmp_path):
    _touch(tmp_path / ".gitignore", "*.log\n")
    _touch(tmp_path / "top.log")
    _touch(tmp_path / "a" / "b" / "deep.log")
    _touch(tmp_path / "a" / "b" / "keep.py")

    result = _names(tmp_path)

    assert "top.log" not in result
    assert "a/b/deep.log" not in result
    assert "a/b/keep.py" in result


def test_within_file_negation_overrides_ignore(tmp_path):
    _touch(tmp_path / ".gitignore", "*.log\n!important.log\n")
    _touch(tmp_path / "noisy.log")
    _touch(tmp_path / "important.log")

    result = _names(tmp_path)

    assert "important.log" in result
    assert "noisy.log" not in result


def test_nested_gitignore_only_applies_to_its_own_subtree(tmp_path):
    _touch(tmp_path / "sub" / ".gitignore", "local.txt\n")
    _touch(tmp_path / "sub" / "local.txt")
    _touch(tmp_path / "local.txt")  # same name, outside sub/ -- must survive

    result = _names(tmp_path)

    assert "sub/local.txt" not in result
    assert "local.txt" in result


def test_gitignored_directory_is_pruned_entirely(tmp_path):
    _touch(tmp_path / ".gitignore", "ignored_dir/\n")
    _touch(tmp_path / "ignored_dir" / "inside.py")
    _touch(tmp_path / "kept.py")

    result = _names(tmp_path)

    assert result == {"kept.py", ".gitignore"}


@pytest.mark.parametrize("dirname", sorted(HARD_SKIP_DIRS))
def test_hard_skip_dirs_are_pruned_without_any_gitignore(tmp_path, dirname):
    _touch(tmp_path / dirname / "file.py")
    _touch(tmp_path / "kept.py")

    result = _names(tmp_path)

    assert result == {"kept.py"}


@pytest.mark.parametrize(
    "filename",
    [
        "bundle.min.js",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Cargo.lock",
        "go.sum",
        "random.lock",
    ],
)
def test_lockfile_and_minified_names_are_hard_skipped(tmp_path, filename):
    _touch(tmp_path / filename)
    _touch(tmp_path / "kept.py")

    result = _names(tmp_path)

    assert result == {"kept.py"}


def test_skips_binary_files(tmp_path):
    (tmp_path / "binary.dat").write_bytes(b"abc\x00def")
    _touch(tmp_path / "kept.py")

    result = _names(tmp_path)

    assert result == {"kept.py"}


def test_skips_files_over_size_cutoff(tmp_path):
    _touch(tmp_path / "big.py", "x" * 200)
    _touch(tmp_path / "small.py", "x" * 10)

    result = _names(tmp_path, max_file_bytes=100)

    assert result == {"small.py"}


def test_stops_at_max_files_and_warns(tmp_path, caplog):
    for i in range(10):
        _touch(tmp_path / f"f{i}.py")

    with caplog.at_level(logging.WARNING, logger="waypost.walk"):
        result = list(iter_files(tmp_path, max_files=3))

    assert len(result) == 3
    assert any("limit" in record.message for record in caplog.records)


def test_no_warning_when_under_the_limit(tmp_path, caplog):
    _touch(tmp_path / "a.py")

    with caplog.at_level(logging.WARNING, logger="waypost.walk"):
        list(iter_files(tmp_path, max_files=100))

    assert caplog.records == []


def test_skips_symlinked_files(tmp_path):
    _touch(tmp_path / "real.py")
    target = tmp_path / "real.py"
    link = tmp_path / "link.py"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported in this environment")

    result = _names(tmp_path)

    assert result == {"real.py"}


def test_empty_repo_yields_nothing(tmp_path):
    assert _names(tmp_path) == set()
