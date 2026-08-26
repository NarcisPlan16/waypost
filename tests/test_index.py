from __future__ import annotations

import json
import logging
from pathlib import Path

from waypost.index import (
    SCHEMA_VERSION,
    build,
    default_index_path,
    load,
    refresh,
    save,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_indexes_supported_files_and_skips_unsupported(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    _write(tmp_path / "README.md", "not source\n")

    index = build(tmp_path)

    assert set(index.files) == {"a.py"}
    assert index.schema == SCHEMA_VERSION
    assert index.files["a.py"].parsed.defs[0].name == "foo"
    assert len(index.files["a.py"].sha256) == 64


def test_save_and_load_round_trip(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    index = build(tmp_path)

    dest = save(index, default_index_path(tmp_path))
    assert dest == default_index_path(tmp_path)

    loaded = load(dest)
    assert loaded is not None
    assert loaded.schema == index.schema
    assert set(loaded.files) == set(index.files)
    assert loaded.files["a.py"].parsed.defs[0].name == "foo"
    assert loaded.files["a.py"].sha256 == index.files["a.py"].sha256


def test_load_missing_file_returns_none(tmp_path):
    assert load(tmp_path / "nope" / "index.json") is None


def test_load_corrupt_json_returns_none(tmp_path, caplog):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert load(path) is None
    assert "corrupt" in caplog.text


def test_load_wrong_schema_returns_none(tmp_path, caplog):
    path = tmp_path / "index.json"
    payload = {"schema": SCHEMA_VERSION + 1, "root": ".", "files": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.INFO):
        assert load(path) is None


def test_refresh_reuses_unchanged_files_and_reparses_changed_ones(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    _write(a, "def foo():\n    pass\n")
    _write(b, "def bar():\n    pass\n")

    first = build(tmp_path)

    _write(b, "def bar():\n    pass\n\ndef baz():\n    pass\n")

    second = refresh(tmp_path, first)

    # unchanged file: same parsed object reused (not reparsed)
    assert second.files["a.py"].parsed is first.files["a.py"].parsed
    # changed file: reparsed, picks up the new symbol
    assert second.files["a.py"].sha256 == first.files["a.py"].sha256
    assert {s.name for s in second.files["b.py"].parsed.defs} == {"bar", "baz"}
    assert second.files["b.py"].sha256 != first.files["b.py"].sha256


def test_refresh_drops_deleted_files(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    _write(a, "def foo():\n    pass\n")
    _write(b, "def bar():\n    pass\n")

    first = build(tmp_path)
    assert set(first.files) == {"a.py", "b.py"}

    b.unlink()
    second = refresh(tmp_path, first)

    assert set(second.files) == {"a.py"}


def test_refresh_with_no_existing_index_behaves_like_build(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")

    result = refresh(tmp_path, None)

    assert set(result.files) == {"a.py"}


def test_refresh_recomputes_ranks_over_final_file_set(tmp_path):
    lib = tmp_path / "lib.py"
    caller = tmp_path / "caller.py"
    _write(lib, "def helper():\n    pass\n")
    _write(caller, "from lib import helper\nhelper()\n")

    first = build(tmp_path)
    assert first.files["lib.py"].rank > 0

    caller.unlink()
    second = refresh(tmp_path, first)

    assert second.files["lib.py"].rank < first.files["lib.py"].rank


# --------------------------------------------------------------------------
# ranking configuration is a property of the index, and survives a refresh
# --------------------------------------------------------------------------


def test_the_index_records_how_it_was_ranked(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")

    index = build(tmp_path, rank_strategy="pagerank", personalize=["a.py"])

    assert index.rank_strategy == "pagerank"
    assert index.focus == ("a.py",)


def test_rank_configuration_survives_a_save_load_round_trip(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    index = build(tmp_path, rank_strategy="pagerank", personalize=["src"])

    loaded = load(save(index, default_index_path(tmp_path)))

    assert loaded is not None
    assert loaded.rank_strategy == "pagerank"
    assert loaded.focus == ("src",)


def test_refresh_keeps_the_strategy_that_built_the_index(tmp_path):
    """A refresh is not an occasion to silently re-score the repository.

    Defaulting `rank_strategy` to "simple" here meant `waypost map --refresh`
    threw away a pagerank index without saying so.
    """
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    first = build(tmp_path, rank_strategy="pagerank", personalize=["a.py"])

    second = refresh(tmp_path, first)

    assert second.rank_strategy == "pagerank"
    assert second.focus == ("a.py",)
    assert second.files["a.py"].rank == first.files["a.py"].rank


def test_refresh_still_honours_an_explicit_override(tmp_path):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    first = build(tmp_path, rank_strategy="pagerank", personalize=["a.py"])

    second = refresh(tmp_path, first, rank_strategy="simple", personalize=())

    assert second.rank_strategy == "simple"
    assert second.focus == ()


def test_load_rejects_a_strategy_this_build_does_not_know(tmp_path, caplog):
    """Better a rebuild now than a ValueError from the middle of a refresh."""
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    path = save(build(tmp_path), default_index_path(tmp_path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rank_strategy"] = "telepathy"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert load(path) is None
    assert "malformed" in caplog.text


def test_default_index_path(tmp_path):
    assert default_index_path(tmp_path) == tmp_path / ".waypost" / "index.json"
