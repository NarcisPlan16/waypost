from __future__ import annotations

import json

import pytest

from bench.cli import main
from bench.context import (
    SymbolProbe,
    build_probe,
    definitions_by_name,
    probe_symbols,
    render_comparison,
    render_probe,
)
from waypost import render
from waypost.index import build, default_index_path, save


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path):
    """A tiny repo with one small symbol buried in a long file."""
    _write(
        tmp_path / "pkg" / "core.py",
        "\n".join(
            [
                '"""Module docstring."""',
                "",
                *[f"# filler line {n} padding the file out" for n in range(60)],
                "",
                "def target(a, b):",
                '    """Add two numbers."""',
                "    return a + b",
                "",
                *[f"# more filler {n}" for n in range(60)],
            ]
        )
        + "\n",
    )
    index = build(tmp_path)
    save(index, default_index_path(tmp_path))
    return index


def test_show_costs_far_less_than_the_file_that_defines_it(tmp_path):
    index = _repo(tmp_path)
    probes = {p.name: p for p in probe_symbols(tmp_path, index)}

    target = probes["target"]
    assert target.show_tokens < target.file_tokens
    # The whole point of `show`: a three-line function does not cost the
    # hundred-line file it happens to live in.
    assert target.ratio > 0.8


def test_each_distinct_name_is_probed_once(tmp_path):
    _write(tmp_path / "a.py", "def shared():\n    pass\n")
    _write(tmp_path / "b.py", "def shared():\n    pass\n")
    index = build(tmp_path)
    save(index, default_index_path(tmp_path))

    probes = probe_symbols(tmp_path, index)
    # `show shared` resolves both definitions in one call, so charging the
    # agent for it twice would invent savings that nobody makes.
    assert [p.name for p in probes] == ["shared"]


def test_ratio_is_zero_rather_than_dividing_by_an_empty_file():
    assert SymbolProbe("x", "x.py", show_tokens=5, file_tokens=0).ratio == 0.0


def test_negative_ratio_when_show_costs_more_than_the_file():
    probe = SymbolProbe("x", "x.py", show_tokens=30, file_tokens=10)
    assert probe.saved == -20
    assert probe.ratio == -2.0


def test_unreadable_file_is_dropped_rather_than_scored_as_free(tmp_path):
    _write(tmp_path / "a.py", "def kept():\n    pass\n")
    index = build(tmp_path)
    save(index, default_index_path(tmp_path))
    (tmp_path / "a.py").unlink()

    # A file that vanished must not appear with file_tokens=0, which would
    # read as "reading it was free" and drag the pooled reduction down.
    assert probe_symbols(tmp_path, index) == []


def test_build_probe_reports_both_halves(tmp_path):
    _repo(tmp_path)
    report = build_probe(tmp_path, label="fixture")

    assert report["label"] == "fixture"
    assert report["symbols"]["count"] >= 1
    assert report["orientation"]["map_tokens"] <= report["map_budget"]
    assert report["orientation"]["repo_tokens"] > report["orientation"]["map_tokens"]


def test_build_probe_without_an_index_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="waypost index"):
        build_probe(tmp_path)


def test_rendered_probe_keeps_the_upper_bound_caveat(tmp_path):
    _repo(tmp_path)
    text = render_probe(build_probe(tmp_path))
    # The number is quotable and wrong to quote bare; the caveat travels with
    # it or someone will paste the percentage into a README.
    assert "upper bound" in text
    assert "not a benchmark result" in text


def test_cli_context_on_a_local_root_emits_json(tmp_path, capsys):
    _repo(tmp_path)
    assert main(["context", "--root", str(tmp_path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["symbols"]["count"] >= 1


def test_cli_context_indexes_a_root_that_has_none(tmp_path, capsys):
    _write(tmp_path / "a.py", "def foo():\n    pass\n")
    assert main(["context", "--root", str(tmp_path)]) == 0
    assert "context probe" in capsys.readouterr().out


def test_cli_context_rejects_an_unknown_repo(tmp_path, capsys):
    repos = tmp_path / "repos.json"
    repos.write_text("{}", encoding="utf-8")
    assert main(["context", "--repo", "nope", "--repos", str(repos)]) == 2
    assert "unknown repo" in capsys.readouterr().err


def test_precomputed_lookup_agrees_with_definitions_symbol_for_symbol(tmp_path):
    # The probe resolves every name in the repository, so it cannot afford
    # `definitions`' full rescan per lookup -- but a faster answer that is a
    # *different* answer would silently change what the probe measures.
    _write(tmp_path / "a.py", "class Client:\n    def request(self):\n        pass\n")
    _write(tmp_path / "b.py", "def request():\n    pass\n")
    _write(tmp_path / "c.py", "class Client:\n    pass\n")
    index = build(tmp_path)

    fast = definitions_by_name(index)
    names = {symbol.name for entry in index.files.values() for symbol in entry.parsed.defs}
    assert names  # a fixture that indexed nothing would pass vacuously
    for name in sorted(names) + ["Client", "request", "absent"]:
        assert render.definitions(index, name) == fast.get(name.lower(), []), name


def test_comparison_table_has_a_row_per_repo_and_keeps_the_caveat(tmp_path):
    _repo(tmp_path)
    first = build_probe(tmp_path, label="one")
    second = build_probe(tmp_path, label="two")

    text = render_comparison([first, second])
    assert "one" in text and "two" in text
    assert "upper bound" in text


def test_cli_context_requires_a_source(capsys):
    with pytest.raises(SystemExit):
        main(["context"])


def test_cli_context_rejects_two_sources(tmp_path):
    with pytest.raises(SystemExit):
        main(["context", "--root", str(tmp_path), "--all"])
