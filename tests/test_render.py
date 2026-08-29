from __future__ import annotations

import pytest

from waypost import render
from waypost.index import build
from waypost.parse import Symbol
from waypost.tokens import TRUNCATION_MARKER, count, get_tokenizer

CLIENT_PY = '''
"""Transport layer."""

import json

from .config import Config

DEFAULT_TIMEOUT = 30
_RETRY_CODES = (429, 503)


class Client:
    """HTTP client with retry and pooling."""

    def request(self, method, url, timeout=None):
        """Send one request."""
        return _encode(method, url)

    def close(self):
        return None


def build_client(config=None):
    """Construct a Client with sensible defaults."""
    return Client()


def _encode(method, url):
    return json.dumps({"method": method, "url": url})
'''

CONFIG_PY = '''
"""Configuration."""


class Config:
    """Runtime configuration."""

    def merge(self, other):
        return other


def load_config(path):
    """Read a config file."""
    return Config()
'''

APP_PY = """
from .client import build_client, Client
from .config import load_config


def main():
    client = build_client(load_config("app.toml"))
    return client.request("GET", "/")
"""


@pytest.fixture
def index(tmp_path):
    pkg = tmp_path / "src" / "app"
    pkg.mkdir(parents=True)
    (pkg / "client.py").write_text(CLIENT_PY, encoding="utf-8")
    (pkg / "config.py").write_text(CONFIG_PY, encoding="utf-8")
    (pkg / "app.py").write_text(APP_PY, encoding="utf-8")
    return build(tmp_path)


def _sym(name, kind="function", *, exported=True, line=1):
    return Symbol(
        name=name, kind=kind, line=line, end_line=line, signature=f"def {name}()", exported=exported
    )


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------


def test_order_symbols_nests_methods_under_their_class():
    defs = [
        _sym("Client", "class", line=10),
        _sym("Client.request", "method", line=12),
        _sym("Client.close", "method", line=20),
        _sym("build_client", line=30),
    ]
    ordered = render.order_symbols(defs)
    assert [(s.name, d) for s, d in ordered] == [
        ("Client", 0),
        ("Client.request", 1),
        ("Client.close", 1),
        ("build_client", 0),
    ]


def test_order_symbols_puts_containers_and_public_callables_first():
    defs = [
        _sym("_helper", line=1, exported=False),
        _sym("MAX", "constant", line=2),
        _sym("run", line=3),
        _sym("Widget", "class", line=4),
    ]
    assert [s.name for s, _ in render.order_symbols(defs)] == ["Widget", "run", "MAX", "_helper"]


def test_a_method_whose_class_lives_elsewhere_is_kept_not_dropped():
    ordered = render.order_symbols([_sym("Elsewhere.patch", "method", line=5)])
    assert [(s.name, d) for s, d in ordered] == [("Elsewhere.patch", 0)]


# --------------------------------------------------------------------------
# map -- the budget is the exit criterion for this sprint
# --------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [0, 1, 5, 12, 30, 60, 120, 300, 900, 5000])
def test_map_never_exceeds_its_budget(index, budget):
    for tokenizer in (get_tokenizer("heuristic"), get_tokenizer()):
        out = render.render_map(index, budget=budget, tokenizer=tokenizer)
        assert tokenizer.count(out) <= budget, (budget, tokenizer.name, out)


def test_map_truncates_at_a_symbol_boundary_never_mid_symbol(index):
    full = render.render_map(index, budget=5000).splitlines()
    clipped = [
        line
        for line in render.render_map(index, budget=60).splitlines()
        if line != TRUNCATION_MARKER
    ]
    # Every surviving line is a whole line from the untruncated rendering,
    # so no signature was cut in half.
    assert set(clipped) <= set(full)


def test_map_says_so_when_it_truncated(index):
    assert render.render_map(index, budget=40).endswith(TRUNCATION_MARKER)
    assert TRUNCATION_MARKER not in render.render_map(index, budget=5000)
    _entries, truncated = render.select_map_entries(index, budget=5000)
    assert truncated is False


def test_map_spends_a_small_budget_across_files_not_on_one(index):
    # The failure this guards against: the top-ranked file's private
    # helpers crowding out every other file in the repository.
    out = render.render_map(index, budget=120)
    named = [line for line in out.splitlines() if line.endswith("loc)")]
    assert len(named) >= 2, out


def test_map_leads_with_the_highest_ranked_file(index):
    ranked = sorted(index.files.items(), key=lambda kv: (-kv[1].rank, kv[0]))
    out = render.render_map(index, budget=2000)
    assert out.splitlines()[0].startswith(ranked[0][0])


def test_map_is_monotone_in_its_budget(index):
    previous: list[str] = []
    for budget in (60, 120, 240, 480, 2000):
        lines = [
            line
            for line in render.render_map(index, budget=budget).splitlines()
            if line != TRUNCATION_MARKER
        ]
        # A bigger budget only ever adds; it never reshuffles or drops.
        assert set(previous) <= set(lines), budget
        previous = lines


def test_focus_promotes_a_path_without_filtering_the_rest(index):
    out = render.render_map(index, budget=2000, focus=["src/app/app.py"])
    assert out.splitlines()[0].startswith("src/app/app.py")
    assert "src/app/client.py" in out


def test_map_never_emits_a_file_header_with_no_symbols(index):
    for budget in range(20, 200, 7):
        lines = [
            line
            for line in render.render_map(index, budget=budget).splitlines()
            if line != TRUNCATION_MARKER
        ]
        for position, line in enumerate(lines):
            if line.endswith("loc)"):
                remaining = lines[position + 1 :]
                assert remaining and remaining[0].startswith("  "), (budget, lines)


def test_map_data_matches_the_text_selection(index):
    data = render.map_data(index, budget=300)
    text = render.render_map(index, budget=300)
    assert data["budget"] == 300
    assert data["tokenizer"] == get_tokenizer().name
    for entry in data["files"]:
        assert entry["path"] in text
        for symbol in entry["symbols"]:
            assert symbol["signature"] in text


def test_an_empty_index_renders_nothing_rather_than_failing(tmp_path):
    empty = build(tmp_path)
    assert render.render_map(empty, budget=500) == ""
    assert render.map_data(empty, budget=500)["files"] == []


# --------------------------------------------------------------------------
# find
# --------------------------------------------------------------------------


def test_find_matches_substrings_and_ranks_exact_first(index):
    hits = render.search(index, "client")
    names = [h.symbol.name for h in hits]
    assert "Client" in names and "build_client" in names
    # `Client` is an exact match; `build_client` merely contains it.
    assert names[0] == "Client"


def test_find_supports_globs(index):
    names = {h.symbol.name for h in render.search(index, "*_client")}
    assert names == {"build_client"}


def test_find_matches_the_last_segment_of_a_qualified_name(index):
    assert any(h.symbol.name == "Client.request" for h in render.search(index, "request"))


def test_find_respects_its_limit(index):
    assert len(render.search(index, "e", limit=2)) == 2


def test_find_reports_a_miss_without_pretending(index):
    out = render.render_find(index, "no_such_symbol_anywhere")
    assert "no symbol matching" in out
    assert render.find_data(index, "no_such_symbol_anywhere")["hits"] == []


@pytest.mark.parametrize("budget", [10, 40, 200])
def test_find_never_exceeds_its_budget(index, budget):
    assert count(render.render_find(index, "c", budget=budget)) <= budget


def test_find_leads_with_the_exact_definition_and_summarises_the_rest(index):
    # The regression this guards: `find Client` printed `build_client` and
    # every other substring match in full, so answering "where is this class"
    # cost 49x what the grep it was meant to replace costs.
    out = render.render_find(index, "Client")

    assert "class Client" in out
    assert "build_client" not in out
    assert "(--all to list)" in out


def test_find_all_still_lists_every_partial_match(index):
    out = render.render_find(index, "Client", all_matches=True)

    assert "class Client" in out
    assert "build_client" in out
    assert "(--all to list)" not in out


def test_find_lists_everything_when_nothing_matches_exactly(index):
    # A discovery query has no exact tier to lead with; answering it with a
    # bare count would make the command useless.
    out = render.render_find(index, "buil")

    assert "build_client" in out
    assert "(--all to list)" not in out


def test_find_json_says_how_many_partial_matches_it_withheld(index):
    data = render.find_data(index, "Client")

    assert [h["name"] for h in data["hits"]] == ["Client"]
    assert data["partial_omitted"] >= 1
    assert data["count"] == len(data["hits"])

    everything = render.find_data(index, "Client", all_matches=True)
    assert everything["partial_omitted"] == 0
    assert len(everything["hits"]) > len(data["hits"])


def test_find_does_not_state_a_partial_count_it_cannot_know(index):
    # Truncated at the limit, the count is a floor, not a total.
    out = render.render_find(index, "Client", limit=2)
    assert "at least" in out


# --------------------------------------------------------------------------
# show -- the span, and only the span
# --------------------------------------------------------------------------


def test_show_prints_the_symbols_own_span_and_nothing_around_it(index):
    text = render.render_show(index, "Client.request")
    assert "def request" in text
    assert "Send one request." in text
    # Neighbours in the same file must not appear.
    assert "def close" not in text
    assert "DEFAULT_TIMEOUT" not in text
    assert "def build_client" not in text


def test_show_numbers_lines_from_the_symbols_real_position(index):
    symbol = next(
        s for s in index.files["src/app/client.py"].parsed.defs if s.name == "Client.request"
    )
    body = [
        line for line in render.render_show(index, "Client.request").splitlines() if "|" in line
    ]
    assert body[0].startswith(f"{symbol.line}|")
    assert len(body) == symbol.end_line - symbol.line + 1


def test_show_suggests_near_misses_before_giving_up(index):
    assert "did you mean" in render.render_show(index, "build_clien")
    assert "no symbol named" in render.render_show(index, "zzz_nothing_like_this")


@pytest.mark.parametrize("budget", [5, 20, 100])
def test_show_never_exceeds_its_budget(index, budget):
    assert count(render.render_show(index, "Client", budget=budget)) <= budget


def test_show_data_carries_the_same_source(index):
    data = render.show_data(index, "build_client")
    assert data["count"] == 1
    assert "def build_client" in data["definitions"][0]["source"]


def test_show_survives_a_stale_index(index, tmp_path):
    # The file shrank since it was indexed. That is a reason to say so, not
    # to raise.
    (tmp_path / "src" / "app" / "client.py").write_text("# gone\n", encoding="utf-8")
    out = render.render_show(index, "build_client")
    assert "shorter than the index expects" in out or "# gone" in out


# --------------------------------------------------------------------------
# refs
# --------------------------------------------------------------------------


def test_refs_reports_definitions_and_referring_files(index):
    out = render.render_refs(index, "build_client")
    assert "src/app/client.py:" in out
    assert "src/app/app.py:" in out


def test_refs_counts_imports_as_references(index):
    data = render.refs_data(index, "load_config")
    referrers = {ref["path"] for ref in data["references"]}
    assert "src/app/app.py" in referrers
    kinds = {site["kind"] for ref in data["references"] for site in ref["sites"]}
    assert "import" in kinds


def test_refs_reports_a_miss(index):
    assert "no definitions or references" in render.render_refs(index, "nothing_at_all")


# --------------------------------------------------------------------------
# outline
# --------------------------------------------------------------------------


def test_outline_lists_one_file_in_full(index):
    out = render.render_outline(index, "src/app/client.py")
    assert "class Client" in out
    assert "def request" in out
    assert "def _encode" in out
    # And nothing from another file.
    assert "load_config" not in out


def test_outline_accepts_an_unambiguous_suffix(index):
    assert render.resolve_path(index, "client.py") == "src/app/client.py"
    assert "class Client" in render.render_outline(index, "client.py")


def test_outline_reports_an_unknown_path(index):
    out = render.render_outline(index, "src/app/missing.py")
    assert "not in the index" in out
    assert render.outline_data(index, "src/app/missing.py")["found"] is False


@pytest.mark.parametrize("budget", [8, 40, 200])
def test_outline_never_exceeds_its_budget(index, budget):
    assert count(render.render_outline(index, "client.py", budget=budget)) <= budget


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------


def test_stats_counts_what_the_index_holds(index):
    data = render.stats_data(index)
    assert data["files"] == 3
    assert data["symbols"] == sum(len(e.parsed.defs) for e in index.files.values())
    assert data["languages"] == {"python": 3}
    assert data["tokenizer"] == get_tokenizer().name
    assert data["index_bytes"] is None  # not persisted in this test


def test_stats_text_names_the_tokenizer_that_measured_the_budgets(index):
    assert get_tokenizer().name in render.render_stats(index)
