"""Tests for symbol extraction.

The headline test is ``test_symbol_recall_meets_exit_criterion``: Sprint 1b's
exit condition is >=95% symbol recall on the fixtures, and that number is
computed here rather than asserted by eye.

Recall alone is a gameable metric -- a parser that emitted every identifier in
the file would score 100% -- so it is paired with
``test_no_unexpected_symbols`` and ``test_excluded_forms_are_not_indexed``,
which fail if the extra symbols show up. Several tests below also pin the two
bundled-query bugs (findings 2 and 6 in ``parse.py``) by asserting that the
raw upstream query is still broken: if a future language-pack release fixes
them, those tests go red and tell us the workaround can be removed, rather
than the workaround silently double-counting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_parser, get_tags_query

from waypost.parse import (
    EXTENSION_LANGUAGES,
    MAX_SIGNATURE_CHARS,
    SUPPORTED_LANGUAGES,
    ParsedFile,
    language_for,
    parse_file,
    parse_source,
    tags_query_source,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parse"
EXPECTED = {
    name: spec
    for name, spec in json.loads((FIXTURE_DIR / "expected.json").read_text()).items()
    if not name.startswith("_")
}
FIXTURE_NAMES = sorted(EXPECTED)

# Sprint 1b exit criterion.
RECALL_TARGET = 0.95


def _parsed(name: str) -> ParsedFile:
    result = parse_file(FIXTURE_DIR / name)
    assert result is not None, f"{name} failed to parse at all"
    return result


def _names(parsed: ParsedFile) -> set[str]:
    return {d.name for d in parsed.defs}


def _by_name(parsed: ParsedFile) -> dict[str, object]:
    return {d.name: d for d in parsed.defs}


# --------------------------------------------------------------------------
# The exit criterion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_symbol_recall_meets_exit_criterion(fixture):
    expected = EXPECTED[fixture]["symbols"]
    found = _names(_parsed(fixture))

    missing = sorted(set(expected) - found)
    recall = (len(expected) - len(missing)) / len(expected)

    assert recall >= RECALL_TARGET, (
        f"{fixture}: recall {recall:.1%} < {RECALL_TARGET:.0%}; missing {missing}"
    )


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_no_unexpected_symbols(fixture):
    """Recall is only meaningful if the parser is not padding its output."""
    expected = EXPECTED[fixture]["symbols"]
    extra = sorted(_names(_parsed(fixture)) - set(expected))

    assert not extra, f"{fixture}: emitted symbols not in the inventory: {extra}"


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_symbol_kinds_are_correct(fixture):
    expected = EXPECTED[fixture]["symbols"]
    actual = {d.name: d.kind for d in _parsed(fixture).defs}

    wrong = {n: (actual[n], k) for n, k in expected.items() if n in actual and actual[n] != k}

    assert not wrong, f"{fixture}: wrong kinds (actual, expected): {wrong}"


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_excluded_forms_are_not_indexed(fixture):
    """Forms the fixture contains that must stay out of the index."""
    excluded = EXPECTED[fixture].get("excluded_by_design", {})
    found = _names(_parsed(fixture))
    leaked = {name: why for name, why in excluded.items() if name in found}

    assert not leaked, f"{fixture}: indexed a form that should be excluded: {leaked}"


def test_fixture_corpus_is_big_enough_to_mean_something():
    """A 95% target on five symbols would be a rounding error, not a criterion."""
    for fixture in FIXTURE_NAMES:
        count = len(EXPECTED[fixture]["symbols"])
        assert count >= 30, f"{fixture} has only {count} expected symbols"


# --------------------------------------------------------------------------
# The two bundled-query bugs this module works around
# --------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_bundled_typescript_query_alone_finds_no_ordinary_declarations(language):
    """Finding 2, pinned. If this goes green-to-red, the JS merge can go.

    The regression this guards against is subtle: dropping the merge does not
    crash and does not fail an import, it just quietly indexes nothing but
    interfaces for every TypeScript file in a repository. The check is written
    against a snippet of *ordinary* TypeScript -- a class, a method and a
    function, none of them TS-only forms -- because that is precisely the set
    the bundled query has no rule for.
    """
    source = b"export class Widget { render(): void {} }\nexport function build(): void {}\n"
    parser = get_parser(language)
    tree = parser.parse(source)

    unmerged = QueryCursor(Query(parser.language, get_tags_query(language)))
    definitions = [c for c in unmerged.captures(tree.root_node) if c.startswith("definition.")]

    assert definitions == [], (
        f"the bundled {language} tags query now returns {definitions} on its own -- "
        "re-check whether parse.py still needs to merge the JavaScript query"
    )


@pytest.mark.parametrize(
    ("language", "fixture"),
    [("typescript", "sample_typescript.ts"), ("tsx", "sample_tsx.tsx")],
)
def test_the_merge_is_what_finds_most_of_the_fixture(language, fixture):
    """The same finding, measured on the real fixture rather than a snippet."""
    parser = get_parser(language)
    tree = parser.parse((FIXTURE_DIR / fixture).read_bytes())

    def names(query_text: str) -> set[str]:
        cursor = QueryCursor(Query(parser.language, query_text))
        found = set()
        for _pattern, caps in cursor.matches(tree.root_node):
            if any(c.startswith("definition.") for c in caps) and caps.get("name"):
                found.add(caps["name"][0].text.decode())
        return found

    unmerged = names(get_tags_query(language))
    merged = names(tags_query_source(language))

    assert merged - unmerged, "the JavaScript merge added nothing"
    assert len(merged) >= 2 * len(unmerged), (
        f"{language}: unmerged found {sorted(unmerged)}, merged found {sorted(merged)}"
    )


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_merged_query_finds_classes_methods_and_functions(language):
    parser = get_parser(language)
    fixture = "sample_typescript.ts" if language == "typescript" else "sample_tsx.tsx"
    tree = parser.parse((FIXTURE_DIR / fixture).read_bytes())

    merged = QueryCursor(Query(parser.language, tags_query_source(language)))
    captures = set(merged.captures(tree.root_node))

    assert {"definition.class", "definition.method", "definition.function"} <= captures


def test_bundled_python_constant_rule_is_still_dead():
    """Finding 6, pinned: the rule targets a node shape the grammar stopped producing."""
    parser = get_parser("python")
    tree = parser.parse(b"CONST = 3\nOTHER = 4\n")

    captures = QueryCursor(Query(parser.language, get_tags_query("python"))).captures(
        tree.root_node
    )

    assert "definition.constant" not in captures, (
        "the bundled Python tags query now captures constants -- parse.py's "
        "supplementary extraction may now be double-counting them"
    )


def test_python_module_constants_are_indexed_anyway():
    parsed = parse_source(b"CONST = 3\nOTHER = 4\n", language="python", path="c.py")

    assert {d.name for d in parsed.defs} == {"CONST", "OTHER"}
    assert all(d.kind == "constant" for d in parsed.defs)


def test_local_variables_are_not_indexed_as_constants():
    """The rule that keeps `map` output from drowning in locals."""
    python = parse_source(
        b"def f():\n    local = 1\n    return local\n", language="python", path="a.py"
    )
    js = parse_source(
        b"function f() { const local = 1; return local; }\n",
        language="javascript",
        path="a.js",
    )

    assert {d.name for d in python.defs} == {"f"}
    assert {d.name for d in js.defs} == {"f"}


def test_typescript_forms_with_no_query_rule_are_still_found():
    """Finding 7: enum, type alias and namespace have no rule at all."""
    parsed = _parsed("sample_typescript.ts")
    kinds = {d.name: d.kind for d in parsed.defs}

    assert kinds["Method"] == "enum"
    assert kinds["RequestBody"] == "type"
    assert kinds["Internals"] == "module"
    assert kinds["Internals.checksum"] == "function"


# --------------------------------------------------------------------------
# Qualification, signatures, docs, exports
# --------------------------------------------------------------------------


def test_nested_names_are_qualified():
    parsed = _parsed("sample_python.py")
    names = _names(parsed)

    assert "Client.Pool.acquire" in names
    assert "acquire" not in names
    assert "parse_payload._attempt" in names


def test_javascript_object_literal_methods_are_qualified_by_their_binding():
    names = _names(_parsed("sample_javascript.js"))

    assert "handlers.onOpen" in names
    assert "onOpen" not in names


def test_signatures_exclude_the_body():
    symbols = _by_name(_parsed("sample_python.py"))

    assert symbols["Client.request"].signature == (
        "def request(self, method: str, url: str, *, timeout: int | None = None)"
    )
    assert symbols["Client"].signature == "class Client(BaseClient)"
    assert "return" not in symbols["Client.request"].signature


def test_typescript_signatures_keep_types_and_drop_braces():
    symbols = _by_name(_parsed("sample_typescript.ts"))

    assert symbols["Client.send"].signature == "send(body: RequestBody): Promise<number>"
    assert symbols["Client"].signature == "class Client implements Transport"
    # `export` is not part of any signature -- neither `function_declaration`
    # nor `variable_declarator` includes it -- so the fact is carried by
    # `exported` instead of being duplicated in the text.
    assert symbols["parseUrl"].signature == "const parseUrl = (raw: string): string =>"
    assert symbols["parseUrl"].exported is True
    assert "{" not in symbols["Config"].signature


def test_multiline_values_are_clipped_to_their_first_line():
    """A 12-line dict must not become a 200-character signature."""
    source = b'MAPPING = {\n    "a": 1,\n    "b": 2,\n}\nNAMES = [\n    "x",\n]\n'

    symbols = {
        d.name: d.signature for d in parse_source(source, language="python", path="c.py").defs
    }

    assert symbols["MAPPING"] == "MAPPING = {...}"
    assert symbols["NAMES"] == "NAMES = [...]"


def test_signature_is_truncated_rather_than_unbounded():
    params = ", ".join(f"argument_number_{i}: string" for i in range(40))
    source = f"export function wide({params}): void {{}}\n".encode()

    parsed = parse_source(source, language="typescript", path="w.ts")
    signature = parsed.defs[0].signature

    assert len(signature) <= MAX_SIGNATURE_CHARS
    assert signature.endswith("...")


def test_python_docstrings_are_extracted_as_one_line():
    symbols = _by_name(_parsed("sample_python.py"))

    assert symbols["Client"].doc == "HTTP client with retry and pooling."
    assert symbols["Client.request"].doc == "Issue a request, retrying on transient failures."
    # A blank doc must be blank, not the first line of the body.
    assert symbols["Client._retry"].doc == ""


def test_javascript_block_comments_are_cleaned_up():
    symbols = _by_name(_parsed("sample_javascript.js"))

    assert symbols["Client"].doc == "HTTP client with retry and pooling."
    assert symbols["Client.send"].doc == "Issue a request, retrying on transient failures."
    assert "*" not in symbols["Config"].doc


def test_export_detection_uses_the_right_convention_per_language():
    python = _by_name(_parsed("sample_python.py"))
    typescript = _by_name(_parsed("sample_typescript.ts"))

    assert python["build_client"].exported is True
    assert python["_normalise"].exported is False
    assert python["Client._retry"].exported is False

    assert typescript["buildClient"].exported is True
    assert typescript["normalise"].exported is False


def test_line_spans_bracket_the_definition():
    symbols = _by_name(_parsed("sample_python.py"))
    lines = (FIXTURE_DIR / "sample_python.py").read_text().splitlines()
    client = symbols["Client"]

    assert lines[client.line - 1].startswith("class Client")
    assert client.end_line > client.line
    assert symbols["Client.Pool"].line > client.line
    assert symbols["Client.Pool"].end_line <= client.end_line


# --------------------------------------------------------------------------
# References, including the import supplement
# --------------------------------------------------------------------------


def test_python_imports_become_references():
    refs = {(r.name, r.kind) for r in _parsed("sample_python.py").refs}

    assert ("json", "import") in refs
    assert ("os.path", "import") in refs
    assert ("dataclass", "import") in refs
    # An alias records the ORIGINAL name: `Any` is defined in another module
    # and can match a definition there; `AnyValue` is defined nowhere.
    assert ("Any", "import") in refs
    assert ("AnyValue", "import") not in refs


def test_javascript_imports_and_requires_both_become_references():
    refs = {(r.name, r.kind) for r in _parsed("sample_javascript.js").refs}

    assert ("node:path", "import") in refs  # require(...)
    assert ("./logger", "import") in refs  # es import
    assert ("Logger", "import") in refs


def test_calls_and_types_become_references():
    python = {(r.name, r.kind) for r in _parsed("sample_python.py").refs}
    typescript = {(r.name, r.kind) for r in _parsed("sample_typescript.ts").refs}

    assert ("basename", "call") in python
    assert ("Config", "class") in typescript  # `new Config(...)`
    assert any(kind == "type" for _, kind in typescript)


def test_every_fixture_produces_a_non_trivial_reference_graph():
    """A silently empty refs list would only surface at the Sprint 6 benchmark."""
    for fixture in FIXTURE_NAMES:
        parsed = _parsed(fixture)
        imports = [r for r in parsed.refs if r.kind == "import"]
        assert len(parsed.refs) >= 10, f"{fixture}: only {len(parsed.refs)} refs"
        assert imports, f"{fixture}: no imports in the reference graph"


# --------------------------------------------------------------------------
# Failing soft
# --------------------------------------------------------------------------


def test_unknown_extension_returns_none(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# hello\n")

    assert parse_file(path) is None


def test_unreadable_file_returns_none_rather_than_raising(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger="waypost.parse"):
        result = parse_file(tmp_path / "missing.py")

    assert result is None
    assert "cannot read" in caplog.text


def test_syntax_errors_yield_partial_symbols_not_a_crash():
    source = b"def good():\n    pass\n\ndef broken(:\n    pass\n\ndef also_good():\n    pass\n"

    parsed = parse_source(source, language="python", path="broken.py")

    assert parsed is not None
    assert parsed.truncated is True
    assert "good" in _names(parsed)


def test_empty_file_parses_to_an_empty_symbol_list():
    parsed = parse_source(b"", language="python", path="empty.py")

    assert parsed is not None
    assert parsed.defs == ()
    assert parsed.loc == 0


def test_unsupported_language_returns_none():
    assert parse_source(b"fn main() {}", language="rust", path="m.rs") is None


def test_invalid_utf8_does_not_raise():
    parsed = parse_source(b"x = '\xff\xfe'\ndef f():\n    pass\n", language="python", path="b.py")

    assert parsed is not None
    assert "f" in _names(parsed)


# --------------------------------------------------------------------------
# Contract details the rest of the tool depends on
# --------------------------------------------------------------------------


def test_output_is_deterministic():
    first = _parsed("sample_typescript.ts")
    second = _parsed("sample_typescript.ts")

    assert first == second
    assert list(first.defs) == sorted(first.defs, key=lambda d: (d.line, d.name, d.kind))


def test_language_for_covers_the_v01_extensions():
    assert language_for("a/b/c.py") == "python"
    assert language_for("Component.TSX") == "tsx"
    assert language_for("script.mjs") == "javascript"
    assert language_for("main.rs") is None
    assert set(EXTENSION_LANGUAGES.values()) == set(SUPPORTED_LANGUAGES)


def test_parse_file_reports_paths_relative_to_root():
    parsed = parse_file("tests/fixtures/parse/sample_python.py", root=Path(__file__).parent.parent)

    assert parsed is not None
    assert parsed.path == "tests/fixtures/parse/sample_python.py"


def test_loc_counts_lines_without_a_trailing_newline():
    assert parse_source(b"a = 1", language="python", path="a.py").loc == 1
    assert parse_source(b"a = 1\n", language="python", path="a.py").loc == 1
    assert parse_source(b"a = 1\nb = 2\n", language="python", path="a.py").loc == 2
