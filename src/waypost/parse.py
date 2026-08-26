"""Symbol extraction via tree-sitter tag queries.

Responsibilities (Sprint 1):
    - Run each language's bundled tags query over a parsed file and map
      ``@definition.*`` captures to ``defs`` and ``@reference.*`` to ``refs``.
    - Qualify nested names, so a method inside ``class Client`` is recorded
      as ``Client.request``.
    - Extract a signature and a single-line doc summary per definition.
    - Fail soft: a parse error on one file logs at debug level and skips it,
      never aborting the index.

Verified findings (measured against tree-sitter 0.26.0 +
tree-sitter-language-pack) that this module MUST account for:

1. Queries are not files on disk. There are no ``.scm`` files in the
   language pack; queries come from accessors::

       from tree_sitter_language_pack import get_tags_query, get_parser

2. TypeScript's tags query is INCOMPLETE ON ITS OWN. It covers only
   TypeScript-specific declaration forms (``function_signature``,
   ``method_signature``, ``abstract_method_signature``,
   ``abstract_class_declaration``, ``interface_declaration``, ``module``).
   It has no rule for ``class_declaration``, ``method_definition`` or
   ``function_declaration`` -- those live in the JavaScript query, because
   the TS grammar inherits from JS but the query does not. Run against
   ordinary TypeScript -- a class, its methods, a function -- it yields ZERO
   definitions. (Precisely: it still matches the TS-only forms above, so a
   file of nothing but interfaces is not empty. What it finds on real
   TypeScript is interfaces and nothing else, which is worse than empty
   because it looks like it worked. ``test_the_merge_is_what_finds_most_of_
   the_fixture`` measures the gap on the fixture rather than asserting it.)

   The fix, verified working, is to concatenate the two queries and run the
   result against the TypeScript parser::

       merged = get_tags_query("javascript") + "\\n" + get_tags_query("typescript")

   Apply the same merge for ``.tsx``.

3. ``@doc`` is JavaScript-only. The JS query captures leading comments as
   ``@doc``; the Python query has no ``@doc`` at all, so Python docstrings
   must be pulled from the AST separately (first string literal in the body).

   Even where ``@doc`` exists it misses the common case: it anchors the
   comment directly against the declaration, but ``export class Client``
   wraps that declaration in an ``export_statement``, so the comment is the
   export's sibling and the capture comes back empty. Relying on ``@doc``
   alone therefore documents private helpers and nothing else. See
   ``_leading_comment``.

4. ``refs`` will be sparse. Python offers only ``@reference.call``; TS adds
   ``@reference.type`` and ``@reference.class``. Neither covers imports.
   Ranking quality depends on the reference graph, so either supplement refs
   from import statements or document the sparsity explicitly -- silently
   under-populating it degrades ``map`` output in a way that will not show up
   until the benchmark.

   Decision taken in Sprint 1b: **supplement**. Imports are extracted from
   the AST (``_imports``) and emitted as ``kind="import"`` references, so a
   file that imports a symbol counts as an inbound reference to whatever
   defines it. Python ``import``/``from ... import``, ES ``import``/re-export
   and CommonJS ``require`` are all covered; an alias records the original
   name, since that is the one some other file actually defines.

Capture names observed for Python::

    @definition.class  @definition.function  @definition.constant
    @name              @reference.call

Three further findings, measured while implementing this module. Each one is
a place where trusting the bundled query would have silently under-populated
the index:

5. ``captures()`` CANNOT be used to build symbols. It returns a flat
   ``dict[capture_name, list[node]]`` with no association between a
   ``@definition.*`` node and its own ``@name``, and its ``@name`` list also
   contains the names of *references*. Pairing must come from
   ``QueryCursor.matches()``, which yields one capture dict per pattern
   match. Standard predicates (``#eq?``, ``#not-eq?``, ``#match?``) are
   applied by tree-sitter itself; the non-standard directives ``#strip!``
   and ``#select-adjacent!`` are NOT, so ``@doc`` is cleaned up here by hand.

6. Python's ``@definition.constant`` rule is DEAD against the current
   grammar. It is written as::

       (module (expression_statement (assignment left: (identifier) @name)))

   but tree-sitter-python 0.26 parses a module-level assignment as a direct
   ``(module (assignment ...))`` with no ``expression_statement`` wrapper.
   The rule therefore never fires and module-level constants are invisible.
   (This corrects the earlier note that ``@definition.constant`` had been
   "observed" -- what was observed was the capture *name* in the query text,
   not a match.)

7. TypeScript's ``@definition.module`` rule is dead for the same class of
   reason: ``namespace NS {}`` parses as ``internal_module``, not ``module``,
   and the only node that *is* a ``module`` (``declare module "m"``) has a
   string name where the rule requires an identifier. TypeScript ``enum`` and
   ``type`` aliases have no rule at all, and ``export const X = 1`` is
   captured only when its value is a function.

Findings 6 and 7 are handled by ``_SUPPLEMENTARY_DEFS``: a small per-language
table of node types extracted directly from the AST, alongside the query.
The table is deliberately narrow -- it covers query *gaps*, not a
reimplementation of the queries.

One upstream exclusion is kept rather than worked around: the JavaScript
query carries ``(#not-eq? @name "constructor")``, so constructors are not
symbols. A constructor is reachable through its class, and emitting one per
class costs tokens in every ``map`` for little localisation value.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from tree_sitter import Node, Query, QueryCursor
from tree_sitter_language_pack import get_parser, get_tags_query

logger = logging.getLogger(__name__)

# Signatures and docs are truncated: this tool exists to spend fewer tokens,
# and a 400-character generic signature helps nobody find anything.
MAX_SIGNATURE_CHARS = 200
MAX_DOC_CHARS = 120

_ELLIPSIS = "..."

# Used to close a bracket that a clipped multi-line signature left open, so
# `EXTENSION_LANGUAGES = {` reads as `EXTENSION_LANGUAGES = {...}`.
_BRACKET_PAIRS = {"{": "}", "(": ")", "[": "]"}

# Extension -> language-pack language name. Deliberately limited to the v0.1
# scope (Python and TS/JS); adding a language means adding fixtures for it.
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}

SUPPORTED_LANGUAGES = frozenset(EXTENSION_LANGUAGES.values())

# Languages whose tags query must be merged with JavaScript's to return
# anything at all. See finding 2.
_MERGE_WITH_JAVASCRIPT = frozenset({"typescript", "tsx"})

# Node types that qualify a nested name. A definition inside one of these
# gets the container's name prepended, so a method of ``class Client`` is
# recorded as ``Client.request``.
_CONTAINER_TYPES = frozenset(
    {
        "class_definition",  # python
        "function_definition",  # python (nested defs)
        "class_declaration",  # js/ts
        "abstract_class_declaration",  # ts
        "class",  # js class expression
        "interface_declaration",  # ts
        "internal_module",  # ts `namespace NS {}`
        "enum_declaration",  # ts
        "function_declaration",  # js/ts (nested functions)
        "variable_declarator",  # js/ts `const handlers = { onOpen: () => ... }`
    }
)

# Function-ish nodes. A definition with one of these between it and the file
# root is inside a function body rather than at module or class scope.
_FUNCTION_TYPES = frozenset(
    {
        "function_declaration",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "function_definition",
    }
)

# Node types that mark a definition as part of the module's public surface.
_EXPORT_TYPES = frozenset({"export_statement"})

# Per-language supplementary definitions: node type -> symbol kind. These
# cover query gaps (findings 6 and 7), not forms the query already handles.
_SUPPLEMENTARY_DEFS: dict[str, dict[str, str]] = {
    "python": {
        "assignment": "constant",
    },
    "typescript": {
        "enum_declaration": "enum",
        "type_alias_declaration": "type",
        "internal_module": "module",
        "variable_declarator": "constant",
    },
    "tsx": {
        "enum_declaration": "enum",
        "type_alias_declaration": "type",
        "internal_module": "module",
        "variable_declarator": "constant",
    },
    "javascript": {
        "variable_declarator": "constant",
    },
}

_PYTHON_IMPORT_TYPES = frozenset({"import_statement", "import_from_statement"})
_JS_IMPORT_TYPES = frozenset({"import_statement", "export_statement", "call_expression"})


@dataclass(frozen=True)
class Symbol:
    """One definition found in a file.

    ``name`` is qualified with its enclosing containers (``Client.request``).
    ``line``/``end_line`` are 1-based and inclusive, so ``show`` can slice the
    symbol's own span without re-reading the file around it.
    """

    name: str
    kind: str
    line: int
    end_line: int
    signature: str
    doc: str = ""
    exported: bool = False


@dataclass(frozen=True)
class Reference:
    """One outbound reference: a call, a type mention, or an import."""

    name: str
    kind: str
    line: int


@dataclass(frozen=True)
class ParsedFile:
    """Everything the indexer needs from one source file."""

    path: str
    language: str
    loc: int
    defs: tuple[Symbol, ...] = field(default_factory=tuple)
    refs: tuple[Reference, ...] = field(default_factory=tuple)
    truncated: bool = False


def language_for(path: Path | str) -> str | None:
    """Return the language-pack name for ``path``, or None if unsupported."""
    return EXTENSION_LANGUAGES.get(Path(path).suffix.lower())


@cache
def tags_query_source(language: str) -> str:
    """The tags query text for ``language``, with the JS merge where needed.

    TypeScript and TSX find no ordinary class, method or function declaration
    on their own -- see finding 2 in the module docstring. This is the single
    place that merge happens.
    """
    query = get_tags_query(language) or ""
    if language in _MERGE_WITH_JAVASCRIPT:
        query = (get_tags_query("javascript") or "") + "\n" + query
    return query


@cache
def _interesting_types(language: str) -> frozenset[str]:
    """Node types the supplementary pass can do anything with.

    Everything else is skipped without a function call. Derived from the two
    tables rather than written out, so adding a supplementary rule cannot
    forget to widen the filter.
    """
    import_types = _PYTHON_IMPORT_TYPES if language == "python" else _JS_IMPORT_TYPES
    return frozenset(_SUPPLEMENTARY_DEFS.get(language, {})) | import_types


@cache
def _compiled(language: str) -> tuple[Any, Query]:
    parser = get_parser(language)
    grammar = parser.language
    if grammar is None:  # pragma: no cover -- the language pack always sets one.
        raise ValueError(f"no grammar for {language!r}")
    return parser, Query(grammar, tags_query_source(language))


def parse_file(path: Path | str, *, root: Path | str | None = None) -> ParsedFile | None:
    """Parse one file; return None if it is unsupported or unreadable.

    ``path`` may be absolute or relative to ``root``. The returned
    ``ParsedFile.path`` is always relative to ``root`` (POSIX-separated) when
    a root is given, matching what :func:`waypost.walk.iter_files` yields.

    Never raises for a bad file: an unreadable path, an unknown extension or
    a tree-sitter failure logs at debug level and returns None, so one bad
    file cannot abort an index.
    """
    path = Path(path)
    abs_path = path if path.is_absolute() or root is None else Path(root) / path
    language = language_for(abs_path)
    if language is None:
        logger.debug("waypost: no parser for %s", abs_path)
        return None

    try:
        source = abs_path.read_bytes()
    except OSError as exc:
        logger.debug("waypost: cannot read %s: %s", abs_path, exc)
        return None

    if root is not None:
        try:
            rel = abs_path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            rel = abs_path.as_posix()
    else:
        rel = path.as_posix()

    return parse_source(source, language=language, path=rel)


def parse_source(source: bytes, *, language: str, path: str = "<memory>") -> ParsedFile | None:
    """Parse ``source`` as ``language``. Returns None only on a hard failure.

    A file with syntax errors is NOT a hard failure: tree-sitter recovers and
    returns a partial tree, and partial symbols beat none. ``truncated`` is
    set so callers can tell the difference.
    """
    if language not in SUPPORTED_LANGUAGES:
        logger.debug("waypost: unsupported language %r for %s", language, path)
        return None

    try:
        parser, query = _compiled(language)
        tree = parser.parse(source)
    except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the index.
        logger.debug("waypost: failed to parse %s: %s", path, exc)
        return None

    root = tree.root_node
    if root.has_error:
        logger.debug("waypost: syntax errors in %s; indexing what parsed", path)

    try:
        defs, refs = _extract(root, source, language, query)
    except Exception:
        # Deliberately louder than the parse failure above, and deliberately
        # with a traceback. A file tree-sitter cannot parse is an expected,
        # uninteresting event -- somebody committed broken syntax. An
        # exception raised *by this module* while walking a tree that parsed
        # fine is a bug in waypost, and at DEBUG it looked identical to the
        # boring case: the file would vanish from the index behind a
        # breadcrumb nobody has logging turned on for. That is the exact
        # silent under-population this module's own docstring warns will not
        # surface until the benchmark.
        logger.warning(
            "waypost: bug extracting symbols from %s; file omitted from the index",
            path,
            exc_info=True,
        )
        return None

    return ParsedFile(
        path=path,
        language=language,
        loc=_line_count(source),
        defs=defs,
        refs=refs,
        truncated=root.has_error,
    )


def _extract(
    root: Node,
    source: bytes,
    language: str,
    query: Query,
) -> tuple[tuple[Symbol, ...], tuple[Reference, ...]]:
    symbols: dict[tuple[str, int, str], Symbol] = {}
    references: dict[tuple[str, str, int], Reference] = {}

    for _pattern, caps in QueryCursor(query).matches(root):
        name_nodes = caps.get("name")
        if not name_nodes:
            continue
        name_node = name_nodes[0]

        for capture, nodes in caps.items():
            if capture.startswith("definition."):
                kind = capture.split(".", 1)[1]
                symbol = _build_symbol(nodes[0], name_node, source, language, kind, caps)
                if symbol is not None:
                    symbols[(symbol.name, symbol.line, symbol.kind)] = symbol
            elif capture.startswith("reference."):
                kind = capture.split(".", 1)[1]
                text = _text(name_node, source)
                if text:
                    ref = Reference(name=text, kind=kind, line=name_node.start_point[0] + 1)
                    references[(ref.name, ref.kind, ref.line)] = ref

    # The supplementary pass acts on a handful of node types but visits every
    # named node, so the filtering has to happen before the function calls,
    # not inside them. Measured over 672 real files (307k LOC, 1.36M named
    # nodes): 2.17s -> 1.54s for this pass, 4.30s -> 3.73s overall.
    #
    # What is left is `_walk` itself -- building a Python wrapper for every
    # node in the tree. It is still ~41% of parse time, more than twice the
    # native query pass, and no amount of filtering fixes that because the
    # traversal is the cost. Expressing these rules as real tree-sitter
    # queries would move the traversal into C. That is a Sprint 2 decision
    # against a Sprint 2 target, recorded here so it is made on a number:
    # 700k LOC extrapolates to ~8.4s against a 15s budget that also has to
    # cover walking, ranking and writing the index.
    supplements = _SUPPLEMENTARY_DEFS.get(language, {})
    interesting = _interesting_types(language)

    for node in _walk(root):
        node_type = node.type
        if node_type not in interesting:
            continue
        supplement = supplements.get(node_type)
        if supplement is not None:
            symbol = _supplementary_symbol(node, source, language, supplement)
            if symbol is not None:
                symbols.setdefault((symbol.name, symbol.line, symbol.kind), symbol)
        for ref in _imports(node, source, language):
            references.setdefault((ref.name, ref.kind, ref.line), ref)

    ordered_defs = tuple(sorted(symbols.values(), key=lambda s: (s.line, s.name, s.kind)))
    ordered_refs = tuple(sorted(references.values(), key=lambda r: (r.line, r.name, r.kind)))
    return ordered_defs, ordered_refs


def _build_symbol(
    node: Node,
    name_node: Node,
    source: bytes,
    language: str,
    kind: str,
    caps: dict[str, list[Node]],
) -> Symbol | None:
    if node.type == "pair" and not _is_navigable_pair(node):
        return None

    name = _text(name_node, source)
    if not name:
        return None
    qualified = _qualify(node, name, source)
    return Symbol(
        name=qualified,
        kind=kind,
        line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        signature=_signature(node, source),
        doc=_doc(node, source, language, caps),
        exported=_is_exported(node, qualified, language),
    )


def _supplementary_symbol(
    node: Node,
    source: bytes,
    language: str,
    kind: str,
) -> Symbol | None:
    """Build a symbol for a form the tags query misses. See findings 6 and 7."""
    if kind == "constant" and not _is_module_level_constant(node, language):
        return None

    if kind == "constant":
        value = node.child_by_field_name("value")
        if value is not None and value.type == "generator_function":
            # `const gen = function* () {}` (and the async form, which is the
            # same node type). The bundled query's variable_declarator rule
            # covers `arrow_function` and `function_expression` but not this,
            # so it arrives here as a fallback -- and calling a generator a
            # constant is a lie that would render as one in every map output.
            kind = "function"

    name_node = node.child_by_field_name("name") or node.child_by_field_name("left")
    if name_node is None or name_node.type not in {"identifier", "type_identifier"}:
        # Destructuring (`const {a, b} = x`) and attribute targets
        # (`obj.attr = 1`) are not module-level constants worth indexing.
        return None

    name = _text(name_node, source)
    if not name:
        return None

    qualified = _qualify(node, name, source)
    return Symbol(
        name=qualified,
        kind=kind,
        line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        signature=_signature(node, source),
        doc=_doc(node, source, language, {}),
        exported=_is_exported(node, qualified, language),
    )


def _is_module_level_constant(node: Node, language: str) -> bool:
    """True for a top-level ``NAME = value`` binding, false for a local one.

    Only module scope counts. A local variable inside a function is not a
    symbol anybody navigates to, and indexing every one of them would bury
    the real symbols in exactly the noise this tool is meant to remove.
    """
    if language == "python":
        return node.parent is not None and node.parent.type == "module"

    if _inside_function_body(node):
        # A namespace body and a function body are both `statement_block`, so
        # the node type alone cannot tell them apart: `const local = 1` inside
        # a function has to be ruled out explicitly.
        return False

    # js/ts: variable_declarator -> lexical_declaration -> [export_statement] -> program
    current = node.parent
    while current is not None and current.type in {
        "lexical_declaration",
        "variable_declaration",
        "export_statement",
    }:
        current = current.parent
    if current is None or not _is_module_scope_body(current):
        return False
    # An arrow/function value is already a @definition.function from the query.
    value = node.child_by_field_name("value")
    return value is None or value.type not in {"arrow_function", "function_expression"}


def _is_module_scope_body(node: Node) -> bool:
    """True for the file body or a namespace body, false for any other block.

    `statement_block` is not specific enough to accept on its own. It is the
    body of a namespace -- where `export const SEED = 1` really is a top-level
    constant of `Internals` -- but it is equally the body of an `if`, `for`,
    `while`, `try` or a bare block, where the binding is scoped to that block
    and is nobody's navigation target::

        if (typeof window !== "undefined") {
            const GUARDED = 1;   // not a module constant
        }

    `_inside_function_body` does not catch these, because an `if` at module
    scope has no function ancestor. Accepting any `statement_block` therefore
    padded the index with block locals -- and because no fixture put a `const`
    inside a bare `if` or `for`, the padding never showed up in a test.
    """
    if node.type == "program":
        return True
    return (
        node.type == "statement_block"
        and node.parent is not None
        and node.parent.type == "internal_module"
    )


def _is_navigable_pair(node: Node) -> bool:
    """True for a key in a named object literal, false for one in an argument.

    `const handlers = { onOpen: () => ... }` at module level is a dispatch
    table worth navigating to, and `handlers.onOpen` is a name that means
    something. The same shape passed straight into a call --
    `app.use({ onError: () => ... })` -- is an implementation detail, and
    `_qualify` has no declarator to hang it off, so it would be indexed under
    the bare key `onError`: a symbol with no owner, one line of noise per
    callback in every map output.

    The test has to be positive -- "is this object bound to a name?" -- rather
    than "is this inside a function?". A config object passed to a top-level
    call is inside no function at all, so the negative test let it through.
    """
    if _inside_function_body(node):
        return False
    current = node
    while current.parent is not None:
        parent = current.parent
        if parent.type in {"object", "pair"}:
            # Nested literal: keep climbing to whatever binds the outermost one.
            current = parent
            continue
        if parent.type == "variable_declarator":
            value = parent.child_by_field_name("value")
            # Compare ids, not identity: the bindings hand out a fresh wrapper
            # object on every access, so `is` is always False here.
            return value is not None and value.id == current.id
        return False
    return False


def _inside_function_body(node: Node) -> bool:
    """True if any ancestor is a function, i.e. the node is not at file/class scope."""
    current = node.parent
    while current is not None:
        if current.type in _FUNCTION_TYPES:
            return True
        current = current.parent
    return False


def _qualify(node: Node, name: str, source: bytes) -> str:
    """Prepend the names of enclosing classes/functions/namespaces."""
    parts: list[str] = []
    current = node.parent
    while current is not None:
        if current.type in _CONTAINER_TYPES:
            container_name = current.child_by_field_name("name")
            if container_name is not None:
                text = _text(container_name, source)
                if text:
                    parts.append(text)
        current = current.parent
    if not parts:
        return name
    return ".".join(reversed(parts)) + "." + name


def _signature(node: Node, source: bytes) -> str:
    """The definition's own line, minus its body, whitespace-collapsed.

    Cutting at the body's start byte is what keeps ``map`` cheap: a class
    with 400 lines in it contributes one line of signature, not 400.

    A definition with no body -- a constant, a type alias -- has no such cut
    point, so it is clipped to its first physical line instead. Without that,
    a 12-line dict literal is flattened into one 200-character signature line,
    which is the opposite of the point: dogfooding on this module's own
    ``EXTENSION_LANGUAGES`` was what surfaced it.
    """
    end = node.end_byte
    body = node.child_by_field_name("body")
    if body is None:
        value = node.child_by_field_name("value")
        if value is not None:
            body = value.child_by_field_name("body")

    elided = False
    if body is not None and node.start_byte < body.start_byte < end:
        end = body.start_byte
    elif node.end_point[0] > node.start_point[0]:
        newline = source.find(b"\n", node.start_byte)
        if newline != -1 and newline < end:
            end = newline
            elided = True

    text = _declaration_keyword(node, source) + " ".join(
        source[node.start_byte : end].decode("utf-8", "replace").split()
    )
    text = _elide(text) if elided else _strip_signature_tail(text)
    return _truncate(text, MAX_SIGNATURE_CHARS)


def _elide(text: str) -> str:
    """Mark a clipped multi-line value, closing the bracket it opened."""
    text = text.rstrip()
    closer = _BRACKET_PAIRS.get(text[-1:])
    if closer is not None:
        return f"{text}{_ELLIPSIS}{closer}"
    return f"{text} {_ELLIPSIS}"


def _declaration_keyword(node: Node, source: bytes) -> str:
    """``const `` / ``let `` / ``var `` for a declarator, else "".

    A ``variable_declarator`` node starts at the name, so without this the
    signature for ``export const arrow = (x) => x`` reads ``arrow = (x)`` --
    which does not tell a reader whether it is a binding or an assignment.
    Only the keyword is prepended, never the whole declaration, so
    ``const a = 1, b = 2`` still yields two separate one-name signatures.
    """
    if node.type != "variable_declarator" or node.parent is None:
        return ""
    if node.parent.type not in {"lexical_declaration", "variable_declaration"}:
        return ""
    keyword = node.parent.children[0] if node.parent.children else None
    if keyword is None or keyword.type not in {"const", "let", "var"}:
        return ""
    return _text(keyword, source) + " "


def _strip_signature_tail(text: str) -> str:
    """Drop the body-opening punctuation, keeping ``=>`` which carries meaning."""
    text = text.rstrip()
    while text and text[-1] in "{;":
        text = text[:-1].rstrip()
    if text.endswith(":"):
        text = text[:-1].rstrip()
    return text


def _doc(node: Node, source: bytes, language: str, caps: dict[str, list[Node]]) -> str:
    """One-line documentation summary, or "" if the definition has none."""
    if language == "python":
        return _python_docstring(node, source)
    return _comment_doc(node, source, caps)


def _python_docstring(node: Node, source: bytes) -> str:
    """First line of the first string literal in the body. See finding 3.

    Both shapes are accepted because tree-sitter-python 0.26 parses a bare
    docstring as a ``string`` directly inside the ``block``, with no
    ``expression_statement`` wrapper -- the same grammar change that killed
    the ``@definition.constant`` rule in finding 6. Matching only the wrapped
    form returns an empty doc for every Python symbol in the index, and no
    test that merely counts symbols would notice.
    """
    body = node.child_by_field_name("body")
    if body is None or not body.named_child_count:
        return ""
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_child_count:
        first = first.named_children[0]
    if first.type != "string":
        return ""
    return _first_line(_string_content(first, source))


def _string_content(string_node: Node, source: bytes) -> str:
    for child in string_node.named_children:
        if child.type == "string_content":
            return _text(child, source)
    return _text(string_node, source).strip("\"'")


def _comment_doc(node: Node, source: bytes, caps: dict[str, list[Node]]) -> str:
    """Clean up the JS ``@doc`` capture by hand -- see finding 5.

    ``#strip!`` and ``#select-adjacent!`` are directives, not predicates, so
    tree-sitter does not apply them: the capture arrives as raw comment nodes,
    possibly several, and the adjacent one is the last.
    """
    doc_nodes = caps.get("doc")
    if doc_nodes:
        return _clean_comment(_text(doc_nodes[-1], source))
    return _leading_comment(node, source)


def _leading_comment(node: Node, source: bytes) -> str:
    """The doc comment of an *exported* declaration, which ``@doc`` never sees.

    The JS query anchors ``(comment)* @doc .`` directly against the
    declaration, but ``export class Client`` wraps that declaration in an
    ``export_statement`` -- so the comment is a sibling of the export, not of
    the class, and the capture comes back empty. Since nearly every documented
    symbol in a TS/JS module is exported, relying on ``@doc`` alone returns a
    blank doc for almost the entire public surface of a repository.
    """
    current = node
    while current.parent is not None and current.parent.type in _EXPORT_TYPES:
        current = current.parent

    sibling = current.prev_named_sibling
    if sibling is None or sibling.type != "comment":
        return ""
    if current.start_point[0] - sibling.end_point[0] > 1:
        return ""  # separated by a blank line: a file banner, not this symbol's doc.
    return _clean_comment(_text(sibling, source))


def _clean_comment(comment: str) -> str:
    """First meaningful line of a ``//`` or ``/** ... */`` comment."""
    if comment.startswith("/*"):
        comment = comment[2:]
        if comment.endswith("*/"):
            comment = comment[:-2]
    for raw in comment.splitlines():
        stripped = raw.strip().lstrip("*/").strip()
        if stripped:
            return _truncate(stripped, MAX_DOC_CHARS)
    return ""


def _is_exported(node: Node, qualified_name: str, language: str) -> bool:
    """Whether the symbol is part of the module's public surface.

    Two different conventions, deliberately: JS/TS has a real ``export``
    keyword, so it is checked syntactically. Python has none, so the
    underscore convention is used -- ``Client.request`` is public,
    ``Client._retry`` and ``_helper`` are not. It is a heuristic, and ranking
    weights it at 0.5 precisely because it is one.
    """
    if language == "python":
        return not any(part.startswith("_") for part in qualified_name.split("."))
    current: Node | None = node
    while current is not None:
        if current.type in _EXPORT_TYPES:
            return True
        current = current.parent
    return False


def _imports(node: Node, source: bytes, language: str) -> Iterator[Reference]:
    """Emit ``kind="import"`` references. See finding 4.

    No tags query captures imports in any of these languages, and imports are
    the single densest signal about which files depend on which -- the
    reference graph that Sprint 2's ranking runs over is mostly this.
    """
    if language == "python":
        if node.type not in _PYTHON_IMPORT_TYPES:
            return
        line = node.start_point[0] + 1
        for child in node.named_children:
            for name in _import_names(child, source):
                yield Reference(name=name, kind="import", line=line)
        return

    if node.type not in _JS_IMPORT_TYPES:
        return
    line = node.start_point[0] + 1

    if node.type == "call_expression":
        function = node.child_by_field_name("function")
        if function is None or _text(function, source) != "require":
            return
        args = node.child_by_field_name("arguments")
        if args is not None:
            for child in args.named_children:
                if child.type == "string":
                    yield Reference(name=_string_content(child, source), kind="import", line=line)
        return

    source_node = node.child_by_field_name("source")
    if source_node is not None and source_node.type == "string":
        yield Reference(name=_string_content(source_node, source), kind="import", line=line)
    for child in node.named_children:
        if child.type == "import_clause":
            for name in _import_names(child, source):
                yield Reference(name=name, kind="import", line=line)
        elif child.type == "export_clause" and source_node is not None:
            # `export { alpha } from "./greek"` is a re-export: this file
            # depends on `alpha` exactly as an import does, and the bindings
            # live in `export_clause`, not `import_clause`. The `source_node`
            # guard is what separates it from a plain `export { alpha }`,
            # where `alpha` is defined *in this file* and recording it as an
            # outbound reference would make the file depend on itself.
            for name in _import_names(child, source):
                yield Reference(name=name, kind="import", line=line)


def _import_names(node: Node, source: bytes) -> Iterator[str]:
    """Identifier-ish names bound by an import clause, dotted names included."""
    if node.type in {"dotted_name", "identifier", "type_identifier"}:
        text = _text(node, source)
        if text:
            yield text
        return
    if node.type in {"aliased_import", "import_specifier", "export_specifier"}:
        # `from typing import Any as AnyValue` records `Any`, not `AnyValue`,
        # and `import { Logger as Log }` records `Logger`, not `Log`. The
        # reference graph matches references against definitions in other
        # files, and it is the original name that is defined somewhere; the
        # local alias is defined nowhere and would never match anything.
        #
        # The JS/TS specifiers hold both identifiers as named children, so
        # recursing into them emitted the alias as well -- one bogus reference
        # per aliased import, each guaranteed to match nothing. Read the
        # `name` field instead of walking the children.
        target = node.child_by_field_name("name")
        if target is not None:
            yield from _import_names(target, source)
        return
    if node.type in {"wildcard_import", "string"}:
        return
    if node.type in {"namespace_import", "namespace_export"}:
        # `import * as NS from "./ns"` binds `NS` locally and nothing else;
        # there is no original name to match against, so the module string
        # already emitted is the whole of the dependency edge.
        return
    for child in node.named_children:
        yield from _import_names(child, source)


def _walk(node: Node) -> Iterator[Node]:
    """Depth-first over every named node, iteratively (deep files recurse far)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.named_children))


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return _truncate(stripped, MAX_DOC_CHARS)
    return ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def _line_count(source: bytes) -> int:
    if not source:
        return 0
    return source.count(b"\n") + (0 if source.endswith(b"\n") else 1)
