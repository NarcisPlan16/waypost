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
   ordinary TypeScript it yields ZERO definitions.

   The fix, verified working, is to concatenate the two queries and run the
   result against the TypeScript parser::

       merged = get_tags_query("javascript") + "\\n" + get_tags_query("typescript")

   Apply the same merge for ``.tsx``.

3. ``@doc`` is JavaScript-only. The JS query captures leading comments as
   ``@doc``; the Python query has no ``@doc`` at all, so Python docstrings
   must be pulled from the AST separately (first string literal in the body).

4. ``refs`` will be sparse. Python offers only ``@reference.call``; TS adds
   ``@reference.type`` and ``@reference.class``. Neither covers imports.
   Ranking quality depends on the reference graph, so either supplement refs
   from import statements or document the sparsity explicitly -- silently
   under-populating it degrades ``map`` output in a way that will not show up
   until the benchmark.

Capture names observed for Python::

    @definition.class  @definition.function  @definition.constant
    @name              @reference.call
"""
