"""Budget-aware rendering of index queries into compact text.

Responsibilities (Sprint 3):
    - ``map``: fill a token budget in rank order -- highest-ranked files
      first, and within a file, classes and exported functions before
      private helpers. Truncate at a symbol boundary, never mid-symbol.
    - ``find`` / ``show`` / ``refs`` / ``outline`` renderers.

The rule that makes this tool worth using: every default output is
token-frugal, and budgets are *measured* with the tokenizer rather than
estimated. ``show`` prints only a symbol's own span, never the surrounding
file. If any command can dump a whole file into an agent's context, the tool
is worse than grep and the benchmark will say so.

Output shape for ``map`` -- indentation and ``:line`` suffixes carry the
structure, so no tokens are spent on JSON punctuation::

    src/client.py (412 loc)
      class Client(BaseClient) :22 - HTTP client with retry and pooling.
        def request(self, method, url, *, timeout=None) -> Response :58
"""
