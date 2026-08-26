"""Tokenizer wrapper and hard budget enforcement.

Responsibilities (Sprint 3):
    - Wrap ``tiktoken`` behind a small interface so the tokenizer is
      swappable and named in the index metadata.
    - Provide the single ``fit(text, budget)`` helper that every renderer
      routes through, so budget enforcement lives in exactly one place.

Budgets are measured, not estimated: ``map --budget N`` must never exceed N
tokens as counted by the configured tokenizer.

Two things worth knowing before changing anything here.

**The count is exact for the configured tokenizer, and an approximation of
any other.** ``cl100k_base`` is not Claude's tokenizer, and nothing published
is. What a budget therefore guarantees is a hard, reproducible ceiling under
a named tokenizer -- which is what makes ``map --budget 2000`` a measurement
rather than a guess -- not a byte-exact prediction of some other model's
context accounting. Every output that quotes a budget also names the
tokenizer that measured it, so a number can always be traced to what
produced it.

**``tiktoken`` downloads its BPE table on first use.** That is a network
call, and this package promises none. It is a one-time fetch into
``tiktoken``'s own cache, not per-invocation telemetry, but on a machine
with no cache and no network it fails -- so a failure to load is not an
error here. :func:`get_tokenizer` falls back to :class:`HeuristicTokenizer`
and logs which one it got. Set ``WAYPOST_TOKENIZER=heuristic`` to skip the
``tiktoken`` attempt entirely (offline CI, air-gapped machines), or set it
to any ``tiktoken`` encoding name to pin a different one.

The fallback deliberately **over**-counts rather than risking an
under-count: rendering less than the budget allowed is a wasted opportunity,
rendering more is a broken promise.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Sequence
from functools import cache
from typing import Protocol

logger = logging.getLogger(__name__)

# The tiktoken encoding used unless WAYPOST_TOKENIZER says otherwise.
DEFAULT_ENCODING = "cl100k_base"

# Environment override: an encoding name, or "heuristic" to force the
# offline fallback without attempting to load tiktoken at all.
TOKENIZER_ENV_VAR = "WAYPOST_TOKENIZER"

HEURISTIC_NAME = "heuristic"

# Appended when `fit` had to cut something, so a truncated map is never
# mistaken for a complete one. It is counted as part of the output it is
# appended to -- the marker lives inside the budget, not on top of it.
TRUNCATION_MARKER = "... (truncated to fit the token budget)"


class Tokenizer(Protocol):
    """Minimal tokenizer interface: a name to report, and a way to count."""

    @property
    def name(self) -> str:  # pragma: no cover -- protocol declaration.
        ...

    def count(self, text: str) -> int:  # pragma: no cover -- protocol declaration.
        ...


class TiktokenTokenizer:
    """Exact counting via a named ``tiktoken`` encoding."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        import tiktoken

        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return f"tiktoken/{self._encoding_name}"

    def count(self, text: str) -> int:
        # `disallowed_special=()` matters: tiktoken raises by default when the
        # text contains a special-token string such as `<|endoftext|>`, and
        # source code is exactly the kind of text that contains one. Counting
        # a repository must never blow up on a string literal.
        return len(self._encoding.encode(text, disallowed_special=()))


class HeuristicTokenizer:
    """Offline fallback that over-counts rather than risking an under-count.

    Words are charged ``ceil(len / 3)`` tokens, punctuation one each, and a
    run of whitespace one. Real BPE merges common words into a single token
    and splits long identifiers into several; charging every word by length
    is therefore an over-estimate on prose and roughly right on the
    identifier-dense text this tool actually renders.

    It is deterministic, has no data files and never touches the network,
    which is what makes it a safe default when ``tiktoken`` cannot load.
    """

    # Word runs, single non-space symbols, and whitespace runs. Every
    # character of the input falls into exactly one of the three, so the
    # count never silently drops part of the text.
    _PIECE_RE = re.compile(r"\w+|[^\w\s]|\s+")

    # Characters per token charged for a word run.
    _CHARS_PER_TOKEN = 3

    @property
    def name(self) -> str:
        return HEURISTIC_NAME

    def count(self, text: str) -> int:
        total = 0
        for piece in self._PIECE_RE.findall(text):
            is_word = piece[0].isalnum() or piece[0] == "_"
            total += math.ceil(len(piece) / self._CHARS_PER_TOKEN) if is_word else 1
        return total


@cache
def get_tokenizer(name: str | None = None) -> Tokenizer:
    """Return the configured tokenizer, cached for the process.

    ``name`` overrides the ``WAYPOST_TOKENIZER`` environment variable, which
    in turn overrides :data:`DEFAULT_ENCODING`. ``"heuristic"`` selects the
    offline fallback directly. Anything else is treated as a ``tiktoken``
    encoding name; if it cannot be loaded -- unknown name, no cached BPE
    table and no network -- this logs why and falls back rather than
    raising, because failing to count is not a reason to fail to answer.
    """
    requested = name or os.environ.get(TOKENIZER_ENV_VAR) or DEFAULT_ENCODING

    if requested == HEURISTIC_NAME:
        return HeuristicTokenizer()

    try:
        return TiktokenTokenizer(requested)
    except Exception as exc:  # Any failure at all means "use the fallback".
        logger.warning(
            "waypost: cannot load tiktoken encoding %r (%s); "
            "falling back to the %s tokenizer, whose counts are approximate",
            requested,
            exc,
            HEURISTIC_NAME,
        )
        return HeuristicTokenizer()


def count(text: str, *, tokenizer: Tokenizer | None = None) -> int:
    """Token count of ``text`` under the configured tokenizer."""
    return (tokenizer or get_tokenizer()).count(text)


def fit_lines(
    lines: Sequence[str],
    budget: int,
    *,
    tokenizer: Tokenizer | None = None,
    marker: str = TRUNCATION_MARKER,
) -> list[str]:
    """Longest prefix of ``lines`` that fits ``budget``, marker included.

    This is the one place a budget is enforced. Renderers choose *what* to
    emit and in what order -- one line per symbol, so a cut lands on a
    symbol boundary rather than mid-signature -- and this decides how much
    of it survives. "Never exceeds the budget" is therefore a property of
    this function, not of each renderer's arithmetic.

    Returns the kept lines, with ``marker`` appended as its own final line
    when anything was dropped. The marker is counted as part of the result
    rather than added on top of it; a budget too small to hold it drops the
    marker instead of the content. Callers can tell truncation happened by
    testing whether the last element equals ``marker``.
    """
    tk = tokenizer or get_tokenizer()

    if budget <= 0:
        return []

    if tk.count("\n".join(lines)) <= budget:
        return list(lines)

    # Whether the marker is affordable at all. Below its own cost, content
    # wins: a budget of four tokens should spend them on a signature.
    with_marker = bool(marker) and tk.count(marker) <= budget

    def assemble(keep: int) -> list[str]:
        out = list(lines[:keep])
        if with_marker:
            out.append(marker)
        return out

    # Largest `keep` whose *assembled* output fits. Counting the assembly
    # rather than the content plus a marker allowance is what makes the
    # guarantee exact: token counts are not additive across a join, so a
    # budget computed from parts can be a token or two wrong -- which, for
    # the one function whose entire job is not exceeding a number, is the
    # whole ballgame.
    low, high = 0, len(lines)
    while low < high:
        mid = (low + high + 1) // 2
        if tk.count("\n".join(assemble(mid))) <= budget:
            low = mid
        else:
            high = mid - 1

    return assemble(low)


def fit(
    text: str,
    budget: int,
    *,
    tokenizer: Tokenizer | None = None,
    marker: str = TRUNCATION_MARKER,
) -> str:
    """Return the longest whole-line prefix of ``text`` within ``budget``.

    Thin wrapper over :func:`fit_lines` for callers holding a finished
    string rather than the lines that produced it.
    """
    return "\n".join(fit_lines(text.split("\n"), budget, tokenizer=tokenizer, marker=marker))
