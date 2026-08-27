"""The only module that talks to the Anthropic API.

The SDK is imported lazily inside :func:`build_client`, so importing
``bench.client`` -- and therefore every module that imports it -- still works
in an environment that only installed the ``dev`` extra. That keeps the whole
harness under test in ordinary CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Cached from the pricing table, in dollars per million tokens. Used only for
# the pre-flight estimate printed before a batch spends anything; nothing in
# the results depends on it.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ApiClient:
    """Adapts the Anthropic SDK to the loop's :class:`MessageClient`.

    Note what is *not* here: no ``cache_control``, anywhere, in any form. The
    loop asserts the responses come back uncached; this is the other half of
    that guarantee.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)

    def count_tokens(self, **kwargs: Any) -> int:
        return int(self._client.messages.count_tokens(**kwargs).input_tokens)


@dataclass
class StubResponse:
    """A canned response with the shape the loop reads."""

    content: list[Any]
    stop_reason: str
    usage: Any


@dataclass
class _StubUsage:
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _StubTextBlock:
    text: str
    type: str = "text"


@dataclass
class StubClient:
    """A client that never calls the API, for ``--dry-run``.

    It answers every request with one text block and no tool calls, so the
    runner walks the entire matrix -- worktrees, indexing, grading, the result
    file -- without spending anything. This is what proves the plumbing before
    real money moves.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> StubResponse:
        self.calls.append(kwargs)
        return StubResponse(
            content=[_StubTextBlock(text="FILES:\n[dry run: no model was called]")],
            stop_reason="end_turn",
            usage=_StubUsage(),
        )

    def count_tokens(self, **kwargs: Any) -> int:
        return 0


def build_client(dry_run: bool = False) -> Any:
    """Return a client for the loop: the real SDK, or the dry-run stub."""
    if dry_run:
        return StubClient()

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            "the anthropic SDK is not installed; run `uv sync --extra bench` "
            "(or use --dry-run, which never calls the API)"
        ) from exc

    return ApiClient(anthropic.Anthropic())


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Dollar estimate for a batch, or ``None`` for an unpriced model."""
    prices = PRICING.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
