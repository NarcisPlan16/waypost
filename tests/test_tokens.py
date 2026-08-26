from __future__ import annotations

import logging

import pytest

from waypost.tokens import (
    TRUNCATION_MARKER,
    HeuristicTokenizer,
    TiktokenTokenizer,
    count,
    fit,
    fit_lines,
    get_tokenizer,
)

# Text with the shapes that actually go through `fit`: signatures, dotted
# names, punctuation runs, unicode, and a special-token string that would
# make a naive tiktoken call raise.
SAMPLE = "\n".join(
    [
        "src/waypost/client.py (412 loc)",
        "  class Client(BaseClient) :22 - HTTP client with retry and pooling.",
        "    def request(self, method, url, *, timeout=None) -> Response :58",
        "    async def _drain(self) -> None :141",
        "  MAX_RETRIES = 3 :17",
        "  def build_client(config: Config | None = None) -> Client :94 - Naïve façade.",
        "  MARKER = '<|endoftext|>' :12",
    ]
)


def _tokenizers():
    """Both implementations, so every property below is checked on each."""
    return [get_tokenizer("heuristic"), get_tokenizer()]


def test_heuristic_tokenizer_is_always_available_without_network():
    tokenizer = get_tokenizer("heuristic")
    assert isinstance(tokenizer, HeuristicTokenizer)
    assert tokenizer.name == "heuristic"
    assert tokenizer.count(SAMPLE) > 0


def test_unknown_encoding_falls_back_instead_of_raising(caplog):
    with caplog.at_level(logging.WARNING):
        tokenizer = get_tokenizer("no-such-encoding-42")
    assert isinstance(tokenizer, HeuristicTokenizer)
    assert "falling back" in caplog.text


def test_special_token_strings_do_not_raise():
    # tiktoken's default `encode` raises on "<|endoftext|>". Source code is
    # exactly the kind of text that contains one, so counting must not care.
    assert count("<|endoftext|> and <|fim_prefix|>") > 0


def test_heuristic_does_not_undercount_tiktoken():
    # The fallback is only safe if it errs toward over-counting: rendering
    # less than the budget allowed is a wasted opportunity, rendering more
    # is a broken promise.
    #
    # Skipped where tiktoken cannot load at all -- an offline machine with
    # no cached BPE table is a supported way to run this tool, not a
    # failing test.
    if not isinstance(get_tokenizer(), TiktokenTokenizer):
        pytest.skip("tiktoken is unavailable here; nothing to compare against")
    exact = TiktokenTokenizer()
    approx = HeuristicTokenizer()
    for text in (SAMPLE, *SAMPLE.splitlines(), "x", "", "a_very_long_identifier_name_here"):
        assert approx.count(text) >= exact.count(text), text


@pytest.mark.parametrize("budget", [0, 1, 3, 7, 12, 25, 60, 200, 10_000])
def test_fit_never_exceeds_the_budget(budget):
    for tokenizer in _tokenizers():
        out = fit(SAMPLE, budget, tokenizer=tokenizer)
        assert tokenizer.count(out) <= budget, (budget, tokenizer.name, out)


def test_fit_returns_the_text_unchanged_when_it_fits():
    assert fit(SAMPLE, 10_000) == SAMPLE


def test_fit_cuts_on_line_boundaries_only():
    original = SAMPLE.splitlines()
    kept = [line for line in fit(SAMPLE, 40).splitlines() if line != TRUNCATION_MARKER]
    assert kept == original[: len(kept)]


def test_the_marker_is_paid_for_out_of_the_budget():
    budget = 40
    out = fit(SAMPLE, budget)
    assert out.endswith(TRUNCATION_MARKER)
    # Not "content within budget, plus a marker on top".
    assert count(out) <= budget


def test_a_budget_too_small_for_the_marker_drops_the_marker_not_the_content():
    out = fit(SAMPLE, 6)
    assert TRUNCATION_MARKER not in out
    assert count(out) <= 6


def test_zero_and_negative_budgets_render_nothing():
    assert fit(SAMPLE, 0) == ""
    assert fit(SAMPLE, -5) == ""
    assert fit_lines(SAMPLE.splitlines(), 0) == []


def test_fit_lines_flags_truncation_by_its_last_element():
    lines = SAMPLE.splitlines()
    assert fit_lines(lines, 10_000) == lines
    truncated = fit_lines(lines, 40)
    assert truncated[-1] == TRUNCATION_MARKER
    assert truncated[:-1] == lines[: len(truncated) - 1]


def test_budgets_are_monotone_once_the_marker_fits():
    # A bigger budget never returns less content. Without this, `--budget`
    # is a knob whose direction an agent cannot reason about.
    #
    # Compared only across budgets that actually emitted the marker. The one
    # step where that is *not* true is real and deliberate: a budget just
    # big enough to afford the marker spends part of itself on it, so a
    # single content line can be displaced by the notice that content was
    # displaced. Silent truncation is the worse failure of the two, so the
    # marker wins that trade.
    previous = 0
    for budget in range(1, 140):
        out = fit(SAMPLE, budget)
        if TRUNCATION_MARKER not in out:
            continue
        kept = [line for line in out.splitlines() if line != TRUNCATION_MARKER]
        assert len(kept) >= previous, budget
        previous = len(kept)
    assert previous > 0
