from __future__ import annotations

import pytest

from bench.client import estimate_cost
from bench.loop import run_task
from bench.models import DEFAULT_THINKING_BUDGET, ModelProfile, profile_for
from tests.test_bench_loop import FakeClient, Response, Text


def test_current_models_get_adaptive_thinking_and_effort():
    extras = profile_for("claude-opus-5").request_extras("high", 16_000)
    assert extras["thinking"] == {"type": "adaptive"}
    assert extras["output_config"] == {"effort": "high"}


def test_haiku_gets_neither_effort_nor_adaptive_thinking():
    """Haiku 4.5 rejects both with a 400.

    This is the whole reason the profile table exists: the batch used to die
    on its first call, after the clone and the index had already run.
    """
    extras = profile_for("claude-haiku-4-5").request_extras("high", 16_000)
    assert "output_config" not in extras
    assert extras["thinking"] == {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET}


def test_an_unknown_model_is_sent_nothing_optional():
    """A self-hosted endpoint is named whatever its server calls it."""
    assert profile_for("qwen3-coder-30b").request_extras("high", 16_000) == {}


def test_a_thinking_budget_above_max_tokens_fails_in_process():
    profile = ModelProfile("tiny", thinking="budget", thinking_budget=8_000)
    with pytest.raises(ValueError, match="below max_tokens"):
        profile.request_extras("high", 4_000)


def test_an_unpriced_model_estimates_nothing_rather_than_zero():
    assert estimate_cost("qwen3-coder-30b", 1_000_000, 1_000_000) is None
    assert estimate_cost("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_the_loop_sends_the_profile_of_the_model_it_was_given():
    client = FakeClient([Response(content=[Text("done")])])
    run_task(
        client,
        system="s",
        tools=[],
        prompt="p",
        execute=lambda name, args: None,  # type: ignore[arg-type,return-value]
        model="claude-haiku-4-5",
    )
    request = client.requests[0]
    assert "output_config" not in request
    assert request["thinking"]["type"] == "enabled"
