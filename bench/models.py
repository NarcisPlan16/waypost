"""Per-model request shape, and the pricing used for the pre-flight estimate.

The loop sends one request shape, but not every model accepts it. Adaptive
thinking and ``output_config.effort`` are current-generation parameters:
Claude Haiku 4.5 rejects both with a 400, so a batch launched against it used
to die on its first call, after cloning and indexing. Hard-coding the newest
shape silently restricts the benchmark to the most expensive models, which is
exactly backwards for a project whose claim is that it saves money.

So the request shape is a property of the model, declared here. An unknown
model gets the conservative profile -- no effort, no thinking, no price -- on
the principle that a benchmark run should fail because the *tool* is wrong,
never because the harness sent a parameter the endpoint had never heard of.

Note what a profile deliberately does not carry: anything that differs
*between arms*. Both arms of a comparison always run the same profile, so
nothing in here can move the measured difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ThinkingMode = Literal["adaptive", "budget", "off"]

# Prices are dollars per million tokens, cached from the pricing table. Only
# the pre-flight estimate reads them; no recorded result depends on a price,
# so a stale entry here can never corrupt a published number.
DEFAULT_THINKING_BUDGET = 4096


@dataclass(frozen=True)
class ModelProfile:
    """What one model accepts, and what it costs."""

    id: str
    thinking: ThinkingMode = "off"
    supports_effort: bool = False
    thinking_budget: int | None = None
    context_window: int | None = None
    input_price: float | None = None
    output_price: float | None = None

    def request_extras(self, effort: str, max_tokens: int) -> dict[str, Any]:
        """The model-dependent half of the request body.

        ``effort`` is accepted even when unsupported rather than rejected: the
        CLI has a default, and refusing to run Haiku because a default the
        user never typed does not apply to it would be obstructive. What must
        not happen is *sending* it.
        """
        extras: dict[str, Any] = {}
        if self.supports_effort:
            extras["output_config"] = {"effort": effort}
        if self.thinking == "adaptive":
            extras["thinking"] = {"type": "adaptive"}
        elif self.thinking == "budget":
            budget = self.thinking_budget or DEFAULT_THINKING_BUDGET
            # The API requires budget_tokens < max_tokens; a 400 here would
            # land after the clone and the index, so catch it in-process.
            if budget >= max_tokens:
                raise ValueError(
                    f"{self.id}: thinking budget {budget} must be below max_tokens {max_tokens}"
                )
            extras["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return extras


# Current-generation models take adaptive thinking and effort. Haiku 4.5 is a
# prior-generation model: effort errors, and thinking needs an explicit token
# budget. Both facts are why this table exists.
PROFILES: dict[str, ModelProfile] = {
    "claude-opus-5": ModelProfile(
        "claude-opus-5",
        thinking="adaptive",
        supports_effort=True,
        context_window=1_000_000,
        input_price=5.0,
        output_price=25.0,
    ),
    "claude-opus-4-8": ModelProfile(
        "claude-opus-4-8",
        thinking="adaptive",
        supports_effort=True,
        context_window=1_000_000,
        input_price=5.0,
        output_price=25.0,
    ),
    "claude-sonnet-5": ModelProfile(
        "claude-sonnet-5",
        thinking="adaptive",
        supports_effort=True,
        context_window=1_000_000,
        input_price=2.0,
        output_price=10.0,
    ),
    "claude-haiku-4-5": ModelProfile(
        "claude-haiku-4-5",
        thinking="budget",
        supports_effort=False,
        thinking_budget=DEFAULT_THINKING_BUDGET,
        context_window=200_000,
        input_price=1.0,
        output_price=5.0,
    ),
}


def profile_for(model: str) -> ModelProfile:
    """The profile for *model*, or a conservative one for an unknown id.

    Unknown ids are expected, not exceptional: a local or self-hosted endpoint
    is named whatever its server calls it. Such a model gets no effort, no
    thinking and no price, which every chat endpoint accepts.
    """
    known = PROFILES.get(model)
    if known is not None:
        return known
    return ModelProfile(model)
