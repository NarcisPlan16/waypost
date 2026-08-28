"""The agentic loop under measurement.

This is deliberately small and deliberately ours. The roadmap forbids
benchmarking through Claude Code: its context management and caching are
opaque, so any reduction measured through it is unattributable. What is
measured here is exactly what this file sends and exactly what the API says it
cost.

Nothing in this module imports the Anthropic SDK. It talks to a
:class:`MessageClient`, which ``bench.client`` implements over the real SDK and
the tests implement with a scripted fake. That is what makes the turn cap,
the token accounting and the cache-leak guard testable offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from bench.models import ModelProfile, profile_for
from bench.tools import ToolOutcome

# From the roadmap: a run that has not finished in 40 turns is a failure, and
# is recorded as one. Silently dropping it would bias whichever arm stalls.
TURN_CAP = 40
MAX_TOKENS = 16_000
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"

# A paused turn is not expected without server tools, but resuming costs one
# line and a silent truncation would look like a short, cheap, wrong run.
MAX_PAUSE_RESUMES = 5


class MessageClient(Protocol):
    """The single API call the loop needs."""

    def create(self, **kwargs: Any) -> Any:  # pragma: no cover - structural type
        ...


ToolExecutor = Callable[[str, dict[str, Any]], ToolOutcome]


class CacheLeakError(RuntimeError):
    """A response reported cached tokens.

    Both arms must run uncached. A cached arm produces a large, entirely fake
    reduction, so this aborts the batch rather than marking one run bad.
    """


@dataclass(frozen=True)
class TurnUsage:
    """What one API response reported it cost."""

    input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens: int

    @property
    def total_input(self) -> int:
        """Every input token the request was billed for.

        Summed from what the API reports rather than counted by hand. This is
        also why waypost's own output is charged to the treatment arm without
        any special handling: a tool result is resent as input on every
        subsequent turn, so it is already in these numbers.
        """
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class LoopResult:
    """The measured outcome of one task run, before grading."""

    turns: int = 0
    usage: list[TurnUsage] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)
    final_text: str = ""
    stop_reason: str | None = None
    failure_reason: str | None = None
    wall_clock_s: float = 0.0

    @property
    def input_tokens(self) -> int:
        return sum(turn.total_input for turn in self.usage)

    @property
    def output_tokens(self) -> int:
        return sum(turn.output_tokens for turn in self.usage)

    @property
    def completed(self) -> bool:
        """Did the loop end because the model was done, rather than cut off?"""
        return self.failure_reason is None


def _usage_of(response: Any) -> TurnUsage:
    usage = getattr(response, "usage", None)

    def field_of(name: str) -> int:
        value = getattr(usage, name, 0) if usage is not None else 0
        return int(value or 0)

    return TurnUsage(
        input_tokens=field_of("input_tokens"),
        cache_read_input_tokens=field_of("cache_read_input_tokens"),
        cache_creation_input_tokens=field_of("cache_creation_input_tokens"),
        output_tokens=field_of("output_tokens"),
    )


def _text_of(response: Any) -> str:
    parts = [
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(parts).strip()


def run_task(
    client: MessageClient,
    *,
    system: str,
    tools: list[dict[str, Any]],
    prompt: str,
    execute: ToolExecutor,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = MAX_TOKENS,
    turn_cap: int = TURN_CAP,
    profile: ModelProfile | None = None,
) -> LoopResult:
    """Drive one task to completion, a turn cap, or a failure.

    Returns a :class:`LoopResult` in every case except a cache leak, which
    raises: a wrong number is worse than a missing one.
    """
    # Resolved once, outside the turn loop: the request shape cannot change
    # part-way through a run without making the run's token total meaningless.
    extras = (profile or profile_for(model)).request_extras(effort, max_tokens)

    result = LoopResult()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    started = time.monotonic()
    pause_resumes = 0

    while True:
        if result.turns >= turn_cap:
            result.failure_reason = "turn_cap"
            break

        response = client.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            **extras,
        )
        result.turns += 1

        turn_usage = _usage_of(response)
        result.usage.append(turn_usage)
        if turn_usage.cache_read_input_tokens or turn_usage.cache_creation_input_tokens:
            raise CacheLeakError(
                "response reported cached tokens "
                f"(read={turn_usage.cache_read_input_tokens}, "
                f"created={turn_usage.cache_creation_input_tokens}); "
                "both arms must run uncached"
            )

        stop_reason = getattr(response, "stop_reason", None)
        result.stop_reason = stop_reason

        # Guarded before the content is read: a refusal carries no answer.
        if stop_reason == "refusal":
            result.failure_reason = "refusal"
            break

        content = list(getattr(response, "content", []))
        messages.append({"role": "assistant", "content": content})

        if stop_reason == "pause_turn":
            pause_resumes += 1
            if pause_resumes > MAX_PAUSE_RESUMES:
                result.failure_reason = "pause_turn"
                break
            continue

        tool_uses = [block for block in content if getattr(block, "type", None) == "tool_use"]
        if not tool_uses:
            result.final_text = _text_of(response)
            if stop_reason == "max_tokens":
                result.failure_reason = "max_tokens"
            break

        # All results go back in a single user message. Splitting them across
        # messages trains the model out of parallel calls, which would change
        # turn counts between arms for a reason unrelated to the tool.
        tool_results: list[dict[str, Any]] = []
        for block in tool_uses:
            name = str(getattr(block, "name", ""))
            tool_input = dict(getattr(block, "input", {}) or {})
            outcome = execute(name, tool_input)
            result.tool_calls[name] = result.tool_calls.get(name, 0) + 1
            entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": getattr(block, "id", ""),
                "content": outcome.content,
            }
            if outcome.is_error:
                entry["is_error"] = True
            tool_results.append(entry)

        messages.append({"role": "user", "content": tool_results})

    result.wall_clock_s = time.monotonic() - started
    return result
