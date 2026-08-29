from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from bench.loop import CacheLeakError, TurnUsage, run_task
from bench.tools import ToolOutcome


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Text:
    text: str
    type: str = "text"


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class Response:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


class FakeClient:
    """Replays a scripted list of responses, recording what it was sent."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        # Snapshot the message list: the loop appends to it in place, so
        # keeping the reference would record every request as the final state.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if self._responses:
            return self._responses.pop(0)
        return Response(content=[Text("done")])


def echo_executor(name: str, tool_input: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(f"{name}:{sorted(tool_input)}")


def run(client, execute=echo_executor, **kwargs):
    return run_task(
        client,
        system="sys",
        tools=[{"name": "bash"}],
        prompt="do the thing",
        execute=execute,
        **kwargs,
    )


def test_total_input_sums_every_billed_input_field():
    usage = TurnUsage(
        input_tokens=100,
        cache_read_input_tokens=10,
        cache_creation_input_tokens=5,
        output_tokens=7,
    )
    assert usage.total_input == 115


def test_a_plain_answer_ends_the_loop_and_is_captured():
    client = FakeClient(
        [Response(content=[Text("FILES:\nsrc/a.py")], usage=Usage(input_tokens=42))]
    )
    result = run(client)

    assert result.turns == 1
    assert result.completed
    assert result.final_text == "FILES:\nsrc/a.py"
    assert result.input_tokens == 42


def test_tool_calls_are_executed_and_counted():
    client = FakeClient(
        [
            Response(
                content=[ToolUse(id="t1", name="waypost", input={"command": "map"})],
                stop_reason="tool_use",
                usage=Usage(input_tokens=100, output_tokens=10),
            ),
            Response(content=[Text("done")], usage=Usage(input_tokens=250, output_tokens=5)),
        ]
    )
    result = run(client)

    assert result.turns == 2
    assert result.tool_calls == {"waypost": 1}
    # Both turns are billed, and the second turn is larger because the tool
    # result was resent as input. That is exactly how waypost's own output ends
    # up charged to the treatment arm without any hand-counting.
    assert result.input_tokens == 350
    assert result.output_tokens == 15


def test_parallel_tool_results_go_back_in_one_user_message():
    client = FakeClient(
        [
            Response(
                content=[
                    ToolUse(id="t1", name="grep", input={"pattern": "x"}),
                    ToolUse(id="t2", name="read_file", input={"path": "a.py"}),
                ],
                stop_reason="tool_use",
            ),
            Response(content=[Text("done")]),
        ]
    )
    run(client)

    # Splitting these across two messages trains the model out of parallel
    # calls, which changes turn counts between arms for no real reason.
    second_request = client.requests[1]["messages"]
    tool_result_messages = [
        m
        for m in second_request
        if m["role"] == "user" and isinstance(m["content"], list) and m["content"]
    ]
    assert len(tool_result_messages) == 1
    assert [block["tool_use_id"] for block in tool_result_messages[0]["content"]] == ["t1", "t2"]


def test_a_failing_tool_is_reported_as_an_error_result_and_the_loop_continues():
    def failing(name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome("Error: nope.", is_error=True)

    client = FakeClient(
        [
            Response(
                content=[ToolUse(id="t1", name="bash", input={"command": "false"})],
                stop_reason="tool_use",
            ),
            Response(content=[Text("recovered")]),
        ]
    )
    result = run(client, execute=failing)

    assert result.completed
    block = client.requests[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True


def test_turn_cap_records_a_failure_rather_than_raising():
    looping = [
        Response(
            content=[ToolUse(id=f"t{i}", name="bash", input={"command": "ls"})],
            stop_reason="tool_use",
        )
        for i in range(10)
    ]
    client = FakeClient(looping)
    result = run(client, turn_cap=5)

    assert result.turns == 5
    assert result.failure_reason == "turn_cap"
    assert not result.completed


def test_cached_tokens_abort_the_run():
    # A cached arm is the easiest way to manufacture a large fake reduction, so
    # this must be loud rather than a flag on the record.
    client = FakeClient(
        [Response(content=[Text("hi")], usage=Usage(input_tokens=10, cache_read_input_tokens=1))]
    )
    with pytest.raises(CacheLeakError):
        run(client)


def test_no_request_ever_carries_cache_control():
    client = FakeClient([Response(content=[Text("hi")])])
    run(client)
    assert "cache_control" not in client.requests[0]


def test_a_refusal_is_recorded_as_a_failure():
    client = FakeClient([Response(content=[Text("")], stop_reason="refusal")])
    result = run(client)
    assert result.failure_reason == "refusal"


def test_hitting_max_tokens_is_a_failure_even_though_text_came_back():
    client = FakeClient([Response(content=[Text("half an ans")], stop_reason="max_tokens")])
    result = run(client)
    assert result.failure_reason == "max_tokens"
    assert result.final_text == "half an ans"


def test_the_trace_records_each_call_in_order_with_its_output_size():
    # The counts alone cannot say whether the model queried waypost and then
    # read the file anyway. The ordered trace can, which is the whole point.
    client = FakeClient(
        [
            Response(
                content=[
                    ToolUse(id="t1", name="waypost", input={"command": "find", "args": "Foo"})
                ],
                stop_reason="tool_use",
            ),
            Response(
                content=[ToolUse(id="t2", name="read_file", input={"path": "src/foo.py"})],
                stop_reason="tool_use",
            ),
            Response(content=[Text("done")]),
        ]
    )
    result = run(client, execute=lambda name, _inp: ToolOutcome("x" * 12))

    assert [(c.turn, c.name, c.arg) for c in result.trace] == [
        (1, "waypost", "find Foo"),
        (2, "read_file", "src/foo.py"),
    ]
    assert [c.output_bytes for c in result.trace] == [12, 12]
    assert not any(c.is_error for c in result.trace)


def test_the_trace_marks_a_failed_call():
    client = FakeClient(
        [
            Response(
                content=[ToolUse(id="t1", name="bash", input={"command": "false"})],
                stop_reason="tool_use",
            ),
            Response(content=[Text("done")]),
        ]
    )
    result = run(client, execute=lambda name, _inp: ToolOutcome("boom", is_error=True))

    assert [(c.name, c.arg, c.is_error) for c in result.trace] == [("bash", "false", True)]


def test_a_long_argument_is_truncated_so_a_forty_turn_trace_stays_readable():
    client = FakeClient(
        [
            Response(
                content=[ToolUse(id="t1", name="bash", input={"command": "echo " + "a" * 500})],
                stop_reason="tool_use",
            ),
            Response(content=[Text("done")]),
        ]
    )
    result = run(client)

    assert len(result.trace[0].arg) == 200
    assert result.trace[0].arg.endswith("...")
