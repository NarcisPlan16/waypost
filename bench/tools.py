"""The tool surface the model under test is given, and its executor.

Both arms share the same four baseline tools. The treatment arm adds exactly
one more, ``waypost``, which shells out to the installed CLI.

Two deliberate choices here decide whether the benchmark is fair:

- **waypost is one tool, not seven.** Every tool definition is resent on every
  request, so seven schemas would charge the treatment arm a fixed per-turn
  tax that has nothing to do with whether the tool saves tokens. One tool with
  a ``command`` argument keeps the definition cost comparable between arms.
- **Output is truncated at the same byte cap in both arms.** The cap is part
  of the measured quantity: a baseline arm allowed to dump unbounded file
  contents into context would look artificially bad.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Big enough that a real file read is not crippled, small enough that a
# runaway `cat` on a vendored bundle cannot dominate a run's token count.
MAX_TOOL_OUTPUT_BYTES = 20_000
TOOL_TIMEOUT_S = 120

WAYPOST_COMMANDS = ("map", "find", "show", "refs", "outline", "stats")


@dataclass(frozen=True)
class ToolOutcome:
    """What the executor hands back for one ``tool_use`` block."""

    content: str
    is_error: bool = False


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_TOOL_OUTPUT_BYTES:
        return text
    kept = encoded[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", errors="ignore")
    dropped = len(encoded) - MAX_TOOL_OUTPUT_BYTES
    return f"{kept}\n... [truncated, {dropped} more bytes]"


BASELINE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the repository root and return its combined "
            "stdout and stderr. Use it for anything the other tools do not cover, "
            "including running the test suite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the repository, optionally a line range of it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repository root."},
                "offset": {"type": "integer", "description": "First line to read, 1-indexed."},
                "limit": {"type": "integer", "description": "How many lines to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": "Search the repository for a regular expression and return matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "Directory or file to search under."},
                "glob": {"type": "string", "description": "Only search files matching this glob."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file. old_string must appear exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the repository root."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Text to replace it with."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]

WAYPOST_TOOL: dict[str, Any] = {
    "name": "waypost",
    "description": (
        "Query the repository's waypost index: a symbol-level map of the codebase "
        "built with tree-sitter, which never returns a whole file. command is one of: "
        "map (ranked symbol map of the repo; accepts --budget N and --focus PATH), "
        "find (locate symbols by name, substring or glob), "
        "show (one symbol's own source span), "
        "refs (where a symbol is defined and which files reference it), "
        "outline (every symbol in one file), "
        "stats (what the index holds). "
        "args are the remaining command-line arguments, already split."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": list(WAYPOST_COMMANDS),
                "description": "The waypost subcommand to run.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments for the subcommand, for example --budget 2000.",
            },
        },
        "required": ["command"],
    },
}


def tools_for_arm(arm: str) -> list[dict[str, Any]]:
    """Tool definitions for ``baseline`` or ``treatment``."""
    if arm == "baseline":
        return [dict(tool) for tool in BASELINE_TOOLS]
    if arm == "treatment":
        return [dict(tool) for tool in BASELINE_TOOLS] + [dict(WAYPOST_TOOL)]
    raise ValueError(f"unknown arm {arm!r}")


class Executor:
    """Runs tool calls against one worktree.

    Every path is resolved and checked to be inside the root: a task that
    wandered into the clone cache would contaminate the next run's tree.
    """

    def __init__(self, root: Path, waypost_cmd: list[str] | None = None) -> None:
        self.root = root.resolve()
        self.waypost_cmd = waypost_cmd or default_waypost_cmd()
        self.calls: dict[str, int] = {}

    def __call__(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        self.calls[name] = self.calls.get(name, 0) + 1
        handler = {
            "bash": self._bash,
            "read_file": self._read_file,
            "grep": self._grep,
            "edit_file": self._edit_file,
            "waypost": self._waypost,
        }.get(name)
        if handler is None:
            return ToolOutcome(f"Error: unknown tool {name!r}.", is_error=True)
        try:
            return handler(tool_input)
        except subprocess.TimeoutExpired:
            return ToolOutcome(f"Error: {name} timed out after {TOOL_TIMEOUT_S}s.", is_error=True)
        except OSError as exc:
            return ToolOutcome(f"Error: {name} failed: {exc}", is_error=True)

    # -- path safety -----------------------------------------------------

    def _resolve(self, raw: Any) -> Path:
        if not isinstance(raw, str) or raw == "":
            raise ValueError("path must be a non-empty string")
        candidate = (self.root / raw).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path {raw!r} escapes the repository root")
        return candidate

    # -- handlers --------------------------------------------------------

    def _run(self, argv: list[str], shell_command: str | None = None) -> ToolOutcome:
        completed = subprocess.run(
            shell_command if shell_command is not None else argv,
            shell=shell_command is not None,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_S,
            env=child_env(),
        )
        output = "".join(part for part in (completed.stdout, completed.stderr) if part)
        if completed.returncode != 0:
            output = f"{output}\n[exit code {completed.returncode}]".lstrip("\n")
        return ToolOutcome(_truncate(output) or "[no output]")

    def _bash(self, tool_input: dict[str, Any]) -> ToolOutcome:
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolOutcome("Error: command must be a non-empty string.", is_error=True)
        return self._run([], shell_command=command)

    def _read_file(self, tool_input: dict[str, Any]) -> ToolOutcome:
        raw_path = tool_input.get("path")
        try:
            path = self._resolve(raw_path)
        except ValueError as exc:
            return ToolOutcome(f"Error: {exc}", is_error=True)
        if not path.is_file():
            return ToolOutcome(f"Error: {raw_path} is not a file.", is_error=True)

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        start = max(1, int(offset)) if isinstance(offset, int) else 1
        end = start + int(limit) if isinstance(limit, int) and limit > 0 else len(lines) + 1
        selected = lines[start - 1 : end - 1]
        numbered = "\n".join(f"{start + i}\t{line}" for i, line in enumerate(selected))
        return ToolOutcome(_truncate(numbered) or "[empty]")

    def _grep(self, tool_input: dict[str, Any]) -> ToolOutcome:
        pattern = tool_input.get("pattern")
        if not isinstance(pattern, str) or pattern == "":
            return ToolOutcome("Error: pattern must be a non-empty string.", is_error=True)

        argv = ["grep", "-rnI", "--exclude-dir=.git"]
        glob = tool_input.get("glob")
        if isinstance(glob, str) and glob:
            argv.append(f"--include={glob}")
        argv += ["-e", pattern]

        target = tool_input.get("path")
        if isinstance(target, str) and target:
            try:
                self._resolve(target)
            except ValueError as exc:
                return ToolOutcome(f"Error: {exc}", is_error=True)
            argv.append(target)
        else:
            argv.append(".")

        completed = subprocess.run(
            argv,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_S,
            env=child_env(),
        )
        # grep exits 1 for "no matches", which is an answer, not a failure.
        if completed.returncode > 1:
            return ToolOutcome(_truncate(completed.stderr) or "Error: grep failed.", is_error=True)
        return ToolOutcome(_truncate(completed.stdout) or "[no matches]")

    def _edit_file(self, tool_input: dict[str, Any]) -> ToolOutcome:
        raw_path = tool_input.get("path")
        try:
            path = self._resolve(raw_path)
        except ValueError as exc:
            return ToolOutcome(f"Error: {exc}", is_error=True)
        old = tool_input.get("old_string")
        new = tool_input.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return ToolOutcome("Error: old_string and new_string must be strings.", is_error=True)
        if not path.is_file():
            return ToolOutcome(f"Error: {raw_path} is not a file.", is_error=True)

        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old)
        if occurrences == 0:
            return ToolOutcome("Error: old_string not found in the file.", is_error=True)
        if occurrences > 1:
            return ToolOutcome(
                f"Error: old_string appears {occurrences} times; make it unique.", is_error=True
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolOutcome("Edit applied.")

    def _waypost(self, tool_input: dict[str, Any]) -> ToolOutcome:
        command = tool_input.get("command")
        if command not in WAYPOST_COMMANDS:
            return ToolOutcome(f"Error: unknown waypost command {command!r}.", is_error=True)
        raw_args = tool_input.get("args") or []
        if not isinstance(raw_args, list) or not all(isinstance(a, str) for a in raw_args):
            return ToolOutcome("Error: args must be an array of strings.", is_error=True)
        return self._run([*self.waypost_cmd, str(command), *raw_args])


def default_waypost_cmd() -> list[str]:
    """How to invoke waypost as a subprocess.

    Prefers the console script, falling back to the current interpreter's
    module entry point so a venv whose Scripts directory is not on PATH still
    works -- the same failure mode the npm wrapper had to handle.
    """
    found = shutil.which("waypost")
    if found:
        return [found]
    return [sys.executable, "-m", "waypost"]


def child_env() -> dict[str, str]:
    """Environment for tool subprocesses.

    The API credentials are stripped: nothing a benchmark task runs should be
    able to spend money, and a task that finds a key is a task that can cheat.
    """
    env = dict(os.environ)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env
