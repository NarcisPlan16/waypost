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
- **The baseline tools must actually work on the host.** They did not: `grep`
  shelled out to a binary that is absent from PATH under PowerShell, and
  `shell=True` on Windows is cmd.exe, so the agent's `grep -rn` failed too.
  Both failures land only on the arm that still has to search, and the Sonnet
  batch of 2026-08-29 measured 8 of 8 baseline greps erroring while treatment
  had none. A control that cannot search does not measure grep-and-read, it
  flatters the tool. `grep` is now implemented in-process, and the shell is
  resolved to a real POSIX one.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Big enough that a real file read is not crippled, small enough that a
# runaway `cat` on a vendored bundle cannot dominate a run's token count.
MAX_TOOL_OUTPUT_BYTES = 20_000

# Directories `grep -rnI --exclude-dir=.git` would have walked but should not.
# `.waypost` is here because the harness builds an index in *both* arms: a
# baseline grep that matches inside the tool-under-test's own generated JSON
# is measuring the benchmark, not the repository.
SKIP_DIRS = frozenset({".git", ".hg", ".svn", "node_modules", "__pycache__", ".waypost"})
TOOL_TIMEOUT_S = 120
# How long to wait for a killed process tree to release its pipes.
KILL_DRAIN_S = 10

WAYPOST_COMMANDS = ("map", "find", "show", "refs", "outline", "stats")


@dataclass(frozen=True)
class ToolOutcome:
    """What the executor hands back for one ``tool_use`` block."""

    content: str
    is_error: bool = False


def _walk_files(base: Path) -> Iterator[Path]:
    """Every file under *base*, skipping VCS and build directories."""
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            yield Path(root) / name


def _text_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Numbered lines of *path*, or nothing at all if it is binary.

    ``grep -I`` skips binary files, and so must this: a match inside a
    compiled artefact is noise, and its bytes would blow the output cap.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if b"\x00" in raw[:8192]:
        return
    text = raw.decode("utf-8", errors="replace")
    yield from enumerate(text.splitlines(), start=1)


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
        "find (a name's definitions, ranked; --all adds partial-name matches), "
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
        self.shell = posix_shell()

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
        # Prefer an explicit POSIX shell over ``shell=True``; see posix_shell.
        # Falling back to the platform shell keeps a host without one running,
        # but such a run is recorded as degraded rather than passed off as a
        # comparison against a working baseline.
        use_platform_shell = shell_command is not None and self.shell is None
        if shell_command is not None and self.shell is not None:
            command: Any = [self.shell, "-c", shell_command]
        elif shell_command is not None:
            command = shell_command
        else:
            command = argv

        code, out, err = _capture(command, shell=use_platform_shell, cwd=self.root)
        output = "".join(part for part in (out, err) if part)
        if code != 0:
            output = f"{output}\n[exit code {code}]".lstrip("\n")
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
        """``grep -rnI``, in-process.

        This used to shell out. It must not: the baseline arm's only search
        tool cannot depend on a binary that may be missing from PATH, and when
        it went missing the failure showed up as a large measured saving for
        waypost rather than as an error anyone would notice.
        """
        pattern = tool_input.get("pattern")
        if not isinstance(pattern, str) or pattern == "":
            return ToolOutcome("Error: pattern must be a non-empty string.", is_error=True)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolOutcome(f"Error: bad regular expression: {exc}", is_error=True)

        target = tool_input.get("path")
        if isinstance(target, str) and target:
            try:
                base = self._resolve(target)
            except ValueError as exc:
                return ToolOutcome(f"Error: {exc}", is_error=True)
        else:
            base = self.root
        if not base.exists():
            return ToolOutcome(f"Error: {target} does not exist.", is_error=True)

        glob = tool_input.get("glob")
        glob = glob if isinstance(glob, str) and glob else None

        files = [base] if base.is_file() else sorted(_walk_files(base))
        hits: list[str] = []
        budget = MAX_TOOL_OUTPUT_BYTES
        for path in files:
            if glob and not fnmatch.fnmatch(path.name, glob):
                continue
            for number, line in _text_lines(path):
                if not regex.search(line):
                    continue
                try:
                    shown = path.relative_to(self.root).as_posix()
                except ValueError:  # pragma: no cover - _resolve keeps us inside
                    shown = path.as_posix()
                hits.append(f"./{shown}:{number}:{line}")
                budget -= len(hits[-1]) + 1
                if budget <= 0:
                    return ToolOutcome(_truncate("\n".join(hits) + "\n"))
        if not hits:
            return ToolOutcome("[no matches]")
        return ToolOutcome(_truncate("\n".join(hits) + "\n"))

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


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOBOBJECT_EXTENDED_LIMIT_INFORMATION = 9


def _win_job_object() -> Any:
    """A job object that kills everything assigned to it when it is closed.

    ``taskkill /T`` is not enough on Windows. MSYS -- which is what Git's bash
    is -- emulates fork, and the grandchildren it spawns do not keep a Windows
    parent link back to the shell: a runaway ``find /`` was observed with a
    parent pid that was never the shell's. Walking the tree therefore misses
    exactly the processes worth killing. A job object catches them regardless
    of re-parenting, which is the only mechanism here that actually holds.
    """
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = EXTENDED_LIMIT()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(
        job, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
    )
    return job


def _kill_tree(process: subprocess.Popen[str], job: Any = None) -> None:
    """Kill *process* and everything it started.

    ``Popen.kill`` kills only the direct child. A shell that has spawned a
    long-running grandchild leaves that grandchild holding the stdout pipe, so
    the drain that follows never returns and the timeout does not exist in
    practice -- which is how one agent's ``find /`` stalled a batch for twelve
    minutes with a 120-second cap configured, and went on scanning the disk
    after the shell above it was killed.
    """
    if sys.platform == "win32":
        if job is not None:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).TerminateJobObject(job, 1)
        process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover - race with exit
        process.kill()


def _capture(command: Any, *, shell: bool, cwd: Path) -> tuple[int, str, str]:
    """Run *command*, returning ``(returncode, stdout, stderr)``.

    Two things this does that ``subprocess.run`` does not: it kills the whole
    process tree when the timeout expires, and it gives the child no stdin. A
    tool that inherits the harness's stdin can block forever on a command that
    happens to read it, and nothing in a benchmark task should be interactive.
    """
    job = _win_job_object() if sys.platform == "win32" else None
    process = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
        start_new_session=sys.platform != "win32",
    )
    try:
        if job is not None:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).AssignProcessToJobObject(
                job,
                int(process._handle),  # type: ignore[attr-defined]
            )
        try:
            out, err = process.communicate(timeout=TOOL_TIMEOUT_S)
            return process.returncode, out, err
        except subprocess.TimeoutExpired:
            _kill_tree(process, job)
            try:
                process.communicate(timeout=KILL_DRAIN_S)
            except subprocess.TimeoutExpired:  # pragma: no cover - tree refused to die
                pass
            raise
    finally:
        if job is not None:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


def posix_shell() -> str | None:
    """A real POSIX shell for the ``bash`` tool, or ``None`` if the host has none.

    The agent under test writes Unix commands. ``shell=True`` runs cmd.exe on
    Windows, so every ``grep -rn`` and multi-line ``python -c`` it writes fails
    -- and only in the arm that still needs to search, which silently converts
    a platform quirk into a measured reduction.

    WSL's ``System32\bash.exe`` is rejected on purpose: it runs in a different
    filesystem namespace, so the Windows worktree path handed to it would not
    resolve, and the tool would fail in a new and more confusing way.
    """
    if os.name != "nt":
        return shutil.which("bash") or shutil.which("sh")

    found = shutil.which("bash")
    candidates = [found] if found and "system32" not in found.lower() else []
    git = shutil.which("git")
    if git:
        # <install>/cmd/git.exe -> <install>/bin/bash.exe
        candidates.append(str(Path(git).parent.parent / "bin" / "bash.exe"))
    candidates += [
        r"C:\Program Files\Gitinash.exe",
        r"C:\Program Files (x86)\Gitinash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def child_env() -> dict[str, str]:
    """Environment for tool subprocesses.

    The API credentials are stripped: nothing a benchmark task runs should be
    able to spend money, and a task that finds a key is a task that can cheat.
    """
    env = dict(os.environ)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env
