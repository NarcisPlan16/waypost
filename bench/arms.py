"""The two arms under test.

The arms are identical except for two things: the tool list, and one appended
block of system-prompt text describing waypost. Everything else -- the task
prompt, ``max_tokens``, effort, the turn cap, the tool output cap -- is
byte-identical, because anything else that differs becomes an alternative
explanation for whatever the benchmark measures.

The treatment text is not written here. It is lifted **verbatim** from the
shipped ``SKILL.md``. Per the roadmap, the main risk to this whole project is
that an agent ignores the tool and greps out of habit -- which would make the
skill *wording* the thing under test. Testing a rewrite of it would answer a
question nobody asked.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.tools import tools_for_arm

ARMS = ("baseline", "treatment")

BASE_SYSTEM = """\
You are a software engineering agent working in a checked-out repository.

Answer the user's task by using the tools available to you. Work in the
repository you have been given; do not assume knowledge of it that you have
not gathered with the tools.

When the task asks you to locate something, end your final message with the
repository-relative path or paths, one per line, under the heading FILES:.
When the task asks you to change something, make the change with edit_file
and say what you changed. When the task asks you to explain something,
explain it.

Be efficient: gather what you need and stop. Do not re-read what you have
already read."""

_SKILL_SECTION = re.compile(
    r"^## When to use it$.*?(?=^## Commands$)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class ArmSpec:
    """Everything that differs between the two arms, resolved."""

    name: str
    system: str
    tools: list[dict[str, Any]]
    skill_sha256: str | None

    def as_record(self) -> dict[str, Any]:
        """The part of the arm worth writing into every run record."""
        return {
            "arm": self.name,
            "system_sha256": _sha256(self.system),
            "skill_sha256": self.skill_sha256,
            "tool_names": [tool["name"] for tool in self.tools],
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_skill_path() -> Path:
    """``SKILL.md`` at the repository root, next to this package."""
    return Path(__file__).resolve().parent.parent / "SKILL.md"


def read_skill_guidance(path: Path | None = None) -> str:
    """Extract the when-to-use / when-not-to-use guidance from ``SKILL.md``.

    Raises rather than falling back to a paraphrase: a benchmark that silently
    substituted different wording for the wording under test would produce a
    number about nothing.
    """
    path = path or default_skill_path()
    text = path.read_text(encoding="utf-8")
    match = _SKILL_SECTION.search(text)
    if match is None:
        raise ValueError(
            f"{path} no longer has a '## When to use it' section ending at '## Commands'; "
            "the treatment prompt is built from it verbatim and cannot be guessed"
        )
    return match.group(0).strip()


def build_arm(name: str, skill_path: Path | None = None) -> ArmSpec:
    """Resolve one arm's system prompt and tool list."""
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}")

    tools = tools_for_arm(name)
    if name == "baseline":
        return ArmSpec(name=name, system=BASE_SYSTEM, tools=tools, skill_sha256=None)

    guidance = read_skill_guidance(skill_path)
    system = (
        f"{BASE_SYSTEM}\n\n"
        "This repository has been indexed with waypost, available to you as the "
        "`waypost` tool. The following is its documentation.\n\n"
        f"{guidance}"
    )
    return ArmSpec(name=name, system=system, tools=tools, skill_sha256=_sha256(guidance))
