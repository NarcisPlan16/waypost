"""Statistics and the report.

The shape of this follows the roadmap's methodology section, and the details
there are not decoration:

- **Median and IQR per task per arm**, not a mean over runs. One pathological
  run must not move the headline.
- **The aggregate is the median of paired per-task reductions**, not the
  reduction of the pooled totals. Pooling lets one enormous task dominate and
  quietly turns the result into a statement about that task.
- **Bootstrap CIs resample tasks, not runs.** Runs within a task are not
  independent; resampling them would report a confidence interval far tighter
  than the evidence supports.
- **Per-category numbers are always printed.** Category E is the control: a
  large reduction there is evidence of a biased harness, not of a working
  tool, and hiding it behind an aggregate defeats the point of having it.
- **The control is excluded from the headline.** Category E tasks are designed
  to be ones waypost cannot help with, so averaging them into the aggregate
  dilutes the measurement with tasks that were never a test of the tool -- in
  either direction. They are reported, in full, as an integrity check instead.

Two rules exist because the first paid smoke run tripped over both:

- **A pair where either arm was cut off is censored from the headline.** Its
  token total is the cost of an unfinished run, so a reduction computed from
  it compares two truncations and measures nothing. The pair is still printed,
  marked, so a batch that is quietly all-censored cannot look like a result.
- **The fixed system-prompt cost is reported separately.** The treatment arm
  carries the SKILL.md block on *every* turn, so on a short run it shows up as
  a large negative "reduction" in the control category, which reads as harness
  bias and is not. Estimating it lets the control line be read correctly.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_ITERATIONS = 10_000
ARM_ORDER = ("baseline", "treatment")

# Category E is the control: its tasks name the file in the prompt, so waypost
# is not supposed to help there. Those tasks exist to catch a biased harness,
# and a harness check is not a result -- see the docstring.
CONTROL_CATEGORY = "E"

# Names the loop records in ``tool_trace``.
WAYPOST_TOOL = "waypost"
READ_TOOL = "read_file"


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} contains no runs")
    return records


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Median, and the interquartile range as (q1, q3).

    Falls back to the extremes for tiny samples, where ``quantiles`` refuses:
    an 8-run smoke test must still produce a readable report.
    """
    ordered = sorted(values)
    median = statistics.median(ordered)
    if len(ordered) < 4:
        return median, ordered[0], ordered[-1]
    q1, _, q3 = statistics.quantiles(ordered, n=4)
    return median, q1, q3


@dataclass(frozen=True)
class TaskSummary:
    """One task's two arms, side by side."""

    task_id: str
    category: str
    baseline_median: float
    treatment_median: float
    baseline_iqr: tuple[float, float]
    treatment_iqr: tuple[float, float]
    runs: int
    baseline_turns: float = 0.0
    treatment_turns: float = 0.0
    censored: bool = False

    @property
    def baseline_per_turn(self) -> float | None:
        """Mean context carried per turn, which is what the tool actually moves."""
        return self.baseline_median / self.baseline_turns if self.baseline_turns else None

    @property
    def treatment_per_turn(self) -> float | None:
        return self.treatment_median / self.treatment_turns if self.treatment_turns else None

    @property
    def reduction(self) -> float | None:
        """Fraction of input tokens saved, or ``None`` if the baseline is zero."""
        if self.baseline_median <= 0:
            return None
        return (self.baseline_median - self.treatment_median) / self.baseline_median


def summarise_tasks(records: list[dict[str, Any]]) -> list[TaskSummary]:
    """Collapse runs to one summary per task that has both arms.

    A task present in only one arm is dropped: an unpaired task cannot
    contribute a paired reduction, and including it in one arm's median only
    would compare different task sets.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    turns: dict[tuple[str, str], list[float]] = {}
    categories: dict[str, str] = {}
    censored: set[str] = set()
    for record in records:
        key = (record["task_id"], record["arm"])
        grouped.setdefault(key, []).append(float(record["input_tokens"]))
        turns.setdefault(key, []).append(float(record.get("turns") or 0))
        categories[record["task_id"]] = record.get("category", "?")
        # `failure_reason` is set only when the loop was cut off -- a run that
        # finished and merely graded False leaves it None -- so this censors
        # truncations without censoring failures.
        if record.get("failure_reason"):
            censored.add(record["task_id"])

    summaries = []
    task_ids = sorted({task_id for task_id, _ in grouped})
    for task_id in task_ids:
        base = grouped.get((task_id, "baseline"))
        treat = grouped.get((task_id, "treatment"))
        if not base or not treat:
            continue
        base_median, base_q1, base_q3 = _quartiles(base)
        treat_median, treat_q1, treat_q3 = _quartiles(treat)
        summaries.append(
            TaskSummary(
                task_id=task_id,
                category=categories[task_id],
                baseline_median=base_median,
                treatment_median=treat_median,
                baseline_iqr=(base_q1, base_q3),
                treatment_iqr=(treat_q1, treat_q3),
                runs=len(base) + len(treat),
                baseline_turns=statistics.median(turns[(task_id, "baseline")]),
                treatment_turns=statistics.median(turns[(task_id, "treatment")]),
                censored=task_id in censored,
            )
        )
    return summaries


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 1,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of *values*.

    *values* must be one entry per task, never one per run.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    size = len(values)
    medians = [
        statistics.median([values[rng.randrange(size)] for _ in range(size)])
        for _ in range(iterations)
    ]
    medians.sort()
    tail = (1.0 - confidence) / 2.0
    low = medians[int(tail * iterations)]
    high = medians[min(iterations - 1, int((1.0 - tail) * iterations))]
    return (low, high)


def success_rate(records: list[dict[str, Any]], arm: str) -> tuple[float | None, int, int]:
    """Success rate for one arm, over the runs that were actually graded.

    Ungraded runs (``success is None``) are excluded from both numerator and
    denominator, and the denominator is returned so the report can say how
    many runs the rate is based on.
    """
    graded = [r for r in records if r["arm"] == arm and r.get("success") is not None]
    if not graded:
        return None, 0, len([r for r in records if r["arm"] == arm])
    wins = sum(1 for r in graded if r["success"])
    return wins / len(graded), len(graded), len([r for r in records if r["arm"] == arm])


def prompt_overhead_per_turn(records: list[dict[str, Any]]) -> float | None:
    """Tokens the treatment arm pays per turn before it does anything.

    Estimated only from task pairs where both arms made the *same* tool calls
    in the same numbers. When the tool use is identical the two runs differ by
    the appended SKILL.md block and nothing else, so the per-turn gap is that
    block. Pairs where the arms behaved differently are unusable for this and
    are skipped rather than averaged in.
    """
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], {})[record["arm"]] = record

    deltas: list[float] = []
    for arms in by_task.values():
        base, treat = arms.get("baseline"), arms.get("treatment")
        if not base or not treat:
            continue
        if base.get("tool_calls") != treat.get("tool_calls"):
            continue
        base_turns, treat_turns = base.get("turns") or 0, treat.get("turns") or 0
        if not base_turns or not treat_turns:
            continue
        deltas.append(treat["input_tokens"] / treat_turns - base["input_tokens"] / base_turns)
    return statistics.median(deltas) if deltas else None


def substitution_stats(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Whether a waypost answer *replaced* a file read, or merely preceded one.

    The first paid runs showed the treatment arm querying waypost and then
    opening the file anyway, which costs a whole extra round trip and is how a
    tool that shrinks per-turn context still loses on total tokens. Total
    tokens cannot show that; only the ordered trace can. So this counts, over
    every treatment run, how often a waypost call is *immediately* followed by
    a read of a file -- the failure -- and which subcommands were reached for.

    Returns ``None`` for results written before ``tool_trace`` existed, rather
    than a zero that would read as "the problem is solved".
    """
    treatment = [r for r in records if r["arm"] == "treatment"]
    traced = [r for r in treatment if r.get("tool_trace")]
    if not traced:
        return None

    subcommands: Counter[str] = Counter()
    calls = followed_by_read = 0
    for record in traced:
        trace = record["tool_trace"]
        for i, entry in enumerate(trace):
            if entry.get("name") != WAYPOST_TOOL:
                continue
            calls += 1
            subcommands[str(entry.get("arg", "")).split(" ")[0] or "?"] += 1
            nxt = trace[i + 1] if i + 1 < len(trace) else None
            if nxt is not None and nxt.get("name") == READ_TOOL:
                followed_by_read += 1

    return {
        "traced_runs": len(traced),
        "untraced_runs": len(treatment) - len(traced),
        "waypost_calls": calls,
        "followed_by_read": followed_by_read,
        "follow_read_rate": (followed_by_read / calls) if calls else None,
        "subcommands": dict(subcommands.most_common()),
    }


def build_report(records: list[dict[str, Any]], seed: int = 1) -> dict[str, Any]:
    """Everything the text report prints, as plain data."""
    summaries = summarise_tasks(records)
    # Two separate exclusions, and they are not the same kind of thing: a
    # censored pair is *unmeasurable*, a control pair is measurable but is not
    # a measurement of the tool. Both are still printed per task.
    uncensored = [s for s in summaries if not s.censored]
    usable = [s for s in uncensored if s.category != CONTROL_CATEGORY]
    reductions = [s.reduction for s in usable if s.reduction is not None]

    aggregate: dict[str, Any] = {
        "tasks_paired": len(summaries),
        "tasks_censored": sum(1 for s in summaries if s.censored),
        "tasks_control": sum(1 for s in summaries if s.category == CONTROL_CATEGORY),
        "tasks_in_headline": len(usable),
    }
    if reductions:
        aggregate["median_reduction"] = statistics.median(reductions)
        low, high = bootstrap_ci(reductions, seed=seed)
        aggregate["ci95"] = [low, high]
    else:
        aggregate["median_reduction"] = None
        aggregate["ci95"] = None

    by_category: dict[str, Any] = {}
    for category in sorted({s.category for s in summaries}):
        values = [
            s.reduction for s in uncensored if s.category == category and s.reduction is not None
        ]
        by_category[category] = {
            "tasks": sum(1 for s in summaries if s.category == category),
            "censored": sum(1 for s in summaries if s.category == category and s.censored),
            "median_reduction": statistics.median(values) if values else None,
        }

    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_records = [r for r in records if r["arm"] == arm]
        rate, graded, total = success_rate(records, arm)
        arms[arm] = {
            "runs": total,
            "graded": graded,
            "success_rate": rate,
            "median_input_tokens": (
                statistics.median([float(r["input_tokens"]) for r in arm_records])
                if arm_records
                else None
            ),
            "failures": sorted({r["failure_reason"] for r in arm_records if r["failure_reason"]}),
        }

    treatment = [r for r in records if r["arm"] == "treatment"]
    integrity = {
        "control_median_reduction": by_category.get(CONTROL_CATEGORY, {}).get("median_reduction"),
        "mean_waypost_calls": (
            sum(int(r.get("waypost_calls", 0)) for r in treatment) / len(treatment)
            if treatment
            else None
        ),
        "treatment_runs_without_waypost": sum(
            1 for r in treatment if not int(r.get("waypost_calls", 0))
        ),
        "cached_runs": sum(1 for r in records if r.get("cache_read_input_tokens")),
        "prompt_overhead_per_turn": prompt_overhead_per_turn(records),
    }

    success_delta = None
    if (
        arms["baseline"]["success_rate"] is not None
        and arms["treatment"]["success_rate"] is not None
    ):
        success_delta = arms["treatment"]["success_rate"] - arms["baseline"]["success_rate"]

    return {
        "runs": len(records),
        "aggregate": aggregate,
        "by_category": by_category,
        "arms": arms,
        "success_delta_pp": None if success_delta is None else success_delta * 100,
        "integrity": integrity,
        "substitution": substitution_stats(records),
        "tasks": [
            {
                "task_id": s.task_id,
                "category": s.category,
                "baseline_median": s.baseline_median,
                "treatment_median": s.treatment_median,
                "baseline_iqr": list(s.baseline_iqr),
                "treatment_iqr": list(s.treatment_iqr),
                "reduction": s.reduction,
                "runs": s.runs,
                "baseline_turns": s.baseline_turns,
                "treatment_turns": s.treatment_turns,
                "baseline_per_turn": s.baseline_per_turn,
                "treatment_per_turn": s.treatment_per_turn,
                "censored": s.censored,
            }
            for s in summaries
        ],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def render_text(report: dict[str, Any]) -> str:
    """The human-readable report."""
    lines: list[str] = []
    lines.append(f"runs: {report['runs']}   paired tasks: {report['aggregate']['tasks_paired']}")
    lines.append("")

    lines.append("input tokens per task (median, IQR)")
    lines.append(f"{'task':<24}{'cat':<5}{'baseline':>22}{'treatment':>22}{'reduction':>12}")
    for task in report["tasks"]:
        base = (
            f"{task['baseline_median']:.0f} "
            f"[{task['baseline_iqr'][0]:.0f}-{task['baseline_iqr'][1]:.0f}]"
        )
        treat = (
            f"{task['treatment_median']:.0f} "
            f"[{task['treatment_iqr'][0]:.0f}-{task['treatment_iqr'][1]:.0f}]"
        )
        lines.append(
            f"{task['task_id']:<24}{task['category']:<5}{base:>22}{treat:>22}"
            f"{_pct(task['reduction']):>12}{' *' if task['censored'] else ''}"
        )
    lines.append("")

    # Total input tokens is turns times context carried, and the two move in
    # opposite directions: the tool shrinks the context but costs turns. A
    # report that prints only the product cannot show which one won.
    lines.append("turns x context per turn")
    lines.append(f"{'task':<24}{'turns b/t':>14}{'ctx/turn b':>14}{'ctx/turn t':>14}")
    for task in report["tasks"]:
        turns = f"{task['baseline_turns']:.0f} / {task['treatment_turns']:.0f}"
        base_pt = task["baseline_per_turn"]
        treat_pt = task["treatment_per_turn"]
        base_text = "n/a" if base_pt is None else f"{base_pt:.0f}"
        treat_text = "n/a" if treat_pt is None else f"{treat_pt:.0f}"
        lines.append(f"{task['task_id']:<24}{turns:>14}{base_text:>14}{treat_text:>14}")
    lines.append("")

    aggregate = report["aggregate"]
    ci = aggregate["ci95"]
    ci_text = "n/a" if ci is None else f"[{_pct(ci[0])}, {_pct(ci[1])}]"
    lines.append(
        f"median paired reduction: {_pct(aggregate['median_reduction'])}   95% CI {ci_text}"
    )
    lines.append(
        f"  (median over {aggregate.get('tasks_in_headline', 0)} task(s); CI bootstrapped over"
        " tasks, not runs; control excluded)"
    )
    control = aggregate.get("tasks_control") or 0
    if control:
        lines.append(
            f"  {control} control task(s) held out of the headline: waypost is not meant to"
            " help there, so they measure the harness, not the tool"
        )
    if not aggregate.get("tasks_in_headline"):
        # Without this an all-censored or all-control batch prints "n/a" and
        # reads like a run that merely had nothing to say.
        lines.append(
            "  NO MEASURABLE TASKS: every pair was censored or was a control."
            " This batch measures nothing."
        )
    censored = aggregate.get("tasks_censored") or 0
    if censored:
        lines.append(
            f"  * {censored} task(s) excluded: an arm was cut off, so the total is the"
            " cost of an unfinished run"
        )
    lines.append("")

    lines.append("by category")
    for category, stats in report["by_category"].items():
        note = (
            "  <- control, held out of the headline; should be ~0 net of prompt overhead"
            if category == CONTROL_CATEGORY
            else ""
        )
        if stats.get("censored"):
            note += f"  ({stats['censored']} censored)"
        reduction = _pct(stats["median_reduction"])
        lines.append(f"  {category}  tasks {stats['tasks']:<3} reduction {reduction}{note}")
    lines.append("")

    lines.append("success")
    for arm in ARM_ORDER:
        stats = report["arms"][arm]
        rate = "n/a" if stats["success_rate"] is None else f"{stats['success_rate'] * 100:.0f}%"
        failures = ", ".join(stats["failures"]) or "none"
        lines.append(
            f"  {arm:<10} {rate:>5} of {stats['graded']} graded / {stats['runs']} runs"
            f"   failures: {failures}"
        )
    delta = report["success_delta_pp"]
    lines.append(f"  delta: {'n/a' if delta is None else f'{delta:+.1f}pp'}")
    lines.append("")

    sub = report.get("substitution")
    if sub is not None:
        lines.append("substitution (treatment arm, from the ordered trace)")
        rate = sub["follow_read_rate"]
        lines.append(
            f"  waypost calls: {sub['waypost_calls']}"
            f"   immediately followed by a file read: {sub['followed_by_read']}"
            f" ({'n/a' if rate is None else f'{rate * 100:.0f}%'})"
        )
        lines.append(
            "  high means the tool located the code and the agent paid a second turn to read"
            " it -- the round trip, not the output size, is the cost"
        )
        used = ", ".join(f"{name} x{count}" for name, count in sub["subcommands"].items())
        lines.append(f"  subcommands: {used or 'none'}")
        if sub["untraced_runs"]:
            lines.append(
                f"  ({sub['untraced_runs']} treatment run(s) predate tool_trace and are not"
                " counted here)"
            )
        lines.append("")

    integrity = report["integrity"]
    lines.append("harness integrity")
    lines.append(
        f"  control (E) reduction:        {_pct(integrity['control_median_reduction'])}"
        "   large here means a biased harness"
    )
    calls = integrity["mean_waypost_calls"]
    lines.append(
        f"  mean waypost calls/treatment: {'n/a' if calls is None else f'{calls:.2f}'}"
        "   near zero means the SKILL.md wording is the bug"
    )
    lines.append(
        f"  treatment runs never calling waypost: {integrity['treatment_runs_without_waypost']}"
    )
    lines.append(f"  runs reporting cached tokens: {integrity['cached_runs']}   must be 0")
    overhead = integrity["prompt_overhead_per_turn"]
    lines.append(
        f"  treatment prompt overhead:    {'n/a' if overhead is None else f'{overhead:+.0f}'}"
        " tokens/turn   paid every turn, before the tool does anything"
    )
    return "\n".join(lines)
