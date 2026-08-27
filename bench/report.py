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
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_ITERATIONS = 10_000
ARM_ORDER = ("baseline", "treatment")


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
    categories: dict[str, str] = {}
    for record in records:
        key = (record["task_id"], record["arm"])
        grouped.setdefault(key, []).append(float(record["input_tokens"]))
        categories[record["task_id"]] = record.get("category", "?")

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


def build_report(records: list[dict[str, Any]], seed: int = 1) -> dict[str, Any]:
    """Everything the text report prints, as plain data."""
    summaries = summarise_tasks(records)
    reductions = [s.reduction for s in summaries if s.reduction is not None]

    aggregate: dict[str, Any] = {"tasks_paired": len(summaries)}
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
            s.reduction for s in summaries if s.category == category and s.reduction is not None
        ]
        by_category[category] = {
            "tasks": sum(1 for s in summaries if s.category == category),
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
        "control_median_reduction": by_category.get("E", {}).get("median_reduction"),
        "mean_waypost_calls": (
            sum(int(r.get("waypost_calls", 0)) for r in treatment) / len(treatment)
            if treatment
            else None
        ),
        "treatment_runs_without_waypost": sum(
            1 for r in treatment if not int(r.get("waypost_calls", 0))
        ),
        "cached_runs": sum(1 for r in records if r.get("cache_read_input_tokens")),
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
            f"{_pct(task['reduction']):>12}"
        )
    lines.append("")

    aggregate = report["aggregate"]
    ci = aggregate["ci95"]
    ci_text = "n/a" if ci is None else f"[{_pct(ci[0])}, {_pct(ci[1])}]"
    lines.append(
        f"median paired reduction: {_pct(aggregate['median_reduction'])}   95% CI {ci_text}"
    )
    lines.append("  (median over tasks; CI bootstrapped over tasks, not runs)")
    lines.append("")

    lines.append("by category")
    for category, stats in report["by_category"].items():
        note = "  <- control, should be ~0" if category == "E" else ""
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
    return "\n".join(lines)
