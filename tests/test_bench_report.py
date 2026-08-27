from __future__ import annotations

import pytest

from bench.report import bootstrap_ci, build_report, render_text, success_rate, summarise_tasks


def record(task_id, arm, tokens, category="A", success=True, **extra):
    return {
        "task_id": task_id,
        "arm": arm,
        "trial": 1,
        "category": category,
        "input_tokens": tokens,
        "success": success,
        "failure_reason": None,
        "waypost_calls": 3 if arm == "treatment" else 0,
        **extra,
    }


def paired(task_id, base, treat, **kwargs):
    return [
        record(task_id, "baseline", base, **kwargs),
        record(task_id, "treatment", treat, **kwargs),
    ]


def test_summaries_need_both_arms():
    records = [*paired("t1", 100, 50), record("t2", "baseline", 100)]
    summaries = summarise_tasks(records)
    # An unpaired task cannot contribute a paired reduction, and counting it in
    # one arm's median only would compare different task sets.
    assert [s.task_id for s in summaries] == ["t1"]
    assert summaries[0].reduction == 0.5


def test_aggregate_is_the_median_of_paired_reductions_not_of_pooled_totals():
    # Pooled totals would be dominated by t3 and report ~50%. The median over
    # tasks reports 20%, which is what the roadmap asks for.
    records = [
        *paired("t1", 100, 80),
        *paired("t2", 100, 90),
        *paired("t3", 100_000, 50_000),
    ]
    report = build_report(records)
    assert report["aggregate"]["median_reduction"] == 0.2


def test_bootstrap_ci_brackets_the_median_and_is_deterministic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.25, 0.35]
    low, high = bootstrap_ci(values, seed=7)
    assert low <= 0.3 <= high
    assert (low, high) == bootstrap_ci(values, seed=7)


def test_bootstrap_ci_of_a_single_task_is_a_point():
    assert bootstrap_ci([0.4]) == (0.4, 0.4)


def test_ungraded_runs_are_excluded_from_the_success_rate():
    records = [
        record("t1", "treatment", 10, success=True),
        record("t2", "treatment", 10, success=None),
        record("t3", "treatment", 10, success=False),
    ]
    rate, graded, total = success_rate(records, "treatment")
    assert (rate, graded, total) == (0.5, 2, 3)


def test_control_category_is_reported_separately():
    records = [*paired("t1", 100, 50, category="A"), *paired("e1", 100, 99, category="E")]
    report = build_report(records)
    assert report["by_category"]["E"]["median_reduction"] == pytest.approx(0.01)
    assert report["integrity"]["control_median_reduction"] == pytest.approx(0.01)


def test_integrity_flags_treatment_runs_that_never_called_waypost():
    records = [
        *paired("t1", 100, 50),
        record("t2", "treatment", 50, waypost_calls=0),
    ]
    integrity = build_report(records)["integrity"]
    assert integrity["treatment_runs_without_waypost"] == 1
    assert integrity["cached_runs"] == 0


def test_text_report_renders_the_headline_and_both_integrity_checks():
    text = render_text(
        build_report([*paired("t1", 100, 50), *paired("e1", 100, 100, category="E")])
    )
    assert "median paired reduction" in text
    assert "control (E) reduction" in text
    assert "mean waypost calls/treatment" in text
    assert "must be 0" in text
