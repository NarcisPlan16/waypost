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


def test_a_pair_with_a_cut_off_arm_is_censored_from_the_headline():
    # t2's treatment run hit the turn cap, so its token total is the cost of an
    # unfinished run. Averaging it in would compare two truncations.
    records = [
        *paired("t1", 100, 80),
        record("t2", "baseline", 400),
        record("t2", "treatment", 800, failure_reason="turn_cap"),
    ]
    report = build_report(records)
    assert report["aggregate"]["tasks_paired"] == 2
    assert report["aggregate"]["tasks_censored"] == 1
    assert report["aggregate"]["median_reduction"] == pytest.approx(0.2)
    assert "* 1 task(s) excluded" in render_text(report)


def test_a_run_that_merely_graded_false_is_not_censored():
    # `failure_reason` is set only for a cut-off. A finished run that got the
    # wrong answer still cost what it cost, and must stay in the sample.
    records = paired("t1", 100, 80, success=False)
    report = build_report(records)
    assert report["aggregate"]["tasks_censored"] == 0
    assert report["aggregate"]["median_reduction"] == pytest.approx(0.2)


def test_prompt_overhead_is_estimated_only_from_identical_tool_use():
    # t1's arms made the same calls, so their per-turn gap is the SKILL.md
    # block: 8000/2 - 6000/2 = 1000. t2 behaved differently and is unusable.
    records = [
        record("t1", "baseline", 6000, turns=2, tool_calls={"read_file": 1}),
        record("t1", "treatment", 8000, turns=2, tool_calls={"read_file": 1}),
        record("t2", "baseline", 6000, turns=2, tool_calls={"read_file": 1}),
        record("t2", "treatment", 90000, turns=9, tool_calls={"waypost": 4}),
    ]
    report = build_report(records)
    assert report["integrity"]["prompt_overhead_per_turn"] == pytest.approx(1000.0)


def test_per_turn_columns_separate_turn_cost_from_context_cost():
    # The tool can shrink context per turn and still lose on the total by
    # spending more turns; the report must show both halves.
    records = [
        record("t1", "baseline", 20000, turns=4),
        record("t1", "treatment", 24000, turns=8),
    ]
    report = build_report(records)
    task = report["tasks"][0]
    assert task["baseline_per_turn"] == pytest.approx(5000.0)
    assert task["treatment_per_turn"] == pytest.approx(3000.0)
    assert "turns x context per turn" in render_text(report)


def test_the_control_category_is_held_out_of_the_headline():
    # E is designed to be a task waypost cannot help with. Averaging it in
    # would dilute the measurement with a task that never tested the tool.
    records = [*paired("t1", 100, 50), *paired("e1", 100, 120, category="E")]
    report = build_report(records)
    assert report["aggregate"]["tasks_in_headline"] == 1
    assert report["aggregate"]["tasks_control"] == 1
    assert report["aggregate"]["median_reduction"] == pytest.approx(0.5)
    # Held out, not hidden: it still has to appear as an integrity check.
    assert report["by_category"]["E"]["median_reduction"] == pytest.approx(-0.2)
    assert report["integrity"]["control_median_reduction"] == pytest.approx(-0.2)


def test_a_batch_of_nothing_but_controls_says_it_measures_nothing():
    # "n/a" alone reads like a run that merely had nothing to say. A batch
    # with no measurable pair must announce that it is not a result.
    report = build_report(paired("e1", 100, 120, category="E"))
    assert report["aggregate"]["tasks_in_headline"] == 0
    assert report["aggregate"]["median_reduction"] is None
    assert "NO MEASURABLE TASKS" in render_text(report)


def test_a_censored_control_is_counted_in_neither_the_headline_nor_the_control_line():
    records = [
        *paired("t1", 100, 50),
        record("e1", "baseline", 100, category="E"),
        record("e1", "treatment", 120, category="E", failure_reason="turn_cap"),
    ]
    report = build_report(records)
    assert report["aggregate"]["tasks_in_headline"] == 1
    assert report["by_category"]["E"]["median_reduction"] is None


def trace(*names):
    return [{"turn": i + 1, "name": n, "arg": n, "output_bytes": 10} for i, n in enumerate(names)]


def test_substitution_counts_waypost_calls_that_did_not_replace_a_read():
    # The failure the first paid runs found: waypost answers where the symbol
    # is, and the agent then spends a whole extra turn reading the file. Only
    # the ordered trace can see it -- the token totals cannot.
    records = [
        record("t1", "baseline", 100, tool_trace=trace("read_file", "read_file")),
        record(
            "t1",
            "treatment",
            80,
            tool_trace=[
                {"turn": 1, "name": "waypost", "arg": "find ['a']", "output_bytes": 10},
                {"turn": 2, "name": "read_file", "arg": "a.py", "output_bytes": 90},
                {"turn": 3, "name": "waypost", "arg": "show ['a']", "output_bytes": 90},
                {"turn": 4, "name": "edit_file", "arg": "a.py", "output_bytes": 5},
            ],
        ),
    ]
    sub = build_report(records)["substitution"]
    # Only the treatment arm is counted, and only the `find` call is a failure.
    assert sub["waypost_calls"] == 2
    assert sub["followed_by_read"] == 1
    assert sub["follow_read_rate"] == pytest.approx(0.5)
    assert sub["subcommands"] == {"find": 1, "show": 1}
    assert "substitution" in render_text(build_report(records))


def test_substitution_is_none_for_results_written_before_tool_trace():
    # A zero here would read as "the round-trip problem is solved" on results
    # that simply never recorded the sequence.
    report = build_report(list(paired("t1", 100, 80)))
    assert report["substitution"] is None
    assert "substitution" not in render_text(report)


def test_substitution_reports_how_many_runs_it_could_not_see():
    records = [
        *paired("t1", 100, 80),
        record("t2", "baseline", 100),
        record("t2", "treatment", 80, tool_trace=trace("waypost", "read_file")),
    ]
    sub = build_report(records)["substitution"]
    assert sub["traced_runs"] == 1
    assert sub["untraced_runs"] == 1
    assert sub["followed_by_read"] == 1
