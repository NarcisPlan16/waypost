from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.arms import ARMS, build_arm, read_skill_guidance
from bench.client import StubClient, estimate_cost
from bench.runner import build_matrix, completed_keys, environment_record, rough_cost
from bench.tasks import parse_task

TASKS = [
    parse_task(
        {
            "id": f"t-{i}",
            "repo": "demo",
            "category": "A",
            "prompt": "p",
            "grade": {"kind": "localization", "expect_files": ["a.py"]},
        }
    )
    for i in range(4)
]


def test_every_cell_of_the_matrix_is_present_exactly_once():
    matrix = build_matrix(TASKS, trials=2, seed=3)
    keys = [k.as_tuple() for k in matrix]
    assert len(keys) == len(TASKS) * 2 * len(ARMS)
    assert len(set(keys)) == len(keys)


def test_the_two_arms_of_a_task_run_adjacently():
    # Adjacency is what keeps API-side drift over a long batch from landing on
    # one arm more than the other.
    matrix = build_matrix(TASKS, trials=2, seed=3)
    for first, second in zip(matrix[::2], matrix[1::2], strict=True):
        assert (first.task_id, first.trial) == (second.task_id, second.trial)
        assert {first.arm, second.arm} == set(ARMS)


def test_which_arm_goes_first_alternates():
    matrix = build_matrix(TASKS, trials=2, seed=3)
    firsts = [pair.arm for pair in matrix[::2]]
    assert set(firsts) == set(ARMS)


def test_order_is_reproducible_from_the_seed():
    assert [k.as_tuple() for k in build_matrix(TASKS, 1, 11)] == [
        k.as_tuple() for k in build_matrix(TASKS, 1, 11)
    ]
    assert [k.as_tuple() for k in build_matrix(TASKS, 1, 11)] != [
        k.as_tuple() for k in build_matrix(TASKS, 1, 12)
    ]


def test_completed_keys_reads_back_what_a_batch_wrote(tmp_path):
    path = tmp_path / "results.jsonl"
    assert completed_keys(path) == set()

    path.write_text(
        json.dumps({"task_id": "t-0", "arm": "baseline", "trial": 1}) + "\n",
        encoding="utf-8",
    )
    assert completed_keys(path) == {("t-0", "baseline", 1)}


def test_rough_cost_is_none_for_an_unpriced_model():
    assert rough_cost("claude-opus-5", 200) > 0
    assert rough_cost("some-future-model", 200) is None
    assert estimate_cost("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)


def test_environment_record_pins_what_the_batch_was():
    record = environment_record("claude-opus-5", "high", 7)
    assert record["model"] == "claude-opus-5"
    assert record["turn_cap"] == 40


def test_the_dry_run_client_never_calls_the_api():
    client = StubClient()
    response = client.create(model="claude-opus-5", messages=[])
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 0
    assert client.calls


def test_treatment_prompt_quotes_the_shipped_skill_verbatim():
    guidance = read_skill_guidance()
    treatment = build_arm("treatment")
    baseline = build_arm("baseline")

    # The roadmap's main risk is that the agent ignores the tool out of habit,
    # which makes the shipped wording the thing under test. A paraphrase here
    # would measure a prompt nobody ships.
    assert guidance in treatment.system
    assert guidance not in baseline.system
    assert baseline.system in treatment.system
    assert treatment.skill_sha256 and baseline.skill_sha256 is None


def test_a_skill_file_without_the_expected_sections_is_an_error(tmp_path):
    broken = tmp_path / "SKILL.md"
    broken.write_text("# waypost\n\nnothing useful here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="When to use it"):
        read_skill_guidance(broken)


def test_arm_record_captures_what_differed():
    record = build_arm("treatment").as_record()
    assert record["arm"] == "treatment"
    assert "waypost" in record["tool_names"]
    assert len(record["system_sha256"]) == 64


def test_the_shipped_repo_table_is_pinned():
    from bench.worktree import load_repos

    repos = load_repos()
    assert set(repos) == {"flask", "hono"}
    for spec in repos.values():
        # A branch name here would silently change the tree under the benchmark.
        assert len(spec.sha) == 40 and spec.sha.isalnum()
        assert Path(spec.url).name


def test_default_repos_file_matches_the_seed_tasks():
    from bench.tasks import load_tasks
    from bench.worktree import load_repos

    repos = set(load_repos())
    assert {task.repo for task in load_tasks(Path("bench/tasks"))} <= repos
