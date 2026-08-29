from __future__ import annotations

from waypost.parse import ParsedFile, Reference, Symbol
from waypost.rank import (
    compute_ranks,
    inbound_reference_counts,
    is_test_path,
    matches_focus,
    pagerank_rank,
    simple_rank,
)


def _file(path, defs=(), refs=(), loc=10) -> ParsedFile:
    return ParsedFile(path=path, language="python", loc=loc, defs=tuple(defs), refs=tuple(refs))


def test_is_test_path():
    assert is_test_path("tests/test_foo.py")
    assert is_test_path("src/foo_test.py")
    assert is_test_path("src/foo.test.ts")
    assert is_test_path("src/foo.spec.ts")
    assert not is_test_path("src/waypost/rank.py")
    assert not is_test_path("src/contest.py")


def test_inbound_reference_counts_matches_by_name_and_ignores_self_refs():
    lib = _file(
        "lib.py",
        defs=[Symbol(name="helper", kind="function", line=1, end_line=2, signature="def helper()")],
    )
    caller = _file(
        "caller.py",
        refs=[Reference(name="helper", kind="call", line=1)],
    )
    lib_with_self_ref = _file(
        "lib.py",
        defs=lib.defs,
        refs=[Reference(name="helper", kind="call", line=5)],
    )

    files = {"lib.py": lib_with_self_ref, "caller.py": caller}
    counts = inbound_reference_counts(files)

    assert counts["lib.py"] == 1  # only caller.py's ref counts, not lib.py's own
    assert counts["caller.py"] == 0


def test_inbound_reference_counts_resolves_qualified_names_by_tail():
    lib = _file(
        "lib.py",
        defs=[
            Symbol(
                name="Client.request",
                kind="function",
                line=3,
                end_line=4,
                signature="def request(self)",
            )
        ],
    )
    caller = _file("caller.py", refs=[Reference(name="request", kind="call", line=1)])

    counts = inbound_reference_counts({"lib.py": lib, "caller.py": caller})

    assert counts["lib.py"] == 1


def test_simple_rank_rewards_inbound_refs_and_exports_penalises_tests():
    lib = _file(
        "lib.py",
        defs=[
            Symbol(
                name="helper",
                kind="function",
                line=1,
                end_line=2,
                signature="def helper()",
                exported=True,
            )
        ],
    )
    caller = _file("caller.py", refs=[Reference(name="helper", kind="call", line=1)])
    test_file = _file(
        "tests/test_lib.py",
        refs=[Reference(name="helper", kind="call", line=1)],
    )

    scores = simple_rank({"lib.py": lib, "caller.py": caller, "tests/test_lib.py": test_file})

    assert scores["lib.py"] == 2.5  # 2 inbound refs + 0.5 * fully exported
    assert scores["caller.py"] == 0.0
    assert scores["tests/test_lib.py"] == -0.3


def test_compute_ranks_dispatches_and_rejects_unknown_strategy():
    files = {"a.py": _file("a.py")}

    assert compute_ranks(files, strategy="simple") == simple_rank(files)

    try:
        compute_ranks(files, strategy="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown strategy")


def test_pagerank_favours_the_file_everything_points_at():
    core_sym = Symbol(name="core", kind="function", line=1, end_line=1, signature="def core()")
    hub = _file("hub.py", defs=[core_sym])
    a = _file("a.py", refs=[Reference(name="core", kind="call", line=1)])
    b = _file("b.py", refs=[Reference(name="core", kind="call", line=1)])

    scores = pagerank_rank({"hub.py": hub, "a.py": a, "b.py": b})

    assert scores["hub.py"] > scores["a.py"]
    assert scores["hub.py"] > scores["b.py"]


def test_pagerank_personalize_biases_toward_focus_files():
    a = _file("a.py")
    b = _file("b.py")

    scores = pagerank_rank({"a.py": a, "b.py": b}, personalize=["a.py"])

    assert scores["a.py"] > scores["b.py"]


def test_pagerank_personalize_accepts_a_directory_not_only_an_exact_file():
    """`--focus src/api` names a directory far more often than a file.

    Matching focus paths by equality made that case personalise toward the
    empty set, i.e. silently do nothing at all.
    """
    api = _file("src/api/routes.py")
    other = _file("src/util/helpers.py")

    scores = pagerank_rank(
        {"src/api/routes.py": api, "src/util/helpers.py": other}, personalize=["src/api"]
    )

    assert scores["src/api/routes.py"] > scores["src/util/helpers.py"]


def test_focus_matches_a_path_prefix_but_not_a_partial_directory_name():
    assert matches_focus("src/api/routes.py", ["src/api"])
    assert matches_focus("src/api/routes.py", ["src/api/routes.py"])
    assert matches_focus("src/api/routes.py", ["./src/api/"])
    assert matches_focus("src/api/routes.py", ["src\\api"])
    # `src/apiary` is not under `src/api`.
    assert not matches_focus("src/apiary/routes.py", ["src/api"])
    assert not matches_focus("src/api/routes.py", [])
    # A leading dot is part of the name, not something to strip.
    assert matches_focus(".github/workflows/ci.yml", [".github"])


def _def(name: str) -> Symbol:
    return Symbol(name=name, kind="function", line=1, end_line=2, signature=f"def {name}()")


def test_ambiguous_reference_splits_its_credit_between_the_definers():
    # `get` is defined in three files, so one reference to it is worth one
    # point in total, not three. This is the Flask defect in miniature.
    files = {
        "a.py": _file("a.py", defs=[_def("get")]),
        "b.py": _file("b.py", defs=[_def("get")]),
        "c.py": _file("c.py", defs=[_def("get")]),
        "caller.py": _file("caller.py", refs=[Reference(name="get", kind="call", line=1)]),
    }

    counts = inbound_reference_counts(files)

    assert counts["a.py"] == counts["b.py"] == counts["c.py"] == 1 / 3
    assert sum(counts.values()) == 1.0


def test_a_unique_name_still_earns_full_credit():
    files = {
        "lib.py": _file("lib.py", defs=[_def("very_specific_helper")]),
        "caller.py": _file(
            "caller.py", refs=[Reference(name="very_specific_helper", kind="call", line=1)]
        ),
    }

    assert inbound_reference_counts(files)["lib.py"] == 1.0


def test_a_uniquely_named_symbol_outranks_a_much_referenced_ambiguous_one():
    # The regression that mattered: a test file defining a method called
    # `get` outranked the source file everything actually depends on.
    defs_get = [_def("get")]
    files = {"noisy.py": _file("noisy.py", defs=defs_get)}
    for i in range(9):
        files[f"other{i}.py"] = _file(f"other{i}.py", defs=defs_get)
    files["core.py"] = _file("core.py", defs=[_def("build_app")])
    files["caller.py"] = _file(
        "caller.py",
        refs=[Reference(name="get", kind="call", line=i) for i in range(5)]
        + [Reference(name="build_app", kind="call", line=99)],
    )

    counts = inbound_reference_counts(files)

    assert counts["noisy.py"] == 0.5  # 5 refs / 10 definers
    assert counts["core.py"] == 1.0
    assert counts["core.py"] > counts["noisy.py"]


def test_one_file_defining_a_name_twice_is_not_paid_twice():
    # `Client.get` and `Server.get` both index the tail `get`, but the file
    # defines that name once.
    lib = _file(
        "lib.py",
        defs=[
            Symbol(name="Client.get", kind="function", line=1, end_line=2, signature="def get()"),
            Symbol(name="Server.get", kind="function", line=4, end_line=5, signature="def get()"),
        ],
    )
    caller = _file("caller.py", refs=[Reference(name="get", kind="call", line=1)])

    assert inbound_reference_counts({"lib.py": lib, "caller.py": caller})["lib.py"] == 1.0


def test_pagerank_spends_less_on_an_ambiguous_edge_than_a_certain_one():
    # `caller.py` references `get` (defined in 3 files) and `build_app`
    # (defined in 1). The certain edge must carry the larger share.
    files = {
        "a.py": _file("a.py", defs=[_def("get")]),
        "b.py": _file("b.py", defs=[_def("get")]),
        "c.py": _file("c.py", defs=[_def("get")]),
        "core.py": _file("core.py", defs=[_def("build_app")]),
        "caller.py": _file(
            "caller.py",
            refs=[
                Reference(name="get", kind="call", line=1),
                Reference(name="build_app", kind="call", line=2),
            ],
        ),
    }

    scores = pagerank_rank(files)

    assert scores["core.py"] > scores["a.py"]
