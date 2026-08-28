"""Benchmark harness for waypost.

This package is *not* part of the shipped wheel. It exists to answer one
question with a number: does giving an agent a budgeted repo map and
symbol-level lookups cut the input tokens it spends on localization-heavy
coding tasks, without costing task success?

The layout is driven by one constraint. Everything except ``client`` must be
importable without the ``anthropic`` SDK, so the loop, the tools, the graders
and the statistics are all covered by ordinary offline tests. Only
``bench.client`` talks to the API.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
