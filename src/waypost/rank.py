"""Importance ranking over the reference graph.

Responsibilities (Sprint 2):
    - Default strategy, cheap and explainable::

          rank = inbound_reference_count
                 + 0.5 * is_exported
                 - 0.3 * is_test

    - Optional ``--rank pagerank``: PageRank over the graph where file A
      points at file B when A references a symbol defined in B, with the
      personalisation vector biased toward ``--focus`` paths or files in the
      current ``git diff``.

    Ship the simple strategy as the default. Whether PageRank becomes the
    default in v0.2 is a question for the Sprint 6 benchmark, not a matter
    of taste.
"""
