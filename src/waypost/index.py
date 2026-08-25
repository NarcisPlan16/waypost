"""Index construction, persistence and incremental refresh.

Responsibilities (Sprint 2):
    - Serialise the schema (version 1) to ``.waypost/index.json``, and
      validate it on read.
    - Refresh incrementally: compare each file's content SHA against the
      stored one, reparse only what changed, drop entries for deleted files,
      and recompute ranks (cheap enough to always do in full).
    - Treat an unknown or older ``schema`` version, or corrupt JSON, as a
      trigger for a silent full rebuild -- never a crash, and never a stale
      wrong answer.
"""
