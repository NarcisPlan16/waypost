"""Tokenizer wrapper and hard budget enforcement.

Responsibilities (Sprint 3):
    - Wrap ``tiktoken`` behind a small interface so the tokenizer is
      swappable and named in the index metadata.
    - Provide the single ``fit(text, budget)`` helper that every renderer
      routes through, so budget enforcement lives in exactly one place.

Budgets are measured, not estimated: ``map --budget N`` must never exceed N
tokens as counted by the configured tokenizer.
"""
