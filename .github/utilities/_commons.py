label_options = [
    ("full pre-commit", "Run `pre-commit` checks for all files"),
    ("run typecheck test", "Run `mypy` typecheck tests"),
    ("full pytest actions", "Run all `pytest` tests and configurations"),
    ("full examples run", "Run all notebook example tests"),
    ("codecov actions", "Run numba-disabled `codecov` tests"),
    (
        "stop pre-commit fixes",
        "Stop automatic `pre-commit` fixes (always disabled for drafts)",
    ),
    ("no numba cache", "Disable numba cache loading"),
]
