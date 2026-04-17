"""Enforce the matching_engine module boundary.

matching_engine.py MUST NOT import from runner, metrics, or reporter. It is a
pure function: (book_state, user_orders, position, limits) ->
(fills, new_book_state, rejections). This is a hard architecture constraint.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.fidelity

FORBIDDEN_MODULES = {"runner", "metrics", "reporter", "sweeper", "data_loader"}


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def test_matching_engine_has_no_forbidden_imports() -> None:
    here = pathlib.Path(__file__).resolve().parents[3]
    engine_path = here / "backtester" / "matching_engine.py"
    assert engine_path.exists(), f"matching_engine.py not found at {engine_path}"
    imports = _imports(engine_path)

    violations = set()
    for name in imports:
        # Check top-level and any submodule reference.
        parts = name.split(".")
        for forbidden in FORBIDDEN_MODULES:
            if forbidden in parts:
                violations.add(name)
    assert violations == set(), (
        f"matching_engine.py imports forbidden module(s): {violations}. "
        "The matching engine must be a pure function with no dependencies on "
        "runner, metrics, reporter, sweeper, or data_loader."
    )
