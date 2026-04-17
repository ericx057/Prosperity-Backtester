"""Shared pytest configuration.

The ``--update-snapshots`` option lets you intentionally rewrite
regression-snapshot JSONs. It's a conftest hook because pytest rejects
``pytest_addoption`` inside a non-conftest module.
"""

from __future__ import annotations


def pytest_addoption(parser) -> None:  # type: ignore[no-untyped-def]
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Overwrite fidelity snapshot files with the current matching engine output.",
    )
