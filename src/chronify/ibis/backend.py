"""Ibis backend abstraction layer for database operations.

This module provides a unified interface for different database backends
(DuckDB, SQLite, Spark) using the Ibis library.

The module re-exports all backend classes and factory functions for convenience.
Individual backend implementations are in separate modules:
- base.py: IbisBackend abstract base class
- duckdb_backend.py: DuckDBBackend
- sqlite_backend.py: SQLiteBackend
- spark_backend.py: SparkBackend
"""

from pathlib import Path
from typing import Any

from chronify.ibis.base import IbisBackend
from chronify.ibis.duckdb_backend import DuckDBBackend
from chronify.ibis.sqlite_backend import SQLiteBackend
from chronify.ibis.spark_backend import SparkBackend


def make_backend(
    backend_name: str = "duckdb",
    file_path: Path | str | None = None,
    **kwargs: Any,
) -> IbisBackend:
    """Factory function to create an IbisBackend instance.

    Parameters
    ----------
    backend_name
        Name of the backend: "duckdb", "sqlite", or "spark"
    file_path
        Path to database file (for duckdb/sqlite), or None for in-memory
    **kwargs
        Additional arguments passed to the backend constructor

    Returns
    -------
    IbisBackend
        Backend instance

    Examples
    --------
    >>> backend = make_backend("duckdb")  # In-memory DuckDB
    >>> backend = make_backend("duckdb", file_path="data.db")  # File-based
    >>> backend = make_backend("sqlite")  # In-memory SQLite
    >>> backend = make_backend("spark")  # Spark (requires spark extra)
    """
    match backend_name:
        case "duckdb":
            return DuckDBBackend(file_path=file_path, **kwargs)
        case "sqlite":
            return SQLiteBackend(file_path=file_path, **kwargs)
        case "spark":
            return SparkBackend(**kwargs)
        case _:
            msg = f"Unknown backend: {backend_name}. Supported: duckdb, sqlite, spark"
            raise NotImplementedError(msg)


# Backwards compatibility alias
get_backend = make_backend

__all__ = [
    "IbisBackend",
    "DuckDBBackend",
    "SQLiteBackend",
    "SparkBackend",
    "make_backend",
    "get_backend",
]
