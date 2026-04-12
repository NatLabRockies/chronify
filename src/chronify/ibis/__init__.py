"""Ibis backend abstraction layer for Chronify."""

from chronify.exceptions import InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType
from chronify.ibis.duckdb_backend import DuckDBBackend
from chronify.ibis.sqlite_backend import SQLiteBackend

__all__ = [
    "DuckDBBackend",
    "IbisBackend",
    "ObjectType",
    "SQLiteBackend",
    "make_backend",
]


def make_backend(
    name: str,
    database: str | None = None,
    **kwargs: object,
) -> IbisBackend:
    """Create an IbisBackend instance.

    Parameters
    ----------
    name
        Backend name: "duckdb", "sqlite", or "spark".
    database
        Database file path, or None for in-memory.
    **kwargs
        Additional keyword arguments passed to the backend constructor.
    """
    match name:
        case "duckdb":
            return DuckDBBackend(database=database or ":memory:", **kwargs)
        case "sqlite":
            return SQLiteBackend(database=database or ":memory:", **kwargs)
        case "spark":
            from chronify.ibis.spark_backend import SparkBackend

            return SparkBackend(**kwargs)
        case _:
            msg = f"Unsupported backend: {name}. Choose from: duckdb, sqlite, spark"
            raise InvalidParameter(msg)
