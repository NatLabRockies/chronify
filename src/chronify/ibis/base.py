"""Abstract base class for Ibis database backends."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger


class ObjectType(Enum):
    TABLE = "table"
    VIEW = "view"


class IbisBackend(ABC):
    """Abstract base class defining the interface for Ibis database backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'duckdb', 'sqlite', 'spark')."""

    @property
    @abstractmethod
    def database(self) -> str | None:
        """Return the database file path, or None for in-memory."""

    @property
    @abstractmethod
    def connection(self) -> ibis.BaseBackend:
        """Return the underlying ibis connection."""

    @abstractmethod
    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table in the database.

        Parameters
        ----------
        name
            Table name.
        obj
            Data to populate the table with.
        schema
            Schema to use if obj is None.
        overwrite
            If True, replace the table if it already exists.

        Returns
        -------
        ir.Table
        """

    @abstractmethod
    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
        """Create a view in the database."""

    @abstractmethod
    def drop_table(self, name: str) -> None:
        """Drop a table from the database."""

    @abstractmethod
    def drop_view(self, name: str) -> None:
        """Drop a view from the database."""

    @abstractmethod
    def list_tables(self) -> list[str]:
        """List all user tables in the database."""

    @abstractmethod
    def table(self, name: str) -> ir.Table:
        """Return an ibis table expression for the named table."""

    @abstractmethod
    def insert(self, name: str, data: pd.DataFrame) -> None:
        """Insert data into an existing table."""

    @abstractmethod
    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        """Delete rows from a table where every column equals its given value.

        Identifiers must be quoted and values must be parameterized to avoid
        SQL injection and to handle values containing quote characters.
        """

    @abstractmethod
    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        """Execute an ibis expression and return a DataFrame."""

    @abstractmethod
    def sql(self, query: str) -> ir.Table:
        """Create an ibis table expression from a raw SQL string."""

    @abstractmethod
    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        """Write an ibis expression result to a Parquet file."""

    @abstractmethod
    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        """Create a view or table backed by a Parquet file.

        Returns the table expression and the type of object created, since some
        backends (e.g., SQLite) must create a table instead of a view.
        """

    def has_table(self, name: str) -> bool:
        """Check whether a table or view exists."""
        return name in self.list_tables()

    def execute_sql(self, query: str) -> Any:
        """Execute a raw SQL statement (no result expected)."""
        logger.trace("execute_sql: {}", query)
        return self.connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query and return a DataFrame."""
        logger.trace("execute_sql_to_df: {}", query)
        return cast(pd.DataFrame, self.connection.raw_sql(query).fetch_df())

    def dispose(self) -> None:
        """Dispose of the backend connection."""
        self.connection.disconnect()

    def reconnect(self) -> None:
        """Reconnect to the database. Subclasses should override if needed."""

    @contextmanager
    def transaction(self) -> Generator[list[tuple[str, ObjectType]], None, None]:
        """Context manager for pseudo-transactions.

        Tracks created objects (tables/views) so they can be cleaned up on failure.
        On success, created objects are kept. On exception, they are dropped.

        Yields a list to which callers should append (name, ObjectType) tuples.
        """
        created: list[tuple[str, ObjectType]] = []
        try:
            yield created
        except Exception:
            for obj_name, obj_type in reversed(created):
                try:
                    if obj_type == ObjectType.TABLE:
                        self.drop_table(obj_name)
                    else:
                        self.drop_view(obj_name)
                    logger.debug("Rolled back {} {}", obj_type.value, obj_name)
                except Exception:
                    logger.warning("Failed to roll back {} {}", obj_type.value, obj_name)
            raise
