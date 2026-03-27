"""Abstract base class for Ibis-based database backends."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from loguru import logger


class IbisBackend(ABC):
    """Abstract base class for Ibis-based database backends."""

    def __init__(self, connection: ibis.BaseBackend) -> None:
        self._connection = connection

    @property
    def connection(self) -> ibis.BaseBackend:
        """Return the underlying Ibis connection."""
        return self._connection

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend (e.g., 'duckdb', 'sqlite', 'spark')."""

    @property
    @abstractmethod
    def database(self) -> str | None:
        """Return the database path/name, or None for in-memory."""

    @abstractmethod
    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table from data.

        Parameters
        ----------
        name
            Name of the table to create
        data
            Data to populate the table
        overwrite
            If True, drop existing table first

        Returns
        -------
        ir.Table
            Ibis table reference
        """

    @abstractmethod
    def create_view(self, name: str, expr: ir.Table) -> None:
        """Create a view from an Ibis expression.

        Parameters
        ----------
        name
            Name of the view
        expr
            Ibis table expression defining the view
        """

    @abstractmethod
    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        """Create a temporary view from user-provided data.

        Parameters
        ----------
        name
            Name of the temporary view
        data
            Data for the view
        """

    @abstractmethod
    def drop_table(self, name: str, if_exists: bool = False) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop
        if_exists
            If True, don't raise error if table doesn't exist
        """

    @abstractmethod
    def drop_view(self, name: str, if_exists: bool = False) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop
        if_exists
            If True, don't raise error if view doesn't exist
        """

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return a list of all tables and views in the database."""

    @abstractmethod
    def table(self, name: str) -> ir.Table:
        """Get a reference to an existing table.

        Parameters
        ----------
        name
            Name of the table

        Returns
        -------
        ir.Table
            Ibis table reference
        """

    @abstractmethod
    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        """Execute an Ibis expression and return results as DataFrame.

        Parameters
        ----------
        expr
            Ibis expression to execute

        Returns
        -------
        pd.DataFrame
            Query results
        """

    @abstractmethod
    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        """Insert data into an existing table.

        Parameters
        ----------
        table_name
            Name of the target table
        data
            Data to insert
        """

    @abstractmethod
    def sql(self, query: str) -> ir.Table:
        """Execute raw SQL and return an Ibis table expression.

        Parameters
        ----------
        query
            SQL query string

        Returns
        -------
        ir.Table
            Query results as Ibis table
        """

    def execute_sql(self, query: str) -> None:
        """Execute raw SQL without returning results.

        Parameters
        ----------
        query
            SQL statement to execute
        """
        self._connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame.

        This method is useful for complex queries that may conflict with Ibis's
        SQL parsing, such as queries with CTEs.

        Parameters
        ----------
        query
            SQL query string

        Returns
        -------
        pd.DataFrame
            Query results
        """
        return cast(pd.DataFrame, self._connection.raw_sql(query).fetchdf())

    @abstractmethod
    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Write query results to a Parquet file.

        Parameters
        ----------
        expr
            Ibis table expression to write
        path
            Output path
        partition_by
            Optional columns to partition by
        overwrite
            If True, overwrite existing file/directory
        """

    @abstractmethod
    def create_view_from_parquet(self, name: str, path: Path) -> None:
        """Create a view that reads from a Parquet file.

        Parameters
        ----------
        name
            Name of the view to create
        path
            Path to Parquet file or directory
        """

    def has_table(self, name: str) -> bool:
        """Check if a table or view exists.

        Parameters
        ----------
        name
            Name of the table/view

        Returns
        -------
        bool
            True if exists
        """
        return name in self.list_tables()

    def dispose(self) -> None:
        """Clean up the backend connection."""
        pass

    def reconnect(self) -> None:
        """Reconnect to the database after dispose."""
        pass

    @contextmanager
    def transaction(self) -> Iterator[list[tuple[str, str]]]:
        """Context manager for pseudo-transaction support.

        Ibis lacks built-in transaction support, so we track created objects
        and clean them up on failure.

        Yields
        ------
        list[tuple[str, str]]
            List to track created objects as (type, name) tuples.
            Callers should append to this list when creating tables/views.
        """
        created_objects: list[tuple[str, str]] = []
        try:
            yield created_objects
        except Exception:
            # Rollback by dropping created objects in reverse order
            for obj_type, name in reversed(created_objects):
                try:
                    if obj_type == "table":
                        self.drop_table(name, if_exists=True)
                    elif obj_type == "view":
                        self.drop_view(name, if_exists=True)
                except Exception as e:
                    logger.warning("Failed to rollback {} {}: {}", obj_type, name, e)
            raise
