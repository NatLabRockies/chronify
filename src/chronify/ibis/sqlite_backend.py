"""SQLite backend implementation."""

from pathlib import Path
from typing import Any, Optional, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa

from chronify.exceptions import InvalidOperation
from chronify.ibis.base import IbisBackend


class SQLiteBackend(IbisBackend):
    """SQLite backend implementation."""

    def __init__(
        self,
        file_path: Optional[Path | str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a SQLite backend.

        Parameters
        ----------
        file_path
            Path to database file, or None for in-memory
        **kwargs
            Additional arguments passed to ibis.sqlite.connect
        """
        self._file_path = str(file_path) if file_path else ":memory:"
        connection = ibis.sqlite.connect(
            self._file_path if self._file_path != ":memory:" else None,
            **kwargs,
        )
        super().__init__(connection)

    @property
    def name(self) -> str:
        return "sqlite"

    @property
    def database(self) -> str | None:
        return None if self._file_path == ":memory:" else self._file_path

    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        if overwrite and self.has_table(name):
            self.drop_table(name, if_exists=True)

        if isinstance(data, ir.Table):
            # SQLite loses datetime type info with CREATE TABLE AS SELECT
            # Execute the query first and create from DataFrame to preserve types
            df = data.execute()
            self._connection.create_table(name, df)
        else:
            self._connection.create_table(name, data)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._connection.raw_sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"CREATE TEMP VIEW {name} AS {sql}")
        else:
            # Create temp table first
            self._connection.create_table(name, data, temp=True)

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP TABLE {if_exists_sql}{name}")

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP VIEW {if_exists_sql}{name}")
        # Also drop the backing table if it exists (created by create_view_from_parquet)
        backing_table = f"_parquet_backing_{name}"
        self._connection.raw_sql(f"DROP TABLE IF EXISTS {backing_table}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables and parquet backing tables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_") and not t.startswith("_parquet_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return cast(pd.DataFrame, expr.execute())

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"INSERT INTO {table_name} {sql}")
        else:
            # For pandas/pyarrow, use parameterized INSERT
            if isinstance(data, pa.Table):
                df = data.to_pandas()
            else:
                df = data
            # Build parameterized INSERT statement
            columns = ",".join(df.columns)
            placeholders = ",".join(["?" for _ in df.columns])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            # Access SQLite's raw connection
            cursor = self._connection.con.cursor()
            cursor.executemany(query, df.values.tolist())
            self._connection.con.commit()

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame."""
        cursor = self._connection.raw_sql(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if partition_by:
            msg = "SQLite does not support partitioned Parquet writes"
            raise InvalidOperation(msg)
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        # Execute and write via pandas/pyarrow
        df = self.execute(expr)
        df.to_parquet(path)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        # SQLite cannot directly read Parquet, so we load data into a backing table
        # and create a view that selects from it
        df = pd.read_parquet(path)
        backing_table = f"_parquet_backing_{name}"
        self._connection.create_table(backing_table, df)
        self._connection.raw_sql(f"CREATE VIEW {name} AS SELECT * FROM {backing_table}")
