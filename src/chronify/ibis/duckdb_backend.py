"""DuckDB backend implementation."""

from pathlib import Path
from typing import Any, Optional, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa

from chronify.exceptions import InvalidOperation
from chronify.ibis.base import IbisBackend


class DuckDBBackend(IbisBackend):
    """DuckDB backend implementation."""

    def __init__(
        self,
        file_path: Optional[Path | str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a DuckDB backend.

        Parameters
        ----------
        file_path
            Path to database file, or None for in-memory
        **kwargs
            Additional arguments passed to ibis.duckdb.connect
        """
        self._file_path = str(file_path) if file_path else ":memory:"
        connection = ibis.duckdb.connect(
            self._file_path,
            **kwargs,
        )
        super().__init__(connection)

    @property
    def name(self) -> str:
        return "duckdb"

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
            # Execute the expression and create table from result
            self._connection.raw_sql(f"CREATE TABLE {name} AS {data.compile()}")
        else:
            self._connection.create_table(name, data)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._connection.raw_sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"CREATE OR REPLACE TEMP VIEW {name} AS {sql}")
        else:
            # Register as a table then create view
            temp_name = f"_temp_{name}"
            self._connection.create_table(temp_name, data, temp=True)
            self._connection.raw_sql(
                f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {temp_name}"
            )

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP TABLE {if_exists_sql}{name}")

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP VIEW {if_exists_sql}{name}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_")
            and not t.startswith("ibis_pyarrow_memtable_")
            and not t.startswith("_temp_")
            and not t.startswith("_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return cast(pd.DataFrame, expr.execute())

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"INSERT INTO {table_name} {sql}")
        elif isinstance(data, pa.Table):
            # Register PyArrow table as a virtual table, insert, then unregister
            self._connection.con.register("__temp_insert_df", data)
            self._connection.con.execute(
                f"INSERT INTO {table_name} SELECT * FROM __temp_insert_df"
            )
            self._connection.con.unregister("__temp_insert_df")
        else:
            # pandas DataFrame - register as a virtual table, insert, then unregister
            self._connection.con.register("__temp_insert_df", data)
            self._connection.con.execute(
                f"INSERT INTO {table_name} SELECT * FROM __temp_insert_df"
            )
            self._connection.con.unregister("__temp_insert_df")

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        sql = expr.compile()
        if partition_by:
            cols = ",".join(partition_by)
            query = f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, PARTITION_BY ({cols}))"
        else:
            query = f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)"
        self._connection.raw_sql(query)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        str_path = f"{path}/**/*.parquet" if path.is_dir() else str(path)
        query = f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{str_path}')"
        self._connection.raw_sql(query)

    def dispose(self) -> None:
        """Close the DuckDB connection to flush data to disk."""
        self._connection.con.close()

    def reconnect(self) -> None:
        """Reconnect to the database after dispose."""
        self._connection = ibis.duckdb.connect(self._file_path)
