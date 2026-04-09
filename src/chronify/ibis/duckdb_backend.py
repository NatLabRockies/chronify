"""DuckDB backend implementation for Ibis."""

from pathlib import Path

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger

from chronify.ibis.base import IbisBackend, ObjectType


class DuckDBBackend(IbisBackend):
    """Ibis backend for DuckDB databases."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        db = str(database)
        self._database = None if db == ":memory:" else db
        self._connection = ibis.duckdb.connect(db)

    @property
    def name(self) -> str:
        return "duckdb"

    @property
    def database(self) -> str | None:
        return self._database

    @property
    def connection(self) -> ibis.BaseBackend:
        return self._connection

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table | None = None,
        schema: ibis.Schema | None = None,
    ) -> ir.Table:
        return self._connection.create_table(name, obj=obj, schema=schema, overwrite=False)

    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
        return self._connection.create_view(name, expr, overwrite=False)

    def drop_table(self, name: str) -> None:
        self._connection.drop_table(name, force=True)

    def drop_view(self, name: str) -> None:
        self._connection.drop_view(name, force=True)

    def list_tables(self) -> list[str]:
        tables = self._connection.list_tables()
        # Filter out internal ibis memtables
        return [t for t in tables if not t.startswith("ibis_pandas_memtable_")]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def insert(self, name: str, data: pd.DataFrame) -> None:
        con = self._connection.con  # raw duckdb connection
        target_columns = list(self.table(name).columns)
        ordered_data = data.reindex(columns=target_columns)
        quoted_columns = ", ".join(f'"{col}"' for col in target_columns)
        con.register("__insert_df", ordered_data)
        try:
            con.execute(
                f"INSERT INTO {name} ({quoted_columns}) "
                f"SELECT {quoted_columns} FROM __insert_df"
            )
        finally:
            con.unregister("__insert_df")
        logger.trace("Inserted {} rows into {}", len(data), name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return self._connection.execute(expr)

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        if partition_by:
            partition_clause = ", ".join(partition_by)
            sql = self._connection.compile(expr)
            self._connection.raw_sql(
                f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, PARTITION_BY ({partition_clause}))"
            )
        else:
            df = self._connection.execute(expr)
            df.to_parquet(path)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        parquet_path = Path(path)
        if parquet_path.is_dir():
            read_path = str(parquet_path / "**" / "*.parquet").replace("\\", "/")
        else:
            read_path = str(parquet_path).replace("\\", "/")
        self._connection.raw_sql(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{read_path}')"
        )
        return self.table(name), ObjectType.VIEW

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        result = self._connection.raw_sql(query)
        return result.fetch_df()

    def dispose(self) -> None:
        self._connection.disconnect()

    def reconnect(self) -> None:
        if self._database is not None:
            self._connection = ibis.duckdb.connect(self._database)
        else:
            logger.warning("Cannot reconnect to an in-memory DuckDB database.")
