"""SQLite backend implementation for Ibis."""

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger

from chronify.ibis.base import IbisBackend, ObjectType


def _adapt_value(v: Any) -> Any:
    """Convert a value for SQLite parameterized insertion.

    Converts datetime/Timestamp objects to ISO-format strings to avoid the
    Python 3.12+ DeprecationWarning about the default datetime adapter.
    Returns None for pd.NaT and other missing-value sentinels.
    """
    if v is pd.NaT or v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


class SQLiteBackend(IbisBackend):
    """Ibis backend for SQLite databases."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        db = str(database)
        self._database = None if db == ":memory:" else db
        self._connection = ibis.sqlite.connect(db)

    @property
    def name(self) -> str:
        return "sqlite"

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
        overwrite: bool = False,
    ) -> ir.Table:
        if isinstance(obj, ir.Table):
            # SQLite CREATE TABLE AS SELECT loses datetime type info.
            # Execute the expression first, then create from the DataFrame.
            df = self._connection.execute(obj)
            return self._connection.create_table(name, obj=df, overwrite=overwrite)
        return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
        return self._connection.create_view(name, expr, overwrite=False)

    def drop_table(self, name: str) -> None:
        self._connection.drop_table(name, force=True)

    def drop_view(self, name: str) -> None:
        self._connection.drop_view(name, force=True)

    def list_tables(self) -> list[str]:
        return cast(list[str], self._connection.list_tables())

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def insert(self, name: str, data: pd.DataFrame) -> None:
        # Use raw SQLite cursor for parameterized inserts
        con = self._connection.con  # raw sqlite3 connection
        table = self._connection.table(name)
        columns = table.columns
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(_quote_identifier(c) for c in columns)
        quoted_name = _quote_identifier(name)
        sql = f"INSERT INTO {quoted_name} ({col_list}) VALUES ({placeholders})"

        ordered = data.reindex(columns=columns)
        rows = [tuple(_adapt_value(v) for v in row) for row in ordered.itertuples(index=False)]
        cursor = con.cursor()
        cursor.executemany(sql, rows)
        con.commit()
        logger.trace("Inserted {} rows into {}", len(data), name)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        con = self._connection.con
        quoted_name = _quote_identifier(name)
        where = " AND ".join(f"{_quote_identifier(c)} = ?" for c in values)
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        con.execute(sql, list(values.values()))
        con.commit()
        logger.trace("Deleted rows from {} matching {}", name, values)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return cast(pd.DataFrame, self._connection.execute(expr))

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        if partition_by:
            msg = "SQLite backend does not support partitioned Parquet writes."
            raise NotImplementedError(msg)
        df = self._connection.execute(expr)
        df.to_parquet(path)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        # SQLite can't read Parquet natively. Load into a table instead.
        df = pd.read_parquet(path)
        return self.create_table(name, obj=df), ObjectType.TABLE

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        con = self._connection.con
        con.execute(query)
        con.commit()

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        con = self._connection.con
        cursor = con.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return pd.DataFrame(rows, columns=columns)

    def dispose(self) -> None:
        self._connection.disconnect()

    def reconnect(self) -> None:
        db = self._database if self._database else ":memory:"
        self._connection = ibis.sqlite.connect(db)


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for SQLite, escaping embedded double quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
