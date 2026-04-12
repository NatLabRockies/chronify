"""DuckDB backend implementation for Ibis."""

import shutil
from pathlib import Path
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from loguru import logger

from chronify.exceptions import ConflictingInputsError, InvalidOperation, InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType, _is_ddl


class DuckDBBackend(IbisBackend):
    """Ibis backend for DuckDB databases."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        connection: ibis.BaseBackend | None = None,
    ) -> None:
        """Construct a DuckDBBackend.

        Parameters
        ----------
        database
            Path to a DuckDB database file, or ``":memory:"`` for an in-memory
            database. Ignored when ``connection`` is provided.
        connection
            Optional pre-existing ibis DuckDB connection. When provided, the
            backend does not own the connection and will not disconnect it on
            ``dispose()``. ``database`` is inferred from the connection when
            possible; otherwise ``backup()`` is unavailable.
        """
        if connection is not None and str(database) != ":memory:":
            msg = f"{database=} and {connection=} cannot both be set"
            raise ConflictingInputsError(msg)

        self._table_cache = None
        self._owns_connection = connection is None
        if connection is None:
            db = str(database)
            self._database = None if db == ":memory:" else db
            self._connection = ibis.duckdb.connect(db)
        else:
            if connection.name != "duckdb":
                msg = f"DuckDBBackend requires a DuckDB ibis connection, got {connection.name!r}"
                raise InvalidParameter(msg)
            self._connection = connection
            self._database = _infer_duckdb_path(connection)

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
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        table = self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)
        self._mark_table_created(name)
        return table

    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
        view = self._connection.create_view(name, expr, overwrite=False)
        self._mark_table_created(name)
        return view

    def drop_table(self, name: str) -> None:
        self._connection.drop_table(name, force=True)
        self._mark_table_dropped(name)

    def drop_view(self, name: str) -> None:
        self._connection.drop_view(name, force=True)
        self._mark_table_dropped(name)

    def list_tables(self) -> list[str]:
        tables = self._connection.list_tables()
        # Filter out internal ibis memtables
        tables = [t for t in tables if not t.startswith("ibis_pandas_memtable_")]
        self._table_cache = set(tables)
        return tables

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def insert(self, name: str, data: pd.DataFrame | pa.Table) -> None:
        con = self._connection.con  # raw duckdb connection
        target_columns = list(self.table(name).columns)
        _validate_insert_columns(name, target_columns, _get_columns(data))
        ordered_data = _select_columns(data, target_columns)
        quoted_columns = ", ".join(f'"{col}"' for col in target_columns)
        quoted_name = _quote_identifier(name)
        con.register("__insert_df", ordered_data)
        try:
            con.execute(
                f"INSERT INTO {quoted_name} ({quoted_columns}) "
                f"SELECT {quoted_columns} FROM __insert_df"
            )
        finally:
            con.unregister("__insert_df")
        logger.trace("Inserted {} rows into {}", _row_count(data), name)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        con = self._connection.con
        quoted_name = _quote_identifier(name)
        where = " AND ".join(f"{_quote_identifier(c)} = ?" for c in values)
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        con.execute(sql, list(values.values()))
        logger.trace("Deleted rows from {} matching {}", name, values)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        # Bypass Ibis's generic pandas materialization and use DuckDB's native
        # cursor.fetch_df(), which is zero-copy from Arrow.
        if isinstance(expr, ir.Table):
            sql = self._connection.compile(expr)
            return cast(pd.DataFrame, self._connection.con.execute(sql).fetch_df())
        return cast(pd.DataFrame, self._connection.execute(expr))

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        escaped_path = path.replace("'", "''")
        sql = self._connection.compile(expr)
        if partition_by:
            partition_clause = ", ".join(_quote_identifier(c) for c in partition_by)
            self._connection.raw_sql(
                f"COPY ({sql}) TO '{escaped_path}' "
                f"(FORMAT PARQUET, PARTITION_BY ({partition_clause}))"
            )
        else:
            self._connection.raw_sql(f"COPY ({sql}) TO '{escaped_path}' (FORMAT PARQUET)")

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        parquet_path = Path(path)
        if parquet_path.is_dir():
            read_path = str(parquet_path / "**" / "*.parquet").replace("\\", "/")
        else:
            read_path = str(parquet_path).replace("\\", "/")
        quoted_name = _quote_identifier(name)
        escaped_path = read_path.replace("'", "''")
        self._connection.raw_sql(
            f"CREATE VIEW {quoted_name} AS SELECT * FROM read_parquet('{escaped_path}')"
        )
        self._mark_table_created(name)
        return self.table(name), ObjectType.VIEW

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._connection.raw_sql(query)
        if _is_ddl(query):
            self._invalidate_table_cache()

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        result = self._connection.raw_sql(query)
        return cast(pd.DataFrame, result.fetch_df())

    def dispose(self) -> None:
        if self._owns_connection:
            self._connection.disconnect()

    def backup(self, dst: str) -> None:
        if self._database is None:
            msg = "backup is only supported with a database backed by a file"
            raise InvalidOperation(msg)
        if not self._owns_connection:
            msg = "backup is not supported for externally-provided DuckDB connections"
            raise InvalidOperation(msg)
        src = self._database
        self._connection.disconnect()
        try:
            shutil.copyfile(src, dst)
        finally:
            self._connection = ibis.duckdb.connect(src)

    def _begin_transaction(self) -> None:
        self._connection.con.execute("BEGIN TRANSACTION")

    def _commit_transaction(self) -> None:
        self._connection.con.execute("COMMIT")

    def _rollback_transaction(self) -> None:
        self._connection.con.execute("ROLLBACK")
        self._invalidate_table_cache()


def _infer_duckdb_path(connection: ibis.BaseBackend) -> str | None:
    """Return the database file path for an ibis DuckDB connection, or None for in-memory."""
    try:
        result = connection.con.execute(
            "SELECT path FROM duckdb_databases() WHERE database_name = current_database()"
        ).fetchone()
    except Exception:
        return None
    if not result:
        return None
    path = result[0]
    return None if not path else str(path)


def _validate_insert_columns(
    table_name: str, target_columns: list[str], data_columns: list[str]
) -> None:
    missing = [c for c in target_columns if c not in data_columns]
    extra = [c for c in data_columns if c not in target_columns]
    if missing or extra:
        msg = (
            f"Insert data columns do not match table {table_name!r}. "
            f"Missing: {missing}. Extra: {extra}."
        )
        raise InvalidParameter(msg)


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB, escaping embedded double quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _get_columns(data: pd.DataFrame | pa.Table) -> list[str]:
    if isinstance(data, pa.Table):
        return cast(list[str], data.column_names)
    return list(data.columns)


def _select_columns(data: pd.DataFrame | pa.Table, columns: list[str]) -> pd.DataFrame | pa.Table:
    if isinstance(data, pa.Table):
        return data.select(columns)
    return data.loc[:, columns]


def _row_count(data: pd.DataFrame | pa.Table) -> int:
    if isinstance(data, pa.Table):
        return cast(int, data.num_rows)
    return len(data)
