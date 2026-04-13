"""DuckDB backend implementation for Ibis."""

import shutil
from pathlib import Path
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger

from chronify.exceptions import ConflictingInputsError, InvalidOperation, InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType


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

    def list_tables(self) -> list[str]:
        tables = self._connection.list_tables()
        # Filter out internal ibis memtables
        return [t for t in tables if not t.startswith("ibis_pandas_memtable_")]

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

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        if partition_by:
            self._connection.to_parquet(expr, path, partition_by=partition_by)
        else:
            self._connection.to_parquet(expr, path)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        parquet_path = Path(path)
        if parquet_path.is_dir():
            read_path = str(parquet_path / "**" / "*.parquet").replace("\\", "/")
        else:
            read_path = str(parquet_path).replace("\\", "/")
        self._connection.create_view(name, self._connection.read_parquet(read_path))
        return self.table(name), ObjectType.VIEW

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


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for DuckDB, escaping embedded double quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


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
