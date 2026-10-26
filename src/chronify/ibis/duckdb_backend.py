"""DuckDB backend implementation for Ibis."""

from pathlib import Path
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger

from chronify.exceptions import ConflictingInputsError, InvalidOperation, InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType, _quote_identifier
from chronify.utils.sql import make_temp_view_name


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

        self._in_transaction = False
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

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        # Bypass Ibis's generic pandas materialization and use DuckDB's native
        # cursor.fetch_df(), which is zero-copy from Arrow.
        if isinstance(expr, ibis.Table):
            # Registers memtables and UDFs, as ibis's own execute does.
            self._connection._run_pre_execute_hooks(expr)
            sql = self._connection.compile(expr)
            return cast(pd.DataFrame, self._connection.con.execute(sql).fetch_df())
        return cast(pd.DataFrame, self._connection.execute(expr))

    def execute_sql_to_df(self, query: str, params: Any = None) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        con = self._connection.con
        cursor = con.execute(query, params) if params is not None else con.execute(query)
        return cast(pd.DataFrame, cursor.fetch_df())

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ibis.Table, ObjectType]:
        # Absolute path because the view is persisted and may be read from a different cwd.
        parquet_path = Path(path).resolve()
        if parquet_path.is_dir():
            read_path = str(parquet_path / "**" / "*.parquet").replace("\\", "/")
        else:
            read_path = str(parquet_path).replace("\\", "/")
        # Reference the file directly. ibis's read_parquet registers a session-only
        # temp view, which would make this view unusable after the connection closes.
        self._connection.con.execute(
            f"CREATE VIEW {_quote_identifier(name)} AS "
            f"SELECT * FROM read_parquet({_quote_literal(read_path)})"
        )
        return self.table(name), ObjectType.VIEW

    def write_parquet(
        self,
        expr: ibis.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        if not partition_by:
            self._connection.to_parquet(expr, path)
            return
        # ibis-duckdb renders partition_by as a Python list repr, which DuckDB
        # rejects, so issue the COPY directly.
        self._connection._run_pre_execute_hooks(expr)
        sql = self._connection.compile(expr)
        cols = ", ".join(_quote_identifier(c) for c in partition_by)
        self._connection.con.execute(
            f"COPY ({sql}) TO {_quote_literal(path)} (FORMAT PARQUET, PARTITION_BY ({cols}))"
        )

    def backup(self, dst: str) -> None:
        if self._database is None:
            msg = "backup is only supported with a database backed by a file"
            raise InvalidOperation(msg)
        if self._in_transaction:
            msg = "backup is not supported inside a transaction"
            raise InvalidOperation(msg)
        # Copy through the open connection so expressions bound to it stay valid.
        con = self._connection.con
        src = con.execute("SELECT current_database()").fetchone()[0]
        alias = _quote_identifier(make_temp_view_name("__chronify_backup"))
        con.execute(f"ATTACH {_quote_literal(dst)} AS {alias}")
        try:
            con.execute(f"COPY FROM DATABASE {_quote_identifier(src)} TO {alias}")
        finally:
            con.execute(f"DETACH {alias}")

    def _begin_transaction(self) -> None:
        self._connection.con.execute("BEGIN TRANSACTION")

    def _commit_transaction(self) -> None:
        self._connection.con.execute("COMMIT")

    def _rollback_transaction(self) -> None:
        self._connection.con.execute("ROLLBACK")

    def _supports_parquet_partitioning(self) -> bool:
        return True


def _quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


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
