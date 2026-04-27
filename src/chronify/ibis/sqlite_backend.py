"""SQLite backend implementation for Ibis."""

import sqlite3
from datetime import datetime
from pathlib import Path
from functools import singledispatchmethod
from typing import Any, Iterable

import ibis
import pandas as pd
import pyarrow as pa
from loguru import logger
from pandas import DatetimeTZDtype

from chronify.exceptions import ConflictingInputsError, InvalidOperation, InvalidParameter
from chronify.ibis.base import (
    IbisBackend,
    ObjectType,
    _DATETIME_RANGES,
    _normalize_timestamps,
)
from chronify.time import TimeDataType
from chronify.time_configs import TimeBaseModel


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


class _NoCommitProxy:
    """Connection proxy that suppresses commit/rollback.

    ibis-sqlite's ``create_table`` (and a few other DDL paths) wraps its work in
    ``with self.begin() as cur:``, which calls ``con.commit()`` on success.
    That commit terminates whatever outer transaction we started in
    :meth:`SQLiteBackend._begin_transaction`, breaking atomicity for DDL.

    Swapping ``connection.con`` for a proxy that no-ops ``commit``/``rollback``
    while still forwarding cursor/execute calls to the real connection lets
    chronify own the transaction lifecycle for the duration of
    :meth:`SQLiteBackend.transaction`. We restore the real connection on
    commit/rollback.
    """

    __slots__ = ("_real",)

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def commit(self) -> None:  # noqa: D401
        return None

    def rollback(self) -> None:  # noqa: D401
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class SQLiteBackend(IbisBackend):
    """Ibis backend for SQLite databases."""

    def __init__(
        self,
        database: str | Path = ":memory:",
        connection: ibis.BaseBackend | None = None,
    ) -> None:
        """Construct a SQLiteBackend.

        Parameters
        ----------
        database
            Path to a SQLite database file, or ``":memory:"`` for an in-memory
            database. Ignored when ``connection`` is provided.
        connection
            Optional pre-existing ibis SQLite connection. When provided, the
            backend does not own the connection and will not disconnect it on
            ``dispose()``.
        """
        if connection is not None and str(database) != ":memory:":
            msg = f"{database=} and {connection=} cannot both be set"
            raise ConflictingInputsError(msg)

        self._in_transaction = False
        self._real_con: sqlite3.Connection | None = None
        self._owns_connection = connection is None
        if connection is None:
            db = str(database)
            self._database = None if db == ":memory:" else db
            self._connection = ibis.sqlite.connect(db)
        else:
            if connection.name != "sqlite":
                msg = f"SQLiteBackend requires a SQLite ibis connection, got {connection.name!r}"
                raise InvalidParameter(msg)
            self._connection = connection
            self._database = _infer_sqlite_path(connection)

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
        obj: pd.DataFrame | pa.Table | ibis.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ibis.Table:
        if isinstance(obj, ibis.Table):
            # SQLite CREATE TABLE AS SELECT loses datetime type info.
            # Execute the expression first, then create from the DataFrame.
            df = self._connection.execute(obj)
            return self._connection.create_table(name, obj=df, overwrite=overwrite)
        return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def insert(self, name: str, data: pd.DataFrame | pa.Table | ibis.Table) -> None:
        if isinstance(data, ibis.Table):
            data = data.execute()
        elif isinstance(data, pa.Table):
            data = data.to_pandas()
        # Use raw SQLite cursor for parameterized inserts
        con = self._connection.con  # raw sqlite3 connection
        table = self._connection.table(name)
        columns = list(table.columns)
        _validate_insert_columns(name, columns, list(data.columns))
        placeholders = ", ".join(["?"] * len(columns))
        col_list = ", ".join(_quote_identifier(c) for c in columns)
        quoted_name = _quote_identifier(name)
        sql = f"INSERT INTO {quoted_name} ({col_list}) VALUES ({placeholders})"

        ordered = data.loc[:, columns]
        rows = [tuple(_adapt_value(v) for v in row) for row in ordered.itertuples(index=False)]
        cursor = con.cursor()
        cursor.executemany(sql, rows)
        self._commit_if_needed()
        logger.trace("Inserted {} rows into {}", len(data), name)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        con = self._connection.con
        quoted_name = _quote_identifier(name)
        where = " AND ".join(f"{_quote_identifier(c)} = ?" for c in values)
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        con.execute(sql, list(values.values()))
        self._commit_if_needed()
        logger.trace("Deleted rows from {} matching {}", name, values)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ibis.Table, ObjectType]:
        # SQLite can't read Parquet natively. Load into a table instead.
        df = pd.read_parquet(path)
        return self.create_table(name, obj=df), ObjectType.TABLE

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._connection.con.execute(query)
        self._commit_if_needed()

    def dispose(self) -> None:
        if self._owns_connection:
            self._connection.disconnect()

    def backup(self, dst: str) -> None:
        if self._database is None:
            msg = "backup is only supported with a database backed by a file"
            raise InvalidOperation(msg)
        dst_con = sqlite3.connect(dst)
        try:
            self._connection.con.backup(dst_con)
        finally:
            dst_con.close()

    def _begin_transaction(self) -> None:
        # Issue BEGIN before swapping in the proxy so a failure (e.g. nested
        # BEGIN, or an outer caller already in a transaction) leaves the
        # backend in its original state rather than permanently pointed at
        # _NoCommitProxy.
        real = self._connection.con
        real.execute("BEGIN")
        # Swap in a proxy so ibis's internal con.commit() during DDL (notably
        # in create_table) is suppressed for the duration of the transaction.
        # The real connection is held aside and used directly for BEGIN /
        # COMMIT / ROLLBACK.
        self._real_con = real
        self._connection.con = _NoCommitProxy(real)  # type: ignore[assignment]
        self._in_transaction = True

    def _commit_transaction(self) -> None:
        assert self._real_con is not None
        real = self._real_con
        # Restore connection state up front so a failure on COMMIT (e.g. the
        # user finalized the transaction inside the block via
        # ``execute_sql("COMMIT")``) doesn't leak the no-commit proxy into
        # the backend's steady state. The ``in_transaction`` guard makes the
        # wrapper a no-op when the underlying transaction has already been
        # closed.
        self._connection.con = real
        self._real_con = None
        self._in_transaction = False
        if real.in_transaction:
            real.execute("COMMIT")

    def _rollback_transaction(self) -> None:
        assert self._real_con is not None
        real = self._real_con
        self._connection.con = real
        self._real_con = None
        self._in_transaction = False
        if real.in_transaction:
            real.execute("ROLLBACK")

    def _commit_if_needed(self) -> None:
        if not self._in_transaction:
            self._connection.con.commit()

    @singledispatchmethod
    def _prepare_write_data(
        self,
        data: Any,
        configs: Iterable[TimeBaseModel],
    ) -> pd.DataFrame:
        """SQLite stores timestamps as text, so joins compare raw strings.

        Canonicalize all tz-aware columns to UTC on write so joins between columns
        written from different source zones (e.g., source table in ``Etc/GMT+5``
        vs. a mapping table localized from tz-naive input) align.
        """
        msg = f"Unsupported data type: {type(data)}"
        raise TypeError(msg)

    @_prepare_write_data.register
    def _(self, data: pa.Table, configs: Iterable[TimeBaseModel]) -> pd.DataFrame:
        return self._prepare_write_data(data.to_pandas(), configs)

    @_prepare_write_data.register
    def _(self, data: ibis.Table, configs: Iterable[TimeBaseModel]) -> pd.DataFrame:
        return self._prepare_write_data(data.execute(), configs)

    @_prepare_write_data.register
    def _(self, data: pd.DataFrame, configs: Iterable[TimeBaseModel]) -> pd.DataFrame:
        data = _normalize_timestamps(data, configs)
        copied = False
        for config in configs:
            if not isinstance(config, _DATETIME_RANGES):
                continue
            if config.dtype != TimeDataType.TIMESTAMP_TZ:
                continue
            if config.time_column not in data.columns:
                continue
            if not isinstance(data[config.time_column].dtype, DatetimeTZDtype):
                continue
            if not copied:
                data = data.copy()
                copied = True
            data[config.time_column] = data[config.time_column].dt.tz_convert("UTC")
        return data

    def _supports_parquet_partitioning(self) -> bool:
        return False


def _infer_sqlite_path(connection: ibis.BaseBackend) -> str | None:
    """Return the database file path for an ibis SQLite connection, or None for in-memory."""
    try:
        row = connection.con.execute("PRAGMA database_list").fetchone()
    except Exception:
        return None
    if not row:
        return None
    # PRAGMA database_list returns (seq, name, file); empty string => in-memory.
    path = row[2]
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
    """Quote a SQL identifier for SQLite, escaping embedded double quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
