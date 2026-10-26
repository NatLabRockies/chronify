"""SQLite backend implementation for Ibis."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from functools import singledispatchmethod
from typing import Any, Iterable

import ibis
import numpy as np
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
    _quote_identifier,
    _validate_insert_columns,
)
from chronify.time import TimeDataType
from chronify.time_configs import TimeBaseModel


def _adapt_value(v: Any) -> Any:
    """Convert a value for SQLite parameterized insertion.

    Converts datetime/Timestamp objects to ISO-format strings to avoid the
    Python 3.12+ DeprecationWarning about the default datetime adapter.
    Returns None for pd.NaT and other missing-value sentinels, and converts
    numpy scalars to Python scalars.
    """
    if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    # sqlite3 binds numpy integers (e.g., from nullable Int64 columns) as BLOBs.
    if isinstance(v, (np.number, np.bool_)):
        return v.item()
    return v


def _adapt_column(s: pd.Series) -> list[Any]:
    """Vectorized equivalent of applying :func:`_adapt_value` to every cell."""
    if pd.api.types.is_datetime64_any_dtype(s) and not s.dt.nanosecond.gt(0).any():
        # Same text as datetime.isoformat(): microseconds only when nonzero,
        # '+HH:MM' offset when tz-aware.
        wall = s.dt.tz_localize(None) if s.dt.tz is not None else s
        out = pd.Series(np.datetime_as_string(wall.to_numpy(), unit="us"), index=s.index)
        out = out.where(s.dt.microsecond != 0, out.str[:19])
        if s.dt.tz is not None:
            # Offsets have few distinct values; let isoformat() render each one.
            delta = wall - s.dt.tz_convert(None)
            offsets = {
                d: datetime(2000, 1, 1, tzinfo=timezone(d)).isoformat()[19:]
                for d in delta.dropna().unique()
            }
            out = out + delta.map(offsets)
        return out.astype(object).where(s.notna(), None).tolist()
    if pd.api.types.infer_dtype(s, skipna=True) in (
        "string",
        "integer",
        "floating",
        "boolean",
        "empty",
    ):
        # astype(object) yields Python scalars, which sqlite3 binds natively.
        return s.astype(object).where(s.notna(), None).tolist()
    return [_adapt_value(v) for v in s]


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
            # SQLite CREATE TABLE AS SELECT loses datetime type info, so create the
            # table from the schema and fill it in the database.
            with self.transaction():
                table = self._connection.create_table(
                    name, schema=schema or obj.schema(), overwrite=overwrite
                )
                self.insert(name, obj)
            return table
        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if obj is None:
            return self._connection.create_table(name, schema=schema, overwrite=overwrite)
        # ibis binds DataFrame values directly and sqlite3 rejects NaT, so create
        # the empty table from the inferred schema and fill it via insert(), which
        # adapts missing values and timestamps.
        with self.transaction():
            table = self._connection.create_table(
                name, schema=schema or ibis.memtable(obj).schema(), overwrite=overwrite
            )
            self.insert(name, obj)
        return table

    def insert(self, name: str, data: pd.DataFrame | pa.Table | ibis.Table) -> None:
        if isinstance(data, pa.Table):
            data = data.to_pandas()
        con = self._connection.con  # raw sqlite3 connection
        columns = list(self._connection.table(name).columns)
        _validate_insert_columns(name, columns, list(data.columns))
        col_list = ", ".join(_quote_identifier(c) for c in columns)
        quoted_name = _quote_identifier(name)
        if isinstance(data, ibis.Table):
            # Stay in the database. Timestamps are copied as the ISO text they are
            # stored as; an expression that computes timestamps with SQLite functions
            # would produce a different text format.
            select = self._connection.compile(data.select(columns))
            con.execute(f"INSERT INTO {quoted_name} ({col_list}) {select}")
            self._commit_if_needed()
            logger.trace("Inserted query results into {}", name)
            return
        # Use raw SQLite cursor for parameterized inserts
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {quoted_name} ({col_list}) VALUES ({placeholders})"

        ordered = data.loc[:, columns]
        rows = list(zip(*(_adapt_column(ordered[c]) for c in columns)))
        with closing(con.cursor()) as cursor:
            cursor.executemany(sql, rows)
        self._commit_if_needed()
        logger.trace("Inserted {} rows into {}", len(data), name)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        con = self._connection.con
        quoted_name = _quote_identifier(name)
        where = " AND ".join(f"{_quote_identifier(c)} = ?" for c in values)
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        # Adapt values so datetime predicates match the stored ISO strings.
        con.execute(sql, [_adapt_value(v) for v in values.values()])
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

    def execute_sql_to_df(self, query: str, params: Any = None) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        return pd.read_sql_query(query, self._connection.con, params=params)

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
        self._connection.con = _NoCommitProxy(real)
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

    def migrate_legacy_timestamps(self, time_columns: dict[str, dict[str, bool]]) -> list[str]:
        """Rewrite timestamps stored by chronify <= 0.7 (SQLAlchemy) into the current format.

        SQLAlchemy stored timestamps as ``YYYY-MM-DD HH:MM:SS.ffffff`` (UTC wall
        clock for tz-aware data) in ``DATETIME`` columns, or ``NUM`` columns for
        tables built with CREATE TABLE AS SELECT. Ibis writes ISO-8601 into
        ``TIMESTAMP`` / ``TIMESTAMPTZ`` columns. SQLite compares timestamps as
        text, so mixing the two silently breaks joins.

        Parameters
        ----------
        time_columns
            Registered time columns: ``{table: {column: is_tz_aware}}``.

        Returns the names of the migrated tables.
        """
        con = self._connection.con
        plans: dict[str, dict[str, bool]] = {}
        for (table,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall():
            declared = {
                row[1]: row[2].upper()
                for row in con.execute(f"PRAGMA table_info({_quote_identifier(table)})")
            }
            registered = time_columns.get(table, {})
            legacy = {
                col: registered.get(col, False)
                for col, typ in declared.items()
                if typ == "DATETIME"
                or (col in registered and typ not in ("TIMESTAMP", "TIMESTAMPTZ"))
            }
            if legacy:
                plans[table] = legacy

        if not plans:
            return []

        con.execute("PRAGMA legacy_alter_table = ON")  # don't validate views during RENAME
        try:
            with self.transaction():
                for table, legacy in plans.items():
                    self._rebuild_with_iso_timestamps(table, legacy)
        finally:
            con.execute("PRAGMA legacy_alter_table = OFF")
        return list(plans)

    def _rebuild_with_iso_timestamps(self, table: str, legacy: dict[str, bool]) -> None:
        con = self._connection.con
        columns = [
            (row[1], row[2])
            for row in con.execute(f"PRAGMA table_info({_quote_identifier(table)})")
        ]
        defs = []
        exprs = []
        for col, typ in columns:
            q = _quote_identifier(col)
            if col not in legacy:
                defs.append(f"{q} {typ}".rstrip())
                exprs.append(q)
                continue
            is_tz = legacy[col]
            defs.append(f"{q} {'TIMESTAMPTZ' if is_tz else 'TIMESTAMP'}")
            # Matches Python's datetime.isoformat(), which is what _adapt_value and
            # ibis write: microseconds only when nonzero, '+00:00' for UTC.
            iso = (
                f"substr({q}, 1, 10) || 'T' || substr({q}, 12, 8) || "
                f"CASE WHEN substr({q}, 21, 6) IN ('', '000000') THEN '' "
                f"ELSE '.' || substr({q}, 21, 6) END" + (" || '+00:00'" if is_tz else "")
            )
            exprs.append(
                f"CASE WHEN {q} GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] "
                f"[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*' THEN {iso} ELSE {q} END"
            )
        tmp = _quote_identifier(f"__chronify_migrate_{table}")
        quoted = _quote_identifier(table)
        con.execute(f"CREATE TABLE {tmp} ({', '.join(defs)})")
        con.execute(f"INSERT INTO {tmp} SELECT {', '.join(exprs)} FROM {quoted}")
        con.execute(f"DROP TABLE {quoted}")
        con.execute(f"ALTER TABLE {tmp} RENAME TO {quoted}")


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
