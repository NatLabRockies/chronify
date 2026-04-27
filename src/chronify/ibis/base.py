"""Abstract base class for Ibis database backends."""

from abc import ABC, abstractmethod
from collections import Counter
from contextlib import contextmanager
from enum import StrEnum
from functools import singledispatch, singledispatchmethod
from typing import Any, Generator, Iterable, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger
from pandas import DatetimeTZDtype

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.time import TimeDataType
from chronify.time_configs import (
    DatetimeRange,
    DatetimeRangeBase,
    DatetimeRangeWithTZColumn,
    TimeBaseModel,
)

_DATETIME_RANGES = (DatetimeRange, DatetimeRangeWithTZColumn)
DatetimeRanges = DatetimeRange | DatetimeRangeWithTZColumn


def _check_one_config_per_datetime_column(configs: Iterable[TimeBaseModel]) -> None:
    time_col_count = Counter(
        config.time_column for config in configs if isinstance(config, DatetimeRangeBase)
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if time_col_dup:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _normalize_timestamps(
    df: pd.DataFrame,
    configs: Iterable[TimeBaseModel],
) -> pd.DataFrame:
    """Normalize datetime columns so their pandas dtype matches the schema config.
    Does not change the caller's DataFrame.
    """
    copied = False
    columns = set(df.columns)
    for config in configs:
        if not isinstance(config, _DATETIME_RANGES):
            continue
        col = config.time_column
        if col not in columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        is_tz_aware = isinstance(df[col].dtype, DatetimeTZDtype)

        if config.dtype == TimeDataType.TIMESTAMP_NTZ and is_tz_aware:
            if not copied:
                df = df.copy()
                copied = True
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        elif config.dtype == TimeDataType.TIMESTAMP_TZ and not is_tz_aware:
            if not copied:
                df = df.copy()
                copied = True
            df[col] = df[col].dt.tz_localize("UTC")

    return df


def _arrow_needs_timestamp_normalization(
    table: pa.Table,
    configs: Iterable[TimeBaseModel],
) -> bool:
    fields = {field.name: field.type for field in table.schema}
    for config in configs:
        if not isinstance(config, _DATETIME_RANGES):
            continue
        arrow_type = fields.get(config.time_column)
        if arrow_type is None or not pa.types.is_timestamp(arrow_type):
            continue
        timezone = arrow_type.tz
        if config.dtype == TimeDataType.TIMESTAMP_NTZ and timezone is not None:
            return True
        if config.dtype == TimeDataType.TIMESTAMP_TZ and timezone is None:
            return True
    return False


def _normalize_arrow_timestamps(
    table: pa.Table,
    configs: Iterable[TimeBaseModel],
) -> pa.Table:
    """Normalize timestamp columns of an Arrow table to match the schema configs.

    Casts tz-aware → tz-naive (preserving UTC instants) and localizes tz-naive →
    UTC, matching the semantics of :func:`_normalize_timestamps` for pandas. Stays
    in Arrow so backends that ingest Arrow natively avoid a pandas round-trip.
    """
    indices = {name: i for i, name in enumerate(table.column_names)}
    for config in configs:
        if not isinstance(config, _DATETIME_RANGES):
            continue
        idx = indices.get(config.time_column)
        if idx is None:
            continue
        arr = table.column(idx)
        if not pa.types.is_timestamp(arr.type):
            continue
        is_tz_aware = arr.type.tz is not None
        if config.dtype == TimeDataType.TIMESTAMP_NTZ and is_tz_aware:
            new_arr = arr.cast(pa.timestamp(arr.type.unit))
        elif config.dtype == TimeDataType.TIMESTAMP_TZ and not is_tz_aware:
            # pyarrow 24 ships incomplete type stubs that omit dynamically-
            # registered compute kernels like assume_timezone, even though
            # the function exists at runtime.
            new_arr = pc.assume_timezone(arr, "UTC")  # type: ignore[attr-defined]
        else:
            continue
        table = table.set_column(idx, table.column_names[idx], new_arr)
    return table


@singledispatch
def _get_columns(data: Any) -> list[str]:
    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


@_get_columns.register
def _(data: pd.DataFrame) -> list[str]:
    return list(data.columns)


@_get_columns.register
def _(data: pa.Table) -> list[str]:
    return cast(list[str], data.column_names)


@_get_columns.register
def _(data: ibis.Table) -> list[str]:
    return list(data.columns)


@singledispatch
def _select_columns(data: Any, columns: list[str]) -> Any:
    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


@_select_columns.register
def _(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data[columns]


@_select_columns.register
def _(data: pa.Table, columns: list[str]) -> pa.Table:
    return data.select(columns)


@_select_columns.register
def _(data: ibis.Table, columns: list[str]) -> ibis.Table:
    return data.select(columns)


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


class ObjectType(StrEnum):
    TABLE = "table"
    VIEW = "view"


class IbisBackend(ABC):
    """Abstract base class defining the interface for Ibis database backends.

    Subclasses must set ``self._in_transaction = False`` in ``__init__``.
    The flag is read by :meth:`transaction` (to make inner blocks
    passthroughs) and on SQLite by ``_commit_if_needed`` (to decide whether
    DML auto-commits).
    """

    _in_transaction: bool

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'duckdb', 'sqlite', 'spark')."""

    @property
    @abstractmethod
    def database(self) -> str | None:
        """Return the database file path, or None for in-memory."""

    @property
    @abstractmethod
    def connection(self) -> ibis.BaseBackend:
        """Return the underlying ibis connection."""

    @abstractmethod
    def _supports_parquet_partitioning(self) -> bool:
        """Return True if the backend supports Hive partitioning of Parquet files."""

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ibis.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ibis.Table:
        """Create a table in the database."""
        return self.connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def create_view(self, name: str, expr: ibis.Table) -> ibis.Table:
        """Create a view in the database."""
        return self.connection.create_view(name, expr, overwrite=False)

    def drop_table(self, name: str) -> None:
        """Drop a table from the database."""
        self.connection.drop_table(name, force=True)

    def drop_view(self, name: str) -> None:
        """Drop a view from the database."""
        self.connection.drop_view(name, force=True)

    def list_tables(self) -> list[str]:
        """List all user tables in the database."""
        return cast(list[str], self.connection.list_tables())

    def table(self, name: str) -> ibis.Table:
        """Return an ibis table expression for the named table."""
        return self.connection.table(name)

    def insert(self, name: str, data: pd.DataFrame | pa.Table | ibis.Table) -> None:
        """Insert data into an existing table.

        Validates that the data columns match the target table, reorders them,
        and delegates to the underlying Ibis connection. Subclasses should
        override when the default does not cooperate with backend-specific
        transaction semantics.
        """
        target_columns = list(self.table(name).columns)
        _validate_insert_columns(name, target_columns, _get_columns(data))
        ordered = _select_columns(data, target_columns)
        self.connection.insert(name, ordered)
        logger.trace("Inserted data into {}", name)

    @abstractmethod
    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        """Delete rows from a table where every column equals its given value.

        Identifiers must be quoted and values must be parameterized to avoid
        SQL injection and to handle values containing quote characters.
        """

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        """Execute an ibis expression and return a DataFrame. Must not be called
        for large tables."""
        return cast(pd.DataFrame, self.connection.execute(expr))

    def sql(self, query: str) -> ibis.Table:
        """Create an ibis table expression from a raw SQL string."""
        return self.connection.sql(query)

    def write_parquet(
        self,
        expr: ibis.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        """Write an ibis expression result to a Parquet file."""
        if partition_by:
            if not self._supports_parquet_partitioning():
                msg = f"{self.name} backend does not support partitioned Parquet writes."
                raise NotImplementedError(msg)
            self.connection.to_parquet(expr, path, partition_by=partition_by)
        else:
            self.connection.to_parquet(expr, path)

    @abstractmethod
    def create_view_from_parquet(self, path: str, name: str) -> tuple[ibis.Table, ObjectType]:
        """Create a view or table backed by a Parquet file.

        Returns the table expression and the type of object created, since some
        backends (e.g., SQLite) must create a table instead of a view.
        """

    def has_table(self, name: str) -> bool:
        """Check whether a table or view exists."""
        return name in self.list_tables()

    def execute_sql(self, query: str) -> None:
        """Execute a raw SQL statement (no result expected)."""
        logger.trace("execute_sql: {}", query)
        self.connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query and return a DataFrame."""
        logger.trace("execute_sql_to_df: {}", query)
        return self.execute(self.sql(query))

    def read_query(self, expr: ibis.Table, config: TimeBaseModel) -> pd.DataFrame:
        """Execute an Ibis expression and return a pandas DataFrame."""
        return self.execute(self.apply_schema_types(expr, config))

    def apply_schema_types(self, expr: ibis.Table, config: TimeBaseModel) -> ibis.Table:
        """Return ``expr`` with backend-specific casts applied so its Ibis type
        matches ``config``.

        Default: no-op. Backends whose schema cannot express the full type
        (e.g. Spark, which has no per-column timezone — only a session-wide
        ``spark.sql.session.timeZone`` — so ``TIMESTAMP`` columns are reported
        as tz-naive even when the values are true UTC instants) should
        override to add lazy ``cast`` expressions. This lets callers get
        correctly-typed pandas output without forcing a
        materialize-then-normalize round-trip.
        """
        return expr

    def write_table(
        self,
        data: pd.DataFrame | pa.Table | ibis.Table,
        name: str,
        configs: Iterable[TimeBaseModel],
        if_exists: str = "append",
    ) -> None:
        """Write tabular data to the database, applying backend-specific normalization.

        ``ibis.Table`` inputs are passed through to the underlying connection so
        materialization can be deferred; backends that cannot ingest an ibis
        expression directly should override :meth:`_prepare_write_data` to
        materialize it.
        """
        _check_one_config_per_datetime_column(configs)
        prepared = self._prepare_write_data(data, configs)
        self._apply_if_exists(prepared, name, if_exists)

    @singledispatchmethod
    def _prepare_write_data(
        self,
        data: Any,
        configs: Iterable[TimeBaseModel],
    ) -> pd.DataFrame | pa.Table | ibis.Table:
        """Normalize data before insert/create_table.

        Default behavior is the DuckDB path: accept Arrow natively when possible,
        otherwise convert to pandas to normalize tz-sensitive columns. Subclasses
        that cannot ingest Arrow directly should convert here.
        """
        msg = f"Unsupported data type: {type(data)}"
        raise TypeError(msg)

    @_prepare_write_data.register
    def _(self, data: pd.DataFrame, configs: Iterable[TimeBaseModel]) -> pd.DataFrame:
        return _normalize_timestamps(data, configs)

    @_prepare_write_data.register
    def _(self, data: pa.Table, configs: Iterable[TimeBaseModel]) -> pa.Table:
        if _arrow_needs_timestamp_normalization(data, configs):
            return _normalize_arrow_timestamps(data, configs)
        return data

    @_prepare_write_data.register
    def _(self, data: ibis.Table, configs: Iterable[TimeBaseModel]) -> ibis.Table:
        return data

    def _apply_if_exists(
        self,
        data: pd.DataFrame | pa.Table | ibis.Table,
        name: str,
        if_exists: str,
    ) -> None:
        match if_exists:
            case "append":
                self.insert(name, data)
            case "replace":
                self.create_table(name, data, overwrite=True)
            case "fail":
                self.create_table(name, data)
            case _:
                msg = f"Invalid if_exists value: {if_exists}"
                raise InvalidOperation(msg)

    def dispose(self) -> None:
        """Dispose of the backend connection."""
        self.connection.disconnect()

    @abstractmethod
    def backup(self, dst: str) -> None:
        """Copy the database to a new location.

        Not supported for in-memory databases or backends without persistent
        file storage (e.g., Spark).
        """

    def _begin_transaction(self) -> None:
        """Start a real database transaction, if the backend supports one."""

    def _commit_transaction(self) -> None:
        """Commit a real database transaction, if one was started."""

    def _rollback_transaction(self) -> None:
        """Roll back a real database transaction, if one was started."""

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for a database transaction.

        On DuckDB and SQLite this issues a real ``BEGIN`` / ``COMMIT`` /
        ``ROLLBACK`` and covers both DML and DDL — work inside the block is
        atomic. Spark has no transaction support, so partial writes inside the
        block are not rolled back; callers that need to clean up after a
        Spark failure must do so themselves.

        Nesting is supported as a passthrough: a ``transaction()`` block
        opened while one is already active does not start a new transaction
        and does not commit or roll back on its own. The outermost block
        controls the lifecycle, so callers can wrap operations that
        themselves use ``transaction()`` (e.g. mapping helpers) without
        savepoints.
        """
        if self._in_transaction:
            yield
            return
        self._begin_transaction()
        self._in_transaction = True
        try:
            yield
        except Exception:
            try:
                self._rollback_transaction()
            finally:
                self._in_transaction = False
            raise
        else:
            try:
                self._commit_transaction()
            finally:
                self._in_transaction = False
