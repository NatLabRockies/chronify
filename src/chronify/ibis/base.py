"""Abstract base class for Ibis database backends."""

from abc import ABC, abstractmethod
from collections import Counter
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Generator, Sequence, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
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


def _check_one_config_per_datetime_column(configs: Sequence[TimeBaseModel]) -> None:
    time_col_count = Counter(
        config.time_column for config in configs if isinstance(config, DatetimeRangeBase)
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if time_col_dup:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _normalize_timestamps(
    df: pd.DataFrame,
    configs: Sequence[TimeBaseModel],
) -> pd.DataFrame:
    """Normalize datetime columns so their pandas dtype matches the schema config."""
    copied = False
    for config in configs:
        if not isinstance(config, _DATETIME_RANGES):
            continue
        col = config.time_column
        if col not in df.columns:
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
    configs: Sequence[TimeBaseModel],
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


class ObjectType(StrEnum):
    TABLE = "table"
    VIEW = "view"


class IbisBackend(ABC):
    """Abstract base class defining the interface for Ibis database backends."""

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

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table in the database."""
        return self.connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
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

    def table(self, name: str) -> ir.Table:
        """Return an ibis table expression for the named table."""
        return self.connection.table(name)

    @abstractmethod
    def insert(self, name: str, data: pd.DataFrame | pa.Table) -> None:
        """Insert data into an existing table."""

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

    def sql(self, query: str) -> ir.Table:
        """Create an ibis table expression from a raw SQL string."""
        return self.connection.sql(query)

    @abstractmethod
    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        """Write an ibis expression result to a Parquet file."""

    @abstractmethod
    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        """Create a view or table backed by a Parquet file.

        Returns the table expression and the type of object created, since some
        backends (e.g., SQLite) must create a table instead of a view.
        """

    def has_table(self, name: str) -> bool:
        """Check whether a table or view exists."""
        return name in self.list_tables()

    def execute_sql(self, query: str) -> Any:
        """Execute a raw SQL statement (no result expected)."""
        logger.trace("execute_sql: {}", query)
        return self.connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query and return a DataFrame."""
        logger.trace("execute_sql_to_df: {}", query)
        return cast(pd.DataFrame, self.connection.raw_sql(query).fetch_df())

    def read_query(self, expr: ir.Table, config: TimeBaseModel) -> pd.DataFrame:
        """Execute an Ibis expression and return a normalized pandas DataFrame."""
        df = self.execute(expr)
        if isinstance(config, _DATETIME_RANGES):
            self._post_read_normalize(df, config)
        return df

    def write_table(
        self,
        data: pd.DataFrame | pa.Table,
        name: str,
        configs: Sequence[TimeBaseModel],
        if_exists: str = "append",
    ) -> None:
        """Write tabular data to the database, applying backend-specific normalization."""
        _check_one_config_per_datetime_column(configs)
        prepared = self._prepare_write_data(data, configs)
        self._apply_if_exists(prepared, name, if_exists)

    def _post_read_normalize(self, df: pd.DataFrame, config: DatetimeRanges) -> None:
        """Backend-specific in-place normalization of a read DataFrame.

        Default: no-op. Backends whose drivers return non-canonical datetime
        types should override.
        """

    def _prepare_write_data(
        self,
        data: pd.DataFrame | pa.Table,
        configs: Sequence[TimeBaseModel],
    ) -> pd.DataFrame | pa.Table:
        """Normalize data before insert/create_table.

        Default behavior is the DuckDB path: accept Arrow natively when possible,
        otherwise convert to pandas to normalize tz-sensitive columns. Subclasses
        that cannot ingest Arrow directly should convert here.
        """
        if isinstance(data, pa.Table) and _arrow_needs_timestamp_normalization(data, configs):
            data = data.to_pandas()
        if isinstance(data, pd.DataFrame):
            data = _normalize_timestamps(data, configs)
        return data

    def _apply_if_exists(
        self,
        data: pd.DataFrame | pa.Table,
        name: str,
        if_exists: str,
    ) -> None:
        match if_exists:
            case "append":
                self.insert(name, data)
            case "replace":
                self.drop_table(name)
                self.create_table(name, data)
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
    def transaction(self) -> Generator[list[tuple[str, ObjectType]], None, None]:
        """Context manager for pseudo-transactions.

        Tracks created objects (tables/views) so they can be cleaned up on failure.
        On success, created objects are kept. On exception, they are dropped.

        Yields a list to which callers should append (name, ObjectType) tuples.
        """
        created: list[tuple[str, ObjectType]] = []
        self._begin_transaction()
        try:
            yield created
        except Exception:
            self._rollback_transaction()
            for obj_name, obj_type in reversed(created):
                try:
                    if obj_type == ObjectType.TABLE:
                        self.drop_table(obj_name)
                    else:
                        self.drop_view(obj_name)
                    logger.debug("Rolled back {} {}", obj_type.value, obj_name)
                except Exception:
                    logger.warning("Failed to roll back {} {}", obj_type.value, obj_name)
            raise
        else:
            self._commit_transaction()
