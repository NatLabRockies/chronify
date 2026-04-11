"""Database I/O functions using Ibis backends."""

from collections import Counter
from pathlib import Path
from typing import Sequence

import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from pandas import DatetimeTZDtype

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType
from chronify.time import TimeDataType
from chronify.time_configs import (
    DatetimeRange,
    DatetimeRangeBase,
    DatetimeRangeWithTZColumn,
    TimeBaseModel,
)
from chronify.utils.path_utils import check_overwrite

DatetimeRanges = DatetimeRange | DatetimeRangeWithTZColumn
_DATETIME_RANGES = (DatetimeRange, DatetimeRangeWithTZColumn)


def read_table(
    backend: IbisBackend,
    table_name: str,
    config: TimeBaseModel,
) -> pd.DataFrame:
    """Read a table from the database."""
    table = backend.table(table_name)
    df = backend.execute(table)

    if backend.name == "sqlite" and isinstance(config, _DATETIME_RANGES):
        _convert_database_output_for_datetime(df, config)
    elif backend.name == "spark" and isinstance(config, _DATETIME_RANGES):
        _convert_spark_output_for_datetime(df, config)

    return df


def read_query(
    backend: IbisBackend,
    expr: ir.Table,
    config: TimeBaseModel,
) -> pd.DataFrame:
    """Execute an Ibis expression and return results."""
    df = backend.execute(expr)

    if backend.name == "sqlite" and isinstance(config, _DATETIME_RANGES):
        _convert_database_output_for_datetime(df, config)
    elif backend.name == "spark" and isinstance(config, _DATETIME_RANGES):
        _convert_spark_output_for_datetime(df, config)

    return df


def _normalize_timestamps(
    df: pd.DataFrame,
    configs: Sequence[TimeBaseModel],
) -> pd.DataFrame:
    """Normalize datetime columns so their pandas dtype matches the schema config.

    - TIMESTAMP_NTZ + tz-aware input  → convert to UTC, then strip timezone
    - TIMESTAMP_TZ  + tz-naive input  → localize as UTC
    - matching dtype                  → no change

    This runs before any backend-specific handling so that all backends receive
    consistently typed data.
    """
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


def write_table(
    backend: IbisBackend,
    df: pd.DataFrame | pa.Table,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_exists: str = "append",
) -> None:
    """Write a DataFrame to the database."""
    if isinstance(df, pa.Table):
        df = df.to_pandas()

    _check_one_config_per_datetime_column(configs)
    df = _normalize_timestamps(df, configs)

    match backend.name:
        case "duckdb":
            _write_to_duckdb(backend, df, table_name, if_exists)
        case "sqlite":
            _write_to_sqlite(backend, df, table_name, configs, if_exists)
        case "spark":
            _write_to_spark(backend, df, table_name, if_exists)
        case _:
            msg = f"Unsupported backend: {backend.name}"
            raise NotImplementedError(msg)


def write_parquet(
    backend: IbisBackend,
    query: str | ir.Table,
    output_file: Path,
    overwrite: bool = False,
    partition_columns: list[str] | None = None,
    config: TimeBaseModel | None = None,
) -> None:
    """Write query results to a Parquet file."""
    check_overwrite(output_file, overwrite)

    if isinstance(query, str):
        expr = backend.sql(query)
    else:
        expr = query

    if backend.name == "spark" and isinstance(config, _DATETIME_RANGES):
        df = backend.execute(expr)
        _convert_spark_output_for_datetime(df, config)
        if partition_columns:
            df.to_parquet(output_file, partition_cols=partition_columns)
        else:
            df.to_parquet(output_file)
        return

    backend.write_parquet(expr, str(output_file), partition_by=partition_columns)


def create_view_from_parquet(
    backend: IbisBackend,
    filename: Path,
    view_name: str,
) -> ObjectType:
    """Create a view (or table for SQLite) from a Parquet file.

    Returns the ObjectType created so callers can clean up correctly.
    """
    _, obj_type = backend.create_view_from_parquet(str(filename), view_name)
    return obj_type


def _check_one_config_per_datetime_column(configs: Sequence[TimeBaseModel]) -> None:
    time_col_count = Counter(
        config.time_column for config in configs if isinstance(config, DatetimeRangeBase)
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if time_col_dup:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _convert_database_input_for_datetime(
    df: pd.DataFrame, config: DatetimeRanges, copied: bool
) -> tuple[pd.DataFrame, bool]:
    """Convert DataFrame datetime columns for SQLite input (store as UTC)."""
    if config.dtype == TimeDataType.TIMESTAMP_NTZ:
        return df, copied

    if not copied:
        df = df.copy()
        copied = True

    if isinstance(df[config.time_column].dtype, DatetimeTZDtype):
        df[config.time_column] = df[config.time_column].dt.tz_convert("UTC")
    else:
        df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")

    return df, copied


def _convert_database_output_for_datetime(df: pd.DataFrame, config: DatetimeRanges) -> None:
    """Convert DataFrame datetime columns after SQLite output."""
    if config.time_column not in df.columns:
        return

    col = df[config.time_column]
    if config.dtype == TimeDataType.TIMESTAMP_TZ:
        if col.dtype == object:
            df[config.time_column] = pd.to_datetime(col, utc=True)
        elif isinstance(col.dtype, DatetimeTZDtype):
            df[config.time_column] = col.dt.tz_convert("UTC")
        else:
            df[config.time_column] = col.dt.tz_localize("UTC")
    else:
        if col.dtype == object:
            df[config.time_column] = pd.to_datetime(col, utc=False)


def _convert_spark_output_for_datetime(df: pd.DataFrame, config: DatetimeRanges) -> None:
    """Convert DataFrame datetime columns after Spark output."""
    if config.time_column not in df.columns:
        return

    col = df[config.time_column]

    if config.dtype == TimeDataType.TIMESTAMP_TZ:
        if not pd.api.types.is_datetime64_any_dtype(col):
            col = pd.to_datetime(col, utc=True)
        elif isinstance(col.dtype, DatetimeTZDtype):
            col = col.dt.tz_convert("UTC")
        else:
            col = col.dt.tz_localize("UTC")
        df[config.time_column] = col.dt.as_unit("us")
    else:
        if not pd.api.types.is_datetime64_any_dtype(col):
            col = pd.to_datetime(col, utc=False)
            df[config.time_column] = col.astype("datetime64[us]")
        if isinstance(col.dtype, DatetimeTZDtype):
            df[config.time_column] = col.dt.tz_convert(None).astype("datetime64[us]")


def _write_to_duckdb(
    backend: IbisBackend,
    df: pd.DataFrame,
    table_name: str,
    if_exists: str,
) -> None:
    match if_exists:
        case "append":
            backend.insert(table_name, df)
        case "replace":
            backend.drop_table(table_name)
            backend.create_table(table_name, df)
        case "fail":
            backend.create_table(table_name, df)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)


def _write_to_sqlite(
    backend: IbisBackend,
    df: pd.DataFrame,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_exists: str,
) -> None:
    # SQLite-specific: ensure TZ timestamps are stored as UTC text.
    # _normalize_timestamps already ran, so NTZ columns are tz-naive and
    # TZ columns are tz-aware UTC. This step converts TZ to UTC for storage.
    copied = False
    for config in configs:
        if isinstance(config, _DATETIME_RANGES):
            df, copied = _convert_database_input_for_datetime(df, config, copied)

    match if_exists:
        case "append":
            backend.insert(table_name, df)
        case "replace":
            backend.drop_table(table_name)
            backend.create_table(table_name, df)
        case "fail":
            backend.create_table(table_name, df)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)


def _write_to_spark(
    backend: IbisBackend,
    df: pd.DataFrame,
    table_name: str,
    if_exists: str,
) -> None:
    match if_exists:
        case "append":
            backend.insert(table_name, df)
        case "replace":
            backend.drop_table(table_name)
            backend.create_table(table_name, df)
        case "fail":
            backend.create_table(table_name, df)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)
