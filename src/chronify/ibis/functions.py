"""Database I/O functions using Ibis backends.

This module provides functions to read and write data efficiently using Ibis.
"""

from collections import Counter
from pathlib import Path
from typing import Sequence

import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from numpy.dtypes import DateTime64DType, ObjectDType
from pandas import DatetimeTZDtype

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.ibis.backend import IbisBackend
from chronify.time import TimeDataType
from chronify.time_configs import DatetimeRangeBase, TimeBaseModel
from chronify.utils.path_utils import check_overwrite


def read_table(
    backend: IbisBackend,
    table_name: str,
    config: TimeBaseModel,
) -> pd.DataFrame:
    """Read a table from the database.

    Parameters
    ----------
    backend
        Ibis backend
    table_name
        Name of table to read
    config
        Time configuration for datetime conversion

    Returns
    -------
    pd.DataFrame
        Table data as DataFrame
    """
    table = backend.table(table_name)
    df = backend.execute(table)

    if backend.name == "sqlite" and isinstance(config, DatetimeRangeBase):
        _convert_database_output_for_datetime(df, config)

    return df


def read_query(
    backend: IbisBackend,
    expr: ir.Table,
    config: TimeBaseModel,
) -> pd.DataFrame:
    """Execute an Ibis expression and return results.

    Parameters
    ----------
    backend
        Ibis backend
    expr
        Ibis table expression to execute
    config
        Time configuration for datetime conversion

    Returns
    -------
    pd.DataFrame
        Query results as DataFrame
    """
    df = backend.execute(expr)

    if backend.name == "sqlite" and isinstance(config, DatetimeRangeBase):
        _convert_database_output_for_datetime(df, config)
    elif backend.name == "spark" and isinstance(config, DatetimeRangeBase):
        _convert_spark_output_for_datetime(df, config)

    return df


def write_table(
    backend: IbisBackend,
    df: pd.DataFrame | pa.Table,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_exists: str = "append",
    scratch_dir: Path | None = None,
) -> None:
    """Write a DataFrame to the database.

    Parameters
    ----------
    backend
        Ibis backend
    df
        DataFrame or PyArrow Table to write
    table_name
        Name of target table
    configs
        Time configurations for datetime handling
    if_exists
        What to do if table exists: "append", "replace", or "fail"
    scratch_dir
        Directory for temporary files (used by some backends)
    """
    match backend.name:
        case "duckdb":
            _write_to_duckdb(backend, df, table_name, if_exists)
        case "sqlite":
            _write_to_sqlite(backend, df, table_name, configs, if_exists)
        case "spark":
            _write_to_pyspark(backend, df, table_name, configs, if_exists, scratch_dir)
        case _:
            msg = f"Unsupported backend: {backend.name}"
            raise NotImplementedError(msg)


def write_parquet(
    backend: IbisBackend,
    query: str | ir.Table,
    output_file: Path,
    overwrite: bool = False,
    partition_columns: list[str] | None = None,
) -> None:
    """Write query results to a Parquet file.

    Parameters
    ----------
    backend
        Ibis backend
    query
        SQL query string or Ibis table expression
    output_file
        Output file path
    overwrite
        If True, overwrite existing file
    partition_columns
        Optional columns to partition by
    """
    check_overwrite(output_file, overwrite)

    if isinstance(query, str):
        expr = backend.sql(query)
    else:
        expr = query

    backend.write_parquet(
        expr,
        output_file,
        partition_by=partition_columns,
        overwrite=overwrite,
    )


def create_view_from_parquet(
    backend: IbisBackend,
    view_name: str,
    filename: Path,
) -> None:
    """Create a view from a Parquet file.

    Parameters
    ----------
    backend
        Ibis backend
    view_name
        Name of view to create
    filename
        Path to Parquet file or directory
    """
    backend.create_view_from_parquet(view_name, filename)


def _check_one_config_per_datetime_column(configs: Sequence[TimeBaseModel]) -> None:
    """Ensure each datetime column has at most one config."""
    time_col_count = Counter(
        config.time_column for config in configs if isinstance(config, DatetimeRangeBase)
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if time_col_dup:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _convert_database_input_for_datetime(
    df: pd.DataFrame, config: DatetimeRangeBase, copied: bool
) -> tuple[pd.DataFrame, bool]:
    """Convert DataFrame datetime columns for SQLite input."""
    if config.dtype == TimeDataType.TIMESTAMP_NTZ:
        return df, copied

    if copied:
        df2 = df
    else:
        df2 = df.copy()
        copied = True

    if isinstance(df2[config.time_column].dtype, DatetimeTZDtype):
        df2[config.time_column] = df2[config.time_column].dt.tz_convert("UTC")
    else:
        df2[config.time_column] = df2[config.time_column].dt.tz_localize("UTC")

    return df2, copied


def _convert_database_output_for_datetime(df: pd.DataFrame, config: DatetimeRangeBase) -> None:
    """Convert DataFrame datetime columns after SQLite output."""
    if config.time_column in df.columns:
        if config.dtype == TimeDataType.TIMESTAMP_TZ:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=True)
            elif isinstance(df[config.time_column].dtype, DatetimeTZDtype):
                # Already tz-aware, convert to UTC
                df[config.time_column] = df[config.time_column].dt.tz_convert("UTC")
            else:
                df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
        else:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=False)


def _convert_spark_output_for_datetime(df: pd.DataFrame, config: DatetimeRangeBase) -> None:
    """Convert DataFrame datetime columns after Spark output.

    Spark stores timestamps in UTC (as configured in SparkBackend).
    This function converts them to match the expected timezone from the config.
    """
    if config.time_column not in df.columns:
        return

    if config.dtype == TimeDataType.TIMESTAMP_TZ:
        # For tz-aware configs, Spark returns UTC timestamps
        # Convert from UTC to the expected timezone
        target_tz = config.start.tzinfo
        if target_tz is not None:
            if isinstance(df[config.time_column].dtype, DatetimeTZDtype):
                # Already tz-aware, convert to target timezone
                df[config.time_column] = df[config.time_column].dt.tz_convert(target_tz)
            else:
                # tz-naive, localize to UTC then convert
                df[config.time_column] = (
                    df[config.time_column].dt.tz_localize("UTC").dt.tz_convert(target_tz)
                )


def _write_to_duckdb(
    backend: IbisBackend,
    df: pd.DataFrame | pa.Table,
    table_name: str,
    if_exists: str,
) -> None:
    """Write DataFrame or PyArrow Table to DuckDB."""
    match if_exists:
        case "append":
            backend.insert(table_name, df)
        case "replace":
            backend.drop_table(table_name, if_exists=True)
            backend.create_table(table_name, df)
        case "fail":
            backend.create_table(table_name, df)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)


def _write_to_sqlite(
    backend: IbisBackend,
    df: pd.DataFrame | pa.Table,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_exists: str,
) -> None:
    """Write DataFrame to SQLite with datetime conversion."""
    _check_one_config_per_datetime_column(configs)

    # Convert PyArrow to pandas for datetime handling
    if isinstance(df, pa.Table):
        df = df.to_pandas()

    copied = False
    for config in configs:
        if isinstance(config, DatetimeRangeBase):
            df, copied = _convert_database_input_for_datetime(df, config, copied)

    match if_exists:
        case "append":
            backend.insert(table_name, df)
        case "replace":
            backend.drop_table(table_name, if_exists=True)
            backend.create_table(table_name, df)
        case "fail":
            backend.create_table(table_name, df)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)


def _write_to_pyspark(
    backend: IbisBackend,
    df: pd.DataFrame | pa.Table,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_exists: str,
    scratch_dir: Path | None,
) -> None:
    """Write DataFrame to Spark.

    Note: For Spark, we typically write to Parquet and create views.
    Direct table writes are limited.
    """
    # Convert PyArrow to pandas for datetime handling
    if isinstance(df, pa.Table):
        df = df.to_pandas()

    # Convert datetime columns for Spark compatibility
    df2 = df.copy()
    for config in configs:
        if isinstance(config, DatetimeRangeBase):
            if isinstance(df2[config.time_column].dtype, DatetimeTZDtype):
                # Spark doesn't like ns precision
                new_dtype = df2[config.time_column].dtype.name.replace(
                    "datetime64[ns", "datetime64[us"
                )
                df2[config.time_column] = df2[config.time_column].astype(new_dtype)  # type: ignore
            elif isinstance(df2[config.time_column].dtype, DateTime64DType):
                df2[config.time_column] = df2[config.time_column].astype("datetime64[us]")  # type: ignore
            else:
                df2[config.time_column] = pd.to_datetime(
                    df2[config.time_column], utc=False, errors="raise"
                ).astype("datetime64[us]")  # type: ignore

    match if_exists:
        case "append":
            msg = "INSERT INTO is not supported with Spark backend"
            raise InvalidOperation(msg)
        case "replace":
            backend.drop_view(table_name, if_exists=True)
            backend.create_temp_view(table_name, df2)
        case "fail":
            backend.create_temp_view(table_name, df2)
        case _:
            msg = f"Invalid if_exists value: {if_exists}"
            raise InvalidOperation(msg)
