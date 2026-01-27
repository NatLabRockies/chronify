"""This file provides functions to read and write the database as efficiently as possible.
The default behavior of sqlalchemy is to convert data into rows of tuples in Python, which
is very slow. This code attempts to bypass Python as much as possible through Arrow tables
in memory.
"""

import atexit
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Optional, TypeAlias, Sequence
from collections import Counter

import pandas as pd
from numpy.dtypes import DateTime64DType, ObjectDType
from pandas import DatetimeTZDtype
from chronify.time import TimeDataType
from sqlalchemy import Connection, Engine, Selectable, text

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.time_configs import (
    DatetimeRangeBase,
    TimeBaseModel,
    DatetimeRange,
    DatetimeRangeWithTZColumn,
)
from chronify.utils.path_utils import check_overwrite, delete_if_exists, to_path

# Copied from Pandas/Polars
DbWriteMode: TypeAlias = Literal["replace", "append", "fail"]
DatetimeRangeWithDtype: TypeAlias = DatetimeRange | DatetimeRangeWithTZColumn


def read_database(
    query: Selectable | str, conn: Connection, config: TimeBaseModel, params: Any = None
) -> pd.DataFrame:
    """Read a database query into a Pandas DataFrame."""
    match conn.engine.name:
        case "duckdb":
            if isinstance(query, str):
                df = conn._dbapi_connection.driver_connection.sql(query, params=params).to_df()  # type: ignore
            else:
                df = conn.execute(query).cursor.fetch_df()  # type: ignore
        case "sqlite":
            df = pd.read_sql(query, conn, params=params)
            if isinstance(config, (DatetimeRange, DatetimeRangeWithTZColumn)):
                _convert_database_output_for_datetime(df, config)
        case "hive":
            df = _read_from_hive(query, conn, config, params)
        case _:
            df = pd.read_sql(query, conn, params=params)
    return df  # type: ignore


def write_database(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_table_exists: DbWriteMode = "append",
    scratch_dir: Path | None = None,
) -> None:
    """Write a Pandas DataFrame to the database.
    configs allows sqlite formatting for more than one datetime columns.

    Note: Writing persistent data with Hive as the backend is not supported.
    This function will write the dataframe to a temporary Parquet file and then create
    a view into that file. This is only to support ephemeral tables, such as for mapping tables.
    """
    match conn.engine.name:
        case "duckdb":
            _write_to_duckdb(df, conn, table_name, if_table_exists)
        case "sqlite":
            _write_to_sqlite(df, conn, table_name, configs, if_table_exists)
        case "hive":
            _write_to_hive(df, conn, table_name, configs, if_table_exists, scratch_dir)
        case _:
            df.to_sql(table_name, conn, if_exists=if_table_exists, index=False)


def _check_one_config_per_datetime_column(configs: Sequence[TimeBaseModel]) -> None:
    time_col_count = Counter(
        [config.time_column for config in configs if isinstance(config, DatetimeRangeBase)]
    )
    time_col_dup = {k: v for k, v in time_col_count.items() if v > 1}
    if len(time_col_dup) > 0:
        msg = f"More than one datetime config found for: {time_col_dup}"
        raise InvalidParameter(msg)


def _convert_database_input_for_datetime(
    df: pd.DataFrame, config: DatetimeRangeWithDtype, copied: bool
) -> tuple[pd.DataFrame, bool]:
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


def _convert_database_output_for_datetime(
    df: pd.DataFrame, config: DatetimeRangeWithDtype
) -> None:
    if config.time_column in df.columns:
        if config.dtype == TimeDataType.TIMESTAMP_TZ:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=True)
            else:
                df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
        else:
            if isinstance(df[config.time_column].dtype, ObjectDType):
                df[config.time_column] = pd.to_datetime(df[config.time_column], utc=False)


def _write_to_duckdb(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    if_table_exists: DbWriteMode,
) -> None:
    assert conn._dbapi_connection is not None
    assert conn._dbapi_connection.driver_connection is not None

    match if_table_exists:
        case "append":
            query = f"INSERT INTO {table_name} SELECT * FROM df"
        case "replace":
            conn._dbapi_connection.driver_connection.sql(f"DROP TABLE IF EXISTS {table_name}")
            query = f"CREATE TABLE {table_name} AS SELECT * FROM df"
        case "fail":
            query = f"CREATE TABLE {table_name} AS SELECT * FROM df"
        case _:
            msg = f"{if_table_exists=}"
            raise InvalidOperation(msg)

    conn._dbapi_connection.driver_connection.sql(query)


def _write_to_hive(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_table_exists: DbWriteMode,
    scratch_dir: Path | None,
) -> None:
    df2 = df.copy()
    for config in configs:
        if isinstance(config, DatetimeRangeBase):
            if isinstance(df2[config.time_column].dtype, DatetimeTZDtype):
                # Spark doesn't like ns. That might change in the future.
                # Pandas might offer a better way to change from ns to us in the future.
                new_dtype = df2[config.time_column].dtype.name.replace(
                    "datetime64[ns", "datetime64[us"
                )
                df2[config.time_column] = df2[config.time_column].astype(new_dtype)  # type: ignore
            elif isinstance(df2[config.time_column].dtype, DateTime64DType):
                new_dtype = "datetime64[us]"
                df2[config.time_column] = df2[config.time_column].astype(new_dtype)  # type: ignore
            else:
                new_dtype = "datetime64[us]"
                df2[config.time_column] = pd.to_datetime(
                    df2[config.time_column], utc=False, errors="raise"
                ).astype(new_dtype)  # type: ignore

    with NamedTemporaryFile(suffix=".parquet", dir=scratch_dir) as f:
        f.close()
        output = Path(f.name)

    df2.to_parquet(output)
    atexit.register(lambda: delete_if_exists(output))
    select_stmt = f"SELECT * FROM parquet.`{output}`"
    # TODO: CREATE TABLE causes DST fallback timestamps to get dropped
    match if_table_exists:
        case "append":
            msg = "INSERT INTO is not supported with write_to_hive"
            raise InvalidOperation(msg)
        case "replace":
            conn.execute(text(f"DROP VIEW IF EXISTS {table_name}"))
            query = f"CREATE VIEW {table_name} AS {select_stmt}"
        case "fail":
            # Let the database fail the operation if the table already exists.
            query = f"CREATE VIEW {table_name} AS {select_stmt}"
        case _:
            msg = f"{if_table_exists=}"
            raise InvalidOperation(msg)
    conn.execute(text(query))


def _read_from_hive(
    query: Selectable | str, conn: Connection, config: TimeBaseModel, params: Any = None
) -> pd.DataFrame:
    df = pd.read_sql_query(query, conn, params=params)
    if (
        isinstance(config, (DatetimeRange, DatetimeRangeWithTZColumn))
        and config.time_column in df.columns
        and config.dtype == TimeDataType.TIMESTAMP_TZ
    ):
        # This is tied to the fact that we set the Spark session to UTC.
        # Otherwise, there is confusion with the computer's local time zone.
        df[config.time_column] = df[config.time_column].dt.tz_localize("UTC")
    return df


def _write_to_sqlite(
    df: pd.DataFrame,
    conn: Connection,
    table_name: str,
    configs: Sequence[TimeBaseModel],
    if_table_exists: DbWriteMode,
) -> None:
    _check_one_config_per_datetime_column(configs)
    copied = False
    for config in configs:
        if isinstance(config, (DatetimeRange, DatetimeRangeWithTZColumn)):
            df, copied = _convert_database_input_for_datetime(df, config, copied)
    df.to_sql(table_name, conn, if_exists=if_table_exists, index=False)


def create_view_from_parquet(conn: Connection, view_name: str, filename: Path) -> None:
    """Create a view from a Parquet file."""
    if conn.engine.name == "duckdb":
        str_path = f"{filename}/**/*.parquet" if filename.is_dir() else str(filename)
        query = f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{str_path}')"
    elif conn.engine.name == "hive":
        query = f"CREATE VIEW {view_name} AS SELECT * FROM parquet.`{filename}`"
    else:
        msg = f"create_view_from_parquet does not support engine={conn.engine.name}"
        raise NotImplementedError(msg)
    conn.execute(text(query))


def write_query_to_parquet(
    engine: Engine,
    query: str,
    output_file: Path,
    overwrite: bool = False,
    partition_columns: Optional[list[str]] = None,
) -> None:
    """Write the query to a Parquet file."""
    output_file = to_path(output_file)
    check_overwrite(output_file, overwrite)
    match engine.name:
        case "duckdb":
            if partition_columns:
                cols = ",".join(partition_columns)
                query = (
                    f"COPY ({query}) TO '{output_file}' (FORMAT PARQUET, PARTITION_BY ({cols}))"
                )
            else:
                query = f"COPY ({query}) TO '{output_file}' (FORMAT PARQUET)"
        case "hive":
            if not overwrite:
                msg = "write_table_to_parquet with Hive requires overwrite=True"
                raise InvalidOperation(msg)
            # TODO: partition columns
            if partition_columns:
                msg = "write_table_to_parquet with Hive doesn't support partition_columns"
                raise InvalidOperation(msg)
            query = f"INSERT OVERWRITE DIRECTORY '{output_file}' USING parquet {query}"
        case _:
            msg = f"{engine.name=}"
            raise NotImplementedError(msg)

    with engine.connect() as conn:
        conn.execute(text(query))
