"""Spark backend implementation for Ibis."""

import uuid
import shutil
from contextlib import contextmanager
from typing import Any, Generator, Sequence, cast
from pathlib import Path
from urllib.parse import urlparse, unquote

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from loguru import logger
from pandas import DatetimeTZDtype

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.ibis.base import (
    DatetimeRanges,
    IbisBackend,
    ObjectType,
    _normalize_timestamps,
)
from chronify.time import TimeDataType
from chronify.time_configs import TimeBaseModel


class SparkBackend(IbisBackend):
    """Ibis backend for PySpark databases.

    Requires pyspark to be installed (pip install chronify[spark]).
    """

    def __init__(self, session: Any = None) -> None:
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            msg = "pyspark is required for SparkBackend. Install with: pip install chronify[spark]"
            raise ImportError(msg) from e

        self._owns_session = session is None
        if session is None:
            session = (
                SparkSession.builder.master("local")
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
                .getOrCreate()
            )
        self._validate_session(session)
        self._session = session
        self._connection = ibis.pyspark.connect(session)

    @property
    def name(self) -> str:
        return "spark"

    @property
    def database(self) -> str | None:
        return None

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
        if isinstance(obj, pd.DataFrame):
            obj = self._prepare_data_for_spark(obj)
        try:
            return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)
        except Exception as exc:
            if "LOCATION_ALREADY_EXISTS" not in str(exc):
                raise
            self._remove_managed_table_location(name)
            return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def insert(self, name: str, data: pd.DataFrame | pa.Table) -> None:
        if isinstance(data, pa.Table):
            data = data.to_pandas()
        # Spark doesn't support INSERT directly -- create a temp view and insert via SQL
        target_columns = list(self.table(name).columns)
        _validate_insert_columns(name, target_columns, list(data.columns))
        data = data.loc[:, target_columns]
        data = self._prepare_data_for_spark(data)
        spark_df = self._session.createDataFrame(data)
        quoted_name = _quote_identifier(name)
        col_list = ", ".join(_quote_identifier(c) for c in target_columns)
        with self._temp_view(spark_df) as tmp_view:
            self._session.sql(
                f"INSERT INTO {quoted_name} ({col_list}) SELECT {col_list} FROM {tmp_view}"
            )
        logger.trace("Inserted {} rows into {}", len(data), name)

    @contextmanager
    def _temp_view(self, spark_df: Any) -> Generator[str, None, None]:
        """Register ``spark_df`` as a uniquely-named temp view; drop on exit."""
        tmp_view = f"__chronify_tmp_{uuid.uuid4().hex}"
        spark_df.createOrReplaceTempView(tmp_view)
        try:
            yield tmp_view
        finally:
            self._session.catalog.dropTempView(tmp_view)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        # Spark 3.4+ supports parameterized SQL via the ``args`` keyword.
        quoted_name = _quote_identifier(name)
        param_names = [f"p{i}" for i in range(len(values))]
        where = " AND ".join(f"{_quote_identifier(c)} = :{p}" for c, p in zip(values, param_names))
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        args = dict(zip(param_names, values.values()))
        try:
            self._session.sql(sql, args=args)
        except Exception as exc:
            if "does not support DELETE" not in str(exc):
                raise
            self._overwrite_without_deleted_rows(name, where, args)
        logger.trace("Deleted rows from {} matching {}", name, values)

    def _overwrite_without_deleted_rows(self, name: str, where: str, args: dict[str, Any]) -> None:
        quoted_name = _quote_identifier(name)
        tmp_name = f"__chronify_delete_{uuid.uuid4().hex}"
        quoted_tmp = _quote_identifier(tmp_name)
        try:
            self._session.sql(
                f"CREATE TABLE {quoted_tmp} AS "
                f"SELECT * FROM {quoted_name} WHERE NOT ({where})",
                args=args,
            )
            self._session.sql(f"INSERT OVERWRITE TABLE {quoted_name} SELECT * FROM {quoted_tmp}")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {quoted_tmp}")
            self._remove_managed_table_location(tmp_name)

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        df = self._to_spark_dataframe(expr)
        writer = df.write.mode("errorifexists")
        if partition_by:
            writer.partitionBy(*partition_by).parquet(path)
        else:
            writer.parquet(path)

    def _to_spark_dataframe(self, expr: ir.Table) -> Any:
        sql = self._connection.compile(expr)
        return self._session.sql(sql)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ir.Table, ObjectType]:
        spark_df = self._session.read.parquet(path)
        spark_df.createOrReplaceTempView(name)
        return self.table(name), ObjectType.VIEW

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._session.sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        return cast(pd.DataFrame, self._session.sql(query).toPandas())

    def dispose(self) -> None:
        self._connection.disconnect()
        if self._owns_session:
            self._session.stop()

    def backup(self, dst: str) -> None:
        msg = "backup is not supported for the Spark backend"
        raise InvalidOperation(msg)

    def _remove_managed_table_location(self, name: str) -> None:
        location = str(self._session.conf.get("spark.sql.warehouse.dir", "spark-warehouse"))
        parsed = urlparse(location)
        if parsed.scheme == "file":
            warehouse = Path(unquote(parsed.path))
        else:
            warehouse = Path(location)
        path = warehouse / name
        if path.exists():
            shutil.rmtree(path)

    def _post_read_normalize(self, df: pd.DataFrame, config: DatetimeRanges) -> None:
        """Spark returns tz-naive nanosecond timestamps; coerce to schema dtype + µs unit."""
        _convert_spark_output_for_datetime(df, config)

    def _prepare_write_data(
        self,
        data: pd.DataFrame | pa.Table,
        configs: Sequence[TimeBaseModel],
    ) -> pd.DataFrame:
        """Spark ingestion goes through createDataFrame(pandas); Arrow must be converted."""
        if isinstance(data, pa.Table):
            data = data.to_pandas()
        return _normalize_timestamps(data, configs)

    @staticmethod
    def _prepare_data_for_spark(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize tz-aware pandas timestamps for Spark ingestion.

        Spark timestamps are timezone-naive and interpreted in the session time
        zone. We require UTC sessions, so convert tz-aware columns to tz-naive
        UTC timestamps before handing them to Spark.
        """
        df = df.copy()
        for col in df.columns:
            if isinstance(df[col].dtype, DatetimeTZDtype):
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    @staticmethod
    def _validate_session(session: Any) -> None:
        time_zone = session.conf.get("spark.sql.session.timeZone", None) or "UTC"
        if time_zone != "UTC":
            msg = (
                "SparkBackend requires spark.sql.session.timeZone=UTC to preserve "
                f"timestamp semantics, got {time_zone!r}."
            )
            raise InvalidParameter(msg)


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
    """Quote a SQL identifier for Spark SQL, escaping embedded backticks."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"
