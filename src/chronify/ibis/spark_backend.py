"""Spark backend implementation for Ibis."""

import uuid
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
from loguru import logger
from pandas import DatetimeTZDtype

from chronify.exceptions import InvalidParameter
from chronify.ibis.base import IbisBackend, ObjectType


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
        obj: pd.DataFrame | ir.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        if isinstance(obj, pd.DataFrame):
            obj = self._prepare_data_for_spark(obj)
        return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

    def create_view(self, name: str, expr: ir.Table) -> ir.Table:
        return self._connection.create_view(name, expr, overwrite=False)

    def drop_table(self, name: str) -> None:
        self._connection.drop_table(name, force=True)

    def drop_view(self, name: str) -> None:
        self._connection.drop_view(name, force=True)

    def list_tables(self) -> list[str]:
        return cast(list[str], self._connection.list_tables())

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def insert(self, name: str, data: pd.DataFrame) -> None:
        # Spark doesn't support INSERT directly -- create a temp view and insert via SQL
        target_columns = list(self.table(name).columns)
        data = data.reindex(columns=target_columns)
        data = self._prepare_data_for_spark(data)
        spark_df = self._session.createDataFrame(data)
        tmp_view = f"__insert_tmp_{uuid.uuid4().hex}"
        spark_df.createOrReplaceTempView(tmp_view)
        quoted_name = _quote_identifier(name)
        col_list = ", ".join(_quote_identifier(c) for c in target_columns)
        try:
            self._session.sql(
                f"INSERT INTO {quoted_name} ({col_list}) SELECT {col_list} FROM {tmp_view}"
            )
        finally:
            self._session.catalog.dropTempView(tmp_view)
        logger.trace("Inserted {} rows into {}", len(data), name)

    def delete_rows(self, name: str, values: dict[str, Any]) -> None:
        # Spark 3.4+ supports parameterized SQL via the ``args`` keyword.
        quoted_name = _quote_identifier(name)
        param_names = [f"p{i}" for i in range(len(values))]
        where = " AND ".join(f"{_quote_identifier(c)} = :{p}" for c, p in zip(values, param_names))
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        args = dict(zip(param_names, values.values()))
        self._session.sql(sql, args=args)
        logger.trace("Deleted rows from {} matching {}", name, values)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return cast(pd.DataFrame, self._connection.execute(expr))

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        df = self._connection.execute(expr)
        if partition_by:
            spark_df = self._session.createDataFrame(df)
            spark_df.write.partitionBy(*partition_by).parquet(path)
        else:
            df.to_parquet(path)

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
        if self._owns_session:
            self._connection.disconnect()

    def reconnect(self) -> None:
        pass  # Spark sessions are long-lived

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


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for Spark SQL, escaping embedded backticks."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"
