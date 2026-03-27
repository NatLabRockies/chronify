"""Spark backend implementation."""

from pathlib import Path
from typing import Any, Literal, cast

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa

from chronify.exceptions import InvalidOperation
from chronify.ibis.base import IbisBackend


class SparkBackend(IbisBackend):
    """Spark backend implementation."""

    def __init__(
        self,
        session: Any = None,
        **kwargs: Any,
    ) -> None:
        """Create a Spark backend.

        Parameters
        ----------
        session
            Existing SparkSession, or None to create a new one
        **kwargs
            Additional arguments passed to ibis.pyspark.connect
        """
        try:
            import pyspark  # noqa: F401
        except ImportError:
            msg = (
                "PySpark is required for PySparkBackend. Install with: pip install chronify[spark]"
            )
            raise ImportError(msg)

        if session is None:
            from pyspark.sql import SparkSession

            session = SparkSession.builder.getOrCreate()

        connection = ibis.pyspark.connect(session, **kwargs)
        super().__init__(connection)
        self._session = session

        # Set timezone to UTC for consistency
        session.conf.set("spark.sql.session.timeZone", "UTC")
        # Use microsecond precision for Parquet timestamps
        session.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")

    @property
    def name(self) -> str:
        return "spark"

    @property
    def database(self) -> str | None:
        return None

    def _prepare_data_for_spark(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Spark by ensuring consistent timezone handling."""
        df = data.copy()

        # Convert timestamps to strings to avoid timezone confusion in Spark
        # Spark will parse the strings according to session timezone (UTC)
        # Naive "2020-01-01 12:00:00" -> 12:00:00 UTC
        # Aware "2020-01-01 12:00:00-05:00" -> 17:00:00 UTC
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        mode: Literal["overwrite", "error"] = "overwrite" if overwrite else "error"
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"CREATE TABLE {name} AS {sql}")
        else:
            df = self._prepare_data_for_spark(data)
            spark_df = self._session.createDataFrame(df)
            spark_df.write.saveAsTable(name, mode=mode)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._session.sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS {sql}")
        else:
            df = self._prepare_data_for_spark(data)
            spark_df = self._session.createDataFrame(df)
            spark_df.createOrReplaceTempView(name)

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        try:
            self._session.sql(f"DROP TABLE {if_exists_sql}{name}")
        except Exception as e:
            if "is a VIEW" in str(e):
                self._session.sql(f"DROP VIEW {if_exists_sql}{name}")
            else:
                raise

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._session.sql(f"DROP VIEW {if_exists_sql}{name}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_")
            and not t.startswith("ibis_pyarrow_memtable_")
            and not t.startswith("_temp_")
            and not t.startswith("_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        # Spark's toPandas() has DST issues when converting timestamps.
        # Work around by using Spark SQL to convert timestamps to strings,
        # then convert back to pandas timestamps after toPandas().
        if isinstance(expr, ir.Table):
            return self._execute_table_with_timestamp_fix(expr)
        return cast(pd.DataFrame, expr.execute())

    def _execute_table_with_timestamp_fix(self, expr: ir.Table) -> pd.DataFrame:
        """Execute a table expression with workaround for Spark timestamp DST issues."""
        schema = expr.schema()
        timestamp_cols = [name for name, dtype in schema.items() if dtype.is_timestamp()]

        if not timestamp_cols:
            return cast(pd.DataFrame, expr.execute())

        # Convert timestamp columns to strings in the query
        sql = expr.compile()
        select_exprs = []
        for col_name in schema.names:
            if col_name in timestamp_cols:
                # Cast to string to avoid DST conversion issues
                select_exprs.append(f"CAST({col_name} AS STRING) AS {col_name}")
            else:
                select_exprs.append(col_name)

        query = f"SELECT {', '.join(select_exprs)} FROM ({sql})"
        df: pd.DataFrame = self._session.sql(query).toPandas()

        # Convert string columns back to timestamps
        for col_name in timestamp_cols:
            df[col_name] = pd.to_datetime(df[col_name])

        return df

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"INSERT INTO {table_name} {sql}")
        else:
            df = self._prepare_data_for_spark(data)
            spark_df = self._session.createDataFrame(df)
            spark_df.write.insertInto(table_name)

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame.

        Uses string conversion for timestamp columns to avoid Spark's
        DST conversion issues during toPandas().
        """
        spark_df = self._session.sql(query)

        # Check for timestamp columns
        timestamp_cols = [
            field.name
            for field in spark_df.schema.fields
            if "timestamp" in field.dataType.simpleString().lower()
        ]

        if not timestamp_cols:
            return cast(pd.DataFrame, spark_df.toPandas())

        # Convert timestamp columns to strings to avoid DST issues
        from pyspark.sql.functions import col

        select_exprs = []
        for field in spark_df.schema.fields:
            if field.name in timestamp_cols:
                select_exprs.append(col(field.name).cast("string").alias(field.name))
            else:
                select_exprs.append(col(field.name))

        df: pd.DataFrame = spark_df.select(*select_exprs).toPandas()

        # Convert string columns back to timestamps
        for col_name in timestamp_cols:
            df[col_name] = pd.to_datetime(df[col_name])

        return df

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        sql = expr.compile()
        if partition_by:
            cols = ",".join(partition_by)
            # Spark uses INSERT OVERWRITE DIRECTORY
            query = (
                f"INSERT OVERWRITE DIRECTORY '{path}' USING parquet PARTITIONED BY ({cols}) {sql}"
            )
        else:
            query = f"INSERT OVERWRITE DIRECTORY '{path}' USING parquet {sql}"
        self._session.sql(query)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        query = f"CREATE VIEW {name} AS SELECT * FROM parquet.`{path}`"
        self._session.sql(query)
