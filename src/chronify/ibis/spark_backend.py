"""Spark backend implementation for Ibis."""

import uuid
import shutil
from contextlib import contextmanager
from typing import Any, Generator
from pathlib import Path
from urllib.parse import urlparse, unquote

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
import pyarrow as pa
from loguru import logger

from chronify.exceptions import InvalidOperation
from chronify.ibis.base import IbisBackend, ObjectType, _DATETIME_RANGES
from chronify.time import TimeDataType
from chronify.time_configs import TimeBaseModel


class SparkBackend(IbisBackend):
    """Ibis backend for PySpark databases.

    Requires pyspark to be installed (pip install chronify[spark]).
    """

    def __init__(self, session: Any = None, *, owns_session: bool | None = None) -> None:
        """Construct a SparkBackend.

        Parameters
        ----------
        session
            Optional pre-existing PySpark session. When provided, the backend
            does not own the session by default and will not stop it on
            ``dispose()``.
        owns_session
            Override whether ``dispose()`` stops ``session``. Defaults to
            ``True`` only when ``session`` is not provided.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            msg = "pyspark is required for SparkBackend. Install with: pip install chronify[spark]"
            raise ImportError(msg) from e

        self._owns_session = session is None if owns_session is None else owns_session
        if session is None:
            session = (
                SparkSession.builder.master("local")
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate()
            )
        self._session = session
        # Arrow preserves TIMESTAMP instants across pandas boundaries; the
        # non-Arrow path converts through JVM Calendar and mis-resolves DST
        # fall-back values (e.g. 2012-11-04 UTC 07:00 collapses into 08:00).
        session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        # ibis.pyspark.connect() forces spark.sql.session.timeZone=UTC as a
        # side effect. Preserve the caller's tz; UTC is re-pinned locally by
        # _pinned_utc_session() around operations that require it.
        tz_key = "spark.sql.session.timeZone"
        prev_tz = session.conf.get(tz_key, None)
        self._connection = ibis.pyspark.connect(session)
        if prev_tz is None:
            session.conf.unset(tz_key)
        elif session.conf.get(tz_key, None) != prev_tz:
            session.conf.set(tz_key, prev_tz)

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
        obj: pd.DataFrame | pa.Table | ibis.Table | None = None,
        schema: ibis.Schema | None = None,
        overwrite: bool = False,
    ) -> ibis.Table:
        try:
            return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)
        except Exception as exc:
            if "LOCATION_ALREADY_EXISTS" not in str(exc):
                raise
            self._remove_managed_table_location(name)
            return self._connection.create_table(name, obj=obj, schema=schema, overwrite=overwrite)

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

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ibis.Table, ObjectType]:
        self._connection.create_view(name, self._connection.read_parquet(path))
        return self.table(name), ObjectType.VIEW

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._session.sql(query)

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

    def apply_schema_types(self, expr: ibis.Table, config: TimeBaseModel) -> ibis.Table:
        """Re-attach the timezone annotation to the time column's Ibis type.

        Spark's schema has no per-column timezone — only a session-wide
        ``spark.sql.session.timeZone`` — so Ibis reports any Spark
        ``TIMESTAMP`` column as ``Timestamp(timezone=None)`` even though the
        underlying values are true UTC instants. The cast to
        ``Timestamp(timezone="UTC")`` is intended as a metadata-only
        re-annotation; this only holds while ``session.timeZone=UTC``, so
        callers that materialize the result must do so under
        :meth:`_pinned_utc_session` (as :meth:`read_query` does).
        """
        if not isinstance(config, _DATETIME_RANGES):
            return expr
        if config.dtype != TimeDataType.TIMESTAMP_TZ:
            return expr
        if config.time_column not in expr.columns:
            return expr
        return expr.mutate(
            **{config.time_column: expr[config.time_column].cast(dt.Timestamp(timezone="UTC"))}
        )

    def read_query(self, expr: ibis.Table, config: TimeBaseModel) -> pd.DataFrame:
        """Execute ``expr`` with the session pinned to UTC for materialization.

        The UTC pin ensures the metadata-only cast applied by
        :meth:`apply_schema_types` is not reinterpreted as a value conversion
        through the caller's ``session.timeZone``.
        """
        with self._pinned_utc_session():
            return self.execute(self.apply_schema_types(expr, config))

    @contextmanager
    def _pinned_utc_session(self) -> Generator[None, None, None]:
        """Temporarily set ``spark.sql.session.timeZone=UTC`` for the block.

        Restores the previous value on exit. Not thread-safe with concurrent
        users of the same Spark session.
        """
        key = "spark.sql.session.timeZone"
        prev = self._session.conf.get(key, None)
        if prev == "UTC":
            yield
            return
        self._session.conf.set(key, "UTC")
        try:
            yield
        finally:
            if prev is None:
                self._session.conf.unset(key)
            else:
                self._session.conf.set(key, prev)

    def _supports_parquet_partitioning(self) -> bool:
        return True


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for Spark SQL, escaping embedded backticks."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"
