"""Spark backend implementation for Ibis."""

import shutil
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, cast
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, unquote

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from loguru import logger

from chronify.exceptions import ConflictingInputsError, InvalidOperation, InvalidParameter
from chronify.ibis.base import (
    IbisBackend,
    ObjectType,
    _DATETIME_RANGES,
    _check_delete_values,
)
from chronify.time import TimeDataType
from chronify.time_configs import TimeBaseModel
from chronify.utils.sql import make_temp_view_name

try:
    from ibis.backends.pyspark import Backend as _IbisPySparkBackend
except ImportError:  # pyspark is not installed; SparkBackend.__init__ raises a clear error.
    _IbisPySparkBackend = object  # type: ignore[misc,assignment]


_SESSION_TZ_KEY = "spark.sql.session.timeZone"


@contextmanager
def _pinned_utc(session: Any) -> Generator[None, None, None]:
    """Temporarily set ``spark.sql.session.timeZone=UTC`` for the block.

    Restores the previous value on exit. Not thread-safe with concurrent
    users of the same Spark session.
    """
    prev = session.conf.get(_SESSION_TZ_KEY, None)
    if prev == "UTC":
        yield
        return
    session.conf.set(_SESSION_TZ_KEY, "UTC")
    try:
        yield
    finally:
        if prev is None:
            session.conf.unset(_SESSION_TZ_KEY)
        else:
            session.conf.set(_SESSION_TZ_KEY, prev)


class _UtcAwarePySparkBackend(_IbisPySparkBackend):  # type: ignore[misc]
    """Ibis PySpark backend that materializes tz-aware timestamps under a UTC session.

    PySpark renders TIMESTAMP values as naive wall-clock times in the session
    time zone, and Ibis then labels them with the column's time zone (UTC; see
    ``SparkBackend.apply_schema_types``). That is only correct under a UTC
    session. Results with only tz-naive timestamps keep the caller's session
    time zone, which is how callers that write tz-naive data under a non-UTC
    session (e.g., dsgrid's ``custom_time_zone``) expect to read it back.
    """

    def execute(self, expr: ir.Expr, /, **kwargs: Any) -> Any:
        if _has_tz_aware_timestamp(expr):
            with _pinned_utc(self._session):
                return super().execute(expr, **kwargs)
        return super().execute(expr, **kwargs)


def _has_tz_aware_timestamp(expr: ir.Expr) -> bool:
    return any(t.is_timestamp() and t.timezone is not None for t in expr.as_table().schema().types)


class SparkBackend(IbisBackend):
    """Ibis backend for PySpark databases.

    Requires pyspark to be installed (pip install chronify[spark]).
    """

    # ibis-pyspark creates views with CREATE TEMPORARY VIEW.
    views_are_temporary = True

    def __init__(
        self,
        session: Any = None,
        *,
        owns_session: bool | None = None,
        warehouse_dir: str | Path | None = None,
    ) -> None:
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
        warehouse_dir
            Directory where Spark stores the tables chronify creates, including
            intermediate tables while mapping (``spark.sql.warehouse.dir``).
            Spark defaults to ``spark-warehouse`` in the current directory. On a
            multi-node cluster this must be on a filesystem shared by all nodes.
            Only applies when this backend creates the session; for a
            pre-existing session, configure ``spark.sql.warehouse.dir`` on it.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            msg = "pyspark is required for SparkBackend. Install with: pip install chronify[spark]"
            raise ImportError(msg) from e

        if session is not None and warehouse_dir is not None:
            msg = (
                "warehouse_dir cannot be combined with session; "
                "set spark.sql.warehouse.dir when building the session instead"
            )
            raise ConflictingInputsError(msg)

        self._in_transaction = False
        self._owns_session = session is None if owns_session is None else owns_session
        if session is None:
            builder = (
                SparkSession.builder.master("local")
                .config("spark.sql.session.timeZone", "UTC")
                .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            )
            if warehouse_dir is not None:
                builder = builder.config(
                    "spark.sql.warehouse.dir", str(Path(warehouse_dir).resolve())
                )
            session = builder.getOrCreate()
        self._session = session
        if warehouse_dir is not None and self._warehouse_path() != Path(warehouse_dir).resolve():
            # getOrCreate() returns an already-running session unchanged, and the
            # warehouse location cannot be changed after a session starts.
            msg = (
                f"Requested warehouse_dir={warehouse_dir} but the active Spark session "
                f"uses {self._warehouse_path()}. Stop the existing session first."
            )
            raise InvalidParameter(msg)
        # Arrow preserves TIMESTAMP instants across pandas boundaries; the
        # non-Arrow path converts through JVM Calendar and mis-resolves DST
        # fall-back values (e.g. 2012-11-04 UTC 07:00 collapses into 08:00).
        session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        # ibis.pyspark.connect() forces spark.sql.session.timeZone=UTC as a
        # side effect. Preserve the caller's tz; UTC is re-pinned locally by
        # _pinned_utc() around operations that require it, including every
        # materialization of tz-aware timestamps (see _UtcAwarePySparkBackend).
        tz_key = _SESSION_TZ_KEY
        prev_tz = session.conf.get(tz_key, None)
        ibis.pyspark  # noqa: B018 -- loading the backend registers ibis.options.pyspark
        self._connection = _UtcAwarePySparkBackend().connect(session)
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
    def session_time_zone(self) -> str | None:
        """Return the Spark session time zone (spark.sql.session.timeZone)."""
        return cast(str | None, self._session.conf.get(_SESSION_TZ_KEY, None))

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
        _check_delete_values(values)
        quoted_name = _quote_identifier(name)
        param_names = [f"p{i}" for i in range(len(values))]
        where = " AND ".join(f"{_quote_identifier(c)} = :{p}" for c, p in zip(values, param_names))
        sql = f"DELETE FROM {quoted_name} WHERE {where}"
        args = {p: self._adapt_arg(v) for p, v in zip(param_names, values.values())}
        try:
            self._session.sql(sql, args=args)
        except Exception as exc:
            if "does not support DELETE" not in str(exc):
                raise
            self._overwrite_without_deleted_rows(name, where, args)
        logger.trace("Deleted rows from {} matching {}", name, values)

    def _overwrite_without_deleted_rows(self, name: str, where: str, args: dict[str, Any]) -> None:
        quoted_name = _quote_identifier(name)
        tmp_name = make_temp_view_name("__chronify_delete")
        quoted_tmp = _quote_identifier(tmp_name)
        try:
            self._session.sql(
                f"CREATE TABLE {quoted_tmp} AS "
                # Keep rows where the predicate is NULL, as DELETE would.
                f"SELECT * FROM {quoted_name} WHERE coalesce(NOT ({where}), true)",
                args=args,
            )
            self._session.sql(f"INSERT OVERWRITE TABLE {quoted_name} SELECT * FROM {quoted_tmp}")
        finally:
            self._session.sql(f"DROP TABLE IF EXISTS {quoted_tmp}")
            self._remove_managed_table_location(tmp_name)

    def create_view_from_parquet(self, path: str, name: str) -> tuple[ibis.Table, ObjectType]:
        # Persistent view over the file itself, unlike create_view. ibis's
        # read_parquet would register an extra ibis_read_parquet_* temp view, and a
        # persistent view can't reference it. Absolute path because the view may be
        # read from a different cwd.
        parquet_path = _quote_identifier(str(Path(path).resolve()))
        self._session.sql(
            f"CREATE VIEW {_quote_identifier(name)} AS SELECT * FROM parquet.{parquet_path}"
        )
        return self.table(name), ObjectType.VIEW

    def write_parquet(
        self,
        expr: ibis.Table,
        path: str,
        partition_by: list[str] | None = None,
    ) -> None:
        """Write to Parquet.

        ibis-pyspark's ``to_parquet`` forwards kwargs to ``DataFrameWriter.save``,
        which takes the camel-cased ``partitionBy`` (snake_case would be silently
        treated as an unknown option).
        """
        self._connection.to_parquet(expr, path, partitionBy=partition_by or None)

    def execute_sql(self, query: str) -> None:
        logger.trace("execute_sql: {}", query)
        self._session.sql(query)

    def execute_sql_to_df(self, query: str, params: Any = None) -> pd.DataFrame:
        logger.trace("execute_sql_to_df: {}", query)
        df = (
            self._session.sql(query, args=params)
            if params is not None
            else self._session.sql(query)
        )
        return cast(pd.DataFrame, df.toPandas())

    def _adapt_arg(self, v: Any) -> Any:
        """Attach the session time zone to naive datetime SQL parameters.

        PySpark converts naive datetime parameters through the driver JVM's local
        time zone, not spark.sql.session.timeZone, so a naive value would bind to
        the wrong instant when the two differ.
        """
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=ZoneInfo(self.session_time_zone or "UTC"))
        return v

    def dispose(self) -> None:
        # ibis-pyspark's disconnect() only stops the session, which must not
        # happen to a caller-provided one.
        if self._owns_session:
            self._connection.disconnect()

    def backup(self, dst: str) -> None:
        msg = "backup is not supported for the Spark backend"
        raise InvalidOperation(msg)

    def _warehouse_path(self) -> Path:
        location = str(self._session.conf.get("spark.sql.warehouse.dir", "spark-warehouse"))
        parsed = urlparse(location)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).resolve()
        if not parsed.scheme:
            return Path(location).resolve()
        return Path(location)

    def _remove_managed_table_location(self, name: str) -> None:
        # Spark lowercases table names for managed-table directories.
        path = self._warehouse_path() / name.lower()
        if path.exists():
            shutil.rmtree(path)

    def apply_schema_types(self, expr: ibis.Table, config: TimeBaseModel) -> ibis.Table:
        """Re-attach the timezone annotation to the time column's Ibis type.

        Spark's schema has no per-column timezone — only a session-wide
        ``spark.sql.session.timeZone`` — so Ibis reports any Spark
        ``TIMESTAMP`` column as ``Timestamp(timezone=None)`` even though the
        underlying values are true UTC instants. The cast to
        ``Timestamp(timezone="UTC")`` is intended as a metadata-only
        re-annotation; this only holds while ``session.timeZone=UTC``, which
        the connection's ``execute`` ensures for tz-aware results.
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

    def _supports_parquet_partitioning(self) -> bool:
        return True


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier for Spark SQL, escaping embedded backticks."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"
