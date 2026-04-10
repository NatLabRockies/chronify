from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os

import pandas as pd
import pytest

from chronify.exceptions import InvalidParameter
from chronify.ibis.spark_backend import SparkBackend
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange


def _require_java_home() -> None:
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Spark tests require JAVA_HOME to be set")


@pytest.fixture
def spark_store(tmp_path: Path) -> Store:
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    warehouse_dir = tmp_path / "spark-warehouse"
    session = (
        pyspark.SparkSession.builder.master("local")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .getOrCreate()
    )
    store = Store(backend=SparkBackend(session=session))
    yield store
    store.dispose()
    session.stop()


def test_spark_round_trip_timestamp_tz_preserves_fractional_seconds(spark_store: Store) -> None:
    schema = TableSchema(
        name="spark_ts_data",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=2,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 00:00:00.123456-05:00",
                    "2020-01-01 01:00:00.654321-05:00",
                ],
                utc=True,
            ),
            "value": [1.0, 2.0],
        }
    )

    spark_store.ingest_table(df, schema, skip_time_checks=True)
    out = spark_store.read_table(schema.name).sort_values("timestamp").reset_index(drop=True)

    expected = pd.to_datetime(
        [
            "2020-01-01 05:00:00.123456+00:00",
            "2020-01-01 06:00:00.654321+00:00",
        ],
        utc=True,
    )
    assert list(out["timestamp"]) == list(expected)


def test_spark_write_table_to_parquet_preserves_timestamp_type(
    spark_store: Store, tmp_path: Path
) -> None:
    schema = TableSchema(
        name="spark_parquet_data",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=1,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    df = pd.DataFrame(
        {
            "id": [1],
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00.123456+00:00"], utc=True),
            "value": [1.0],
        }
    )

    spark_store.ingest_table(df, schema, skip_time_checks=True)
    outfile = tmp_path / "spark_ts.parquet"
    spark_store.write_table_to_parquet(schema.name, outfile, overwrite=True)

    out = pd.read_parquet(outfile)
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00.123456+00:00")


def test_spark_write_query_to_parquet_preserves_timestamp_tz(
    spark_store: Store, tmp_path: Path
) -> None:
    """write_query_to_parquet must preserve tz semantics when name is supplied."""
    schema = TableSchema(
        name="spark_query_tz",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=2,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"], utc=True
            ),
            "value": [1.0, 2.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)

    outfile = tmp_path / "query_tz.parquet"
    expr = spark_store.get_table(schema.name)
    spark_store.write_query_to_parquet(expr, outfile, overwrite=True, name=schema.name)

    out = pd.read_parquet(outfile)
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00+00:00")


def test_spark_ingest_normalizes_tz_aware_to_ntz(spark_store: Store) -> None:
    """A TIMESTAMP_NTZ schema with tz-aware input should strip tz on all backends."""
    schema = TableSchema(
        name="spark_ntz",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1),
            length=2,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"], utc=True
            ),
            "value": [1.0, 2.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)
    out = spark_store.read_table(schema.name)
    # Should be tz-naive after round-trip
    assert not isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00")


def test_spark_ingest_normalizes_tz_naive_to_tz(spark_store: Store) -> None:
    """A TIMESTAMP_TZ schema with tz-naive input should localize to UTC on all backends."""
    schema = TableSchema(
        name="spark_tz",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=2,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            "value": [1.0, 2.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)
    out = spark_store.read_table(schema.name)
    assert isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00+00:00")


def test_spark_time_zone_conversion(spark_store: Store) -> None:
    """Time zone conversion should work on Spark the same as other backends."""
    schema = TableSchema(
        name="spark_tzconv",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=24,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )
    from chronify.datetime_range_generator import DatetimeRangeGenerator

    timestamps = DatetimeRangeGenerator(schema.time_config).list_timestamps()
    df = pd.DataFrame(
        {
            "id": [1] * len(timestamps),
            "timestamp": pd.to_datetime(timestamps),
            "value": range(len(timestamps)),
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)

    to_tz = ZoneInfo("US/Eastern")
    dst_schema = spark_store.convert_time_zone(schema.name, to_tz)
    out = spark_store.read_table(dst_schema.name)
    expected = df["timestamp"].dt.tz_convert(to_tz).dt.tz_localize(None)
    out_sorted = out.sort_values("timestamp").reset_index(drop=True)
    assert list(out_sorted["timestamp"]) == list(expected)


def test_spark_backend_rejects_non_utc_session() -> None:
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    session = (
        pyspark.SparkSession.builder.master("local")
        .config("spark.sql.session.timeZone", "America/Denver")
        .getOrCreate()
    )
    try:
        with pytest.raises(InvalidParameter, match="spark.sql.session.timeZone=UTC"):
            SparkBackend(session=session)
    finally:
        session.stop()
