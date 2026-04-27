from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os

import pandas as pd
import pytest

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
    out = (
        spark_store.read_table(schema.name)
        .execute()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

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
    out = spark_store.read_table(schema.name).execute()
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
    out = spark_store.read_table(schema.name).execute()
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
    out = spark_store.read_table(dst_schema.name).execute()
    expected = df["timestamp"].dt.tz_convert(to_tz).dt.tz_localize(None)
    out_sorted = out.sort_values("timestamp").reset_index(drop=True)
    assert list(out_sorted["timestamp"]) == list(expected)


def test_spark_delete_rows(spark_store: Store) -> None:
    """delete_rows should remove matching rows and return the count."""
    schema = TableSchema(
        name="spark_del",
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
            "id": [1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01 01:00:00+00:00",
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01 01:00:00+00:00",
                ],
                utc=True,
            ),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)
    count = spark_store.delete_rows(schema.name, {"id": 1})
    assert count == 2
    out = spark_store.read_table(schema.name).execute()
    assert len(out) == 2
    assert set(out["id"]) == {2}


def test_spark_write_parquet_partitioned(spark_store: Store, tmp_path: Path) -> None:
    """write_parquet with partition_by should produce partitioned output."""
    schema = TableSchema(
        name="spark_part",
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
            "id": [1, 1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01 01:00:00+00:00",
                    "2020-01-01 00:00:00+00:00",
                    "2020-01-01 01:00:00+00:00",
                ],
                utc=True,
            ),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)
    outdir = tmp_path / "partitioned_output"
    spark_store.backend.write_parquet(
        spark_store.backend.table(schema.name),
        str(outdir),
        partition_by=["id"],
    )
    # Partitioned parquet creates subdirectories
    assert outdir.exists()
    subdirs = [p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("id=")]
    assert len(subdirs) == 2


def test_spark_create_view_from_parquet(spark_store: Store, tmp_path: Path) -> None:
    """create_view_from_parquet should create a readable view."""
    schema = TableSchema(
        name="spark_pq_src",
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
                ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"],
                utc=True,
            ),
            "value": [1.0, 2.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)

    # Write to parquet then create a view from it
    outfile = tmp_path / "view_src.parquet"
    spark_store.write_table_to_parquet(schema.name, outfile, overwrite=True)

    from chronify.ibis.base import ObjectType

    table_expr, obj_type = spark_store.backend.create_view_from_parquet(str(outfile), "pq_view")
    assert obj_type == ObjectType.VIEW
    result = spark_store.backend.execute(table_expr)
    assert len(result) == 2


def test_spark_create_and_drop_view(spark_store: Store) -> None:
    """create_view and drop_view should work correctly."""
    schema = TableSchema(
        name="spark_view_src",
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
                ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"],
                utc=True,
            ),
            "value": [1.0, 2.0],
        }
    )
    spark_store.ingest_table(df, schema, skip_time_checks=True)

    expr = spark_store.backend.table(schema.name)
    spark_store.backend.create_view("test_view", expr)
    assert spark_store.backend.has_table("test_view")

    spark_store.backend.drop_view("test_view")
    assert not spark_store.backend.has_table("test_view")


def test_spark_dispose(tmp_path: Path) -> None:
    """dispose should not raise on an owned session."""
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    warehouse_dir = tmp_path / "spark-warehouse-dispose"
    session = (
        pyspark.SparkSession.builder.master("local")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .getOrCreate()
    )
    backend = SparkBackend(session=session, owns_session=True)
    backend.dispose()


def test_spark_backend_accepts_non_utc_session() -> None:
    """Non-UTC session tz is allowed; UTC is pinned only for read_query."""
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    session = pyspark.SparkSession.builder.master("local").getOrCreate()
    prev_tz = session.conf.get("spark.sql.session.timeZone", None)
    session.conf.set("spark.sql.session.timeZone", "America/Denver")
    try:
        backend = SparkBackend(session=session, owns_session=False)
        assert session.conf.get("spark.sql.session.timeZone") == "America/Denver"
        with backend._pinned_utc_session():
            assert session.conf.get("spark.sql.session.timeZone") == "UTC"
        assert session.conf.get("spark.sql.session.timeZone") == "America/Denver"
        backend.dispose()
    finally:
        if prev_tz is None:
            session.conf.unset("spark.sql.session.timeZone")
        else:
            session.conf.set("spark.sql.session.timeZone", prev_tz)
