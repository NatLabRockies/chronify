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
def spark_store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    """Parametrize indirectly with a session time zone; defaults to UTC."""
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    warehouse_dir = tmp_path / "spark-warehouse"
    session = (
        pyspark.SparkSession.builder.master("local")
        .config("spark.sql.session.timeZone", getattr(request, "param", "UTC"))
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
    spark_store.write_query_to_parquet(expr, outfile, overwrite=True)

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


def test_spark_backend_warehouse_dir(tmp_path: Path) -> None:
    """Tables chronify creates are stored under warehouse_dir."""
    _require_java_home()
    pyspark = pytest.importorskip("pyspark.sql")
    from chronify.exceptions import ConflictingInputsError, InvalidParameter

    # warehouse_dir is static per Spark session; start from a clean slate.
    active = pyspark.SparkSession.getActiveSession()
    if active is not None:
        active.stop()

    warehouse_dir = tmp_path / "warehouse"
    backend = SparkBackend(warehouse_dir=warehouse_dir)
    try:
        store = Store(backend=backend)
        schema = TableSchema(
            name="devices",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=datetime(2020, 1, 1),
                length=3,
                resolution=timedelta(hours=1),
            ),
            time_array_id_columns=["id"],
        )
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=3, freq="h", unit="us"),
                "id": 1,
                "value": [1.0, 2.0, 3.0],
            }
        )
        store.ingest_table(df, schema)
        assert (warehouse_dir / "devices").is_dir()

        # The running session can't switch warehouses; refuse rather than ignore.
        with pytest.raises(InvalidParameter, match="warehouse_dir"):
            SparkBackend(warehouse_dir=tmp_path / "other")
        with pytest.raises(ConflictingInputsError):
            SparkBackend(session=backend._session, warehouse_dir=warehouse_dir)
    finally:
        backend.dispose()


def test_spark_delete_rows_keeps_null_ids(spark_store: Store) -> None:
    """The non-DELETE fallback must keep rows whose key is NULL, as DELETE would."""
    schema = TableSchema(
        name="spark_del_null",
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
            "id": pd.array([1, 1, 2, 2, None, None], dtype="Int64"),
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"] * 3, utc=True
            ),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    spark_store.ingest_table(df, schema)
    assert spark_store.delete_rows(schema.name, {"id": 1}) == 2
    ids = spark_store.read_table(schema.name).execute()["id"]
    assert sorted(ids.dropna().tolist()) == [2, 2]
    assert ids.isna().sum() == 2


def _three_hours_schema(name: str, tz: ZoneInfo | None) -> TableSchema:
    return TableSchema(
        name=name,
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=tz),
            length=3,
            resolution=timedelta(hours=1),
        ),
        time_array_id_columns=["id"],
    )


def _three_hours_frame(tz: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": 1,
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="h", tz=tz),
            "value": [1.0, 2.0, 3.0],
        }
    )


@pytest.mark.parametrize("spark_store", ["America/Denver"], indirect=True)
def test_spark_non_utc_session_reads_tz_aware_as_utc(spark_store: Store) -> None:
    """PySpark renders timestamps in the session tz; tz-aware reads must not shift."""
    schema = _three_hours_schema("tz_aware", ZoneInfo("UTC"))
    df = _three_hours_frame("UTC")
    spark_store.ingest_table(df, schema)
    table = spark_store.read_table(schema.name)
    assert table.execute()["timestamp"].tolist() == df["timestamp"].tolist()
    expected = [x.to_pydatetime() for x in df["timestamp"]]
    assert table.to_pyarrow()["timestamp"].to_pylist() == expected
    session = spark_store.backend._session  # type: ignore[attr-defined]
    assert session.conf.get("spark.sql.session.timeZone") == "America/Denver"


@pytest.mark.parametrize("spark_store", ["America/Denver"], indirect=True)
def test_spark_non_utc_session_keeps_tz_naive_wall_clock(
    spark_store: Store, tmp_path: Path
) -> None:
    """dsgrid writes tz-naive data under a non-UTC session (custom_time_zone) and
    maps it with chronify under the same session. Those values must not shift."""
    session = spark_store.backend._session  # type: ignore[attr-defined]
    df = _three_hours_frame(None)
    src = tmp_path / "src.parquet"
    session.createDataFrame(df).write.parquet(str(src))
    schema = _three_hours_schema("tz_naive", None)
    spark_store.create_view_from_parquet(src, schema, bypass_checks=True)
    assert spark_store.read_table(schema.name).execute()["timestamp"].tolist() == (
        df["timestamp"].tolist()
    )

    out = tmp_path / "out.parquet"
    spark_store.localize_time_zone(schema.name, ZoneInfo("EST"), output_file=out)
    expected = (df["timestamp"] + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    assert pd.read_parquet(out)["timestamp"].tolist() == expected.tolist()
