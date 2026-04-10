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
