from datetime import timedelta, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from sqlalchemy import MetaData

from chronify.time_configs import (
    YearMonthDayHourTimeNTZ,
    MonthDayHourTimeNTZ,
    DatetimeRange,
    TimeConfig,
    YearMonthDayPeriodTimeNTZ,
)
from chronify.models import TableSchema, PivotedTableSchema
from chronify.store import Store
from chronify.sqlalchemy.functions import write_database, read_database
from chronify.time_series_mapper import map_time


@pytest.fixture
def iter_store(iter_engines):
    return Store(engine=iter_engines)


def ingest_csv(csv_file: Path, conn, name: str, time_configs: list[TimeConfig]):
    data = pd.read_csv(csv_file)
    write_database(data, conn, name, time_configs, if_table_exists="replace")


def test_MDH_mapper(time_series_NMDH, iter_store: Store):
    from_schema = TableSchema(
        name="test_MDH",
        value_column="value",
        time_config=MonthDayHourTimeNTZ(
            length=48,
            year=2023,
            month_column="month",
            day_column="day",
            hour_columns=["hour"],
        ),
        time_array_id_columns=["name"],
    )
    pivoted_input_schema = PivotedTableSchema(
        pivoted_dimension_name="hour",
        value_columns=[str(x) for x in range(1, 25)],
        time_config=MonthDayHourTimeNTZ(
            length=48,
            year=2023,
            month_column="month",
            day_column="day",
        ),
    )
    data = pd.read_csv(time_series_NMDH)
    iter_store.ingest_pivoted_table(data, pivoted_input_schema, from_schema)

    metadata = MetaData()
    metadata.reflect(iter_store.engine, views=True)

    to_schema = TableSchema(
        name="test_MDH_datetime",
        value_column="value",
        time_config=DatetimeRange(
            length=48,
            start=datetime(2023, 1, 1),
            resolution=timedelta(hours=1),
            time_column="timestamp",
        ),
        time_array_id_columns=["name"],
    )

    map_time(iter_store.engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)

    with iter_store.engine.connect() as conn:
        mapped_table = read_database(
            f"SELECT * FROM {to_schema.name}", conn, to_schema.time_config
        )
        assert np.array_equal(mapped_table["value"].to_numpy(), np.arange(25, 73))


def test_YMDH_mapper(time_series_NYMDH, iter_store):
    from_schema = TableSchema(
        name="test_YMDH",
        value_column="value",
        time_config=YearMonthDayHourTimeNTZ(
            length=48,
            year=2023,
            year_column="year",
            month_column="month",
            day_column="day",
            hour_columns=["hour"],
        ),
        time_array_id_columns=["name"],
    )
    pivoted_input_schema = PivotedTableSchema(
        pivoted_dimension_name="hour",
        value_columns=[str(x) for x in range(1, 25)],
        time_config=YearMonthDayHourTimeNTZ(
            length=48,
            year=2023,
            year_column="year",
            month_column="month",
            day_column="day",
        ),
    )
    data = pd.read_csv(time_series_NYMDH)
    iter_store.ingest_pivoted_table(data, pivoted_input_schema, from_schema)

    metadata = MetaData()
    metadata.reflect(iter_store.engine, views=True)

    to_schema = TableSchema(
        name="test_YMDH_datetime",
        value_column="value",
        time_config=DatetimeRange(
            length=48,
            start=datetime(2023, 1, 1),
            resolution=timedelta(hours=1),
            time_column="timestamp",
        ),
        time_array_id_columns=["name"],
    )

    map_time(iter_store.engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)

    with iter_store.engine.connect() as conn:
        mapped_table = read_database(
            f"SELECT * FROM {to_schema.name}", conn, to_schema.time_config
        )
        assert np.array_equal(mapped_table["value"].to_numpy(), np.arange(25, 73))


def test_NYMDPV_mapper(time_series_NYMDPV, iter_store: Store):
    from_schema = TableSchema(
        name="test_NYMDPV",
        value_column="value",
        time_config=YearMonthDayPeriodTimeNTZ(
            length=2,
            year=2023,
            year_column="year",
            month_column="month",
            day_column="day",
            hour_columns=["period"],
        ),
        time_array_id_columns=["name"],
    )

    data = pd.read_csv(time_series_NYMDPV)
    iter_store.ingest_table(data, from_schema, skip_time_checks=True)

    metadata = MetaData()
    metadata.reflect(iter_store.engine, views=True)

    to_schema = TableSchema(
        name="test_YMDH_datetime",
        value_column="value",
        time_config=DatetimeRange(
            length=48,
            start=datetime(2023, 1, 1),
            resolution=timedelta(hours=1),
            time_column="timestamp",
        ),
        time_array_id_columns=["name"],
    )

    map_time(iter_store.engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)

    with iter_store.engine.connect() as conn:
        mapped_table = read_database(
            f"SELECT * FROM {to_schema.name}", conn, to_schema.time_config
        ).sort_values("timestamp")
        values = np.concatenate(
            [
                np.ones(5) * 100,
                np.ones(7) * 200,
                np.ones(12) * 300,
                np.ones(24) * 400,
            ]
        )
        assert np.array_equal(mapped_table["value"].to_numpy(), values)
