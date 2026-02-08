from zoneinfo import ZoneInfo
import pytest
from datetime import datetime, timedelta, tzinfo
from typing import Any, Optional

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeIntervalType
from chronify.datetime_range_generator import DatetimeRangeGenerator


def generate_datetime_data(time_config: DatetimeRange) -> pd.DatetimeIndex:
    return pd.to_datetime(DatetimeRangeGenerator(time_config).list_timestamps())


def get_datetime_schema(year: int, tzinfo: tzinfo | None) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year + 1, month=1, day=1, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution
    schema = TableSchema(
        name="mapped_data",
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
        time_array_id_columns=["id"],
        value_column="value",
    )
    return schema


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: Optional[tuple[Any, str]],
) -> None:
    # Ingest
    metadata = MetaData()
    with engine.begin() as conn:
        write_database(
            df, conn, from_schema.name, [from_schema.time_config], if_table_exists="replace"
        )
    metadata.reflect(engine, views=True)

    # Map
    if error:
        with pytest.raises(error[0], match=error[1]):
            map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)
    else:
        map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)

        # Check mapped table
        with engine.connect() as conn:
            query = f"select * from {to_schema.name}"
            queried = read_database(query, conn, to_schema.time_config)
        queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)

        truth = generate_datetime_data(to_schema.time_config)
        check_mapped_timestamps(queried, truth)

        # handles time shift
        if from_schema.time_config.interval_type == to_schema.time_config.interval_type:
            time_delta = timedelta(0)  # sec
        elif from_schema.time_config.interval_type == TimeIntervalType.PERIOD_BEGINNING:
            # datetime is period_ending, working backward, 2am pd-end >> 1am pd-beg
            time_delta = -to_schema.time_config.resolution
        elif from_schema.time_config.period_ending:
            time_delta = to_schema.time_config.resolution
        check_mapped_values(queried, df, time_delta=time_delta)


def check_mapped_timestamps(df: pd.DataFrame, ts: pd.Series) -> None:
    res = sorted(df["timestamp"].drop_duplicates().to_list())
    tru = sorted(ts)
    assert res == tru, "wrong unique timestamps"

    if "time_zone" in df.columns:
        res = df.groupby(["time_zone"])["timestamp"].count().unique().tolist()
        tru = [len(ts)]
        assert res == tru, "wrong number of timestamps"


def check_mapped_values(dfo: pd.DataFrame, dfi: pd.DataFrame, time_delta: timedelta) -> None:
    dfr = dfo.groupby("id").sample(2, random_state=10)
    if dfr["timestamp"].iloc[0].tzinfo is None:
        dfr["local_time"] = dfr["timestamp"].copy()
    else:
        dfr["local_time"] = dfr.apply(
            lambda x: x.timestamp.tz_convert(x.time_zone), axis=1
        )  # obj dtype
    dfr["local_time"] += time_delta
    dfr["month"] = dfr["local_time"].apply(lambda x: pd.Timestamp(x).month)
    dfr["hour"] = dfr["local_time"].apply(lambda x: pd.Timestamp(x).hour)
    dfr["day_of_week"] = dfr["local_time"].apply(lambda x: pd.Timestamp(x).day_of_week)
    dfr["is_weekday"] = dfr["day_of_week"].map(lambda x: True if x < 5 else False)

    keys = [x for x in dfi.columns if x != "value"]
    dfr = pd.merge(dfr, dfi.rename(columns={"value": "mapped_value"}), on=keys)
    diff = dfr["value"].compare(dfr["mapped_value"])
    assert len(diff) == 0, f"Mapped values are inconsistent. {diff}"


@pytest.mark.parametrize("interval_shift", [False, True])
def test_one_week_per_month_by_hour_tz_naive(
    iter_engines: Engine,
    one_week_per_month_by_hour_table: tuple[pd.DataFrame, int, TableSchema],
    interval_shift: bool,
) -> None:
    tzinfo = None
    df, _, schema = one_week_per_month_by_hour_table
    # For tz-naive case, time_zone can exist in the table and not get used,
    # but it needs to be added to time_array_id.
    df["time_zone"] = df["id"].map(
        dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
    )

    to_schema = get_datetime_schema(2020, tzinfo)
    if interval_shift:
        to_schema.time_config.interval_type = TimeIntervalType.PERIOD_ENDING
    error = None
    run_test(iter_engines, df, schema, to_schema, error)


@pytest.mark.parametrize("interval_shift", [False, True])
def test_one_week_per_month_by_hour_tz_aware(
    iter_engines: Engine,
    one_week_per_month_by_hour_table_tz: tuple[pd.DataFrame, int, TableSchema],
    interval_shift: bool,
) -> None:
    tzinfo = ZoneInfo("US/Pacific")
    df, _, schema = one_week_per_month_by_hour_table_tz

    to_schema = get_datetime_schema(2020, tzinfo)
    if interval_shift:
        to_schema.time_config.interval_type = TimeIntervalType.PERIOD_ENDING
    error = None
    run_test(iter_engines, df, schema, to_schema, error)


@pytest.mark.parametrize("interval_shift", [False, True])
def test_one_weekday_day_and_one_weekend_day_per_month_by_hour_tz_naive(
    iter_engines: Engine,
    one_weekday_day_and_one_weekend_day_per_month_by_hour_table: tuple[
        pd.DataFrame, int, TableSchema
    ],
    interval_shift: bool,
) -> None:
    tzinfo = None
    df, _, schema = one_weekday_day_and_one_weekend_day_per_month_by_hour_table

    to_schema = get_datetime_schema(2020, tzinfo)
    if interval_shift:
        to_schema.time_config.interval_type = TimeIntervalType.PERIOD_ENDING
    error = None
    run_test(iter_engines, df, schema, to_schema, error)


@pytest.mark.parametrize("interval_shift", [False, True])
def test_one_weekday_day_and_one_weekend_day_per_month_by_hour_tz_aware(
    iter_engines: Engine,
    one_weekday_day_and_one_weekend_day_per_month_by_hour_table_tz: tuple[
        pd.DataFrame, int, TableSchema
    ],
    interval_shift: bool,
) -> None:
    tzinfo = ZoneInfo("US/Eastern")
    df, _, schema = one_weekday_day_and_one_weekend_day_per_month_by_hour_table_tz

    to_schema = get_datetime_schema(2020, tzinfo)
    if interval_shift:
        to_schema.time_config.interval_type = TimeIntervalType.PERIOD_ENDING
    error = None
    run_test(iter_engines, df, schema, to_schema, error)


def test_instantaneous_interval_type(
    iter_engines: Engine,
    one_week_per_month_by_hour_table: tuple[pd.DataFrame, int, TableSchema],
) -> None:
    df, _, schema = one_week_per_month_by_hour_table
    schema.time_config.interval_type = TimeIntervalType.INSTANTANEOUS
    to_schema = get_datetime_schema(2020, None)
    to_schema.time_config.interval_type = TimeIntervalType.INSTANTANEOUS
    error = None
    run_test(iter_engines, df, schema, to_schema, error)
