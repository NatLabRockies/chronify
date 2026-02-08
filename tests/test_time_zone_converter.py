from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, tzinfo
import numpy as np
import pytest
from typing import Any

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_zone_converter import (
    TimeZoneConverter,
    TimeZoneConverterByColumn,
    convert_time_zone,
    convert_time_zone_by_column,
)
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeDataType, TimeIntervalType
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.exceptions import InvalidParameter


def generate_datetime_dataframe(schema: TableSchema) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            schema.time_config.time_column: pd.to_datetime(
                DatetimeRangeGenerator(schema.time_config).list_timestamps()
            )
        }
    )
    for i, x in enumerate(schema.time_array_id_columns):
        df[x] = i
    df[schema.value_column] = np.random.rand(len(df))
    return df


def generate_dataframe_with_tz_col(schema: TableSchema) -> pd.DataFrame:
    df = generate_datetime_dataframe(schema).drop(columns=["id"])
    time_zones = [
        ZoneInfo("US/Eastern"),
        ZoneInfo("US/Central"),
        ZoneInfo("US/Mountain"),
        None,
    ]
    time_zones = [tz.key if tz else "None" for tz in time_zones]
    dfo = pd.merge(
        df, pd.DataFrame({"id": range(len(time_zones)), "time_zone": time_zones}), how="cross"
    )
    dfo = (
        dfo.drop(columns=["time_zone_x"])
        .rename(columns={"time_zone_y": "time_zone"})
        .reset_index()
    )
    return dfo


def get_datetime_schema(
    year: int,
    tzinfo: tzinfo | None,
    interval_type: TimeIntervalType,
    name: str,
    has_tz_col: bool = False,
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year, month=1, day=2, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    cols = ["id"]
    cols += ["time_zone"] if has_tz_col else []
    schema = TableSchema(
        name=name,
        time_config=DatetimeRange(
            dtype=TimeDataType.TIMESTAMP_TZ if tzinfo else TimeDataType.TIMESTAMP_NTZ,
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        ),
        time_array_id_columns=cols,
        value_column="value",
    )
    return schema


def ingest_data(
    engine: Engine,
    metadata: MetaData,
    df: pd.DataFrame,
    schema: TableSchema,
) -> None:
    with engine.begin() as conn:
        write_database(df, conn, schema.name, [schema.time_config], if_table_exists="replace")
    metadata.reflect(engine, views=True)


def get_mapped_dataframe(
    engine: Engine,
    table_name: str,
    time_config: DatetimeRange,
) -> pd.DataFrame:
    with engine.connect() as conn:
        query = f"select * from {table_name}"
        queried = read_database(query, conn, time_config)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)
    return queried


def run_conversion(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_time_zone: tzinfo | None,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    to_schema = convert_time_zone(
        engine, metadata, from_schema, to_time_zone, check_mapped_timestamps=True
    )
    dfo = get_mapped_dataframe(engine, to_schema.name, to_schema.time_config)
    assert df["value"].equals(dfo["value"])
    if to_time_zone is None:
        expected = df["timestamp"].dt.tz_localize(None)
    else:
        expected = df["timestamp"].dt.tz_convert(to_time_zone).dt.tz_localize(None)
    assert (dfo["timestamp"] == expected).prod() == 1


def run_conversion_to_column_time_zones(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    wrap_time_allowed: bool,
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    to_schema = convert_time_zone_by_column(
        engine,
        metadata,
        from_schema,
        "time_zone",
        wrap_time_allowed=wrap_time_allowed,
        check_mapped_timestamps=True,
    )
    dfo = get_mapped_dataframe(engine, to_schema.name, to_schema.time_config)
    dfo = dfo[df.columns].sort_values(by="index").reset_index(drop=True)
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"])  # needed for engine 2, not sure why

    assert df["value"].equals(dfo["value"])
    if wrap_time_allowed:
        assert set(dfo["timestamp"].value_counts()) == {4}
        expected = [x.replace(tzinfo=None) for x in sorted(set(df["timestamp"]))]
        assert set(dfo["timestamp"]) == set(expected)
    else:
        for i in range(len(df)):
            tzn = df.loc[i, "time_zone"]
            if tzn == "None":
                ts = df.loc[i, "timestamp"].replace(tzinfo=None)
            else:
                ts = df.loc[i, "timestamp"].tz_convert(ZoneInfo(tzn)).replace(tzinfo=None)
            assert dfo.loc[i, "timestamp"] == ts


def run_conversion_with_error(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    use_tz_col: bool,
    error: tuple[Any, str],
) -> None:
    metadata = MetaData()
    ingest_data(engine, metadata, df, from_schema)
    with pytest.raises(error[0], match=error[1]):
        if use_tz_col:
            tzc = TimeZoneConverterByColumn(
                engine, metadata, from_schema, "time_zone", wrap_time_allowed=False
            )
            tzc.convert_time_zone(check_mapped_timestamps=True)
        else:
            tzc2 = TimeZoneConverter(engine, metadata, from_schema, None)
            tzc2.convert_time_zone(check_mapped_timestamps=True)


def test_src_table_no_time_zone(iter_engines: Engine) -> None:
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    error = (InvalidParameter, "Source schema time config start time must be timezone-aware")
    run_conversion_with_error(iter_engines, df, from_schema, False, error)


@pytest.mark.parametrize(
    "to_time_zone", [None, ZoneInfo("US/Central"), ZoneInfo("America/Los_Angeles")]
)
def test_time_conversion(iter_engines: Engine, to_time_zone: tzinfo | None) -> None:
    from_schema = get_datetime_schema(
        2018, ZoneInfo("US/Mountain"), TimeIntervalType.PERIOD_BEGINNING, "base_table"
    )
    df = generate_datetime_dataframe(from_schema)
    run_conversion(iter_engines, df, from_schema, to_time_zone)


@pytest.mark.parametrize("wrap_time_allowed", [False, True])
def test_time_conversion_to_column_time_zones(
    iter_engines: Engine, wrap_time_allowed: bool
) -> None:
    from_schema = get_datetime_schema(
        2018,
        ZoneInfo("US/Mountain"),
        TimeIntervalType.PERIOD_BEGINNING,
        "base_table",
        has_tz_col=True,
    )
    df = generate_dataframe_with_tz_col(from_schema)
    run_conversion_to_column_time_zones(iter_engines, df, from_schema, wrap_time_allowed)
