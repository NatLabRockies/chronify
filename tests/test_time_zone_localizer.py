from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, tzinfo
import numpy as np
import pytest
from typing import Any

import pandas as pd

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query, write_table
from chronify.time_utils import get_standard_time_zone
from chronify.time_zone_localizer import (
    TimeZoneLocalizer,
    TimeZoneLocalizerByColumn,
    localize_time_zone,
    localize_time_zone_by_column,
)
from chronify.time_configs import DatetimeRange, DatetimeRangeWithTZColumn, DatetimeRangeBase
from chronify.models import TableSchema
from chronify.time import TimeDataType, TimeIntervalType
from chronify.datetime_range_generator import (
    DatetimeRangeGenerator,
    DatetimeRangeGeneratorExternalTimeZone,
)
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
    time_col = schema.time_config.time_column
    ts_dct = DatetimeRangeGeneratorExternalTimeZone(
        schema.time_config
    ).list_timestamps_by_time_zone()
    dfo_list = []
    for i, (time_zone, data) in enumerate(ts_dct.items()):
        dfo_list.append(
            pd.DataFrame(
                {x: i for x in schema.time_array_id_columns}
                | {
                    time_col: pd.to_datetime(data).tz_localize(None),
                    "time_zone": time_zone,
                    schema.value_column: np.random.rand(len(data)),
                }
            )
        )
    dfo = pd.concat(dfo_list, ignore_index=True)
    dfo = dfo.reset_index()
    return dfo


def get_datetime_schema(
    year: int,
    tzinfo: tzinfo | None,
    interval_type: TimeIntervalType,
    name: str,
) -> TableSchema:
    start = datetime(year=year, month=3, day=11, tzinfo=tzinfo)
    end = datetime(year=year, month=3, day=12, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    cols = ["id"]
    time_config = DatetimeRange(
        dtype=TimeDataType.TIMESTAMP_TZ if tzinfo else TimeDataType.TIMESTAMP_NTZ,
        start=start,
        resolution=resolution,
        length=length,
        interval_type=interval_type,
        time_column="timestamp",
    )
    schema = TableSchema(
        name=name,
        time_config=time_config,
        time_array_id_columns=cols,
        value_column="value",
    )
    return schema


def get_datetime_with_tz_col_schema(
    year: int,
    tzinfo: tzinfo | None,
    interval_type: TimeIntervalType,
    name: str,
    standard_tz: bool = False,
) -> TableSchema:
    start = datetime(year=year, month=3, day=11, tzinfo=tzinfo)
    end = datetime(year=year, month=3, day=12, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    cols = ["id"]
    time_zones = [
        ZoneInfo("US/Eastern"),
        ZoneInfo("US/Central"),
        ZoneInfo("US/Mountain"),
    ]
    if standard_tz:
        time_zones = [get_standard_time_zone(tz) for tz in time_zones]
    time_config: DatetimeRangeBase = DatetimeRangeWithTZColumn(
        dtype=TimeDataType.TIMESTAMP_NTZ,
        start=start,
        resolution=resolution,
        length=length,
        interval_type=interval_type,
        time_column="timestamp",
        time_zone_column="time_zone",
        time_zones=time_zones,
    )
    schema = TableSchema(
        name=name,
        time_config=time_config,
        time_array_id_columns=cols,
        value_column="value",
    )
    return schema


def ingest_data(
    backend: IbisBackend,
    df: pd.DataFrame,
    schema: TableSchema,
) -> None:
    write_table(backend, df, schema.name, [schema.time_config], if_exists="replace")


def get_mapped_dataframe(
    backend: IbisBackend,
    table_name: str,
    time_config: DatetimeRangeBase,
) -> pd.DataFrame:
    table = backend.table(table_name)
    queried = read_query(backend, table, time_config)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)
    return queried


def run_localization(
    backend: IbisBackend,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_time_zone: tzinfo | None,
) -> None:
    ingest_data(backend, df, from_schema)
    to_schema = localize_time_zone(
        backend, from_schema, to_time_zone, check_mapped_timestamps=True
    )
    dfo = get_mapped_dataframe(backend, to_schema.name, to_schema.time_config)
    assert df["value"].equals(dfo["value"])
    if to_time_zone is None:
        expected = df["timestamp"]
    else:
        std_tz = get_standard_time_zone(to_time_zone)
        expected = df["timestamp"].dt.tz_localize(std_tz).dt.tz_convert(to_time_zone)

    assert (dfo["timestamp"] == expected).prod() == 1


def run_localization_to_column_time_zones(
    backend: IbisBackend,
    df: pd.DataFrame,
    from_schema: TableSchema,
) -> None:
    ingest_data(backend, df, from_schema)
    to_schema = localize_time_zone_by_column(
        backend,
        from_schema,
        check_mapped_timestamps=True,
    )
    dfo = get_mapped_dataframe(backend, to_schema.name, to_schema.time_config)
    dfo = dfo[df.columns].sort_values(by="index").reset_index(drop=True)
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"])  # needed for engine 2, not sure why
    assert df["value"].equals(dfo["value"])
    for i in range(len(dfo)):
        tzn = dfo.loc[i, "time_zone"]
        if tzn == "None":
            ts = dfo.loc[i, "timestamp"].replace(tzinfo=None)
        else:
            # source data is in local standard time
            ts = dfo.loc[i, "timestamp"].tz_convert(tzn).replace(tzinfo=None)

        assert df.loc[i, "timestamp"] == ts


def run_localization_with_error(
    backend: IbisBackend,
    df: pd.DataFrame,
    from_schema: TableSchema,
    error: tuple[Any, str],
) -> None:
    ingest_data(backend, df, from_schema)

    with pytest.raises(error[0], match=error[1]):
        TimeZoneLocalizer(backend, from_schema, None).localize_time_zone(
            check_mapped_timestamps=True
        )


def run_localization_by_column_with_error(
    backend: IbisBackend,
    df: pd.DataFrame,
    from_schema: TableSchema,
    error: tuple[Any, str],
    time_zone_column: str | None = None,
) -> None:
    ingest_data(backend, df, from_schema)

    with pytest.raises(error[0], match=error[1]):
        TimeZoneLocalizerByColumn(
            backend,
            from_schema,
            time_zone_column=time_zone_column,
        ).localize_time_zone(check_mapped_timestamps=True)


@pytest.mark.parametrize("to_time_zone", [None, ZoneInfo("EST")])
def test_time_localization(iter_backends: IbisBackend, to_time_zone: tzinfo | None) -> None:
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    run_localization(iter_backends, df, from_schema, to_time_zone)


@pytest.mark.parametrize("from_time_tz", [None, ZoneInfo("US/Mountain"), ZoneInfo("MST")])
def test_time_localization_by_column(
    iter_backends: IbisBackend, from_time_tz: tzinfo | None
) -> None:
    from_schema = get_datetime_with_tz_col_schema(
        2018,
        from_time_tz,
        TimeIntervalType.PERIOD_BEGINNING,
        "base_table",
        standard_tz=True,
    )
    df = generate_dataframe_with_tz_col(from_schema)
    run_localization_to_column_time_zones(iter_backends, df, from_schema)


# Error tests for TimeZoneLocalizer
def test_time_localizer_to_dst_time_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizer raises error when to_time_zone is a non standard time zone"""
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    to_time_zone = ZoneInfo("US/Mountain")  # has DST
    ingest_data(iter_backends, df, from_schema)
    with pytest.raises(
        InvalidParameter, match="TimeZoneLocalizer only supports standard time zones"
    ):
        localize_time_zone(iter_backends, from_schema, to_time_zone, check_mapped_timestamps=True)


def test_time_localizer_with_tz_aware_config_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizer raises error when start time is tz-aware"""
    from_schema = get_datetime_schema(
        2018, ZoneInfo("US/Mountain"), TimeIntervalType.PERIOD_BEGINNING, "base_table"
    )
    df = generate_datetime_dataframe(from_schema)
    error = (InvalidParameter, "Source schema time config start time must be tz-naive")
    run_localization_with_error(iter_backends, df, from_schema, error)


def test_time_localizer_with_wrong_dtype_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizer raises error when dtype is not TIMESTAMP_NTZ"""
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    # Manually change dtype to TIMESTAMP_TZ to trigger error
    from_schema.time_config = from_schema.time_config.model_copy(
        update={"dtype": TimeDataType.TIMESTAMP_TZ}
    )
    df = generate_datetime_dataframe(from_schema)
    error = (InvalidParameter, "Source schema time config dtype must be TIMESTAMP_NTZ")
    run_localization_with_error(iter_backends, df, from_schema, error)


def test_time_localizer_with_datetime_range_with_tz_col_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizer raises error when time config is DatetimeRangeWithTZColumn"""
    from_schema = get_datetime_with_tz_col_schema(
        2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table", standard_tz=True
    )
    df = generate_dataframe_with_tz_col(from_schema)
    error = (InvalidParameter, "try using TimeZoneLocalizerByColumn")
    run_localization_with_error(iter_backends, df, from_schema, error)


# Error tests for TimeZoneLocalizerByColumn
def test_time_localizer_by_column_to_dst_time_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizerByColumn raises error when to_time_zone is a non standard time zone"""
    from_schema = get_datetime_with_tz_col_schema(
        2018,
        None,
        TimeIntervalType.PERIOD_BEGINNING,
        "base_table",
        standard_tz=False,
    )
    df = generate_dataframe_with_tz_col(from_schema)
    ingest_data(iter_backends, df, from_schema)
    with pytest.raises(
        InvalidParameter, match="TimeZoneLocalizerByColumn only supports standard time zones"
    ):
        localize_time_zone_by_column(iter_backends, from_schema, check_mapped_timestamps=True)


def test_time_localizer_by_column_missing_tz_column_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizerByColumn raises error when time_zone_column is missing for DatetimeRange"""
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    error = (InvalidParameter, "time_zone_column must be provided")
    run_localization_by_column_with_error(iter_backends, df, from_schema, error)


def test_time_localizer_by_column_wrong_dtype_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizerByColumn raises error when dtype is not TIMESTAMP_NTZ"""
    from_schema = get_datetime_with_tz_col_schema(
        2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table", standard_tz=True
    )
    # Change dtype to TIMESTAMP_TZ to trigger error
    from_schema.time_config = from_schema.time_config.model_copy(
        update={"dtype": TimeDataType.TIMESTAMP_TZ}
    )
    df = generate_dataframe_with_tz_col(from_schema)
    error = (InvalidParameter, "Source schema time config dtype must be TIMESTAMP_NTZ")
    run_localization_by_column_with_error(iter_backends, df, from_schema, error)


def test_time_localizer_by_column_non_standard_tz_error(iter_backends: IbisBackend) -> None:
    """Test that TimeZoneLocalizerByColumn raises error when time zones are not standard"""
    from_schema = get_datetime_with_tz_col_schema(
        2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table", standard_tz=False
    )
    df = generate_dataframe_with_tz_col(from_schema)
    error = (InvalidParameter, "is not a standard time zone")
    run_localization_by_column_with_error(iter_backends, df, from_schema, error)


def test_localize_time_zone_by_column_missing_tz_column_error(iter_backends: IbisBackend) -> None:
    """Test that localize_time_zone_by_column raises error when time_zone_column is None for DatetimeRange"""
    from_schema = get_datetime_schema(2018, None, TimeIntervalType.PERIOD_BEGINNING, "base_table")
    df = generate_datetime_dataframe(from_schema)
    error = (
        Exception,
        "time_zone_column must be provided when source schema time config is of type DatetimeRange",
    )
    run_localization_by_column_with_error(
        iter_backends, df, from_schema, error, time_zone_column=None
    )
