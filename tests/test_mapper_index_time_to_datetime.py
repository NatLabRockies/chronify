import pandas as pd
from sqlalchemy import Engine, MetaData
import pytest
from datetime import timedelta
from zoneinfo import ZoneInfo
from typing import Any, Optional

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRange,
    IndexTimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.exceptions import ConflictingInputsError
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, DaylightSavingAdjustmentType
from chronify.time_utils import get_standard_time_zone


def output_dst_schema(
    interval_type: TimeIntervalType,
    standard_time: bool = False,
    interval_resolution: timedelta = timedelta(hours=1),
) -> TableSchema:
    if standard_time:
        tz = "MST"
    else:
        tz = "US/Mountain"
    nts = int(8760 * 60 * 60 / interval_resolution.seconds)
    return TableSchema(
        name="output_data",
        time_config=DatetimeRange(
            start=pd.Timestamp("2018-01-01 00:00", tz=tz),
            resolution=interval_resolution,
            length=nts,
            interval_type=interval_type,
            time_column="timestamp",
        ),
        time_array_id_columns=[],
        value_column="value",
    )


def data_for_simple_mapping(
    tz_naive: bool = False,
    interval_shift: bool = False,
    standard_time: bool = False,
    interval_resolution: timedelta = timedelta(hours=1),
) -> tuple[pd.DataFrame, TableSchema, TableSchema]:
    start_timestamp = pd.Timestamp("2018-01-01 00:00")
    end_timestamp = pd.Timestamp("2018-12-31 23:59")
    timeseries = pd.date_range(start_timestamp, end_timestamp, freq=interval_resolution)
    nts = len(timeseries)
    src_df = pd.DataFrame({"index_time": range(1, nts + 1), "value": range(1, nts + 1)})

    if tz_naive:
        st_ts = start_timestamp
    else:
        st_ts = start_timestamp.tz_localize("US/Mountain")

    time_config = IndexTimeRange(
        start=1,
        length=nts,
        start_timestamp=st_ts,
        resolution=interval_resolution,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="index_time",
    )
    src_schema = TableSchema(
        name="input_data",
        time_config=time_config,
        time_array_id_columns=[],
        value_column="value",
    )
    if interval_shift:
        interval_type = TimeIntervalType.PERIOD_ENDING
    else:
        interval_type = TimeIntervalType.PERIOD_BEGINNING

    dst_schema = output_dst_schema(
        interval_type, standard_time=standard_time, interval_resolution=interval_resolution
    )
    return src_df, src_schema, dst_schema


def data_for_unaligned_time_mapping(
    interval_shift: bool = False,
    standard_time: bool = False,
    interval_resolution: timedelta = timedelta(hours=1),
) -> tuple[pd.DataFrame, TableSchema, TableSchema]:
    time_array_len = int(8760 * 60 * 60 / interval_resolution.seconds)
    src_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "index_time": range(1, time_array_len + 1),
                    "value": range(1, time_array_len + 1),
                    "time_zone": ["US/Mountain"] * time_array_len,
                    "id": [1] * time_array_len,
                },
            ),
            pd.DataFrame(
                {
                    "index_time": range(1, time_array_len + 1),
                    "value": range(10, time_array_len * 10 + 10, 10),
                    "time_zone": ["US/Central"] * time_array_len,
                    "id": [2] * time_array_len,
                },
            ),
        ]
    )

    time_config = IndexTimeRangeWithTZColumn(
        start=1,
        length=time_array_len,
        start_timestamp=pd.Timestamp("2018-01-01 00:00"),
        resolution=interval_resolution,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="index_time",
        time_zone_column="time_zone",
    )
    src_schema = TableSchema(
        name="input_data",
        time_config=time_config,
        time_array_id_columns=["id"],
        value_column="value",
    )
    if interval_shift:
        interval_type = TimeIntervalType.PERIOD_ENDING
    else:
        interval_type = TimeIntervalType.PERIOD_BEGINNING
    dst_schema = output_dst_schema(
        interval_type, standard_time=standard_time, interval_resolution=interval_resolution
    )
    dst_schema.time_array_id_columns = ["id"]
    return src_df, src_schema, dst_schema


def run_test(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: Optional[tuple[Any, str]],
    data_adjustment: Optional[TimeBasedDataAdjustment] = None,
    wrap_time_allowed: bool = False,
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
            map_time(
                engine,
                metadata,
                from_schema,
                to_schema,
                data_adjustment=data_adjustment,
                wrap_time_allowed=wrap_time_allowed,
                check_mapped_timestamps=True,
            )
    else:
        map_time(
            engine,
            metadata,
            from_schema,
            to_schema,
            data_adjustment=data_adjustment,
            wrap_time_allowed=wrap_time_allowed,
            check_mapped_timestamps=True,
        )


def get_output_table(engine: Engine, to_schema: TableSchema) -> pd.DataFrame:
    with engine.connect() as conn:
        query = f"select * from {to_schema.name}"
        queried = read_database(query, conn, to_schema.time_config)
    return queried


@pytest.mark.parametrize("src_tz_naive", [True, False])
@pytest.mark.parametrize("interval_shift", [False, True])
@pytest.mark.parametrize("dst_std_time", [False, True])
def test_simple_mapping(
    iter_engines: Engine, src_tz_naive: bool, interval_shift: bool, dst_std_time: bool
) -> None:
    src_df, src_schema, dst_schema = data_for_simple_mapping(
        tz_naive=src_tz_naive, interval_shift=interval_shift, standard_time=dst_std_time
    )
    error = None
    run_test(iter_engines, src_df, src_schema, dst_schema, error)

    dfo = get_output_table(iter_engines, dst_schema)
    assert sorted(dfo["value"]) == sorted(src_df["value"])


@pytest.mark.parametrize("interval_shift", [False, True])
@pytest.mark.parametrize("dst_std_time", [False, True])
def test_unaligned_time_mapping(
    iter_engines: Engine, interval_shift: bool, dst_std_time: bool
) -> None:
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping(
        interval_shift=interval_shift, standard_time=dst_std_time
    )
    error = None
    wrap_time_allowed = True
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
        wrap_time_allowed=wrap_time_allowed,
    )

    dfo = get_output_table(iter_engines, dst_schema)
    assert sorted(dfo["value"]) == sorted(src_df["value"])


def test_unaligned_time_mapping_without_wrap_time(iter_engines: Engine) -> None:
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping()
    error = (
        ConflictingInputsError,
        "Length must match between",
    )
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
    )


@pytest.mark.parametrize("interval_shift", [False, True])
@pytest.mark.parametrize("dst_std_time", [False, True])
@pytest.mark.parametrize("interpolate_fallback", [False, True])
def test_industrial_time_mapping(
    iter_engines: Engine,
    interval_shift: bool,
    dst_std_time: bool,
    interpolate_fallback: bool,
) -> None:
    """I.e., unaligned time mapping with data_adjustment"""
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping(
        interval_shift=interval_shift, standard_time=dst_std_time
    )
    error = None
    if interpolate_fallback:
        data_adjustment = TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_INTERPOLATE_FALLBACK
        )
    else:
        data_adjustment = TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_DUPLICATE_FALLBACK
        )
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
        data_adjustment=data_adjustment,
        wrap_time_allowed=True,
    )

    dfo = get_output_table(iter_engines, dst_schema)
    dfo = dfo.sort_values(by=["time_zone", "value"]).reset_index(drop=True)

    # Check value associated with springforward hour is dropped
    # If the following fail, print to debug: dfo.loc[1656:1659], dfo.loc[1656+8760:1659+8760]
    assert 1659 not in dfo["value"].values
    assert 16590 not in dfo["value"].values

    # Check value associated with fallback hour
    if interpolate_fallback:
        if interval_shift:
            fallback = pd.Timestamp("'2018-11-04 02:00:00")
        else:
            fallback = pd.Timestamp("'2018-11-04 01:00:00")

        cond1 = dfo["value"] == 7370.5
        tz1 = ZoneInfo(dfo.loc[cond1, "time_zone"].tolist()[0])
        tz1_std = get_standard_time_zone(tz1)
        truth1 = fallback.tz_localize(tz1_std)
        assert dfo.loc[cond1, "timestamp"].tolist()[0] == truth1

        cond2 = dfo["value"] == 73705
        tz2 = ZoneInfo(dfo.loc[cond2, "time_zone"].tolist()[0])
        tz2_std = get_standard_time_zone(tz2)
        truth2 = fallback.tz_localize(tz2_std)
        assert dfo.loc[cond2, "timestamp"].tolist()[0] == truth2
    else:
        assert dfo.loc[7367:7370]["value"].value_counts()[73700] == 2
        assert dfo.loc[7367 + 8760 : 7370 + 8760]["value"].value_counts()[7370] == 2


@pytest.mark.parametrize("dst_std_time", [False, True])
@pytest.mark.parametrize("interpolate_fallback", [False, True])
def test_industrial_time_subhourly(
    iter_engines: Engine,
    dst_std_time: bool,
    interpolate_fallback: bool,
) -> None:
    src_df, src_schema, dst_schema = data_for_unaligned_time_mapping(
        standard_time=dst_std_time, interval_resolution=timedelta(minutes=30)
    )
    error = None
    if interpolate_fallback:
        data_adjustment = TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_INTERPOLATE_FALLBACK
        )
    else:
        data_adjustment = TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_DUPLICATE_FALLBACK
        )
    run_test(
        iter_engines,
        src_df,
        src_schema,
        dst_schema,
        error,
        data_adjustment=data_adjustment,
        wrap_time_allowed=True,
    )

    dfo = get_output_table(iter_engines, dst_schema)
    dfo = dfo.sort_values(by=["time_zone", "timestamp"]).reset_index(drop=True)
    dfo.loc[dfo["timestamp"].dt.date.astype(str) == "2018-11-04"]

    assert set([3317, 3318, 33170, 33180]).intersection(set(dfo["value"].values)) == set()

    # Check value associated with fallback hour
    if interpolate_fallback:
        for val in [14740, 147400]:
            cond = (dfo["value"] > val) & (dfo["value"] < val + 1)
            dfoi = dfo.loc[cond, "value"].tolist()
            assert dfoi == [val + x / (len(dfoi) + 1) for x in range(1, len(dfoi) + 1)]
    else:
        assert dfo.loc[32254:32303]["value"].value_counts()[[14739, 14740]].tolist() == [2, 2]
        assert dfo.loc[14730:14760]["value"].value_counts()[[147390, 147400]].tolist() == [2, 2]
