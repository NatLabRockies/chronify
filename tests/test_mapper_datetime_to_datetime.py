from zoneinfo import ZoneInfo
from datetime import tzinfo
import pytest
from datetime import datetime, timedelta
from typing import Any
import numpy as np

import pandas as pd
from sqlalchemy import Engine, MetaData

from chronify.sqlalchemy.functions import read_database, write_database
from chronify.time_series_mapper import map_time
from chronify.time_configs import DatetimeRange
from chronify.models import TableSchema
from chronify.time import TimeIntervalType, MeasurementType
from chronify.exceptions import ConflictingInputsError, InvalidParameter
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.time_utils import (
    shifted_interval_timestamps,
    rolled_interval_timestamps,
    wrapped_time_timestamps,
)


def generate_datetime_data(time_config: DatetimeRange) -> pd.DatetimeIndex:  # type: ignore
    return pd.to_datetime(DatetimeRangeGenerator(time_config).list_timestamps())


def generate_datetime_dataframe(schema: TableSchema) -> pd.DataFrame:
    df = pd.DataFrame({schema.time_config.time_column: generate_datetime_data(schema.time_config)})

    for i, x in enumerate(schema.time_array_id_columns):
        df[x] = i
    df[schema.value_column] = np.random.rand(len(df))
    return df


def get_datetime_schema(
    year: int, tzinfo: tzinfo | None, interval_type: TimeIntervalType, name: str
) -> TableSchema:
    start = datetime(year=year, month=1, day=1, tzinfo=tzinfo)
    end = datetime(year=year + 1, month=1, day=1, tzinfo=tzinfo)
    resolution = timedelta(hours=1)
    length = (end - start) / resolution + 1
    schema = TableSchema(
        name=name,
        time_config=DatetimeRange(
            start=start,
            resolution=resolution,
            length=length,
            interval_type=interval_type,
            time_column="timestamp",
        ),
        time_array_id_columns=["id"],
        value_column="value",
    )
    return schema


def ingest_data(
    engine: Engine,
    df: pd.DataFrame,
    schema: TableSchema,
) -> None:
    metadata = MetaData()
    with engine.begin() as conn:
        write_database(df, conn, schema.name, [schema.time_config], if_table_exists="replace")
    metadata.reflect(engine, views=True)


def run_test_with_error(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
    error: tuple[Any, str],
) -> None:
    metadata = MetaData()
    ingest_data(engine, df, from_schema)
    with pytest.raises(error[0], match=error[1]):
        map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)


def get_mapped_results(
    engine: Engine,
    df: pd.DataFrame,
    from_schema: TableSchema,
    to_schema: TableSchema,
) -> pd.DataFrame:
    metadata = MetaData()
    ingest_data(engine, df, from_schema)
    map_time(engine, metadata, from_schema, to_schema, check_mapped_timestamps=True)

    with engine.connect() as conn:
        query = f"select * from {to_schema.name}"
        queried = read_database(query, conn, to_schema.time_config)
    queried = queried.sort_values(by=["id", "timestamp"]).reset_index(drop=True)[df.columns]

    return queried


def check_time_shift_timestamps(
    dfi: pd.DataFrame, dfo: pd.DataFrame, to_time_config: DatetimeRange
) -> None:
    assert not dfo.equals(dfi)
    df_truth = generate_datetime_data(to_time_config)
    assert (dfo[to_time_config.time_column] == df_truth).all()


def check_time_shift_values(
    dfi: pd.DataFrame,
    dfo: pd.DataFrame,
    from_time_config: DatetimeRange,
    to_time_config: DatetimeRange,
) -> None:
    for idx in [5, 50, 500]:
        row = dfi.loc[idx]
        ftz, ttz = from_time_config.start.tzinfo, to_time_config.start.tzinfo
        if None in (ftz, ttz):
            ts = row["timestamp"].tz_localize(ttz)
        else:
            ts = row["timestamp"].tz_convert(ttz)
        fint, tint = from_time_config.interval_type, to_time_config.interval_type
        match fint, tint:
            case TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING:
                mult = 1
            case TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING:
                mult = -1
            case TimeIntervalType.INSTANTANEOUS, TimeIntervalType.INSTANTANEOUS:
                mult = 0
        ts += from_time_config.resolution * mult
        assert row["value"] == dfo.loc[dfo["timestamp"] == ts, "value"].iloc[0]


def test_roll_time_using_shift_and_wrap() -> None:
    from_schema = get_datetime_schema(2024, None, TimeIntervalType.PERIOD_ENDING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2024, None, TimeIntervalType.PERIOD_BEGINNING, "to_table")
    data = generate_datetime_data(to_schema.time_config)

    df["rolled"] = rolled_interval_timestamps(
        df[from_schema.time_config.time_column].tolist(),
        from_schema.time_config.interval_type,
        to_schema.time_config.interval_type,
        data,
    )
    df["rolled2"] = shifted_interval_timestamps(
        df[from_schema.time_config.time_column].tolist(),
        from_schema.time_config.interval_type,
        to_schema.time_config.interval_type,
    )
    df["rolled2"] = wrapped_time_timestamps(
        df["rolled2"].tolist(),
        data,
    )
    assert df["rolled"].equals(df["rolled2"])
    assert set(data) == set(df["rolled"].tolist())


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test_time_interval_shift(
    iter_engines: Engine,
    tzinfo: tzinfo | None,
) -> None:
    from_schema = get_datetime_schema(
        2020, tzinfo, TimeIntervalType.PERIOD_BEGINNING, "from_table"
    )
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, tzinfo, TimeIntervalType.PERIOD_ENDING, "to_table")

    queried = get_mapped_results(iter_engines, df, from_schema, to_schema)
    check_time_shift_timestamps(df, queried, to_schema.time_config)
    check_time_shift_values(df, queried, from_schema.time_config, to_schema.time_config)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("US/Eastern"), None])
def test_time_interval_shift_different_time_ranges(
    iter_engines: Engine,
    tzinfo: tzinfo | None,
) -> None:
    from_schema = get_datetime_schema(
        2020, tzinfo, TimeIntervalType.PERIOD_BEGINNING, "from_table"
    )
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, tzinfo, TimeIntervalType.PERIOD_ENDING, "to_table")
    to_schema.time_config.start += to_schema.time_config.resolution

    queried = get_mapped_results(iter_engines, df, from_schema, to_schema)
    check_time_shift_timestamps(df, queried, to_schema.time_config)
    assert df["value"].equals(queried["value"])


@pytest.mark.parametrize(
    "tzinfo_tuple",
    [
        # (ZoneInfo("US/Eastern"), None),
        (None, ZoneInfo("Etc/GMT+5")),
        # (ZoneInfo("US/Eastern"), ZoneInfo("US/Mountain")),
    ],
)
def test_time_shift_different_timezones(
    iter_engines: Engine, tzinfo_tuple: tuple[tzinfo | None]
) -> None:
    from_schema = get_datetime_schema(
        2020, tzinfo_tuple[0], TimeIntervalType.PERIOD_BEGINNING, "from_table"
    )
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(
        2020, tzinfo_tuple[1], TimeIntervalType.PERIOD_ENDING, "to_table"
    )

    queried = get_mapped_results(iter_engines, df, from_schema, to_schema)
    check_time_shift_timestamps(df, queried, to_schema.time_config)
    check_time_shift_values(df, queried, from_schema.time_config, to_schema.time_config)


def test_instantaneous_interval_type(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.INSTANTANEOUS, "to_table")
    error = (ConflictingInputsError, "If instantaneous time interval is used")
    run_test_with_error(iter_engines, df, from_schema, to_schema, error)


def test_schema_compatibility(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_ENDING, "to_table")
    to_schema.time_array_id_columns += ["extra_column"]
    error = (ConflictingInputsError, ".* cannot produce the columns")
    run_test_with_error(iter_engines, df, from_schema, to_schema, error)


def test_measurement_type_consistency(
    iter_engines: Engine,
) -> None:
    from_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(from_schema)
    to_schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_ENDING, "to_table")
    to_schema.time_config.measurement_type = MeasurementType.MAX
    error = (ConflictingInputsError, "Inconsistent measurement_types")
    run_test_with_error(iter_engines, df, from_schema, to_schema, error)


def test_duplicated_configs_in_write_database(
    iter_engines: Engine,
) -> None:
    schema = get_datetime_schema(2020, None, TimeIntervalType.PERIOD_BEGINNING, "from_table")
    df = generate_datetime_dataframe(schema)
    configs = [schema.time_config, schema.time_config]

    # Ingest
    with iter_engines.connect() as conn:
        if conn.engine.name == "sqlite":
            with pytest.raises(InvalidParameter, match="More than one datetime config found"):
                write_database(df, conn, schema.name, configs, if_table_exists="replace")
        else:
            write_database(df, conn, schema.name, configs, if_table_exists="replace")
