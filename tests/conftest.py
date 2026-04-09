from typing import Generator
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import pytest

from chronify.ibis import IbisBackend, make_backend
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat
from chronify.time_configs import RepresentativePeriodTimeNTZ, RepresentativePeriodTimeTZ


BACKEND_NAMES = ["duckdb", "sqlite"]


@pytest.fixture
def create_duckdb_backend() -> IbisBackend:
    """Return a DuckDB backend."""
    return make_backend("duckdb")


@pytest.fixture(params=BACKEND_NAMES)
def iter_backends(request) -> Generator[IbisBackend, None, None]:
    """Return an iterable of in-memory backends to test."""
    backend = make_backend(request.param)
    yield backend
    backend.dispose()


@pytest.fixture(params=BACKEND_NAMES)
def iter_stores_by_engine(request) -> Generator[Store, None, None]:
    """Return an iterable of stores with different backends to test."""
    backend = make_backend(request.param)
    store = Store(backend=backend)
    yield store
    store.dispose()


@pytest.fixture(params=BACKEND_NAMES)
def iter_stores_by_engine_no_data_ingestion(request) -> Generator[Store, None, None]:
    """Return an iterable of stores with different backends to test."""
    backend = make_backend(request.param)
    store = Store(backend=backend)
    yield store
    store.dispose()


@pytest.fixture(params=BACKEND_NAMES)
def iter_backends_file(request, tmp_path) -> Generator[tuple[IbisBackend, str], None, None]:
    """Return an iterable of file-based backends to test."""
    file_path = tmp_path / "store.db"
    backend = make_backend(request.param, database=str(file_path))
    yield backend, request.param
    backend.dispose()


@pytest.fixture(params=BACKEND_NAMES)
def iter_backend_names(request) -> Generator[str, None, None]:
    """Return an iterable of backend names."""
    yield request.param


def one_week_per_month_by_hour_data(tz_aware: bool = False) -> tuple[pd.DataFrame, int]:
    hours_per_year = 12 * 7 * 24
    num_time_arrays = 3
    df = pd.DataFrame(
        {
            "id": np.repeat(range(1, 1 + num_time_arrays), hours_per_year),
            "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
            # 0: Monday, 6: Sunday
            "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
            "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
            "value": np.random.random(hours_per_year * num_time_arrays),
        }
    )
    if tz_aware:
        df["time_zone"] = df["id"].map(
            dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
        )
    return df, num_time_arrays


@pytest.fixture
def one_week_per_month_by_hour_table() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour tz-naive data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_week_per_month_by_hour_data()

    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeNTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


@pytest.fixture
def one_week_per_month_by_hour_table_tz() -> tuple[pd.DataFrame, int, TableSchema]:
    """Return a table suitable for testing one_week_per_month_by_hour tz-aware data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_week_per_month_by_hour_data(tz_aware=True)

    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
            time_zone_column="time_zone",
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


def one_weekday_day_and_one_weekend_day_per_month_by_hour_data(
    tz_aware: bool = False,
) -> tuple[pd.DataFrame, int]:
    hours_per_year = 12 * 2 * 24
    num_time_arrays = 3
    df = pd.DataFrame(
        {
            "id": np.repeat(range(1, 1 + num_time_arrays), hours_per_year),
            "month": np.tile(np.repeat(range(1, 13), 2 * 24), num_time_arrays),
            # 0: Monday, 6: Sunday
            "is_weekday": np.tile(np.tile(np.repeat([True, False], 24), 12), num_time_arrays),
            "hour": np.tile(np.tile(range(24), 12 * 2), num_time_arrays),
            "value": np.random.random(hours_per_year * num_time_arrays),
        }
    )
    if tz_aware:
        df["time_zone"] = df["id"].map(
            dict(zip([1, 2, 3], ["US/Central", "US/Mountain", "US/Pacific"]))
        )
    return df, num_time_arrays


@pytest.fixture
def one_weekday_day_and_one_weekend_day_per_month_by_hour_table() -> (
    tuple[pd.DataFrame, int, TableSchema]
):
    """Return a table suitable for testing one_weekday_day_and_one_weekend_day_per_month_by_hour tz-naive data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_weekday_day_and_one_weekend_day_per_month_by_hour_data()
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeNTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR,
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


@pytest.fixture
def one_weekday_day_and_one_weekend_day_per_month_by_hour_table_tz() -> (
    tuple[pd.DataFrame, int, TableSchema]
):
    """Return a table suitable for testing one_weekday_day_and_one_weekend_day_per_month_by_hour tz-aware data.

    Returns
    -------
    DataFrame, number of time arrays in DataFrame
    """
    df, num_time_arrays = one_weekday_day_and_one_weekend_day_per_month_by_hour_data(tz_aware=True)
    schema = TableSchema(
        name="ev_charging",
        value_column="value",
        time_config=RepresentativePeriodTimeTZ(
            time_format=RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR,
            time_zone_column="time_zone",
        ),
        time_array_id_columns=["id"],
    )
    return df, num_time_arrays, schema


def temp_csv_file(data: str):
    with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        tmp_file = Path(tmp_file.name)
        yield tmp_file

    tmp_file.unlink()


@pytest.fixture
def time_series_NMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    yield from temp_csv_file(
        f"name,month,day,{hours}\nGeneration,1,1,{load1}\nGeneration,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    yield from temp_csv_file(
        f"name,year,month,day,{hours}\ntest_generator,2023,1,1,{load1}\ntest_generator,2023,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDPV():
    header = "name,year,month,day,period,value\n"
    data = "test_generator,2023,1,1,H1-5,100\ntest_generator,2023,1,1,H6-12,200\ntest_generator,2023,1,1,H13-24,300\ntest_generator,2023,1,2,H1-24,400"
    yield from temp_csv_file(header + data)
