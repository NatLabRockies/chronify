from typing import Generator
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from chronify.ibis.backend import IbisBackend, make_backend
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat
from chronify.time_configs import RepresentativePeriodTimeNTZ, RepresentativePeriodTimeTZ


BACKENDS = ("duckdb", "sqlite", "spark")


@pytest.fixture
def create_duckdb_backend() -> IbisBackend:
    """Return an Ibis backend for DuckDB."""
    return make_backend("duckdb")


@pytest.fixture(params=[x for x in BACKENDS if x != "spark"])
def iter_backends(request) -> Generator[IbisBackend, None, None]:
    """Return an iterable of Ibis backends to test."""
    yield make_backend(request.param)


@pytest.fixture(params=[x for x in BACKENDS if x != "spark"])
def iter_stores_by_backend(request) -> Generator[Store, None, None]:
    """Return an iterable of stores with different backends to test.
    Will only return backends that support data ingestion.
    """
    backend = request.param
    store = Store(backend_name=backend)
    yield store
    store.dispose()


@pytest.fixture(params=BACKENDS)
def iter_stores_by_backend_no_data_ingestion(request, tmp_path) -> Generator[Store, None, None]:
    """Return an iterable of stores with different backends to test."""
    backend = request.param
    orig_tables_and_views: set[str] | None = set()
    if backend == "spark":
        # Use a temp directory for the spark warehouse to avoid conflicts
        warehouse_dir = tmp_path / "spark-warehouse"
        spark = (
            SparkSession.builder.appName("chronify_test")
            .config("spark.sql.warehouse.dir", str(warehouse_dir))
            .getOrCreate()
        )
        store = Store.create_new_spark_store(session=spark)
        for name in store._backend.list_tables():
            orig_tables_and_views.add(name)
    else:
        store = Store(backend_name=backend)
        orig_tables_and_views = None
    yield store
    if backend == "spark" and orig_tables_and_views is not None:
        # Cleanup views and tables created during test
        for name in store._backend.list_tables():
            if name not in orig_tables_and_views:
                try:
                    store._backend.drop_view(name, if_exists=True)
                except Exception:
                    pass
                try:
                    store._backend.drop_table(name, if_exists=True)
                except Exception:
                    pass


@pytest.fixture(params=[x for x in BACKENDS if x != "spark"])
def iter_backends_file(request, tmp_path) -> Generator[IbisBackend, None, None]:
    """Return an iterable of Ibis file-based backends to test."""
    backend = request.param
    file_path = tmp_path / "store.db"
    yield make_backend(backend, file_path=file_path)


@pytest.fixture(params=[x for x in BACKENDS if x != "spark"])
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
