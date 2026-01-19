import fileinput
import gc
import shutil
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo
from itertools import chain

import duckdb
import ibis.expr.datatypes as dt
import numpy as np
import pandas as pd
import pytest

from chronify.csv_io import read_csv
from chronify.duckdb.functions import unpivot
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidOperation,
    InvalidParameter,
    InvalidTable,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.ibis.backend import IbisBackend, make_backend
from chronify.models import ColumnDType, CsvTableSchema, PivotedTableSchema, TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType, DaylightSavingAdjustmentType
from chronify.time_configs import (
    DatetimeRange,
    DatetimeRangeWithTZColumn,
    IndexTimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_checker import check_timestamp_lists
from chronify.utils.sql import make_temp_view_name


GENERATOR_TIME_SERIES_FILE = "tests/data/gen.csv"


@pytest.fixture
def generators_schema():
    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("EST")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
    )
    dst_schema = TableSchema(
        name="generators",
        time_config=time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    yield Path(GENERATOR_TIME_SERIES_FILE), src_schema, dst_schema


@pytest.fixture
def multiple_tables():
    resolution = timedelta(hours=1)
    df_base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", "2020-12-31 23:00:00", freq=resolution),
            "value": np.random.random(8784),
        }
    )
    df1 = df_base.copy()
    df2 = df_base.copy()
    df1["id"] = 1
    df2["id"] = 2
    tables = [df1, df2]
    schema = TableSchema(
        name="devices",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, 0),
            length=8784,
            resolution=timedelta(hours=1),
        ),
        time_array_id_columns=["id"],
    )
    yield tables, schema


@pytest.mark.parametrize("use_time_zone", [True, False])
def test_ingest_csv(iter_stores_by_backend: Store, tmp_path, generators_schema, use_time_zone):
    store = iter_stores_by_backend
    src_file, src_schema, dst_schema = generators_schema
    src_schema.column_dtypes[0] = ColumnDType(
        name="timestamp", dtype=dt.Timestamp(timezone="UTC") if use_time_zone else dt.Timestamp()
    )
    if use_time_zone:
        new_src_file = tmp_path / "gen_tz.csv"
        duckdb.sql(
            f"""
            SELECT timezone('EST', timestamp) as timestamp, gen1, gen2, gen3
            FROM read_csv('{src_file}')
        """
        ).to_df().to_csv(new_src_file, index=False)
        src_file = new_src_file
    store.ingest_from_csv(src_file, src_schema, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3

    new_file = tmp_path / "gen2.csv"
    shutil.copyfile(src_file, new_file)
    with fileinput.input([new_file], inplace=True) as f:
        for line in f:
            new_line = line.replace("gen1", "g1b").replace("gen2", "g2b").replace("gen3", "g3b")
            print(new_line, end="")

    timestamp_generator = make_time_range_generator(dst_schema.time_config)
    expected_timestamps = timestamp_generator.list_timestamps()

    # Test addition of new generators to the same table.
    src_schema2 = CsvTableSchema(
        time_config=src_schema.time_config,
        column_dtypes=[
            ColumnDType(
                name="timestamp",
                dtype=dt.Timestamp(timezone="UTC") if use_time_zone else dt.Timestamp(),
            ),
            ColumnDType(name="g1b", dtype=dt.Float64()),
            ColumnDType(name="g2b", dtype=dt.Float64()),
            ColumnDType(name="g3b", dtype=dt.Float64()),
        ],
        value_columns=["g1b", "g2b", "g3b"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    store.ingest_from_csv(new_file, src_schema2, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3 * 2
    all(df.timestamp.unique() == expected_timestamps)

    # Read a subset of the table.
    df2 = store.read_query(
        dst_schema.name, f"SELECT * FROM {dst_schema.name} WHERE generator = 'gen2'"
    )
    assert len(df2) == 8784
    df_gen2 = df[df["generator"] == "gen2"]
    assert all((df2.values == df_gen2.values)[0])

    # Adding the same rows should fail.
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema2, dst_schema)


def test_ingest_csvs_with_rollback(tmp_path, multiple_tables):
    # Note: Ibis doesn't support transactions in the same way as SQLAlchemy.
    # This test verifies basic table creation behavior.
    store = Store(backend_name="duckdb")
    tables, dst_schema = multiple_tables
    src_file1 = tmp_path / "file1.csv"
    src_file2 = tmp_path / "file2.csv"
    tables[0].to_csv(src_file1)
    tables[1].to_csv(src_file2)
    src_schema = CsvTableSchema(
        time_config=dst_schema.time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="id", dtype=dt.Int64()),
            ColumnDType(name="value", dtype=dt.Float64()),
        ],
        value_columns=[dst_schema.value_column],
        time_array_id_columns=dst_schema.time_array_id_columns,
    )

    store.ingest_from_csvs((src_file1, src_file2), src_schema, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == len(tables[0]) + len(tables[1])
    assert len(df.id.unique()) == 2


@pytest.mark.parametrize("existing_connection", [False])
def test_ingest_multiple_tables(
    iter_stores_by_backend: Store, multiple_tables, existing_connection: bool
):
    store = iter_stores_by_backend
    tables, schema = multiple_tables
    store.ingest_tables(tables, schema)
    query = "SELECT * FROM devices WHERE id = 2"
    df = store.read_query("devices", query)
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    assert df.equals(tables[1])


def test_ingest_multiple_tables_error(iter_stores_by_backend: Store, multiple_tables):
    store = iter_stores_by_backend
    tables, schema = multiple_tables
    orig_value = tables[1].loc[8783]["id"]
    tables[1].loc[8783] = (tables[1].loc[8783]["timestamp"], 0.1, 99)
    with pytest.raises(InvalidTable):
        store.ingest_tables(tables, schema)
    assert not store.has_table(schema.name)

    tables[1].loc[8783] = (tables[1].loc[8783]["timestamp"], 0.1, orig_value)
    store.ingest_tables(tables, schema)
    df = store.read_query(schema.name, f"select * from {schema.name} where id=2")
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    assert df.equals(tables[1])


@pytest.mark.parametrize("use_pandas", [False, True])
def test_ingest_pivoted_table(iter_stores_by_backend: Store, generators_schema, use_pandas: bool):
    store = iter_stores_by_backend
    src_file, src_schema, dst_schema = generators_schema
    pivoted_schema = PivotedTableSchema(**src_schema.model_dump(exclude={"column_dtypes"}))
    rel = read_csv(src_file, src_schema)
    input_table = rel.to_df() if use_pandas else rel
    store.ingest_pivoted_table(input_table, pivoted_schema, dst_schema)
    table = store._backend.table(dst_schema.name)
    query = table.filter(table["generator"] == "gen1")
    df = store._backend.execute(query)
    assert len(df) == 8784


def test_ingest_invalid_csv(iter_stores_by_backend: Store, tmp_path, generators_schema):
    store = iter_stores_by_backend
    src_file, src_schema, dst_schema = generators_schema
    lines = src_file.read_text().splitlines()[:-10]
    new_file = tmp_path / "data.csv"
    with open(new_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema, dst_schema)
    with pytest.raises(TableNotStored):
        store.read_table(dst_schema.name)


def test_invalid_schema(iter_stores_by_backend: Store, generators_schema):
    store = iter_stores_by_backend
    src_file, src_schema, dst_schema = generators_schema
    src_schema.value_columns = ["g1", "g2", "g3"]
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(src_file, src_schema, dst_schema)


def test_ingest_one_week_per_month_by_hour(
    iter_stores_by_backend: Store, one_week_per_month_by_hour_table
):
    store = iter_stores_by_backend
    df, num_time_arrays, schema = one_week_per_month_by_hour_table

    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert len(df2["id"].unique()) == num_time_arrays
    assert len(df2) == 24 * 7 * 12 * num_time_arrays
    columns = schema.time_config.list_time_columns()
    columns.insert(0, "value")
    assert all(df.sort_values(columns)["value"] == df2.sort_values(columns)["value"])


def test_ingest_one_week_per_month_by_hour_invalid(
    iter_stores_by_backend: Store, one_week_per_month_by_hour_table
):
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    df_filtered = df[df["hour"] != 5]
    assert len(df_filtered) < len(df)

    with pytest.raises(InvalidTable):
        store.ingest_table(df_filtered, schema)


def test_load_parquet(iter_stores_by_backend_no_data_ingestion: Store, tmp_path):
    store = iter_stores_by_backend_no_data_ingestion
    if store._backend.name == "sqlite":
        # SQLite doesn't support parquet
        return

    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("EST")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    dst_schema = TableSchema(
        name="generators",
        time_config=time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841
    out_file = tmp_path / "gen2.parquet"
    rel2.to_parquet(str(out_file))
    store.create_view_from_parquet(out_file, dst_schema)
    df = store.read_table(dst_schema.name)
    assert len(df) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df.timestamp.unique() == expected_timestamps)

    # This adds test coverage for Spark.
    as_dict = dst_schema.model_dump()
    as_dict["name"] = "test_view"
    schema2 = TableSchema(**as_dict)
    store.create_view_from_parquet(out_file, schema2)
    df2 = store.read_table(schema2.name)
    assert schema2.name in store.list_tables()
    assert len(df2) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df2.timestamp.unique() == expected_timestamps)
    store.drop_view(schema2.name)
    assert schema2.name not in store.list_tables()
    assert dst_schema.name in store.list_tables()
    df3 = store.read_table(dst_schema.name)
    assert len(df3) == 8784 * 3


@pytest.mark.parametrize(
    "params",
    [
        (True, 2020),
        (True, 2021),
        (False, 2020),
        (False, 2021),
    ],
)
def test_map_one_week_per_month_by_hour_to_datetime(
    tmp_path,
    iter_stores_by_backend_no_data_ingestion: Store,
    one_week_per_month_by_hour_table,
    one_week_per_month_by_hour_table_tz,
    params: tuple[bool, int],
):
    store = iter_stores_by_backend_no_data_ingestion
    use_time_zone, year = params
    if use_time_zone:
        df, num_time_arrays, src_schema = one_week_per_month_by_hour_table_tz
    else:
        df, num_time_arrays, src_schema = one_week_per_month_by_hour_table
    tzinfo = ZoneInfo("America/Denver") if use_time_zone else None
    time_array_len = 8784 if year % 4 == 0 else 8760
    dst_schema = TableSchema(
        name="ev_charging_datetime",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(year, 1, 1, 0, tzinfo=tzinfo),
            length=time_array_len,
            resolution=timedelta(hours=1),
        ),
        time_array_id_columns=["id"],
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        df.to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(df, src_schema)
    store.map_table_time_config(src_schema.name, dst_schema, check_mapped_timestamps=True)
    df2 = store.read_table(dst_schema.name)
    assert len(df2) == time_array_len * num_time_arrays
    actual = sorted(df2["timestamp"].unique())
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    if use_time_zone:
        expected = [pd.Timestamp(x) for x in expected]
    check_timestamp_lists(actual, expected)

    out_file = tmp_path / "out.parquet"
    assert not out_file.exists()
    store.write_table_to_parquet(dst_schema.name, out_file, overwrite=True)
    assert out_file.exists()

    with pytest.raises(TableAlreadyExists):
        store.map_table_time_config(src_schema.name, dst_schema, check_mapped_timestamps=True)


@pytest.mark.parametrize("tzinfo", [ZoneInfo("EST"), None])
def test_map_datetime_to_datetime(
    tmp_path, iter_stores_by_backend_no_data_ingestion: Store, tzinfo
):
    store = iter_stores_by_backend_no_data_ingestion
    time_array_len = 8784
    year = 2020

    src_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=0, tzinfo=tzinfo),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )
    dst_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=1, tzinfo=tzinfo),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_ENDING,
        time_column="timestamp",
    )

    src_csv_schema = CsvTableSchema(
        time_config=src_time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    dst_schema = TableSchema(
        name="generators_pe",
        time_config=dst_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        rel2.to_df().to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(rel2, src_schema)

    if tzinfo is None and store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None
    store.map_table_time_config(
        src_schema.name, dst_schema, output_file=output_file, check_mapped_timestamps=True
    )
    if output_file is None or store._backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name)
    else:
        df2 = pd.read_parquet(output_file)
    assert len(df2) == time_array_len * 3
    actual = sorted(df2["timestamp"].unique())
    assert isinstance(src_schema.time_config, DatetimeRange)
    assert actual[0] == src_schema.time_config.start + timedelta(hours=1)
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    check_timestamp_lists(actual, expected)


def test_map_index_time_to_datetime(
    tmp_path: Path, iter_stores_by_backend_no_data_ingestion: Store
) -> None:
    store = iter_stores_by_backend_no_data_ingestion
    year = 2018
    time_array_len = 8760
    src_schema = TableSchema(
        name="generators_index",
        time_array_id_columns=["generator", "time_zone"],
        value_column="value",
        time_config=IndexTimeRangeWithTZColumn(
            start=0,
            length=time_array_len,
            start_timestamp=pd.Timestamp(f"{year}-01-01 00:00"),
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="index_time",
            time_zone_column="time_zone",
        ),
    )
    dst_schema = TableSchema(
        name="generators_datetime",
        time_array_id_columns=["generator"],
        value_column="value",
        time_config=DatetimeRange(
            start=datetime(year=year, month=1, day=1, hour=1, tzinfo=ZoneInfo("EST")),
            resolution=timedelta(hours=1),
            length=time_array_len,
            interval_type=TimeIntervalType.PERIOD_ENDING,
            time_column="timestamp",
        ),
    )
    time_zones = ("US/Eastern", "US/Central", "US/Mountain", "US/Pacific")
    src_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "index_time": range(time_array_len),
                    "value": range(i, i + time_array_len),
                    "time_zone": [time_zone] * time_array_len,
                    "generator": [f"gen{i}"] * time_array_len,
                },
            )
            for i, time_zone in enumerate(time_zones)
        ]
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        src_df.to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(src_df, src_schema)

    if store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None
    store.map_table_time_config(
        src_schema.name,
        dst_schema,
        output_file=output_file,
        check_mapped_timestamps=True,
        wrap_time_allowed=True,
        data_adjustment=TimeBasedDataAdjustment(
            daylight_saving_adjustment=DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_DUPLICATE_FALLBACK
        ),
    )
    if output_file is None or store._backend.name == "sqlite":
        result = store.read_table(dst_schema.name)
    else:
        result = pd.read_parquet(output_file)

    # Format data for display
    result = result.sort_values(by=["generator", "timestamp"]).reset_index(drop=True)[
        ["generator", "timestamp", "value"]
    ]
    result["timestamp"] = result["timestamp"].dt.tz_convert(dst_schema.time_config.start.tzinfo)

    # Check skips, dups, and time-wrapped values
    for i in range(len(time_zones)):
        df = result.loc[result["generator"] == f"gen{i}"].reset_index(drop=True)
        val_skipped, val_dupped = 1658 + i, 7369 + i
        expected_values = np.roll(
            list(
                chain(
                    range(i, val_skipped),
                    range(val_skipped + 1, val_dupped + 1),
                    range(val_dupped, i + time_array_len),
                )
            ),
            i,
        )
        assert np.array_equal(df["value"].values, expected_values)


def test_to_parquet(tmp_path, generators_schema):
    src_file, src_schema, dst_schema = generators_schema
    store = Store()
    store.ingest_from_csv(src_file, src_schema, dst_schema)
    filename = tmp_path / "data.parquet"
    table = store._backend.table(dst_schema.name)
    query = table.filter(table["generator"] == "gen2")
    store.write_query_to_parquet(query, filename, overwrite=True)
    assert filename.exists()
    df = pd.read_parquet(filename)
    assert len(df) == 8784


def test_load_existing_store(iter_backends_file, one_week_per_month_by_hour_table):
    backend = iter_backends_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(backend=backend)
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert df2.equals(df)
    file_path = Path(backend._file_path)
    assert file_path.exists()
    store2 = Store.load_from_file(backend_name=backend.name, file_path=file_path)
    df3 = store2.read_table(schema.name)
    assert df3.equals(df2)
    with pytest.raises(FileNotFoundError):
        Store.load_from_file(backend_name=backend.name, file_path="./invalid/path")


def test_create_methods(iter_backend_names, tmp_path):
    path = tmp_path / "data.db"
    assert not path.exists()
    Store.create_file_db(backend_name=iter_backend_names, file_path=path)
    gc.collect()
    assert path.exists()
    with pytest.raises(InvalidOperation):
        Store.create_file_db(backend_name=iter_backend_names, file_path=path)
    Store.create_file_db(backend_name=iter_backend_names, file_path=path, overwrite=True)
    Store.create_in_memory_db(backend_name=iter_backend_names)


def test_invalid_backend():
    with pytest.raises(NotImplementedError):
        Store(backend_name="invalid_backend")


def test_create_with_existing_backend():
    backend = make_backend("duckdb")
    store = Store(backend=backend)
    assert store._backend is backend


def test_create_with_sqlite():
    Store(backend_name="sqlite")


def test_create_with_conflicting_parameters():
    with pytest.raises(ConflictingInputsError):
        Store(backend=make_backend("duckdb"), backend_name="duckdb")


def test_backup(iter_backends_file: IbisBackend, one_week_per_month_by_hour_table, tmp_path):
    backend = iter_backends_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(backend=backend)
    store.ingest_table(df, schema)
    dst_file = tmp_path / "backup.db"
    assert not dst_file.exists()
    store.backup(dst_file)
    assert dst_file.exists()
    store2 = Store(backend_name=backend.name, file_path=dst_file)
    df2 = store2.read_table(schema.name)
    assert df2.equals(df)

    with pytest.raises(InvalidOperation):
        store.backup(dst_file)
    dst_file2 = tmp_path / "backup2.db"
    dst_file2.touch()
    store.backup(dst_file2, overwrite=True)

    # Make sure the original still works.
    df3 = store.read_table(schema.name)
    assert df3.equals(df)


def test_backup_not_allowed(one_week_per_month_by_hour_table, tmp_path):
    backend = make_backend("duckdb")
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(backend=backend)
    store.ingest_table(df, schema)
    dst_file = tmp_path / "backup.db"
    assert not dst_file.exists()
    with pytest.raises(InvalidOperation):
        store.backup(dst_file)
    assert not dst_file.exists()


def test_delete_rows(iter_stores_by_backend: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name)
    assert df2.equals(df)
    assert sorted(df2["id"].unique()) == [1, 2, 3]
    with pytest.raises(InvalidParameter):
        store.delete_rows(schema.name, {})
    store.delete_rows(schema.name, {"id": 2})
    df3 = store.read_table(schema.name)
    assert sorted(df3["id"].unique()) == [1, 3]
    store.delete_rows(schema.name, {"id": 1})
    df4 = store.read_table(schema.name)
    assert sorted(df4["id"].unique()) == [3]
    store.delete_rows(schema.name, {"id": 3})
    with pytest.raises(TableNotStored):
        store.read_table(schema.name)
    with pytest.raises(TableNotStored):
        store.delete_rows(schema.name, {"id": 3})


def test_drop_table(iter_stores_by_backend: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    assert not store.list_tables()
    store.ingest_table(df, schema)
    assert store.read_table(schema.name).equals(df)
    assert store.list_tables() == [schema.name]
    store.drop_table(schema.name)
    with pytest.raises(TableNotStored):
        store.read_table(schema.name)
    assert not store.list_tables()
    with pytest.raises(TableNotStored):
        store.drop_table(schema.name)


def test_drop_view(iter_stores_by_backend: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    table = store._backend.table(schema.name)
    query = table.filter(table["id"] == 1)
    inputs = schema.model_dump()
    inputs["name"] = make_temp_view_name()
    schema2 = TableSchema(**inputs)
    store.create_view(schema2, query)
    assert schema2.name in store.list_tables()
    store.drop_view(schema2.name)
    assert schema2.name not in store.list_tables()


def test_read_raw_query(iter_stores_by_backend: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)

    query = f"SELECT * FROM {schema.name}"
    df2 = store.read_raw_query(query)
    assert df2.equals(df)


def test_check_timestamps(iter_stores_by_backend: Store, one_week_per_month_by_hour_table) -> None:
    store = iter_stores_by_backend
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    store.check_timestamps(schema.name)


@pytest.mark.parametrize("to_time_zone", [ZoneInfo("US/Eastern"), ZoneInfo("US/Mountain"), None])
def test_convert_time_zone(
    tmp_path, iter_stores_by_backend_no_data_ingestion: Store, to_time_zone: tzinfo | None
):
    store = iter_stores_by_backend_no_data_ingestion
    time_array_len = 8784
    year = 2020
    tzinfo = ZoneInfo("EST")

    src_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=0, tzinfo=tzinfo),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_csv_schema = CsvTableSchema(
        time_config=src_time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        rel2.to_df().to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(rel2, src_schema)

    if tzinfo is None and store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.convert_time_zone(
        src_schema.name, to_time_zone, output_file=output_file, check_mapped_timestamps=True
    )
    if output_file is None or store._backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name)
    else:
        df2 = pd.read_parquet(output_file)
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    assert len(df2) == time_array_len * 3
    actual = sorted(df2["timestamp"].unique())
    assert isinstance(dst_schema.time_config, DatetimeRangeWithTZColumn)
    if to_time_zone:
        expected_start = src_time_config.start.astimezone(to_time_zone).replace(tzinfo=None)
    else:
        expected_start = src_time_config.start.replace(tzinfo=None)
    assert dst_schema.time_config.start == expected_start
    assert pd.Timestamp(actual[0]) == dst_schema.time_config.start
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    expected = sorted(set(expected))
    check_timestamp_lists(actual, expected)


@pytest.mark.parametrize("wrapped_time_allowed", [False, True])
def test_convert_time_zone_by_column(
    tmp_path, iter_stores_by_backend_no_data_ingestion: Store, wrapped_time_allowed: bool
):
    store = iter_stores_by_backend_no_data_ingestion
    time_array_len = 8784
    year = 2020
    tzinfo = ZoneInfo("EST")

    src_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=0, tzinfo=tzinfo),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_csv_schema = CsvTableSchema(
        time_config=src_time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841
    # add time_zone column
    stmt = ", ".join(rel2.columns)
    tz_col_stmt = "CASE WHEN generator='gen1' THEN 'US/Eastern' WHEN generator='gen2' THEN 'US/Central' ELSE 'None' END AS time_zone"
    stmt += f", {tz_col_stmt}"
    rel2 = rel2.project(stmt)

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator", "time_zone"],
        value_column="value",
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        rel2.to_df().to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(rel2, src_schema)

    if tzinfo is None and store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.convert_time_zone_by_column(
        src_schema.name,
        "time_zone",
        output_file=output_file,
        wrap_time_allowed=wrapped_time_allowed,
        check_mapped_timestamps=True,
    )
    if output_file is None or store._backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name)
    else:
        df2 = pd.read_parquet(output_file)
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df_stats = df2.groupby(["time_zone"])["timestamp"].agg(["min", "max", "count"])
    assert set(df_stats["count"]) == {time_array_len}
    if wrapped_time_allowed:
        assert set(df_stats["min"]) == {dst_schema.time_config.start.replace(tzinfo=None)}
    else:
        assert (df_stats.loc["US/Eastern"] == df_stats.loc["None"]).prod() == 1
        assert df_stats.loc["US/Central", "min"] == dst_schema.time_config.start.astimezone(
            ZoneInfo("US/Central")
        ).replace(tzinfo=None)
    assert isinstance(dst_schema.time_config, DatetimeRangeWithTZColumn)
    expected_dct = make_time_range_generator(dst_schema.time_config).list_timestamps_by_time_zone()
    for tz, expected in expected_dct.items():
        actual = sorted(df2.loc[df2["time_zone"] == tz, "timestamp"])
        check_timestamp_lists(actual, expected)


@pytest.mark.parametrize("to_time_zone", [ZoneInfo("UTC"), ZoneInfo("EST"), None])
def test_localize_time_zone(
    tmp_path, iter_stores_by_backend_no_data_ingestion: Store, to_time_zone: tzinfo | None
):
    """Test time zone localization of tz-naive timestamps to a specified time zone."""
    store = iter_stores_by_backend_no_data_ingestion
    time_array_len = 8784
    year = 2020

    # Create tz-naive source time config
    src_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=0, tzinfo=None),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_csv_schema = CsvTableSchema(
        time_config=src_time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        rel2.to_df().to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(rel2, src_schema)

    if to_time_zone is None and store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.localize_time_zone(
        src_schema.name,
        to_time_zone,
        output_file=output_file,
        check_mapped_timestamps=True,
    )
    if output_file is None or store._backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name)
    else:
        df2 = pd.read_parquet(output_file)
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    assert len(df2) == time_array_len * 3
    actual = sorted(df2["timestamp"].unique())
    assert isinstance(dst_schema.time_config, DatetimeRange)
    if to_time_zone:
        # Should be localized to the target time zone
        assert dst_schema.time_config.start_time_is_tz_naive() is False
    else:
        # Should remain tz-naive
        assert dst_schema.time_config.start_time_is_tz_naive() is True
    assert pd.Timestamp(actual[0]) == dst_schema.time_config.start
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    expected = sorted(set(expected))
    check_timestamp_lists(actual, expected)


def test_localize_time_zone_by_column(tmp_path, iter_stores_by_backend_no_data_ingestion: Store):
    """Test time zone localization based on a time zone column."""
    store = iter_stores_by_backend_no_data_ingestion
    time_array_len = 8784
    year = 2020

    # Create tz-naive source time config
    src_time_config = DatetimeRange(
        start=datetime(year=year, month=1, day=1, hour=0, tzinfo=None),
        resolution=timedelta(hours=1),
        length=time_array_len,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_csv_schema = CsvTableSchema(
        time_config=src_time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp()),
            ColumnDType(name="gen1", dtype=dt.Float64()),
            ColumnDType(name="gen2", dtype=dt.Float64()),
            ColumnDType(name="gen3", dtype=dt.Float64()),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    rel = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema)
    rel2 = unpivot(rel, ("gen1", "gen2", "gen3"), "generator", "value")  # noqa: F841
    # add time_zone column with standard time zones (not DST)
    stmt = ", ".join(rel2.columns)
    tz_col_stmt = "CASE WHEN generator='gen1' THEN 'Etc/GMT+5' WHEN generator='gen2' THEN 'Etc/GMT+6' ELSE 'Etc/GMT+7' END AS time_zone"
    stmt += f", {tz_col_stmt}"
    rel2 = rel2.project(stmt)

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator", "time_zone"],
        value_column="value",
    )
    if store._backend.name == "spark":
        out_file = tmp_path / "data.parquet"
        rel2.to_df().to_parquet(out_file)
        store.create_view_from_parquet(out_file, src_schema)
    else:
        store.ingest_table(rel2, src_schema)

    if store._backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.localize_time_zone_by_column(
        src_schema.name,
        "time_zone",
        output_file=output_file,
        check_mapped_timestamps=True,
    )
    if output_file is None or store._backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name)
    else:
        df2 = pd.read_parquet(output_file)
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df_stats = df2.groupby(["time_zone"])["timestamp"].agg(["min", "max", "count"])
    assert set(df_stats["count"]) == {time_array_len}
    assert isinstance(dst_schema.time_config, DatetimeRangeWithTZColumn)
    # Verify that each time zone has tz-aware localized timestamps
    assert df2["timestamp"].dt.tz is not None
    expected_dct = make_time_range_generator(dst_schema.time_config).list_timestamps_by_time_zone()
    for tz, expected in expected_dct.items():
        actual = sorted(df2.loc[df2["time_zone"] == tz, "timestamp"])
        check_timestamp_lists(actual, expected)
