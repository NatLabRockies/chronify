import fileinput
import gc
import shutil
from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo
from itertools import chain

import duckdb
import numpy as np
import pandas as pd
import pytest

from chronify import time_series_mapper_base
from chronify.csv_io import read_csv
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidOperation,
    InvalidParameter,
    InvalidTable,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.ibis import make_backend
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
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("Etc/GMT+5")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
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
def test_ingest_csv(iter_stores_by_engine: Store, tmp_path, generators_schema, use_time_zone):
    store = iter_stores_by_engine
    src_file, src_schema, dst_schema = generators_schema
    import ibis.expr.datatypes as dt

    if use_time_zone:
        new_src_file = tmp_path / "gen_tz.csv"
        duckdb.sql(
            f"""
            SELECT timezone('Etc/GMT+5', timestamp) as timestamp, gen1, gen2, gen3
            FROM read_csv('{src_file}')
        """
        ).to_df().to_csv(new_src_file, index=False)
        src_file = new_src_file
        src_schema.column_dtypes[0] = ColumnDType(
            name="timestamp", dtype=dt.Timestamp(timezone="Etc/GMT+5")
        )
    store.ingest_from_csv(src_file, src_schema, dst_schema)
    df = store.read_table(dst_schema.name).execute()
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
    ts_dtype = dt.Timestamp(timezone="Etc/GMT+5") if use_time_zone else "datetime"
    src_schema2 = CsvTableSchema(
        time_config=src_schema.time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=ts_dtype),
            ColumnDType(name="g1b", dtype="float"),
            ColumnDType(name="g2b", dtype="float"),
            ColumnDType(name="g3b", dtype="float"),
        ],
        value_columns=["g1b", "g2b", "g3b"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    store.ingest_from_csv(new_file, src_schema2, dst_schema)
    df = store.read_table(dst_schema.name).execute()
    assert len(df) == 8784 * 3 * 2
    all(df.timestamp.unique() == expected_timestamps)

    # Read a subset of the table.
    df2 = store.read_query(f"SELECT * FROM {dst_schema.name} WHERE generator = 'gen2'").execute()
    assert len(df2) == 8784
    df_gen2 = df[df["generator"] == "gen2"]
    assert all((df2.values == df_gen2.values)[0])

    # Adding the same rows should fail.
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(new_file, src_schema2, dst_schema)


def test_ingest_csvs_with_rollback(tmp_path, multiple_tables):
    # The new ibis-based backend uses pseudo-transactions that track created objects.
    # Real SQL rollbacks are not supported.
    store = Store(backend_name="duckdb")
    tables, dst_schema = multiple_tables
    src_file1 = tmp_path / "file1.csv"
    src_file2 = tmp_path / "file2.csv"
    tables[0].to_csv(src_file1)
    tables[1].to_csv(src_file2)
    src_schema = CsvTableSchema(
        time_config=dst_schema.time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="id", dtype="int"),
            ColumnDType(name="value", dtype="float"),
        ],
        value_columns=[dst_schema.value_column],
        time_array_id_columns=dst_schema.time_array_id_columns,
    )

    store.ingest_from_csvs((src_file1, src_file2), src_schema, dst_schema)
    df = store.read_table(dst_schema.name).execute()
    assert len(df) == len(tables[0]) + len(tables[1])
    assert len(df.id.unique()) == 2


def test_ingest_multiple_tables(iter_stores_by_engine: Store, multiple_tables):
    store = iter_stores_by_engine
    tables, schema = multiple_tables
    store.ingest_tables(tables, schema)
    df = store.read_query("SELECT * FROM devices WHERE id = 2").execute()
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    assert df.equals(tables[1])


def test_ingest_multiple_tables_error(iter_stores_by_engine: Store, multiple_tables):
    store = iter_stores_by_engine
    tables, schema = multiple_tables
    orig_value = tables[1].loc[8783]["id"]
    tables[1].loc[8783] = (tables[1].loc[8783]["timestamp"], 0.1, 99)
    with pytest.raises(InvalidTable):
        store.ingest_tables(tables, schema)
    assert not store.has_table(schema.name)

    tables[1].loc[8783] = (tables[1].loc[8783]["timestamp"], 0.1, orig_value)
    store.ingest_tables(tables, schema)
    df = store.read_query(f"select * from {schema.name} where id=2").execute()
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    assert df.equals(tables[1])


def test_ingest_cleanup_failure_does_not_mask_original_error(monkeypatch, multiple_tables):
    """A cleanup-side failure during post-ingest rollback must not replace
    the user-visible ingest error. Regression: a Spark connectivity blip
    inside ``drop_table`` would otherwise hide the InvalidTable that
    originally caused the rollback.
    """
    store = Store.create_in_memory_db()
    tables, schema = multiple_tables
    tables[1].loc[8783] = (tables[1].loc[8783]["timestamp"], 0.1, 99)

    real_drop = store._backend.drop_table

    def _exploding_drop(name):
        if name == schema.name:
            msg = "simulated cleanup failure"
            raise RuntimeError(msg)
        return real_drop(name)

    monkeypatch.setattr(store._backend, "drop_table", _exploding_drop)

    # The user must see the InvalidTable from validation, not the RuntimeError
    # from cleanup.
    with pytest.raises(InvalidTable):
        store.ingest_tables(tables, schema)


@pytest.mark.parametrize("use_pandas", [False, True])
def test_ingest_pivoted_table(iter_stores_by_engine: Store, generators_schema, use_pandas: bool):
    import ibis

    store = iter_stores_by_engine
    src_file, src_schema, dst_schema = generators_schema
    pivoted_schema = PivotedTableSchema(**src_schema.model_dump(exclude={"column_dtypes"}))
    df = read_csv(src_file, src_schema).to_df()
    input_table = df if use_pandas else ibis.memtable(df)
    store.ingest_pivoted_table(input_table, pivoted_schema, dst_schema)
    table = store.get_table(dst_schema.name)
    stmt = table.filter(table.generator == "gen1")
    df = store.read_query(stmt).execute()
    assert len(df) == 8784


def test_ingest_invalid_csv(iter_stores_by_engine: Store, tmp_path, generators_schema):
    store = iter_stores_by_engine
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
        store.read_table(dst_schema.name).execute()


def test_invalid_schema(iter_stores_by_engine: Store, generators_schema):
    store = iter_stores_by_engine
    src_file, src_schema, dst_schema = generators_schema
    src_schema.value_columns = ["g1", "g2", "g3"]
    with pytest.raises(InvalidTable):
        store.ingest_from_csv(src_file, src_schema, dst_schema)


def test_ingest_one_week_per_month_by_hour(
    iter_stores_by_engine: Store, one_week_per_month_by_hour_table
):
    store = iter_stores_by_engine
    df, num_time_arrays, schema = one_week_per_month_by_hour_table

    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name).execute()
    assert len(df2["id"].unique()) == num_time_arrays
    assert len(df2) == 24 * 7 * 12 * num_time_arrays
    columns = schema.time_config.list_time_columns()
    columns.insert(0, "value")
    assert all(df.sort_values(columns)["value"] == df2.sort_values(columns)["value"])


def test_ingest_one_week_per_month_by_hour_invalid(
    iter_stores_by_engine: Store, one_week_per_month_by_hour_table
):
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    df_filtered = df[df["hour"] != 5]
    assert len(df_filtered) < len(df)

    with pytest.raises(InvalidTable):
        store.ingest_table(df_filtered, schema)


def test_load_parquet(iter_stores_by_engine_no_data_ingestion: Store, tmp_path):
    store = iter_stores_by_engine_no_data_ingestion
    if store.backend.name == "sqlite":
        # SQLite doesn't support parquet
        return

    time_config = DatetimeRange(
        start=datetime(year=2020, month=1, day=1, tzinfo=ZoneInfo("Etc/GMT+5")),
        resolution=timedelta(hours=1),
        length=8784,
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
        time_column="timestamp",
    )

    src_schema = CsvTableSchema(
        time_config=time_config,
        column_dtypes=[
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
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
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )
    out_file = tmp_path / "gen2.parquet"
    df2.to_parquet(str(out_file))
    store.create_view_from_parquet(out_file, dst_schema)
    df = store.read_table(dst_schema.name).execute()
    assert len(df) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df.timestamp.unique() == expected_timestamps)

    # This adds test coverage for views.
    as_dict = dst_schema.model_dump()
    as_dict["name"] = "test_view"
    schema2 = TableSchema(**as_dict)
    store.create_view_from_parquet(out_file, schema2)
    df2 = store.read_table(schema2.name).execute()
    assert schema2.name in store.list_tables()
    assert len(df2) == 8784 * 3
    timestamp_generator = make_time_range_generator(time_config)
    expected_timestamps = timestamp_generator.list_timestamps()
    all(df2.timestamp.unique() == expected_timestamps)
    store.drop_view(schema2.name)
    assert schema2.name not in store.list_tables()
    assert dst_schema.name in store.list_tables()
    df3 = store.read_table(dst_schema.name).execute()
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
    iter_stores_by_engine_no_data_ingestion: Store,
    one_week_per_month_by_hour_table,
    one_week_per_month_by_hour_table_tz,
    params: tuple[bool, int],
):
    store = iter_stores_by_engine_no_data_ingestion
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
    store.ingest_table(df, src_schema)
    store.map_table_time_config(src_schema.name, dst_schema, check_mapped_timestamps=True)
    df2 = store.read_table(dst_schema.name).execute()
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


@pytest.mark.parametrize("tzinfo", [ZoneInfo("Etc/GMT+5"), None])
def test_map_datetime_to_datetime(
    tmp_path, iter_stores_by_engine_no_data_ingestion: Store, tzinfo
):
    store = iter_stores_by_engine_no_data_ingestion
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
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
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
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    store.ingest_table(df2, src_schema)

    if tzinfo is None and store.backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None
    store.map_table_time_config(
        src_schema.name, dst_schema, output_file=output_file, check_mapped_timestamps=True
    )
    if output_file is None or store.backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name).execute()
    else:
        df2 = pd.read_parquet(output_file)
    assert len(df2) == time_array_len * 3
    actual = sorted(df2["timestamp"].unique())
    assert isinstance(src_schema.time_config, DatetimeRange)
    assert actual[0] == src_schema.time_config.start + timedelta(hours=1)
    expected = make_time_range_generator(dst_schema.time_config).list_timestamps()
    check_timestamp_lists(actual, expected)


def test_map_index_time_to_datetime(
    tmp_path: Path, iter_stores_by_engine_no_data_ingestion: Store
) -> None:
    store = iter_stores_by_engine_no_data_ingestion
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
            start=datetime(year=year, month=1, day=1, hour=1, tzinfo=ZoneInfo("Etc/GMT+5")),
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
    store.ingest_table(src_df, src_schema)

    if store.backend.name != "sqlite":
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
    if output_file is None or store.backend.name == "sqlite":
        result = store.read_table(dst_schema.name).execute()
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
    table = store.get_table(dst_schema.name)
    stmt = table.filter(table.generator == "gen2")
    store.write_query_to_parquet(stmt, filename, overwrite=True)
    assert filename.exists()
    df = pd.read_parquet(filename)
    assert len(df) == 8784


def test_load_existing_store(iter_backends_file, one_week_per_month_by_hour_table):
    backend, backend_name = iter_backends_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(backend=backend)
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name).execute()
    assert df2.equals(df)
    file_path = Path(backend.database)
    assert file_path.exists()
    store2 = Store.load_from_file(backend_name=backend_name, file_path=file_path)
    df3 = store2.read_table(schema.name).execute()
    assert df3.equals(df2)
    with pytest.raises(FileNotFoundError):
        Store.load_from_file(backend_name=backend_name, file_path="./invalid/path")


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
    with pytest.raises(InvalidParameter):
        Store(backend_name="hive")


def test_create_with_existing_backend():
    backend = make_backend("duckdb")
    store = Store(backend=backend)
    assert store.backend is backend


def test_create_with_sqlite():
    Store(backend_name="sqlite")


def test_create_with_conflicting_parameters():
    with pytest.raises(ConflictingInputsError):
        Store(backend=make_backend("duckdb"), backend_name="duckdb")


def test_backup(iter_backends_file, one_week_per_month_by_hour_table, tmp_path):
    backend, backend_name = iter_backends_file
    df, _, schema = one_week_per_month_by_hour_table
    store = Store(backend=backend)
    store.ingest_table(df, schema)
    dst_file = tmp_path / "backup.db"
    assert not dst_file.exists()
    store.backup(dst_file)
    assert dst_file.exists()
    store2 = Store(backend_name=backend_name, file_path=dst_file)
    df2 = store2.read_table(schema.name).execute()
    assert df2.equals(df)

    with pytest.raises(InvalidOperation):
        store.backup(dst_file)
    dst_file2 = tmp_path / "backup2.db"
    dst_file2.touch()
    store.backup(dst_file2, overwrite=True)

    # Make sure the original still works.
    df3 = store.read_table(schema.name).execute()
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


def test_delete_rows(iter_stores_by_engine: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    df2 = store.read_table(schema.name).execute()
    assert df2.equals(df)
    assert sorted(df2["id"].unique()) == [1, 2, 3]
    with pytest.raises(InvalidParameter):
        store.delete_rows(schema.name, {})
    store.delete_rows(schema.name, {"id": 2})
    df3 = store.read_table(schema.name).execute()
    assert sorted(df3["id"].unique()) == [1, 3]
    store.delete_rows(schema.name, {"id": 1})
    df4 = store.read_table(schema.name).execute()
    assert sorted(df4["id"].unique()) == [3]
    store.delete_rows(schema.name, {"id": 3})
    with pytest.raises(TableNotStored):
        store.read_table(schema.name).execute()
    with pytest.raises(TableNotStored):
        store.delete_rows(schema.name, {"id": 3})


def test_drop_table(iter_stores_by_engine: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    assert not store.list_tables()
    store.ingest_table(df, schema)
    assert store.read_table(schema.name).execute().equals(df)
    assert store.list_tables() == [schema.name]
    store.drop_table(schema.name)
    with pytest.raises(TableNotStored):
        store.read_table(schema.name).execute()
    assert not store.list_tables()
    with pytest.raises(TableNotStored):
        store.drop_table(schema.name)


def test_drop_view(iter_stores_by_engine: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    table = store.get_table(schema.name)
    stmt = table.filter(table.id == 1)
    inputs = schema.model_dump()
    inputs["name"] = make_temp_view_name()
    schema2 = TableSchema(**inputs)
    store.create_view(schema2, stmt)
    assert schema2.name in store.list_tables()
    store.drop_view(schema2.name)
    assert schema2.name not in store.list_tables()


def test_read_raw_query(iter_stores_by_engine: Store, one_week_per_month_by_hour_table):
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)

    query = f"SELECT * FROM {schema.name}"
    df2 = store.read_raw_query(query)
    assert df2.equals(df)

    query = f"SELECT * FROM {schema.name} where id = 2"
    df2 = store.read_raw_query(query)
    assert df2.equals(df[df["id"] == 2].reset_index(drop=True))


def test_check_timestamps(iter_stores_by_engine: Store, one_week_per_month_by_hour_table) -> None:
    store = iter_stores_by_engine
    df, _, schema = one_week_per_month_by_hour_table
    store.ingest_table(df, schema)
    store.check_timestamps(schema.name)


@pytest.mark.parametrize("to_time_zone", [ZoneInfo("US/Eastern"), ZoneInfo("US/Mountain"), None])
def test_convert_time_zone(
    tmp_path, iter_stores_by_engine_no_data_ingestion: Store, to_time_zone: tzinfo | None
):
    store = iter_stores_by_engine_no_data_ingestion
    time_array_len = 8784
    year = 2020
    tzinfo = ZoneInfo("Etc/GMT+5")

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
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    store.ingest_table(df2, src_schema)

    if tzinfo is None and store.backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.convert_time_zone(
        src_schema.name, to_time_zone, output_file=output_file, check_mapped_timestamps=True
    )
    if output_file is None or store.backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name).execute()
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
    tmp_path, iter_stores_by_engine_no_data_ingestion: Store, wrapped_time_allowed: bool
):
    store = iter_stores_by_engine_no_data_ingestion
    time_array_len = 8784
    year = 2020
    tzinfo = ZoneInfo("Etc/GMT+5")

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
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )
    df2["time_zone"] = (
        df2["generator"].map({"gen1": "US/Eastern", "gen2": "US/Central"}).fillna("None")
    )

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator", "time_zone"],
        value_column="value",
    )
    store.ingest_table(df2, src_schema)

    if tzinfo is None and store.backend.name != "sqlite":
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
    if output_file is None or store.backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name).execute()
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


@pytest.mark.parametrize("to_time_zone", [ZoneInfo("UTC"), ZoneInfo("Etc/GMT+5"), None])
def test_localize_time_zone(
    tmp_path, iter_stores_by_engine_no_data_ingestion: Store, to_time_zone: tzinfo | None
):
    """Test time zone localization of tz-naive timestamps to a specified time zone."""
    store = iter_stores_by_engine_no_data_ingestion
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
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator"],
        value_column="value",
    )
    store.ingest_table(df2, src_schema)

    if to_time_zone is None and store.backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.localize_time_zone(
        src_schema.name,
        to_time_zone,
        output_file=output_file,
        check_mapped_timestamps=True,
    )
    if output_file is None or store.backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name).execute()
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


def test_localize_time_zone_by_column(tmp_path, iter_stores_by_engine_no_data_ingestion: Store):
    """Test time zone localization based on a time zone column."""
    store = iter_stores_by_engine_no_data_ingestion
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
            ColumnDType(name="timestamp", dtype="datetime"),
            ColumnDType(name="gen1", dtype="float"),
            ColumnDType(name="gen2", dtype="float"),
            ColumnDType(name="gen3", dtype="float"),
        ],
        value_columns=["gen1", "gen2", "gen3"],
        pivoted_dimension_name="generator",
        time_array_id_columns=[],
    )
    df = read_csv(GENERATOR_TIME_SERIES_FILE, src_csv_schema).to_df()
    df2 = df.melt(
        id_vars=["timestamp"],
        value_vars=["gen1", "gen2", "gen3"],
        var_name="generator",
        value_name="value",
    )
    df2["time_zone"] = (
        df2["generator"].map({"gen1": "Etc/GMT+5", "gen2": "Etc/GMT+6"}).fillna("Etc/GMT+7")
    )

    src_schema = TableSchema(
        name="generators_pb",
        time_config=src_time_config,
        time_array_id_columns=["generator", "time_zone"],
        value_column="value",
    )
    store.ingest_table(df2, src_schema)

    if store.backend.name != "sqlite":
        output_file = tmp_path / "mapped_data"
    else:
        output_file = None

    dst_schema = store.localize_time_zone_by_column(
        src_schema.name,
        "time_zone",
        output_file=output_file,
        check_mapped_timestamps=True,
    )
    if output_file is None or store.backend.name == "sqlite":
        df2 = store.read_table(dst_schema.name).execute()
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


def test_map_table_preserves_existing_parquet_on_failure(tmp_path, monkeypatch):
    """A failing remap to an existing parquet must not destroy the original."""
    store = Store.create_in_memory_db()
    year = 2020
    length = 24
    src_schema = TableSchema(
        name="src_preserve",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(year, 1, 1), periods=length, freq="h"),
            "id": 1,
            "value": list(range(length)),
        }
    )
    store.ingest_table(df, src_schema)

    output_file = tmp_path / "out.parquet"
    sentinel = b"PRE-EXISTING-CONTENT"
    output_file.write_bytes(sentinel)

    dst_schema = TableSchema(
        name="dst_preserve",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_ENDING,
            time_column="timestamp",
        ),
    )

    def _fail(*args, **kwargs):
        msg = "forced failure"
        raise InvalidTable(msg)

    monkeypatch.setattr(time_series_mapper_base, "check_timestamps", _fail)

    with pytest.raises(InvalidTable, match="forced"):
        store.map_table_time_config(
            src_schema.name,
            dst_schema,
            output_file=output_file,
            check_mapped_timestamps=True,
        )

    assert output_file.read_bytes() == sentinel
    # No staging files left behind in tmp_path either.
    leftover = [p for p in tmp_path.iterdir() if p.name != output_file.name]
    assert leftover == []


def test_map_table_preserves_existing_parquet_on_promotion_failure(tmp_path, monkeypatch):
    """If the staging→target rename fails after the transaction commits, the
    pre-existing target must survive. Regression for an earlier
    delete-then-rename window where a crash between the two operations
    would leave the user with neither the old nor the new output.
    """
    store = Store.create_in_memory_db()
    year = 2020
    length = 24
    src_schema = TableSchema(
        name="src_promo_fail",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(year, 1, 1), periods=length, freq="h"),
            "id": 1,
            "value": list(range(length)),
        }
    )
    store.ingest_table(df, src_schema)

    output_file = tmp_path / "out.parquet"
    sentinel = b"PRE-EXISTING-DO-NOT-DESTROY"
    output_file.write_bytes(sentinel)

    dst_schema = TableSchema(
        name="dst_promo_fail",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_ENDING,
            time_column="timestamp",
        ),
    )

    # The promotion sequence is: target→backup, then staging→target, then
    # delete backup. Inject a failure on the second Path.replace so we
    # exercise the restore path.
    real_replace = Path.replace
    n = [0]

    def _flaky_replace(self, target):
        n[0] += 1
        if n[0] == 2:
            msg = "simulated promotion failure"
            raise OSError(msg)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _flaky_replace)

    with pytest.raises(OSError, match="simulated"):
        store.map_table_time_config(
            src_schema.name,
            dst_schema,
            output_file=output_file,
        )

    # Original content must survive — restored from the backup after the
    # second rename failed.
    assert output_file.read_bytes() == sentinel
    # And no debris left in the directory.
    leftover = sorted(p.name for p in tmp_path.iterdir() if p.name != output_file.name)
    assert leftover == []


def test_map_table_succeeds_when_backup_cleanup_fails(tmp_path, monkeypatch):
    """A failure to remove the post-promotion backup is cosmetic debris; the
    mapping must still succeed and the schema must still be registered.

    Reviewer regression: previously a backup-cleanup OSError caused
    ``apply_mapping`` to re-raise, so ``map_table_time_config`` skipped its
    ``add_schema`` call and the user was left with a new parquet on disk
    that the store didn't know about.
    """
    store = Store.create_in_memory_db()
    year = 2020
    length = 24
    src_schema = TableSchema(
        name="src_backup_fail",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(year, 1, 1), periods=length, freq="h"),
            "id": 1,
            "value": list(range(length)),
        }
    )
    store.ingest_table(df, src_schema)

    output_file = tmp_path / "out.parquet"
    output_file.write_bytes(b"ORIGINAL")

    dst_schema = TableSchema(
        name="dst_backup_fail",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_ENDING,
            time_column="timestamp",
        ),
    )

    real_delete = time_series_mapper_base.delete_if_exists

    def _fail_on_backup(path):
        if ".backup." in path.name:
            msg = "simulated backup cleanup failure"
            raise OSError(msg)
        return real_delete(path)

    monkeypatch.setattr(time_series_mapper_base, "delete_if_exists", _fail_on_backup)

    # Mapping must succeed despite the backup-cleanup OSError.
    store.map_table_time_config(src_schema.name, dst_schema, output_file=output_file)

    # New parquet content is in place.
    new = pd.read_parquet(output_file)
    assert len(new) == length
    # And the schema is registered, so the store and on-disk state agree.
    assert store._schema_mgr.get_schema(dst_schema.name).name == dst_schema.name


def test_map_table_cleanup_handles_directory_staging(tmp_path, monkeypatch):
    """Spark writes parquet as a directory, not a file. Cleanup of a failed
    map must use a directory-safe delete instead of unlink()."""
    from chronify.ibis.base import IbisBackend

    # Override write_parquet so the staging output is a directory, mirroring
    # how pyspark's Backend.to_parquet writes (a directory of part-* files).
    def _write_dir(self, expr, path, partition_by=None):  # noqa: ARG001
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "part-0.parquet").write_bytes(b"not a real parquet")

    monkeypatch.setattr(IbisBackend, "write_parquet", _write_dir)

    store = Store.create_in_memory_db()
    year = 2020
    length = 24
    src_schema = TableSchema(
        name="src_dir_cleanup",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
            time_column="timestamp",
        ),
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(year, 1, 1), periods=length, freq="h"),
            "id": 1,
            "value": list(range(length)),
        }
    )
    store.ingest_table(df, src_schema)

    dst_schema = TableSchema(
        name="dst_dir_cleanup",
        value_column="value",
        time_array_id_columns=["id"],
        time_config=DatetimeRange(
            start=datetime(year, 1, 1, 1),
            resolution=timedelta(hours=1),
            length=length,
            interval_type=TimeIntervalType.PERIOD_ENDING,
            time_column="timestamp",
        ),
    )

    output_file = tmp_path / "spark_style_out"

    # The fake parquet content fails downstream (in create_view_from_parquet
    # or check_timestamps), exercising the cleanup path on a directory
    # staging output. Without the directory-safe delete this raises
    # IsADirectoryError and masks the original error.
    with pytest.raises(Exception):  # noqa: B017
        store.map_table_time_config(
            src_schema.name,
            dst_schema,
            output_file=output_file,
            check_mapped_timestamps=True,
        )

    assert not output_file.exists()
    leftover = sorted(p.name for p in tmp_path.iterdir())
    assert leftover == []
