"""Regression tests for issues found in the SQLAlchemy-to-ibis migration review."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import ibis
import pandas as pd
import pytest

from chronify.exceptions import InvalidParameter, InvalidTable
from chronify.ibis import make_backend
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange
from chronify.time_series_mapper_base import _apply_mapping


def _make_tz_schema(name: str, length: int = 3) -> TableSchema:
    return TableSchema(
        name=name,
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=length,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )


def _make_tz_df(num_ids: int = 1, length: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=length, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "id": [i for i in range(1, num_ids + 1) for _ in range(length)],
            "timestamp": list(timestamps) * num_ids,
            "value": [float(x) for x in range(num_ids * length)],
        }
    )


def test_sqlite_timestamp_storage_format_is_consistent():
    """Creation-path and append-path rows must store timestamps in the same
    ISO-8601 format so SQL string
    comparisons (joins, deletes) work across all rows."""
    backend = make_backend("sqlite")
    store = Store(backend=backend)
    schema = _make_tz_schema("t1")
    store.ingest_table(_make_tz_df(num_ids=1), schema, skip_time_checks=True)
    # Append through the same schema.
    df2 = _make_tz_df(num_ids=1)
    df2["id"] = 2
    store.ingest_table(df2, schema, skip_time_checks=True)

    raw = [
        r[0]
        for r in backend.connection.con.execute("SELECT DISTINCT timestamp FROM t1").fetchall()
    ]
    assert raw == [
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T01:00:00+00:00",
        "2020-01-01T02:00:00+00:00",
    ]
    out = store.read_table("t1").execute()
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_sqlite_delete_rows_with_timestamp_predicate():
    """delete_rows must adapt datetime values to the stored string format."""
    backend = make_backend("sqlite")
    store = Store(backend=backend)
    schema = _make_tz_schema("t1")
    store.ingest_table(_make_tz_df(num_ids=1), schema, skip_time_checks=True)

    backend.delete_rows("t1", {"timestamp": pd.Timestamp("2020-01-01 01:00:00+00:00")})
    remaining = backend.connection.con.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
    assert remaining == 2

    # Plain datetime values must be adapted to the stored format too.
    backend.delete_rows("t1", {"timestamp": datetime(2020, 1, 1, 2, tzinfo=ZoneInfo("UTC"))})
    remaining = backend.connection.con.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
    assert remaining == 1


def test_create_view_from_parquet_failure_allows_retry(tmp_path: Path):
    """A failed create_view_from_parquet must not leave an orphaned schema
    registration that blocks a retry with corrected data."""
    store = Store(backend_name="duckdb")
    schema = _make_tz_schema("pq_table")

    bad_df = _make_tz_df(num_ids=1)
    bad_df = bad_df.iloc[:-1]  # one timestamp short of the configured length
    bad_file = tmp_path / "bad.parquet"
    bad_df.to_parquet(bad_file)
    with pytest.raises(InvalidTable):
        store.create_view_from_parquet(bad_file, schema)
    assert not store.has_table(schema.name)

    good_file = tmp_path / "good.parquet"
    _make_tz_df(num_ids=1).to_parquet(good_file)
    store.create_view_from_parquet(good_file, schema)
    assert store.has_table(schema.name)
    assert len(store.read_table(schema.name).execute()) == 3


def test_apply_mapping_casts_string_keys_numerically(iter_backends):
    """A string data column joined against a numeric mapping key must coerce
    numerically ('01' matches 1) instead of comparing unequal strings and
    silently dropping rows."""
    backend = iter_backends
    from_schema = _make_tz_schema("src_str_key")
    to_schema = _make_tz_schema("dst_str_key")
    df = _make_tz_df(num_ids=1)
    df["month"] = "01"
    backend.create_table(from_schema.name, df)
    backend.create_table("map_str_key", pd.DataFrame({"from_month": [1]}))

    _apply_mapping("map_str_key", from_schema, to_schema, backend)
    result = backend.execute(backend.table(to_schema.name))
    assert len(result) == 3


@pytest.mark.parametrize("case", ["string_source", "naive_source", "naive_mapping"])
def test_apply_mapping_joins_mismatched_timestamp_keys(iter_all_backends, case):
    """Timestamp join keys of different types must still match. On SQLite, casting
    one side reformats the stored text, which silently dropped every row."""
    backend = iter_all_backends
    df = _make_tz_df(num_ids=1)
    timestamps = df["timestamp"]
    df_mapping = pd.DataFrame({"from_timestamp": timestamps, "factor": 1.0})
    if case == "string_source":
        df["timestamp"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S")
    elif case == "naive_source":
        df["timestamp"] = timestamps.dt.tz_localize(None)
    else:
        df_mapping["from_timestamp"] = timestamps.dt.tz_localize(None)
    backend.create_table("src_ts_key", df)
    backend.create_table("map_ts_key", df_mapping)

    _apply_mapping(
        "map_ts_key", _make_tz_schema("src_ts_key"), _make_tz_schema("dst_ts_key"), backend
    )
    result = backend.execute(backend.table("dst_ts_key"))
    assert len(result) == 3


def test_read_raw_query_supports_non_select_statements(iter_stores_by_engine):
    """read_raw_query is a raw escape hatch: backend-specific statements that
    are not SELECT-shaped must work."""
    store = iter_stores_by_engine
    schema = _make_tz_schema("raw_table")
    store.ingest_table(_make_tz_df(num_ids=1), schema, skip_time_checks=True)

    if store.backend.name == "duckdb":
        df = store.read_raw_query("SHOW TABLES")
        assert "raw_table" in set(df["name"])
        df = store.read_raw_query("PRAGMA table_info('raw_table')")
    else:
        df = store.read_raw_query("PRAGMA table_info('raw_table')")
    assert "timestamp" in set(df["name"])


def test_sqlite_execute_sql_to_df_in_transaction_does_not_warn():
    """Inside a transaction, pandas must get the real sqlite3 connection, not the
    no-commit proxy, which it doesn't recognize."""
    import warnings

    backend = make_backend("sqlite")
    with backend.transaction(), warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backend.execute_sql_to_df("SELECT 1 AS x")["x"].tolist() == [1]
    backend.dispose()


def test_make_spark_backend_rejects_database():
    """Spark has no database file; a path must not be silently ignored."""
    with pytest.raises(InvalidParameter, match="does not use a database file"):
        make_backend("spark", database="my.db")


def test_read_raw_query_supports_params(iter_stores_by_engine):
    store = iter_stores_by_engine
    schema = _make_tz_schema("raw_params")
    store.ingest_table(_make_tz_df(num_ids=2), schema, skip_time_checks=True)

    df = store.read_raw_query("SELECT * FROM raw_params WHERE id = ?", params=(2,))
    assert len(df) == 3
    assert set(df["id"]) == {2}


def test_read_raw_query_sqlite_returns_raw_strings():
    """Unlike read_query, read_raw_query performs no timestamp conversion."""
    store = Store(backend_name="sqlite")
    schema = _make_tz_schema("raw_strings")
    store.ingest_table(_make_tz_df(num_ids=1), schema, skip_time_checks=True)

    df = store.read_raw_query("SELECT timestamp FROM raw_strings ORDER BY timestamp")
    assert df["timestamp"].iloc[0] == "2020-01-01T00:00:00+00:00"


def test_duckdb_execute_handles_memtable_expressions(create_duckdb_backend):
    """execute() must not bypass ibis's memtable registration."""
    backend = create_duckdb_backend
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = backend.execute(ibis.memtable(df))
    assert list(result["a"]) == [1, 2, 3]


def test_base_execute_sql_to_df_rejects_params_when_unsupported(create_duckdb_backend):
    """DuckDB supports params; the generic ibis fallback raises instead of
    silently ignoring them."""
    from chronify.ibis.base import IbisBackend

    backend = create_duckdb_backend
    with pytest.raises(InvalidParameter):
        IbisBackend.execute_sql_to_df(backend, "SELECT 1", params=(1,))


def test_sqlite_parquet_with_non_utc_timestamps_maps_all_rows(tmp_path: Path):
    """SQLite copies Parquet into a table; tz-aware values must be stored as UTC
    text, or joins against mapping tables silently match nothing."""
    store = Store(backend_name="sqlite")
    tz = ZoneInfo("Etc/GMT+5")
    start = datetime(2020, 1, 1, tzinfo=tz)
    df = pd.DataFrame(
        {
            "id": 1,
            "timestamp": pd.date_range(start, periods=3, freq="1h"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "tz.parquet"
    df.to_parquet(path)
    schema = TableSchema(
        name="src",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp", start=start, length=3, resolution=timedelta(hours=1)
        ),
        time_array_id_columns=["id"],
    )
    store.create_view_from_parquet(path, schema)
    dst = store.convert_time_zone("src", ZoneInfo("US/Mountain"))
    assert len(store.read_table(dst.name).execute()) == 3
