"""Tests for Store error paths and edge cases."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from chronify.exceptions import (
    InvalidParameter,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.ibis import make_backend
from chronify.models import TableSchema
from chronify.store import Store
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange


def _make_tz_schema(name: str = "generators") -> TableSchema:
    return TableSchema(
        name=name,
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
            length=3,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )


def _make_ntz_schema(name: str = "generators") -> TableSchema:
    return TableSchema(
        name=name,
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1),
            length=3,
            resolution=timedelta(hours=1),
            interval_type=TimeIntervalType.PERIOD_BEGINNING,
        ),
        time_array_id_columns=["id"],
    )


def _make_store() -> Store:
    return Store(backend=make_backend("duckdb"))


def _make_tz_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
                utc=True,
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )


class TestGetTable:
    def test_get_table_not_stored(self):
        store = _make_store()
        with pytest.raises(TableNotStored, match="nonexistent"):
            store.get_table("nonexistent")
        store.dispose()

    def test_try_get_table_returns_none(self):
        store = _make_store()
        result = store.try_get_table("nonexistent")
        assert result is None
        store.dispose()

    def test_try_get_table_returns_table(self):
        store = _make_store()
        schema = _make_tz_schema()
        store.ingest_table(_make_tz_df(), schema)
        result = store.try_get_table(schema.name)
        assert result is not None
        store.dispose()


class TestDropTableErrors:
    def test_drop_table_not_stored(self):
        store = _make_store()
        with pytest.raises(TableNotStored, match="nonexistent"):
            store.drop_table("nonexistent")
        store.dispose()

    def test_drop_table_if_exists_no_error(self):
        store = _make_store()
        store.drop_table("nonexistent", if_exists=True)
        store.dispose()

    def test_drop_view_not_stored(self):
        store = _make_store()
        with pytest.raises(TableNotStored, match="nonexistent"):
            store.drop_view("nonexistent")
        store.dispose()

    def test_drop_view_if_exists_no_error(self):
        store = _make_store()
        store.drop_view("nonexistent", if_exists=True)
        store.dispose()


class TestDeleteRows:
    def test_delete_rows_table_not_stored(self):
        store = _make_store()
        with pytest.raises(TableNotStored, match="nonexistent"):
            store.delete_rows("nonexistent", {"id": 1})
        store.dispose()

    def test_delete_rows_empty_values(self):
        store = _make_store()
        schema = _make_tz_schema()
        store.ingest_table(_make_tz_df(), schema)
        with pytest.raises(InvalidParameter, match="cannot be empty"):
            store.delete_rows(schema.name, {})
        store.dispose()

    def test_delete_rows_wrong_columns(self):
        store = _make_store()
        schema = _make_tz_schema()
        store.ingest_table(_make_tz_df(), schema)
        with pytest.raises(InvalidParameter, match="must match the schema columns"):
            store.delete_rows(schema.name, {"wrong_column": 1})
        store.dispose()

    def test_delete_rows_no_matching_rows(self):
        store = _make_store()
        schema = _make_tz_schema()
        store.ingest_table(_make_tz_df(), schema)
        with pytest.raises(InvalidParameter, match="Failed to delete rows"):
            store.delete_rows(schema.name, {"id": 999})
        store.dispose()


class TestWriteParquetErrors:
    def test_write_table_to_parquet_not_stored(self, tmp_path):
        store = _make_store()
        with pytest.raises(TableNotStored, match="nonexistent"):
            store.write_table_to_parquet("nonexistent", tmp_path / "out.parquet")
        store.dispose()


class TestConvertTimeZoneAlreadyExists:
    def test_convert_time_zone_dst_exists(self):
        store = _make_store()
        schema = _make_tz_schema()
        store.ingest_table(_make_tz_df(), schema)

        # First conversion creates the destination table
        store.convert_time_zone(schema.name, ZoneInfo("US/Eastern"))

        # Second conversion to the same tz should fail because dst table exists
        with pytest.raises(TableAlreadyExists):
            store.convert_time_zone(schema.name, ZoneInfo("US/Eastern"))
        store.dispose()

    def test_localize_time_zone_dst_exists(self):
        store = _make_store()
        schema = _make_ntz_schema()
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
                ),
                "value": [1.0, 2.0, 3.0],
            }
        )
        store.ingest_table(df, schema)

        # First localization (must use standard tz without DST)
        store.localize_time_zone(schema.name, ZoneInfo("EST"))

        # Second localization to the same tz should fail
        with pytest.raises(TableAlreadyExists):
            store.localize_time_zone(schema.name, ZoneInfo("EST"))
        store.dispose()


class TestSchemaManager:
    def test_schema_manager_property(self):
        store = _make_store()
        mgr = store.schema_manager
        assert mgr is not None
        store.dispose()
