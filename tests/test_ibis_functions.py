"""Tests for ibis/functions.py edge cases and uncovered branches."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pytest

from chronify.exceptions import InvalidOperation, InvalidParameter
from chronify.ibis import make_backend
from chronify.ibis.functions import (
    _check_one_config_per_datetime_column,
    _convert_database_output_for_datetime,
    _convert_spark_output_for_datetime,
    _normalize_timestamps,
    write_table,
)
from chronify.time import TimeIntervalType
from chronify.time_configs import DatetimeRange


def _make_tz_config(col: str = "timestamp") -> DatetimeRange:
    return DatetimeRange(
        time_column=col,
        start=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
        length=2,
        resolution=timedelta(hours=1),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
    )


def _make_ntz_config(col: str = "timestamp") -> DatetimeRange:
    return DatetimeRange(
        time_column=col,
        start=datetime(2020, 1, 1),
        length=2,
        resolution=timedelta(hours=1),
        interval_type=TimeIntervalType.PERIOD_BEGINNING,
    )


class TestNormalizeTimestamps:
    def test_tz_aware_to_ntz(self):
        config = _make_ntz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"], utc=True
                ),
            }
        )
        result = _normalize_timestamps(df, [config])
        assert not isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_tz_naive_to_tz(self):
        config = _make_tz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            }
        )
        result = _normalize_timestamps(df, [config])
        assert isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_column_not_in_df(self):
        config = _make_tz_config(col="missing_col")
        df = pd.DataFrame({"other": [1, 2]})
        result = _normalize_timestamps(df, [config])
        assert list(result.columns) == ["other"]

    def test_non_datetime_column_skipped(self):
        config = _make_tz_config()
        df = pd.DataFrame({"timestamp": ["not", "datetime"]})
        result = _normalize_timestamps(df, [config])
        assert list(result["timestamp"]) == ["not", "datetime"]


class TestCheckOneConfigPerDatetimeColumn:
    def test_duplicate_config_raises(self):
        configs = [_make_tz_config("timestamp"), _make_tz_config("timestamp")]
        with pytest.raises(InvalidParameter, match="More than one datetime config"):
            _check_one_config_per_datetime_column(configs)


class TestConvertDatabaseOutputForDatetime:
    def test_tz_with_object_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame({"timestamp": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"]})
        _convert_database_output_for_datetime(df, config)
        assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_tz_with_tz_aware_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00+05:00", "2020-01-01 01:00:00+05:00"]
                ),
            }
        )
        _convert_database_output_for_datetime(df, config)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_tz_with_naive_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            }
        )
        _convert_database_output_for_datetime(df, config)
        assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_ntz_with_object_dtype(self):
        config = _make_ntz_config()
        df = pd.DataFrame({"timestamp": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"]})
        _convert_database_output_for_datetime(df, config)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_missing_column_is_noop(self):
        config = _make_tz_config()
        df = pd.DataFrame({"other": [1, 2]})
        _convert_database_output_for_datetime(df, config)
        assert list(df.columns) == ["other"]


class TestConvertSparkOutputForDatetime:
    def test_tz_with_object_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame({"timestamp": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"]})
        _convert_spark_output_for_datetime(df, config)
        assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_tz_with_tz_aware_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00+05:00", "2020-01-01 01:00:00+05:00"]
                ),
            }
        )
        _convert_spark_output_for_datetime(df, config)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_tz_with_naive_dtype(self):
        config = _make_tz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            }
        )
        _convert_spark_output_for_datetime(df, config)
        assert isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_ntz_with_object_dtype(self):
        config = _make_ntz_config()
        df = pd.DataFrame({"timestamp": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"]})
        _convert_spark_output_for_datetime(df, config)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_ntz_strips_tz_from_aware(self):
        config = _make_ntz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00:00+00:00", "2020-01-01 01:00:00+00:00"], utc=True
                ),
            }
        )
        _convert_spark_output_for_datetime(df, config)
        assert not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype)

    def test_missing_column_is_noop(self):
        config = _make_tz_config()
        df = pd.DataFrame({"other": [1, 2]})
        _convert_spark_output_for_datetime(df, config)
        assert list(df.columns) == ["other"]


class TestWriteTable:
    def test_pyarrow_table_input(self):
        backend = make_backend("duckdb")
        config = _make_ntz_config()
        pa_table = pa.table(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
                "value": [1.0, 2.0],
            }
        )
        write_table(backend, pa_table, "pa_test", [config], if_exists="fail")
        assert backend.has_table("pa_test")
        df = backend.execute(backend.table("pa_test"))
        assert len(df) == 2
        backend.dispose()

    def test_pyarrow_table_input_stays_arrow_for_duckdb(self, monkeypatch):
        backend = make_backend("duckdb")
        config = _make_ntz_config()
        pa_table = pa.table(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
                "value": [1.0, 2.0],
            }
        )
        seen_arrow = False

        def create_table(name, obj=None, schema=None, overwrite=False):
            nonlocal seen_arrow
            seen_arrow = isinstance(obj, pa.Table)
            return backend.connection.create_table(
                name, obj=obj, schema=schema, overwrite=overwrite
            )

        monkeypatch.setattr(backend, "create_table", create_table)
        write_table(backend, pa_table, "pa_test_arrow", [config], if_exists="fail")
        assert seen_arrow
        backend.dispose()

    def test_invalid_if_exists_duckdb(self):
        backend = make_backend("duckdb")
        config = _make_ntz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
                "value": [1.0, 2.0],
            }
        )
        write_table(backend, df, "test_tbl", [config], if_exists="fail")
        with pytest.raises(InvalidOperation, match="Invalid if_exists"):
            write_table(backend, df, "test_tbl", [config], if_exists="invalid")
        backend.dispose()

    def test_unsupported_backend(self):
        """A backend with an unknown name should raise NotImplementedError."""
        from unittest.mock import MagicMock

        backend = MagicMock()
        backend.name = "unknown_db"
        config = _make_ntz_config()
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"]),
            }
        )
        with pytest.raises(NotImplementedError, match="Unsupported backend"):
            write_table(backend, df, "test", [config], if_exists="fail")
