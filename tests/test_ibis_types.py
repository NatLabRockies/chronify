"""Tests for ibis type conversion utilities."""

import ibis.expr.datatypes as dt
import pytest

from chronify.ibis.types import get_ibis_type_from_string


class TestGetIbisTypeFromString:
    def test_valid_types(self):
        assert get_ibis_type_from_string("int") == dt.Int64()
        assert get_ibis_type_from_string("float") == dt.Float64()
        assert get_ibis_type_from_string("str") == dt.String()
        assert get_ibis_type_from_string("bool") == dt.Boolean()
        assert get_ibis_type_from_string("datetime") == dt.Timestamp(timezone=None)
        assert get_ibis_type_from_string("datetime_tz") == dt.Timestamp(timezone="UTC")

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported type name"):
            get_ibis_type_from_string("invalid")


def test_read_csv_accepts_any_ibis_dtype(tmp_path):
    """read_csv must accept ibis types beyond a handful of common ones."""
    from datetime import datetime, timedelta
    from decimal import Decimal

    from chronify.csv_io import read_csv
    from chronify.models import ColumnDType, CsvTableSchema
    from chronify.time_configs import DatetimeRange

    path = tmp_path / "data.csv"
    path.write_text("timestamp,count,price\n2020-01-01 00:00:00,1,1.25\n")
    schema = CsvTableSchema(
        time_config=DatetimeRange(
            start=datetime(2020, 1, 1),
            resolution=timedelta(hours=1),
            length=1,
            time_column="timestamp",
        ),
        column_dtypes=[
            ColumnDType(name="timestamp", dtype=dt.Timestamp(scale=6)),
            ColumnDType(name="count", dtype=dt.UInt32()),
            ColumnDType(name="price", dtype=dt.Decimal(10, 2)),
        ],
        value_columns=["count", "price"],
    )
    df = read_csv(path, schema).to_df()
    assert df["count"].tolist() == [1]
    assert df["price"].tolist() == [Decimal("1.25")]
