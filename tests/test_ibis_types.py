"""Tests for ibis type conversion utilities."""

import ibis.expr.datatypes as dt
import pytest

from chronify.ibis.types import (
    get_duckdb_type_from_ibis,
    get_ibis_type_from_string,
)


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


class TestGetDuckdbTypeFromIbis:
    def test_common_types(self):
        assert get_duckdb_type_from_ibis(dt.Boolean()) == "BOOLEAN"
        assert get_duckdb_type_from_ibis(dt.Int64()) == "BIGINT"
        assert get_duckdb_type_from_ibis(dt.Float64()) == "DOUBLE"
        assert get_duckdb_type_from_ibis(dt.String()) == "VARCHAR"

    def test_timestamp_with_timezone(self):
        assert get_duckdb_type_from_ibis(dt.Timestamp(timezone="UTC")) == "TIMESTAMPTZ"

    def test_timestamp_without_timezone(self):
        assert get_duckdb_type_from_ibis(dt.Timestamp(timezone=None)) == "TIMESTAMP"

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported Ibis type for DuckDB"):
            get_duckdb_type_from_ibis(dt.Binary())
