"""Tests for ibis type conversion utilities."""

import ibis.expr.datatypes as dt
import pyarrow as pa
import pytest

from chronify.ibis.types import (
    get_duckdb_type_from_ibis,
    get_ibis_type_from_duckdb,
    get_ibis_type_from_string,
    ibis_to_pyarrow_type,
    pyarrow_to_ibis_type,
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


class TestGetIbisTypeFromDuckdb:
    def test_common_types(self):
        assert get_ibis_type_from_duckdb("BOOLEAN") == dt.Boolean()
        assert get_ibis_type_from_duckdb("INTEGER") == dt.Int32()
        assert get_ibis_type_from_duckdb("BIGINT") == dt.Int64()
        assert get_ibis_type_from_duckdb("DOUBLE") == dt.Float64()
        assert get_ibis_type_from_duckdb("VARCHAR") == dt.String()
        assert get_ibis_type_from_duckdb("TIMESTAMP") == dt.Timestamp(timezone=None)
        assert get_ibis_type_from_duckdb("TIMESTAMPTZ") == dt.Timestamp(timezone="UTC")

    def test_case_insensitive(self):
        assert get_ibis_type_from_duckdb("boolean") == dt.Boolean()
        assert get_ibis_type_from_duckdb("integer") == dt.Int32()

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported DuckDB type"):
            get_ibis_type_from_duckdb("BLOB")


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


class TestPyarrowConversion:
    def test_pyarrow_to_ibis(self):
        result = pyarrow_to_ibis_type(pa.int64())
        assert isinstance(result, dt.Int64)

    def test_ibis_to_pyarrow(self):
        result = ibis_to_pyarrow_type(dt.Int64())
        assert result == pa.int64()

    def test_roundtrip(self):
        original = dt.Float64()
        arrow_type = ibis_to_pyarrow_type(original)
        back = pyarrow_to_ibis_type(arrow_type)
        assert isinstance(back, dt.Float64)
