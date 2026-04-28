"""Type conversion utilities for Ibis backends."""

import ibis.expr.datatypes as dt
import pyarrow as pa

# Mapping from user-facing string type names to Ibis data types
_COLUMN_TYPES: dict[str, dt.DataType] = {
    "bool": dt.Boolean(),
    "int": dt.Int64(),
    "bigint": dt.Int64(),
    "float": dt.Float64(),
    "double": dt.Float64(),
    "str": dt.String(),
    "datetime": dt.Timestamp(timezone=None),
    "datetime_tz": dt.Timestamp(timezone="UTC"),
}

# Mapping from DuckDB type names to Ibis data types
_DUCKDB_TYPE_MAP: dict[str, dt.DataType] = {
    "BOOLEAN": dt.Boolean(),
    "TINYINT": dt.Int8(),
    "SMALLINT": dt.Int16(),
    "INTEGER": dt.Int32(),
    "BIGINT": dt.Int64(),
    "FLOAT": dt.Float32(),
    "DOUBLE": dt.Float64(),
    "VARCHAR": dt.String(),
    "TIMESTAMP": dt.Timestamp(timezone=None),
    "TIMESTAMP WITH TIME ZONE": dt.Timestamp(timezone="UTC"),
    "TIMESTAMPTZ": dt.Timestamp(timezone="UTC"),
    "TIMESTAMP_NS": dt.Timestamp(timezone=None),
    "DATE": dt.Date(),
}

# Reverse mapping from Ibis types to DuckDB type strings
_IBIS_TO_DUCKDB_MAP: dict[type[dt.DataType], str] = {
    dt.Boolean: "BOOLEAN",
    dt.Int8: "TINYINT",
    dt.Int16: "SMALLINT",
    dt.Int32: "INTEGER",
    dt.Int64: "BIGINT",
    dt.Float32: "FLOAT",
    dt.Float64: "DOUBLE",
    dt.String: "VARCHAR",
    dt.Date: "DATE",
}


def get_ibis_type_from_string(type_name: str) -> dt.DataType:
    """Convert a string type name to an Ibis DataType.

    Parameters
    ----------
    type_name
        One of: "bool", "int", "bigint", "float", "double", "str", "datetime", "datetime_tz"
    """
    if type_name not in _COLUMN_TYPES:
        msg = f"Unsupported type name: {type_name}. Valid types: {sorted(_COLUMN_TYPES.keys())}"
        raise ValueError(msg)
    return _COLUMN_TYPES[type_name]


def get_ibis_type_from_duckdb(duckdb_type: str) -> dt.DataType:
    """Convert a DuckDB type string to an Ibis DataType."""
    upper = duckdb_type.upper()
    if upper in _DUCKDB_TYPE_MAP:
        return _DUCKDB_TYPE_MAP[upper]
    msg = f"Unsupported DuckDB type: {duckdb_type}"
    raise ValueError(msg)


def get_duckdb_type_from_ibis(ibis_type: dt.DataType) -> str:
    """Convert an Ibis DataType to a DuckDB type string."""
    if isinstance(ibis_type, dt.Timestamp):
        if ibis_type.timezone is not None:
            return "TIMESTAMPTZ"
        return "TIMESTAMP"
    for cls, duckdb_name in _IBIS_TO_DUCKDB_MAP.items():
        if isinstance(ibis_type, cls):
            return duckdb_name
    msg = f"Unsupported Ibis type for DuckDB: {ibis_type}"
    raise ValueError(msg)


def pyarrow_to_ibis_type(arrow_type: pa.DataType) -> dt.DataType:
    """Convert a PyArrow type to an Ibis DataType."""
    return dt.DataType.from_pyarrow(arrow_type)


def ibis_to_pyarrow_type(ibis_type: dt.DataType) -> pa.DataType:
    """Convert an Ibis DataType to a PyArrow type."""
    return ibis_type.to_pyarrow()
