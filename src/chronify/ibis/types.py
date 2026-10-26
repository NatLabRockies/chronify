"""Type conversion utilities for Ibis backends."""

import ibis.expr.datatypes as dt

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
