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
