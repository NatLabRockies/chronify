"""Type conversion utilities for Ibis backends."""


import duckdb
import ibis
import ibis.expr.datatypes as dt
import pandas as pd
import pyarrow as pa
from duckdb.typing import DuckDBPyType

from chronify.exceptions import InvalidParameter


# Mapping from string type names to Ibis types
_COLUMN_TYPES = {
    "bool": dt.Boolean(),
    "datetime": dt.Timestamp(),
    "float": dt.Float64(),
    "int": dt.Int64(),
    "bigint": dt.Int64(),
    "str": dt.String(),
}


# Mapping from DuckDB type IDs to Ibis types
_DUCKDB_TO_IBIS_TYPES = {
    duckdb.typing.BIGINT.id: dt.Int64(),  # type: ignore
    duckdb.typing.BOOLEAN.id: dt.Boolean(),  # type: ignore
    duckdb.typing.DOUBLE.id: dt.Float64(),  # type: ignore
    duckdb.typing.FLOAT.id: dt.Float32(),  # type: ignore
    duckdb.typing.INTEGER.id: dt.Int32(),  # type: ignore
    duckdb.typing.SMALLINT.id: dt.Int16(),  # type: ignore
    duckdb.typing.TINYINT.id: dt.Int8(),  # type: ignore
    duckdb.typing.VARCHAR.id: dt.String(),  # type: ignore
}


def get_ibis_type_from_duckdb(duckdb_type: DuckDBPyType) -> dt.DataType:
    """Return the Ibis type for a DuckDB type.

    Parameters
    ----------
    duckdb_type
        DuckDB type

    Returns
    -------
    dt.DataType
        Ibis data type
    """
    match duckdb_type:
        case duckdb.typing.TIMESTAMP_TZ:  # type: ignore
            return dt.Timestamp(timezone="UTC")
        case (
            duckdb.typing.TIMESTAMP  # type: ignore
            | duckdb.typing.TIMESTAMP_MS  # type: ignore
            | duckdb.typing.TIMESTAMP_NS  # type: ignore
            | duckdb.typing.TIMESTAMP_S  # type: ignore
        ):
            return dt.Timestamp()
        case _:
            ibis_type = _DUCKDB_TO_IBIS_TYPES.get(duckdb_type.id)
            if ibis_type is None:
                msg = f"There is no Ibis mapping for {duckdb_type=}"
                raise InvalidParameter(msg)
            return ibis_type


def get_ibis_type_from_string(type_name: str) -> dt.DataType:
    """Return the Ibis type for a string type name.

    Parameters
    ----------
    type_name
        Type name (e.g., "int", "str", "datetime")

    Returns
    -------
    dt.DataType
        Ibis data type
    """
    ibis_type = _COLUMN_TYPES.get(type_name)
    if ibis_type is None:
        msg = f"Unknown type name: {type_name}. Valid options: {list(_COLUMN_TYPES.keys())}"
        raise InvalidParameter(msg)
    return ibis_type


def get_ibis_types_from_dataframe(
    df: pd.DataFrame | pa.Table,
) -> list[dt.DataType]:
    """Return a list of Ibis types from a DataFrame or PyArrow table.

    Parameters
    ----------
    df
        Input DataFrame or PyArrow table

    Returns
    -------
    list[dt.DataType]
        List of Ibis types for each column
    """
    # Use DuckDB to infer types (most robust approach)
    if isinstance(df, pa.Table):
        short_df = df.slice(0, 1)
    else:
        short_df = df.head(1)  # noqa: F841

    duckdb_types = duckdb.sql("SELECT * FROM short_df").dtypes
    return [get_ibis_type_from_duckdb(t) for t in duckdb_types]


def get_ibis_schema_from_dataframe(
    df: pd.DataFrame | pa.Table,
) -> ibis.Schema:
    """Return an Ibis schema from a DataFrame or PyArrow table.

    Parameters
    ----------
    df
        Input DataFrame or PyArrow table

    Returns
    -------
    ibis.Schema
        Ibis schema
    """
    if isinstance(df, pa.Table):
        columns = df.column_names
    else:
        columns = list(df.columns)

    types = get_ibis_types_from_dataframe(df)
    return ibis.schema(dict(zip(columns, types)))


def pyarrow_to_ibis_type(pa_type: pa.DataType) -> dt.DataType:
    """Convert a PyArrow type to an Ibis type.

    Parameters
    ----------
    pa_type
        PyArrow data type

    Returns
    -------
    dt.DataType
        Ibis data type
    """
    # Use Ibis's built-in conversion
    return dt.from_pyarrow(pa_type)


def ibis_to_pyarrow_type(ibis_type: dt.DataType) -> pa.DataType:
    """Convert an Ibis type to a PyArrow type.

    Parameters
    ----------
    ibis_type
        Ibis data type

    Returns
    -------
    pa.DataType
        PyArrow data type
    """
    return ibis_type.to_pyarrow()


# Mapping from Ibis types to DuckDB types
_IBIS_TO_DUCKDB_TYPES: dict[type[dt.DataType], DuckDBPyType] = {
    dt.Int8: duckdb.typing.TINYINT,  # type: ignore
    dt.Int16: duckdb.typing.SMALLINT,  # type: ignore
    dt.Int32: duckdb.typing.INTEGER,  # type: ignore
    dt.Int64: duckdb.typing.BIGINT,  # type: ignore
    dt.Float32: duckdb.typing.FLOAT,  # type: ignore
    dt.Float64: duckdb.typing.DOUBLE,  # type: ignore
    dt.Boolean: duckdb.typing.BOOLEAN,  # type: ignore
    dt.String: duckdb.typing.VARCHAR,  # type: ignore
    dt.Timestamp: duckdb.typing.TIMESTAMP,  # type: ignore
}


def get_duckdb_type_from_ibis(ibis_type: dt.DataType) -> DuckDBPyType:
    """Return the DuckDB type for an Ibis type.

    Parameters
    ----------
    ibis_type
        Ibis data type

    Returns
    -------
    DuckDBPyType
        DuckDB type
    """
    # Check for timestamp with timezone
    if isinstance(ibis_type, dt.Timestamp) and ibis_type.timezone is not None:
        return duckdb.typing.TIMESTAMP_TZ  # type: ignore

    duckdb_type = _IBIS_TO_DUCKDB_TYPES.get(type(ibis_type))
    if duckdb_type is None:
        msg = f"There is no DuckDB mapping for {ibis_type=}"
        raise InvalidParameter(msg)
    return duckdb_type
