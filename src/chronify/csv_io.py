from pathlib import Path
from typing import Any

import duckdb
from duckdb import DuckDBPyRelation

from chronify.models import CsvTableSchema
from chronify.ibis.types import get_duckdb_type_from_ibis
from chronify.time_configs import DatetimeRange


def read_csv(path: Path | str, schema: CsvTableSchema, **kwargs: Any) -> DuckDBPyRelation:
    """Read a CSV file into a DuckDB relation."""
    if schema.column_dtypes:
        dtypes = {x.name: get_duckdb_type_from_ibis(x.dtype).id for x in schema.column_dtypes}
        rel = duckdb.read_csv(str(path), dtype=dtypes, **kwargs)
    else:
        rel = duckdb.read_csv(str(path), **kwargs)

    time_config = schema.time_config
    exprs = []
    for i, column in enumerate(rel.columns):
        expr = column
        if isinstance(time_config, DatetimeRange) and column == time_config.time_column:
            time_type = rel.types[i]
            if time_type == duckdb.typing.TIMESTAMP and not time_config.start_time_is_tz_naive():  # type: ignore
                expr = f"timezone('{time_config.start.tzinfo.key}', {column}) AS {column}"  # type: ignore
        exprs.append(expr)

    expr = ",".join(exprs)
    return duckdb.sql(f"SELECT {expr} FROM rel")
