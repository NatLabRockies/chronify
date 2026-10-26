from pathlib import Path
from typing import Any

import duckdb
from duckdb import DuckDBPyRelation
from ibis.backends.sql.datatypes import DuckDBType

from chronify.models import CsvTableSchema
from chronify.time_configs import DatetimeRange


def read_csv(path: Path | str, schema: CsvTableSchema, **kwargs: Any) -> DuckDBPyRelation:
    """Read a CSV file into a DuckDB relation."""
    if schema.column_dtypes:
        dtypes = {x.name: DuckDBType.to_string(x.dtype) for x in schema.column_dtypes}
        rel = duckdb.read_csv(str(path), dtype=dtypes, **kwargs)
    else:
        rel = duckdb.read_csv(str(path), **kwargs)

    time_config = schema.time_config
    exprs = []
    for i, column in enumerate(rel.columns):
        expr = column
        if isinstance(time_config, DatetimeRange) and column == time_config.time_column:
            time_type = rel.types[i]
            if (
                time_type == duckdb.sqltype("TIMESTAMP")
                and not time_config.start_time_is_tz_naive()
            ):
                expr = f"timezone('{time_config.start.tzinfo.key}', {column}) AS {column}"  # type: ignore
        exprs.append(expr)

    return rel.project(",".join(exprs))
