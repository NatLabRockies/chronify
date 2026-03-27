from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from chronify.ibis.backend import IbisBackend


def create_materialized_view(
    query: str,
    dst_table: str,
    backend: IbisBackend,
    scratch_dir: Optional[Path] = None,
) -> None:
    """Create a materialized view with a Parquet file. This is a workaround for an undiagnosed
    problem with timestamps and time zones with Spark/Hive.

    The Parquet file will be written to scratch_dir. Callers must ensure that the directory
    persists for the duration of the work.
    """
    with NamedTemporaryFile(dir=scratch_dir, suffix=".parquet") as f:
        f.close()
        output = Path(f.name)

    write_query = f"""
        INSERT OVERWRITE DIRECTORY
            '{output}'
            USING parquet
            ({query})
    """
    backend.execute_sql(write_query)
    view_query = f"CREATE VIEW {dst_table} AS SELECT * FROM parquet.`{output}`"
    backend.execute_sql(view_query)
