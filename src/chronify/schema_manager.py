import json

import ibis.expr.types as ir
from loguru import logger

from chronify.exceptions import TableNotStored
from chronify.ibis.backend import IbisBackend
from chronify.models import TableSchema


class SchemaManager:
    """Manages schemas for the Store. Provides a cache to avoid repeated database reads."""

    SCHEMAS_TABLE = "schemas"

    def __init__(self, backend: IbisBackend):
        self._backend = backend
        # Caching is not necessary if using SQLite, which provides very fast performance (~1 us)
        # for checking schemas in the **tiny** schemas table.
        # The same lookups in DuckDB are taking over 100 us.
        self._cache: dict[str, TableSchema] = {}

        if self.SCHEMAS_TABLE in self._backend.list_tables():
            logger.info("Loaded existing database {}", self._backend.database)
            self.rebuild_cache()
        else:
            self._create_schemas_table()
            logger.info("Initialized new database: {}", self._backend.database)

    def _create_schemas_table(self) -> None:
        """Create the schemas table."""
        # Use raw SQL to create table with proper types across all backends
        if self._backend.name == "duckdb":
            self._backend.execute_sql(
                f"CREATE TABLE {self.SCHEMAS_TABLE}(name VARCHAR, schema VARCHAR)"
            )
        elif self._backend.name == "spark":
            # Spark 4.0+ requires VARCHAR to have a length; use STRING instead
            self._backend.execute_sql(
                f"CREATE TABLE {self.SCHEMAS_TABLE}(name STRING, schema STRING)"
            )
        else:
            # SQLite uses TEXT
            self._backend.execute_sql(f"CREATE TABLE {self.SCHEMAS_TABLE}(name TEXT, schema TEXT)")

    def _get_schema_table(self) -> ir.Table:
        """Get the schemas table as an Ibis table."""
        return self._backend.table(self.SCHEMAS_TABLE)

    def add_schema(self, schema: TableSchema) -> None:
        """Add the schema to the store."""
        import pandas as pd

        df = pd.DataFrame({"name": [schema.name], "schema": [schema.model_dump_json()]})
        self._backend.insert(self.SCHEMAS_TABLE, df)

        # If there is a rollback after this addition to cache, things _should_ still be OK.
        # The table will be deleted and any attempted reads will fail with an error.
        # There will be a stale entry in cache, but it will be overwritten if the user ever
        # adds a new table with the same name.
        self._cache[schema.name] = schema
        logger.trace("Added schema for table {}", schema.name)

    def get_schema(self, name: str) -> TableSchema:
        """Retrieve the schema for the table with name."""
        schema = self._cache.get(name)
        if schema is None:
            self.rebuild_cache()

        schema = self._cache.get(name)
        if schema is None:
            msg = f"{name=}"
            raise TableNotStored(msg)

        return self._cache[name]

    def remove_schema(self, name: str) -> None:
        """Remove the schema from the store."""
        if self._backend.name == "spark":
            # Spark doesn't support DELETE, so we recreate the table
            table = self._get_schema_table()
            # Filter out the row to delete and get remaining rows
            remaining = table.filter(table.name != name)
            df = self._backend.execute(remaining)

            self._backend.execute_sql(f"DROP TABLE {self.SCHEMAS_TABLE}")
            self._backend.execute_sql(
                f"CREATE TABLE {self.SCHEMAS_TABLE}(name STRING, schema STRING)"
            )
            for _, row in df.iterrows():
                escaped_name = _escape_sql_string(row["name"])
                escaped_schema = _escape_sql_string(row["schema"])
                self._backend.execute_sql(
                    f"INSERT INTO {self.SCHEMAS_TABLE} VALUES({escaped_name}, {escaped_schema})"
                )
        else:
            # DuckDB and SQLite support DELETE
            escaped_name = _escape_sql_string(name)
            self._backend.execute_sql(
                f"DELETE FROM {self.SCHEMAS_TABLE} WHERE name = {escaped_name}"
            )

        self._cache.pop(name, None)

    def rebuild_cache(self) -> None:
        """Rebuild the cache of schemas."""
        self._cache.clear()
        table = self._get_schema_table()
        df = self._backend.execute(table)
        for _, row in df.iterrows():
            name = row["name"]
            json_text = row["schema"]
            schema = TableSchema(**json.loads(json_text))
            assert name == schema.name
            assert name not in self._cache
            self._cache[name] = schema


def _escape_sql_string(value: str) -> str:
    """Escape a string value for safe inclusion in SQL statements.

    Parameters
    ----------
    value
        The string value to escape

    Returns
    -------
    str
        SQL-safe quoted string
    """
    # Escape single quotes by doubling them
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
