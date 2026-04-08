import json

import pandas as pd
from loguru import logger

from chronify.exceptions import TableNotStored
from chronify.ibis.base import IbisBackend
from chronify.models import TableSchema


class SchemaManager:
    """Manages schemas for the Store. Provides a cache to avoid repeated database reads."""

    SCHEMAS_TABLE = "schemas"

    def __init__(self, backend: IbisBackend) -> None:
        self._backend = backend
        self._cache: dict[str, TableSchema] = {}

        if self._backend.has_table(self.SCHEMAS_TABLE):
            logger.info("Loaded existing database {}", self._backend.database)
            self.rebuild_cache()
        else:
            self._create_schemas_table()
            logger.info("Initialized new database: {}", self._backend.database)

    def _create_schemas_table(self) -> None:
        import ibis

        schema = ibis.schema({"name": "string", "schema": "string"})
        self._backend.create_table(self.SCHEMAS_TABLE, schema=schema)

    def add_schema(self, schema: TableSchema) -> None:
        """Add the schema to the store."""
        df = pd.DataFrame({"name": [schema.name], "schema": [schema.model_dump_json()]})
        self._backend.insert(self.SCHEMAS_TABLE, df)
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

        return schema

    def remove_schema(self, name: str) -> None:
        """Remove the schema from the store."""
        self._backend.execute_sql(f"DELETE FROM {self.SCHEMAS_TABLE} WHERE name = '{name}'")
        self._cache.pop(name, None)

    def rebuild_cache(self) -> None:
        """Rebuild the cache of schemas."""
        self._cache.clear()
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        df = self._backend.execute_sql_to_df(f"SELECT * FROM {self.SCHEMAS_TABLE}")
        for _, row in df.iterrows():
            name = row["name"]
            schema = TableSchema(**json.loads(row["schema"]))
            assert name == schema.name
            assert name not in self._cache
            self._cache[name] = schema
