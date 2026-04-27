import json

import ibis
import pandas as pd
from loguru import logger

from chronify.exceptions import InvalidOperation, InvalidParameter, TableNotStored
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
        # Uniqueness of `name` is enforced in `add_schema` rather than via a
        # unique index, since Spark SQL does not support CREATE UNIQUE INDEX.
        schema = ibis.schema({"name": "string", "schema": "string"})
        try:
            self._backend.create_table(self.SCHEMAS_TABLE, schema=schema)
        except Exception as exc:
            # On Spark, a stale warehouse directory can cause
            # LOCATION_ALREADY_EXISTS even though list_tables() didn't find
            # the table. Drop the stale remnant and retry. Other failures
            # (permissions, connectivity, etc.) must propagate.
            if self._backend.name != "spark" or "LOCATION_ALREADY_EXISTS" not in str(exc):
                raise
            logger.debug("Retrying schemas table creation after dropping stale Spark remnant.")
            self._backend.drop_table(self.SCHEMAS_TABLE)
            self._backend.create_table(self.SCHEMAS_TABLE, schema=schema)

    def add_schema(self, schema: TableSchema) -> None:
        """Add the schema to the store."""
        if schema.name in self._cache or self._schema_row_exists(schema.name):
            msg = f"A schema with name={schema.name!r} is already registered"
            raise InvalidParameter(msg)
        df = pd.DataFrame({"name": [schema.name], "schema": [schema.model_dump_json()]})
        self._backend.insert(self.SCHEMAS_TABLE, df)
        self._cache[schema.name] = schema
        logger.trace("Added schema for table {}", schema.name)

    def _schema_row_exists(self, name: str) -> bool:
        table = self._backend.table(self.SCHEMAS_TABLE)
        df = self._backend.execute(table.filter(table["name"] == name).limit(1))
        return not df.empty

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
        self._backend.delete_rows(self.SCHEMAS_TABLE, {"name": name})
        self._cache.pop(name, None)

    def rebuild_cache(self) -> None:
        """Rebuild the cache of schemas."""
        self._cache.clear()
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        df = self._backend.execute(self._backend.table(self.SCHEMAS_TABLE))
        for _, row in df.iterrows():
            name = row["name"]
            schema = TableSchema(**json.loads(row["schema"]))
            if name != schema.name:
                msg = (
                    f"schemas table is corrupt: row name={name!r} does not match "
                    f"schema.name={schema.name!r}"
                )
                raise InvalidOperation(msg)
            if name in self._cache:
                msg = f"schemas table is corrupt: duplicate entry for name={name!r}"
                raise InvalidOperation(msg)
            self._cache[name] = schema
