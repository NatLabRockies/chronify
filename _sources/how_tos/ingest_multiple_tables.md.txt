# How to Ingest Multiple Tables Efficiently

There are a few important considerations when ingesting many tables:
- Avoid loading all tables into memory at once, if possible.
- Ensure additions are atomic. If anything fails, the final state should be the same as the initial
state.

**Setup**

The input data are in CSV files. Each file contains a timestamp column and one value column per
device.

```python
from datetime import datetime, timedelta

from chronify import (
    ColumnDType,
    CsvTableSchema,
    DatetimeRange,
    Store,
    TableSchema,
)

store = Store.create_in_memory_db()
time_config = DatetimeRange(
    time_column="timestamp",
    start=datetime(2020, 1, 1, 0),
    length=8784,
    resolution=timedelta(hours=1),
)
src_schema = CsvTableSchema(
    time_config=time_config,
    column_dtypes=[
        ColumnDType(name="timestamp", dtype="datetime"),
        ColumnDType(name="device1", dtype="float"),
        ColumnDType(name="device2", dtype="float"),
        ColumnDType(name="device3", dtype="float"),
    ],
    value_columns=["device1", "device2", "device3"],
    pivoted_dimension_name="device",
)
dst_schema = TableSchema(
    name="devices",
    time_config=time_config,
    value_column="value",
    time_array_id_columns=["device"],
)
```

## Automated through chronify
Chronify will manage the database connection and errors.
```python
store.ingest_from_csvs(
    (
        "/path/to/file1.csv",
        "/path/to/file2.csv",
        "/path/to/file3.csv",
    ),
    src_schema,
    dst_schema,
)

```

## Self-Managed
Wrap the additions in a backend transaction. Any tables or views created within the block are
automatically dropped if an exception is raised.
```python
with store.backend.transaction():
    store.ingest_from_csv("/path/to/file1.csv", src_schema, dst_schema)
    store.ingest_from_csv("/path/to/file2.csv", src_schema, dst_schema)
    store.ingest_from_csv("/path/to/file3.csv", src_schema, dst_schema)
```

```{note}
Real database transaction semantics depend on the backend. The DuckDB and SQLite backends issue
a real `BEGIN` / `COMMIT` / `ROLLBACK` around the block, so partial inserts to existing tables
are rolled back on failure. The Spark backend does not support transactions; the context
manager falls back to best-effort cleanup that drops any tables or views created inside the
block when an exception is raised, but rows appended to pre-existing tables cannot be
rolled back.
```
