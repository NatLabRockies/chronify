# How to Ingest Multiple Tables Efficiently

There are a few important considerations when ingesting many tables:
- Use one database connection.
- Avoid loading all tables into memory at once, if possible.
- Ensure additions are atomic. If anything fails, the final state should be the same as the initial
state.

**Setup**

The input data are in CSV files. Each file contains a timestamp column and one value column per
device.

```python
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from chronify import DatetimeRange, Store, TableSchema, CsvTableSchema

store = Store.create_in_memory_db()
resolution = timedelta(hours=1)
time_config = DatetimeRange(
    time_column="timestamp",
    start=datetime(2020, 1, 1, 0),
    length=8784,
    resolution=timedelta(hours=1),
)
src_schema = CsvTableSchema(
    time_config=time_config,
    column_dtypes=[
        ColumnDType(name="timestamp", dtype=DateTime(timezone=False)),
        ColumnDType(name="device1", dtype=Double()),
        ColumnDType(name="device2", dtype=Double()),
        ColumnDType(name="device3", dtype=Double()),
    ],
    value_columns=["device1", "device2", "device3"],
    pivoted_dimension_name="device",
)
dst_schema = TableSchema(
    name="devices",
    value_column="value",
    time_array_id_columns=["id"],
)
```

## Automated through chronfiy
Chronify will manage the database connection and errors.
```python
store.ingest_from_csvs(
    src_schema,
    dst_schema,
    (
        "/path/to/file1.csv",
        "/path/to/file2.csv",
        "/path/to/file3.csv",
    ),
 )

```

## Self-Managed
Open one connection to the database for the duration of your additions. Handle errors.
```python
with store.engine.connect() as conn:
    try:
        store.ingest_from_csv(src_schema, dst_schema, "/path/to/file1.csv")
        store.ingest_from_csv(src_schema, dst_schema, "/path/to/file2.csv")
        store.ingest_from_csv(src_schema, dst_schema, "/path/to/file3.csv")
    except Exception:
        conn.rollback()
```
