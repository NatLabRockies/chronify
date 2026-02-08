# How to Map Time
This recipe demonstrates how to map a table's time configuration from one type to another.

**Source table**: data is stored in representative time where there is one week of data per month by
hour for one year.

**Destination table**: data is stored with `datetime` timestamps for each hour of the year.

**Workflow**:
- Add the source table to the database.
- Call `Store.map_table_time_config()`
- Chronify adds the destination table to the database.

This example creates a representative time table used in chronify's tests.

1. Ingest the source data.

```python
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from chronify import (
    DatetimeRange,
    RepresentativePeriodFormat,
    RepresentativePeriodTimeNTZ,
    Store,
    CsvTableSchema,
    TableSchema,
)

src_table_name = "ev_charging"
dst_table_name = "ev_charging_datetime"
hours_per_year = 12 * 7 * 24
num_time_arrays = 3
df = pd.DataFrame({
    "id": np.concatenate([np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]),
    "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
    "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
    "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
    "value": np.random.random(hours_per_year * num_time_arrays),
})
schema = TableSchema(
    name=src_table_name,
    value_column="value",
    time_config=RepresentativePeriodTimeNTZ(
        time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
    ),
    time_array_id_columns=["id"],
)
store = Store.create_in_memory_db()
store.ingest_table(df, schema)
store.read_query(src_table_name, f"SELECT * FROM {src_table_name} LIMIT 5").head()
```

```
   id  month  day_of_week  hour     value
0   1      1            0     0  0.578496
1   1      1            0     1  0.092271
2   1      1            0     2  0.111521
3   1      1            0     3  0.671668
4   1      1            0     4  0.782365
```

2. Map the table's time to datetime.
```python
dst_schema = TableSchema(
    name=dst_table_name,
    value_column="value",
    time_array_id_columns=["id"],
    time_config=DatetimeRange(
        time_column="timestamp",
        start=datetime(2020, 1, 1, 0),
        length=8784,
        resolution=timedelta(hours=1),
    )
)
store.map_table_time_config(src_table_name, dst_schema)
store.read_query(dst_table_name, f"SELECT * FROM {dst_table_name} LIMIT 5").head()
```

```
   id     value           timestamp
0   3  0.006213 2020-01-01 00:00:00
1   3  0.865765 2020-01-01 01:00:00
2   3  0.187256 2020-01-01 02:00:00
3   3  0.336157 2020-01-01 03:00:00
4   3  0.582281 2020-01-01 04:00:00
```
