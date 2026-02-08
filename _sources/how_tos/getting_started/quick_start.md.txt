# Quick Start

```python

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from chronify import DatetimeRange, Store, TableSchema

store = Store.create_file_db(file_path="time_series.db")
resolution = timedelta(hours=1)
time_range = pd.date_range("2020-01-01", "2020-12-31 23:00:00", freq=resolution)
store.ingest_tables(
    (
        pd.DataFrame({"timestamp": time_range, "value": np.random.random(8784), "id": 1}),
        pd.DataFrame({"timestamp": time_range, "value": np.random.random(8784), "id": 2}),
    ),
    TableSchema(
        name="devices",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=datetime(2020, 1, 1, 0),
            length=8784,
            resolution=timedelta(hours=1),
        ),
        time_array_id_columns=["id"],
    )
 )
query = "SELECT timestamp, value FROM devices WHERE id = ?"
df = store.read_query("devices", query, params=(2,))
df.head()
```

```
            timestamp     value  id
0 2020-01-01 00:00:00  0.594748   2
1 2020-01-01 01:00:00  0.608295   2
2 2020-01-01 02:00:00  0.297535   2
3 2020-01-01 03:00:00  0.870238   2
4 2020-01-01 04:00:00  0.376144   2
```
