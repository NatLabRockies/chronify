# Apache Spark Backend
Download Spark from https://spark.apache.org/downloads.html and install it. Spark provides startup
scripts for UNIX operating systems (not Windows).

## Install chronify with Spark support
```
$ pip install "chronify[spark]"
```

## Installation on a development computer
Installation can be as simple as
```
$ tar -xzf spark-4.0.1-bin-hadoop3.tgz
$ export SPARK_HOME=$(pwd)/spark-4.0.1-bin-hadoop3
```

Start a Thrift server. This allows JDBC clients to send SQL queries to an in-process Spark cluster
running in local mode.
```
$ $SPARK_HOME/sbin/start-thriftserver.sh --master=spark://$(hostname):7077
```

The URL to connect to this server is `hive://localhost:10000/default`

## Installation on an HPC
The chronify development team uses these
[scripts](https://github.com/NREL/HPC/tree/master/applications/spark) to run Spark on NREL's HPC.

## Chronify Usage
This example creates a chronify Store with Spark as the backend and then adds a view to a Parquet
file. Chronify will run its normal time checks.

First, create the Parquet file and chronify schema.

```python
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from chronify import DatetimeRange, Store, TableSchema, CsvTableSchema

initial_time = datetime(2020, 1, 1)
end_time = datetime(2020, 12, 31, 23)
resolution = timedelta(hours=1)
timestamps = pd.date_range(initial_time, end_time, freq=resolution, unit="us")
dfs = []
for i in range(1, 4):
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "id": i,
            "value": np.random.random(len(timestamps)),
        }
    )
    dfs.append(df)
df = pd.concat(dfs)
df.to_parquet("data.parquet", index=False)
schema = TableSchema(
    name="devices",
    value_column="value",
    time_config=DatetimeRange(
        time_column="timestamp",
        start=initial_time,
        length=len(timestamps),
        resolution=resolution,
    ),
    time_array_id_columns=["id"],
)
```

```python
from chronify import Store

store = Store.create_new_hive_store("hive://localhost:10000/default")
store.create_view_from_parquet("data.parquet")
```

Verify the data:
```python
store.read_table(schema.name).head()
```
```
            timestamp  id     value
0 2020-01-01 00:00:00   1  0.785399
1 2020-01-01 01:00:00   1  0.102756
2 2020-01-01 02:00:00   1  0.178587
3 2020-01-01 03:00:00   1  0.326194
4 2020-01-01 04:00:00   1  0.994851
```

## Time configuration mapping
The primary use case for Spark is to map datasets that are larger than can be processed by DuckDB
on one computer. In such a workflow a user would call
```python
store.map_table_time_config(src_table_name, dst_schema, output_file="mapped_data.parquet")
```
