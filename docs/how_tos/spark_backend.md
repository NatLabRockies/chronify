# Apache Spark Backend

This guide shows how to use Chronify with Apache Spark for processing large time series datasets
that are too large for DuckDB on a single machine.

## Prerequisites

Download Spark from https://spark.apache.org/downloads.html and install it. Spark provides startup
scripts for UNIX operating systems (not Windows).

## Install chronify with Spark support
```bash
pip install chronify[spark]
```

## Installation on a development computer

Installation can be as simple as:
```bash
tar -xzf spark-4.0.0-bin-hadoop3.tgz
export SPARK_HOME=$(pwd)/spark-4.0.0-bin-hadoop3
```

## Installation on an HPC
The chronify development team uses the
[package](https://github.com/NREL/sparkctl) to run Spark on NLR's HPC.

## Chronify Usage

This example creates a chronify Store with Spark as the backend and then adds a view to a Parquet
file. Chronify will run its normal time checks.

**Note:** The Spark backend is designed primarily for reading data from Parquet files via views.
Direct data ingestion (inserting DataFrames) is not supported with Spark.

### Step 1: Create a Parquet file and schema

```python
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from chronify import DatetimeRange, Store, TableSchema

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

### Step 2: Create a Spark store and add a view

```python
from chronify import Store

# Create a new Spark store (this will create a local SparkSession)
store = Store.create_new_spark_store()

# Or pass an existing SparkSession:
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("chronify").getOrCreate()
# store = Store.create_new_spark_store(session=spark)

# Create a view from the Parquet file
store.create_view_from_parquet("data.parquet", schema)
```

### Step 3: Verify the data

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

## Registering Ibis tables

If you're using Ibis to load and transform data before passing it to chronify for time validation,
you must use `create_view`. This is useful when integrating with
other packages like dsgrid that use Ibis for data loading and transformation.

### Validation only (recommended for most workflows)

Use `create_view` when you only need time validation and don't need the data to persist
in chronify after the session ends:

```python
import ibis
from chronify import Store, TableSchema, DatetimeRange
from datetime import datetime, timedelta

# Load data with Ibis (can be from any Ibis-supported backend)
conn = ibis.pyspark.connect(spark_session)
table = conn.read_parquet("data.parquet")

# Optionally transform the data (rename columns, filter, etc.)
table = table.rename({"old_timestamp": "timestamp"})

# Create the chronify schema
schema = TableSchema(
    name="devices",
    value_column="value",
    time_config=DatetimeRange(
        time_column="timestamp",
        start=datetime(2020, 1, 1),
        length=8784,
        resolution=timedelta(hours=1),
    ),
    time_array_id_columns=["id"],
)

# Register with chronify for time validation only
store = Store.create_new_spark_store(session=spark_session)
store.create_view(schema, table)
```

Both the temp view and schema are registered and available during the session.

### Persistent registration (requires Hive)

Use `create_view` when you need the view and schema to persist across sessions.
This requires Spark to be configured with Hive metastore support.

#### Configuring Hive metastore

To enable Hive support, create your SparkSession with `enableHiveSupport()`:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("chronify") \
    .config("spark.sql.warehouse.dir", "/path/to/spark-warehouse") \
    .enableHiveSupport() \
    .getOrCreate()

store = Store.create_new_spark_store(session=spark)
```

## Time configuration mapping

The primary use case for Spark is to map datasets that are larger than can be processed by DuckDB
on one computer. In such a workflow a user would call:

```python
store.map_table_time_config(src_table_name, dst_schema, output_file="mapped_data.parquet")
```

This writes the mapped data to a Parquet file that can then be used with any backend.

## Backend differences

The Spark backend differs from DuckDB and SQLite in the following ways:

| Feature | DuckDB/SQLite | Spark |
|---------|---------------|-------|
| Data ingestion (`ingest_table`) | Yes | No |
| View from Parquet | Yes | Yes |
| Register Ibis table (`create_view`) | Yes | Yes |
| Delete rows | Yes | No |
| In-memory storage | Yes | No (views only) |
| Time mapping to Parquet | Yes | Yes |

For smaller datasets that fit in memory, DuckDB offers better performance and full feature support.
Use Spark when working with datasets too large for a single machine or when integrating with
existing Spark infrastructure.
