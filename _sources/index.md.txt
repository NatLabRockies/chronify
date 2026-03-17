# Chronify

This package implements validation, mapping, and storage of time series data in support of
Python-based modeling packages.

## Features
- Stores time series data in any database supported by SQLAlchemy.
- Supports data ingestion in a variety of file formats and configurations.
- Supports efficient retrieval of time series through SQL queries.
- Validates consistency of timestamps and resolution.
- Provides mappings between different time configurations.

```{eval-rst}
.. toctree::
    :maxdepth: 2
    :caption: Contents:
    :hidden:

    how_tos/index
    tutorials/index
    reference/index
    explanation/index
```

## Supported Backends
While chronify should work with any database supported by SQLAlchemy, it has been tested with
the following:

- DuckDB (default)
- SQLite
- Apache Spark through Apache Thrift Server

DuckDB and SQLite are fully supported.

Because of limitations in the backend software, chronify functionality with Spark is limited to
the following:

- Create a view into an existing Parquet file (or directory).
- Perform time series checks.
- Map between time configurations.
- Write output data to Parquet files.

There is no support for creating tables and ingesting data with Spark.

## How to use this guide
- Refer to [How Tos](#how-tos-page) for step-by-step instructions for creating store and ingesting data.
- Refer to [Tutorials](#tutorials-page) examples of ingesting different types of data and mapping
between time configurations.
- Refer to [Reference](#reference-page) for API reference material.
- Refer to [Explanation](#explanation-page) for descriptions and behaviors of the time series store.

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
