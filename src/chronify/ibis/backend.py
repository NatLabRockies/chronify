"""Ibis backend abstraction layer for database operations."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import ibis
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
from loguru import logger

from chronify.exceptions import InvalidOperation


class IbisBackend(ABC):
    """Abstract base class for Ibis-based database backends."""

    def __init__(self, connection: ibis.BaseBackend) -> None:
        self._connection = connection

    @property
    def connection(self) -> ibis.BaseBackend:
        """Return the underlying Ibis connection."""
        return self._connection

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend (e.g., 'duckdb', 'sqlite', 'spark')."""

    @property
    @abstractmethod
    def database(self) -> str | None:
        """Return the database path/name, or None for in-memory."""

    @abstractmethod
    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table from data.

        Parameters
        ----------
        name
            Name of the table to create
        data
            Data to populate the table
        overwrite
            If True, drop existing table first

        Returns
        -------
        ir.Table
            Ibis table reference
        """

    @abstractmethod
    def create_view(self, name: str, expr: ir.Table) -> None:
        """Create a view from an Ibis expression.

        Parameters
        ----------
        name
            Name of the view
        expr
            Ibis table expression defining the view
        """

    @abstractmethod
    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        """Create a temporary view from user-provided data.

        Parameters
        ----------
        name
            Name of the temporary view
        data
            Data for the view
        """

    @abstractmethod
    def drop_table(self, name: str, if_exists: bool = False) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop
        if_exists
            If True, don't raise error if table doesn't exist
        """

    @abstractmethod
    def drop_view(self, name: str, if_exists: bool = False) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop
        if_exists
            If True, don't raise error if view doesn't exist
        """

    @abstractmethod
    def list_tables(self) -> list[str]:
        """Return a list of all tables and views in the database."""

    @abstractmethod
    def table(self, name: str) -> ir.Table:
        """Get a reference to an existing table.

        Parameters
        ----------
        name
            Name of the table

        Returns
        -------
        ir.Table
            Ibis table reference
        """

    @abstractmethod
    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        """Execute an Ibis expression and return results as DataFrame.

        Parameters
        ----------
        expr
            Ibis expression to execute

        Returns
        -------
        pd.DataFrame
            Query results
        """

    @abstractmethod
    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        """Insert data into an existing table.

        Parameters
        ----------
        table_name
            Name of the target table
        data
            Data to insert
        """

    @abstractmethod
    def sql(self, query: str) -> ir.Table:
        """Execute raw SQL and return an Ibis table expression.

        Parameters
        ----------
        query
            SQL query string

        Returns
        -------
        ir.Table
            Query results as Ibis table
        """

    def execute_sql(self, query: str) -> None:
        """Execute raw SQL without returning results.

        Parameters
        ----------
        query
            SQL statement to execute
        """
        self._connection.raw_sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame.

        This method is useful for complex queries that may conflict with Ibis's
        SQL parsing, such as queries with CTEs.

        Parameters
        ----------
        query
            SQL query string

        Returns
        -------
        pd.DataFrame
            Query results
        """
        return self._connection.raw_sql(query).fetchdf()

    @abstractmethod
    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Write query results to a Parquet file.

        Parameters
        ----------
        expr
            Ibis table expression to write
        path
            Output path
        partition_by
            Optional columns to partition by
        overwrite
            If True, overwrite existing file/directory
        """

    @abstractmethod
    def create_view_from_parquet(self, name: str, path: Path) -> None:
        """Create a view that reads from a Parquet file.

        Parameters
        ----------
        name
            Name of the view to create
        path
            Path to Parquet file or directory
        """

    def has_table(self, name: str) -> bool:
        """Check if a table or view exists.

        Parameters
        ----------
        name
            Name of the table/view

        Returns
        -------
        bool
            True if exists
        """
        return name in self.list_tables()

    def dispose(self) -> None:
        """Clean up the backend connection."""
        pass

    def reconnect(self) -> None:
        """Reconnect to the database after dispose."""
        pass

    @contextmanager
    def transaction(self) -> Iterator[list[tuple[str, str]]]:
        """Context manager for pseudo-transaction support.

        Ibis lacks built-in transaction support, so we track created objects
        and clean them up on failure.

        Yields
        ------
        list[tuple[str, str]]
            List to track created objects as (type, name) tuples.
            Callers should append to this list when creating tables/views.
        """
        created_objects: list[tuple[str, str]] = []
        try:
            yield created_objects
        except Exception:
            # Rollback by dropping created objects in reverse order
            for obj_type, name in reversed(created_objects):
                try:
                    if obj_type == "table":
                        self.drop_table(name, if_exists=True)
                    elif obj_type == "view":
                        self.drop_view(name, if_exists=True)
                except Exception as e:
                    logger.warning("Failed to rollback {} {}: {}", obj_type, name, e)
            raise


class DuckDBBackend(IbisBackend):
    """DuckDB backend implementation."""

    def __init__(
        self,
        file_path: Optional[Path | str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a DuckDB backend.

        Parameters
        ----------
        file_path
            Path to database file, or None for in-memory
        **kwargs
            Additional arguments passed to ibis.duckdb.connect
        """
        self._file_path = str(file_path) if file_path else ":memory:"
        connection = ibis.duckdb.connect(
            self._file_path,
            **kwargs,
        )
        super().__init__(connection)

    @property
    def name(self) -> str:
        return "duckdb"

    @property
    def database(self) -> str | None:
        return None if self._file_path == ":memory:" else self._file_path

    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        if overwrite and self.has_table(name):
            self.drop_table(name, if_exists=True)

        if isinstance(data, ir.Table):
            # Execute the expression and create table from result
            self._connection.raw_sql(f"CREATE TABLE {name} AS {data.compile()}")
        else:
            self._connection.create_table(name, data)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._connection.raw_sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"CREATE OR REPLACE TEMP VIEW {name} AS {sql}")
        else:
            # Register as a table then create view
            temp_name = f"_temp_{name}"
            self._connection.create_table(temp_name, data, temp=True)
            self._connection.raw_sql(
                f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {temp_name}"
            )

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP TABLE {if_exists_sql}{name}")

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP VIEW {if_exists_sql}{name}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_")
            and not t.startswith("ibis_pyarrow_memtable_")
            and not t.startswith("_temp_")
            and not t.startswith("_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return expr.execute()

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"INSERT INTO {table_name} {sql}")
        elif isinstance(data, pa.Table):
            # Register PyArrow table as a virtual table, insert, then unregister
            self._connection.con.register("__temp_insert_df", data)
            self._connection.con.execute(
                f"INSERT INTO {table_name} SELECT * FROM __temp_insert_df"
            )
            self._connection.con.unregister("__temp_insert_df")
        else:
            # pandas DataFrame - register as a virtual table, insert, then unregister
            self._connection.con.register("__temp_insert_df", data)
            self._connection.con.execute(
                f"INSERT INTO {table_name} SELECT * FROM __temp_insert_df"
            )
            self._connection.con.unregister("__temp_insert_df")

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        sql = expr.compile()
        if partition_by:
            cols = ",".join(partition_by)
            query = f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, PARTITION_BY ({cols}))"
        else:
            query = f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)"
        self._connection.raw_sql(query)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        str_path = f"{path}/**/*.parquet" if path.is_dir() else str(path)
        query = f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{str_path}')"
        self._connection.raw_sql(query)

    def dispose(self) -> None:
        """Close the DuckDB connection to flush data to disk."""
        self._connection.con.close()

    def reconnect(self) -> None:
        """Reconnect to the database after dispose."""
        self._connection = ibis.duckdb.connect(self._file_path)


class SQLiteBackend(IbisBackend):
    """SQLite backend implementation."""

    def __init__(
        self,
        file_path: Optional[Path | str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a SQLite backend.

        Parameters
        ----------
        file_path
            Path to database file, or None for in-memory
        **kwargs
            Additional arguments passed to ibis.sqlite.connect
        """
        self._file_path = str(file_path) if file_path else ":memory:"
        connection = ibis.sqlite.connect(
            self._file_path if self._file_path != ":memory:" else None,
            **kwargs,
        )
        super().__init__(connection)

    @property
    def name(self) -> str:
        return "sqlite"

    @property
    def database(self) -> str | None:
        return None if self._file_path == ":memory:" else self._file_path

    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        if overwrite and self.has_table(name):
            self.drop_table(name, if_exists=True)

        if isinstance(data, ir.Table):
            # SQLite loses datetime type info with CREATE TABLE AS SELECT
            # Execute the query first and create from DataFrame to preserve types
            df = data.execute()
            self._connection.create_table(name, df)
        else:
            self._connection.create_table(name, data)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._connection.raw_sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"CREATE TEMP VIEW {name} AS {sql}")
        else:
            # Create temp table first
            self._connection.create_table(name, data, temp=True)

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP TABLE {if_exists_sql}{name}")

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._connection.raw_sql(f"DROP VIEW {if_exists_sql}{name}")
        # Also drop the backing table if it exists (created by create_view_from_parquet)
        backing_table = f"_parquet_backing_{name}"
        self._connection.raw_sql(f"DROP TABLE IF EXISTS {backing_table}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables and parquet backing tables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_") and not t.startswith("_parquet_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        return expr.execute()

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._connection.raw_sql(f"INSERT INTO {table_name} {sql}")
        else:
            # For pandas/pyarrow, use parameterized INSERT
            if isinstance(data, pa.Table):
                df = data.to_pandas()
            else:
                df = data
            # Build parameterized INSERT statement
            columns = ",".join(df.columns)
            placeholders = ",".join(["?" for _ in df.columns])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            # Access SQLite's raw connection
            cursor = self._connection.con.cursor()
            cursor.executemany(query, df.values.tolist())
            self._connection.con.commit()

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame."""
        cursor = self._connection.raw_sql(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if partition_by:
            msg = "SQLite does not support partitioned Parquet writes"
            raise InvalidOperation(msg)
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        # Execute and write via pandas/pyarrow
        df = self.execute(expr)
        df.to_parquet(path)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        # SQLite cannot directly read Parquet, so we load data into a backing table
        # and create a view that selects from it
        df = pd.read_parquet(path)
        backing_table = f"_parquet_backing_{name}"
        self._connection.create_table(backing_table, df)
        self._connection.raw_sql(f"CREATE VIEW {name} AS SELECT * FROM {backing_table}")


class SparkBackend(IbisBackend):
    """Spark backend implementation."""

    def __init__(
        self,
        session: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Create a Spark backend.

        Parameters
        ----------
        session
            Existing SparkSession, or None to create a new one
        **kwargs
            Additional arguments passed to ibis.pyspark.connect
        """
        try:
            import pyspark  # noqa: F401
        except ImportError:
            msg = (
                "PySpark is required for PySparkBackend. Install with: pip install chronify[spark]"
            )
            raise ImportError(msg)

        if session is None:
            from pyspark.sql import SparkSession

            session = SparkSession.builder.getOrCreate()

        connection = ibis.pyspark.connect(session, **kwargs)
        super().__init__(connection)
        self._session = session

        # Set timezone to UTC for consistency
        session.conf.set("spark.sql.session.timeZone", "UTC")
        # Use microsecond precision for Parquet timestamps
        session.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")

    @property
    def name(self) -> str:
        return "spark"

    @property
    def database(self) -> str | None:
        return None

    def create_table(
        self,
        name: str,
        data: pd.DataFrame | pa.Table | ir.Table,
        overwrite: bool = False,
    ) -> ir.Table:
        mode: Literal["overwrite", "error"] = "overwrite" if overwrite else "error"
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"CREATE TABLE {name} AS {sql}")
        elif isinstance(data, pa.Table):
            spark_df = self._session.createDataFrame(data.to_pandas())
            spark_df.write.saveAsTable(name, mode=mode)
        else:
            spark_df = self._session.createDataFrame(data)
            spark_df.write.saveAsTable(name, mode=mode)

        return self.table(name)

    def create_view(self, name: str, expr: ir.Table) -> None:
        sql = expr.compile()
        self._session.sql(f"CREATE VIEW {name} AS {sql}")

    def create_temp_view(self, name: str, data: ir.Table | pd.DataFrame | pa.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS {sql}")
        elif isinstance(data, pa.Table):
            spark_df = self._session.createDataFrame(data.to_pandas())
            spark_df.createOrReplaceTempView(name)
        else:
            spark_df = self._session.createDataFrame(data)
            spark_df.createOrReplaceTempView(name)

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._session.sql(f"DROP TABLE {if_exists_sql}{name}")

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        if_exists_sql = "IF EXISTS " if if_exists else ""
        self._session.sql(f"DROP VIEW {if_exists_sql}{name}")

    def list_tables(self) -> list[str]:
        # Filter out internal Ibis memtables
        return [
            t
            for t in self._connection.list_tables()
            if not t.startswith("ibis_pandas_memtable_")
            and not t.startswith("ibis_pyarrow_memtable_")
            and not t.startswith("_temp_")
            and not t.startswith("_backing_")
        ]

    def table(self, name: str) -> ir.Table:
        return self._connection.table(name)

    def execute(self, expr: ir.Expr) -> pd.DataFrame:
        # Spark's toPandas() has DST issues when converting timestamps.
        # Work around by using Spark SQL to convert timestamps to strings,
        # then convert back to pandas timestamps after toPandas().
        if isinstance(expr, ir.Table):
            return self._execute_table_with_timestamp_fix(expr)
        return expr.execute()

    def _execute_table_with_timestamp_fix(self, expr: ir.Table) -> pd.DataFrame:
        """Execute a table expression with workaround for Spark timestamp DST issues."""
        schema = expr.schema()
        timestamp_cols = [name for name, dtype in schema.items() if dtype.is_timestamp()]

        if not timestamp_cols:
            return expr.execute()

        # Convert timestamp columns to strings in the query
        sql = expr.compile()
        select_exprs = []
        for col_name in schema.names:
            if col_name in timestamp_cols:
                # Cast to string to avoid DST conversion issues
                select_exprs.append(f"CAST({col_name} AS STRING) AS {col_name}")
            else:
                select_exprs.append(col_name)

        query = f"SELECT {', '.join(select_exprs)} FROM ({sql})"
        df = self._session.sql(query).toPandas()

        # Convert string columns back to timestamps
        for col_name in timestamp_cols:
            df[col_name] = pd.to_datetime(df[col_name])

        return df

    def insert(self, table_name: str, data: pd.DataFrame | pa.Table | ir.Table) -> None:
        if isinstance(data, ir.Table):
            sql = data.compile()
            self._session.sql(f"INSERT INTO {table_name} {sql}")
        elif isinstance(data, pa.Table):
            spark_df = self._session.createDataFrame(data.to_pandas())
            spark_df.write.insertInto(table_name)
        else:
            spark_df = self._session.createDataFrame(data)
            spark_df.write.insertInto(table_name)

    def sql(self, query: str) -> ir.Table:
        return self._connection.sql(query)

    def execute_sql_to_df(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return results as a DataFrame.

        Uses string conversion for timestamp columns to avoid Spark's
        DST conversion issues during toPandas().
        """
        spark_df = self._session.sql(query)

        # Check for timestamp columns
        timestamp_cols = [
            field.name
            for field in spark_df.schema.fields
            if "timestamp" in field.dataType.simpleString().lower()
        ]

        if not timestamp_cols:
            return spark_df.toPandas()

        # Convert timestamp columns to strings to avoid DST issues
        from pyspark.sql.functions import col

        select_exprs = []
        for field in spark_df.schema.fields:
            if field.name in timestamp_cols:
                select_exprs.append(col(field.name).cast("string").alias(field.name))
            else:
                select_exprs.append(col(field.name))

        df = spark_df.select(*select_exprs).toPandas()

        # Convert string columns back to timestamps
        for col_name in timestamp_cols:
            df[col_name] = pd.to_datetime(df[col_name])

        return df

    def write_parquet(
        self,
        expr: ir.Table,
        path: Path,
        partition_by: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and path.exists():
            msg = f"File already exists: {path}"
            raise InvalidOperation(msg)

        sql = expr.compile()
        if partition_by:
            cols = ",".join(partition_by)
            # Spark uses INSERT OVERWRITE DIRECTORY
            query = (
                f"INSERT OVERWRITE DIRECTORY '{path}' USING parquet PARTITIONED BY ({cols}) {sql}"
            )
        else:
            query = f"INSERT OVERWRITE DIRECTORY '{path}' USING parquet {sql}"
        self._session.sql(query)

    def create_view_from_parquet(self, name: str, path: Path) -> None:
        query = f"CREATE VIEW {name} AS SELECT * FROM parquet.`{path}`"
        self._session.sql(query)


def make_backend(
    backend_name: str = "duckdb",
    file_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> IbisBackend:
    """Factory function to create an IbisBackend instance.

    Parameters
    ----------
    backend_name
        Name of the backend: "duckdb", "sqlite", or "spark"
    file_path
        Path to database file (for duckdb/sqlite), or None for in-memory
    **kwargs
        Additional arguments passed to the backend constructor

    Returns
    -------
    IbisBackend
        Backend instance

    Examples
    --------
    >>> backend = make_backend("duckdb")  # In-memory DuckDB
    >>> backend = make_backend("duckdb", file_path="data.db")  # File-based
    >>> backend = make_backend("sqlite")  # In-memory SQLite
    >>> backend = make_backend("spark")  # Spark (requires spark extra)
    """
    match backend_name:
        case "duckdb":
            return DuckDBBackend(file_path=file_path, **kwargs)
        case "sqlite":
            return SQLiteBackend(file_path=file_path, **kwargs)
        case "spark":
            return SparkBackend(**kwargs)
        case _:
            msg = f"Unknown backend: {backend_name}. Supported: duckdb, sqlite, spark"
            raise NotImplementedError(msg)


# Backwards compatibility alias
get_backend = make_backend
