from collections.abc import Iterable
from pathlib import Path
import shutil
from typing import Any, Optional
from chronify.utils.sql import make_temp_view_name
from datetime import tzinfo

import duckdb
import pandas as pd
from duckdb import DuckDBPyRelation
from loguru import logger
from sqlalchemy import (
    Column,
    Connection,
    Engine,
    MetaData,
    Selectable,
    Table,
    create_engine,
    delete,
    func,
    select,
    text,
)

import chronify.duckdb.functions as ddbf
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidOperation,
    InvalidParameter,
    InvalidTable,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.csv_io import read_csv
from chronify.models import (
    CsvTableSchema,
    PivotedTableSchema,
    TableSchema,
    get_duckdb_types_from_pandas,
    get_sqlalchemy_type_from_duckdb,
)
from chronify.sqlalchemy.functions import (
    create_view_from_parquet,
    read_database,
    write_database,
    write_query_to_parquet,
)
from chronify.schema_manager import SchemaManager
from chronify.time_configs import DatetimeRange, IndexTimeRangeBase, TimeBasedDataAdjustment
from chronify.time_series_checker import check_timestamps
from chronify.time_series_mapper import map_time
from chronify.time_zone_converter import TimeZoneConverter, TimeZoneConverterByColumn
from chronify.time_zone_localizer import TimeZoneLocalizer, TimeZoneLocalizerByColumn
from chronify.utils.path_utils import check_overwrite, to_path
from chronify.utils.sqlalchemy_view import create_view


class Store:
    """Data store for time series data"""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_name: Optional[str] = None,
        file_path: Optional[Path | str] = None,
        **connect_kwargs: Any,
    ) -> None:
        """Construct the Store.

        Parameters
        ----------
        engine
            Optional, defaults to a engine connected to an in-memory DuckDB database.
        engine_name
            Optional, name of engine to use ('duckdb', 'sqlite'). Mutually exclusive with engine.
        file_path
            Optional, use this file for the database. If the file does not exist, create a new
            database. If the file exists, load that existing database.
            Defaults to a new in-memory database.

        Examples
        --------
        >>> from sqlalchemy
        >>> store1 = Store()
        >>> store2 = Store(engine=Engine("duckdb:///time_series.db"))
        >>> store3 = Store(engine=Engine("sqlite:///time_series.db"))
        >>> store4 = Store(engine_name="sqlite")
        """
        self._metadata = MetaData()
        if engine and engine_name:
            msg = f"{engine=} and {engine_name=} cannot both be set"
            raise ConflictingInputsError(msg)
        filename = ":memory:" if file_path is None else str(file_path)
        if engine is None:
            name = engine_name or "duckdb"
            match name:
                case "duckdb" | "sqlite":
                    engine_path = f"{name}:///{filename}"
                case _:
                    msg = f"{engine_name=}"
                    raise NotImplementedError(msg)
            self._engine = create_engine(engine_path, **connect_kwargs)
        else:
            self._engine = engine

        self._schema_mgr = SchemaManager(self._engine, self._metadata)
        if self._engine.url.database != ":memory:":
            self.update_metadata()

    @classmethod
    def create_in_memory_db(
        cls,
        engine_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with an in-memory database."""
        return Store(engine=create_engine(f"{engine_name}:///:memory:", **connect_kwargs))

    @classmethod
    def create_file_db(
        cls,
        file_path: Path | str = "time_series.db",
        engine_name: str = "duckdb",
        overwrite: bool = False,
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with a file-based database."""
        path = to_path(file_path)
        check_overwrite(path, overwrite)
        return Store(engine=create_engine(f"{engine_name}:///{path}", **connect_kwargs))

    @classmethod
    def create_new_hive_store(
        cls,
        url: str,
        drop_schema: bool = True,
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a new Store in a Hive database.
        Recommended usage is to create views from Parquet files. Ingesting data into tables
        from files or DataFrames is not supported.

        This has been tested with Apache Spark running an Apache Thrift Server.

        Parameters
        ----------
        url
            Thrift server URL
        drop_schema
            If True, drop the schema table if it's already there.

        Examples
        --------
        >>> store = Store.create_new_hive_store("hive://localhost:10000/default")

        See also
        --------
        create_view_from_parquet
        """
        # We don't currently expect to need to load an existing hive-based store, but it could
        # be added.
        if "hive://" not in url:
            msg = f"Expected 'hive://' to be in url: {url}"
            raise InvalidParameter(msg)
        engine = create_engine(url, **connect_kwargs)
        metadata = MetaData()
        metadata.reflect(engine, views=True)
        with engine.begin() as conn:
            # Workaround for ambiguity of time zones in the read path.
            conn.execute(text("SET TIME ZONE 'UTC'"))
            # Workaround for the fact that Spark uses a non-standard format for timestamps
            # in Parquet files. Pandas/DuckDB can't interpret them properly.
            conn.execute(text("SET spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS"))

            if drop_schema:
                if SchemaManager.SCHEMAS_TABLE in metadata.tables:
                    conn.execute(text(f"DROP TABLE {SchemaManager.SCHEMAS_TABLE}"))

        return cls(engine=engine)

    @classmethod
    def load_from_file(
        cls,
        file_path: Path | str,
        engine_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Load an existing store from a database."""
        path = to_path(file_path)
        if not path.exists():
            msg = str(path)
            raise FileNotFoundError(msg)
        return Store(engine=create_engine(f"{engine_name}:///{path}", **connect_kwargs))

    def dispose(self) -> None:
        """Call self.engine.dispose() in order to dispose of the current connections."""
        self._engine.dispose()

    def get_table(self, name: str) -> Table:
        """Return the sqlalchemy Table object."""
        if not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)

        return Table(name, self._metadata)

    def has_table(self, name: str) -> bool:
        """Return True if the database has a table with the given name."""
        return name in self._metadata.tables

    def list_tables(self) -> list[str]:
        """Return a list of user tables in the database."""
        return [x for x in self._metadata.tables if x != SchemaManager.SCHEMAS_TABLE]

    def try_get_table(self, name: str) -> Table | None:
        """Return the sqlalchemy Table object or None if it is not stored."""
        if not self.has_table(name):
            return None
        return Table(name, self._metadata)

    def update_metadata(self) -> None:
        """Update the sqlalchemy metadata for table schema. Call this method if you add tables
        in the sqlalchemy engine outside of this class or perform a rollback
        in the same transaction in which chronify added tables.
        """
        # Create a new object because sqlalchemy does not detect dropped tables in reflect.
        metadata = MetaData()
        metadata.reflect(self._engine, views=True)
        logger.trace(
            "Updated metadata, added: {}, dropped: {}",
            sorted(set(metadata.tables).difference(self._metadata.tables)),
            sorted(set(self._metadata.tables).difference(metadata.tables)),
        )
        self._metadata = metadata
        self._schema_mgr.rebuild_cache()

    def backup(self, dst: Path | str, overwrite: bool = False) -> None:
        """Copy the database to a new location. Not yet supported for in-memory databases."""
        self._engine.dispose()
        path = to_path(dst)
        check_overwrite(path, overwrite)
        match self._engine.name:
            case "duckdb" | "sqlite":
                if self._engine.url.database is None or self._engine.url.database == ":memory:":
                    msg = "backup is only supported with a database backed by a file"
                    raise InvalidOperation(msg)
                src_file = Path(self._engine.url.database)
                shutil.copyfile(src_file, path)
                logger.info("Copied database to {}", path)
            case _:
                msg = self._engine.name
                raise NotImplementedError(msg)

    @property
    def engine(self) -> Engine:
        """Return the sqlalchemy engine."""
        return self._engine

    @property
    def metadata(self) -> MetaData:
        """Return the sqlalchemy metadata."""
        return self._metadata

    @property
    def schema_manager(self) -> SchemaManager:
        """Return the store's schema manager."""
        return self._schema_mgr

    def check_timestamps(self, name: str, connection: Connection | None = None) -> None:
        """Check the timestamps in the table.

        This is useful if you call a :meth:`ingest_table` many times with skip_time_checks=True
        and then want to check the final table.

        Parameters
        ----------
        name
            Name of the table to check.

        Raises
        ------
        InvalidTable
            Raised if the timestamps do not match the schema.
        """
        table = self.get_table(name)
        schema = self._schema_mgr.get_schema(name)
        if connection is None:
            with self._engine.connect() as conn:
                check_timestamps(conn, table, schema)
        else:
            check_timestamps(connection, table, schema)

    def create_view_from_parquet(
        self, path: Path, schema: TableSchema, bypass_checks: bool = False
    ) -> None:
        """Load a table into the database."""
        self._create_view_from_parquet(path, schema)
        try:
            with self._engine.connect() as conn:
                table = self.get_table(schema.name)
                if not bypass_checks:
                    check_timestamps(conn, table, schema)
        except InvalidTable:
            # This doesn't use conn.rollback because we can't update the sqlalchemy metadata
            # for this view inside the connection.
            self.drop_view(schema.name)
            raise

    def _create_view_from_parquet(self, path: Path | str, schema: TableSchema) -> None:
        """Create a view in the database from a Parquet file.

        Parameters
        ----------
        schema
            Defines the schema of the view to create in the database. Must match the input data.
        path
            Path to Parquet file.

        Raises
        ------
        InvalidTable
            Raised if the schema does not match the input data.

        Examples
        --------
        >>> store = Store()
        >>> store.create_view_from_parquet(
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ...     "table.parquet",
        ... )
        """
        with self._engine.begin() as conn:
            create_view_from_parquet(conn, schema.name, to_path(path))
            self._schema_mgr.add_schema(conn, schema)

        self.update_metadata()

    def ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> bool:
        """Ingest data from a CSV file.

        Parameters
        ----------
        path
            Source data file
        src_schema
            Defines the schema of the source file.
        dst_schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Returns
        -------
        bool
            Return True if a table was created.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> resolution = timedelta(hours=1)
        >>> time_config = DatetimeRange(
        ...     time_column="timestamp",
        ...     start=datetime(2020, 1, 1, 0),
        ...     length=8784,
        ...     resolution=timedelta(hours=1),
        ... )
        >>> store = Store()
        >>> store.ingest_from_csv(
        ...     "data.csv",
        ...     CsvTableSchema(
        ...         time_config=time_config,
        ...         pivoted_dimension_name="device",
        ...         value_columns=["device1", "device2", "device3"],
        ...     ),
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=time_config,
        ...         time_array_id_columns=["device"],
        ...     ),
        ... )

        See Also
        --------
        ingest_from_csvs
        """
        return self.ingest_from_csvs((path,), src_schema, dst_schema, connection=connection)

    def ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> bool:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it. This is faster than calling :meth:`ingest_from_csv` many times.
        Each file is loaded into memory one at a time.
        If any error occurs, all added data will be removed and the state of the database will
        be the same as the original state.

        Parameters
        ----------
        path
            Source data files
        src_schema
            Defines the schema of the source files.
        dst_schema
            Defines the destination table in the database.
        conn
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Returns
        -------
        bool
            Return True if a table was created.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        See Also
        --------
        ingest_from_csv
        """
        try:
            if connection is None:
                with self._engine.begin() as conn:
                    created_table = self._ingest_from_csvs(conn, paths, src_schema, dst_schema)
            else:
                created_table = self._ingest_from_csvs(connection, paths, src_schema, dst_schema)
        except Exception:
            # TODO:
            # 1. The implicit rollback does not remove tables from our sqlalchemy metadata object.
            #    This means that the metadata object could be out-of-date if the user
            #    is self-managing the connection.
            # 2. Python sqlite3 does not appear to support rollbacks with DDL statements.
            #    See discussion at https://bugs.python.org/issue10740.
            self._handle_sqlite_error_case(dst_schema.name, connection)
            if dst_schema.name in self._metadata.tables:
                self._metadata.remove(Table(dst_schema.name, self._metadata))
            raise

        return created_table

    def _ingest_from_csvs(
        self,
        conn: Connection,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        created_table = False
        if not paths:
            return created_table

        for path in paths:
            if self._ingest_from_csv(conn, path, src_schema, dst_schema):
                created_table = True
        table = Table(dst_schema.name, self._metadata)
        check_timestamps(conn, table, dst_schema)
        return created_table

    def _ingest_from_csv(
        self,
        conn: Connection,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        rel = read_csv(path, src_schema)
        columns = set(src_schema.list_columns())
        check_columns(rel.columns, columns)

        # TODO
        if isinstance(src_schema.time_config, IndexTimeRangeBase):
            if isinstance(dst_schema.time_config, DatetimeRange):
                raise NotImplementedError
                # timestamps = IndexTimeRangeGenerator(src_schema.time_config).list_timestamps()
                # rel = ddbf.add_datetime_column(
                #    rel=rel,
                #    start=dst_schema.time_config.start,
                #    resolution=dst_schema.time_config.resolution,
                #    length=dst_schema.time_config.length,
                #    time_array_id_columns=src_schema.time_array_id_columns,
                #    time_column=dst_schema.time_config.time_column,
                #    timestamps=timestamps,
                # )
            else:
                cls_name = dst_schema.time_config.__class__.__name__
                msg = f"{src_schema.time_config.__class__.__name__} cannot be converted to {cls_name}"
                raise NotImplementedError(msg)

        if src_schema.pivoted_dimension_name is not None:
            return self._ingest_pivoted_table(conn, rel, src_schema, dst_schema)

        return self._ingest_table(conn, rel, dst_schema)

    def ingest_pivoted_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> bool:
        """Ingest pivoted data into the table specifed by schema. If the table does not exist,
        create it. Chronify will unpivot the data before ingesting it.

        Parameters
        ----------
        data
            Input data to ingest into the database.
        src_schema
            Defines the schema of the input data.
        dst_schema
            Defines the destination table in the database.
        conn
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Returns
        -------
        bool
            Return True if a table was created.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> resolution = timedelta(hours=1)
        >>> df = pd.DataFrame(
        ...     {
        ...         "timestamp": pd.date_range(
        ...             "2020-01-01", "2020-12-31 23:00:00", freq=resolution
        ...         ),
        ...         "device1": np.random.random(8784),
        ...         "device2": np.random.random(8784),
        ...         "device3": np.random.random(8784),
        ...     }
        ... )
        >>> time_config = DatetimeRange(
        ...     time_column="timestamp",
        ...     start=datetime(2020, 1, 1, 0),
        ...     length=8784,
        ...     resolution=timedelta(hours=1),
        ... )
        >>> store = Store()
        >>> store.ingest_pivoted_table(
        ...     df,
        ...     PivotedTableSchema(
        ...         time_config=time_config,
        ...         pivoted_dimension_name="device",
        ...         value_columns=["device1", "device2", "device3"],
        ...     ),
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=time_config,
        ...         time_array_id_columns=["device"],
        ...     ),
        ... )

        See Also
        --------
        ingest_pivoted_tables
        """
        return self.ingest_pivoted_tables((data,), src_schema, dst_schema, connection=connection)

    def ingest_pivoted_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
        connection: Optional[Connection] = None,
    ) -> bool:
        """Ingest pivoted data into the table specifed by schema.

        If the table does not exist, create it. Unpivot the data before ingesting it.
        This is faster than calling :meth:`ingest_pivoted_table` many times.
        If any error occurs, all added data will be removed and the state of the database will
        be the same as the original state.

        Parameters
        ----------
        data
            Data to ingest into the database.
        src_schema
            Defines the schema of all input tables.
        dst_schema
            Defines the destination table in the database.
        conn
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Returns
        -------
        bool
            Return True if a table was created.

        See Also
        --------
        ingest_pivoted_table
        """
        try:
            if connection is None:
                with self._engine.begin() as conn:
                    created_table = self._ingest_pivoted_tables(conn, data, src_schema, dst_schema)
            else:
                created_table = self._ingest_pivoted_tables(
                    connection, data, src_schema, dst_schema
                )
        except Exception:
            self._handle_sqlite_error_case(dst_schema.name, connection)
            if dst_schema.name in self._metadata.tables:
                self._metadata.remove(Table(dst_schema.name, self._metadata))
            raise

        return created_table

    def _ingest_pivoted_tables(
        self,
        conn: Connection,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        created_table = False
        for table in data:
            if self._ingest_pivoted_table(conn, table, src_schema, dst_schema):
                created_table = True
        check_timestamps(conn, Table(dst_schema.name, self._metadata), dst_schema)
        return created_table

    def _ingest_pivoted_table(
        self,
        conn: Connection,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        if isinstance(data, pd.DataFrame):
            # This is a shortcut for registering a temporary view.
            tmp_df = data  # noqa: F841
            rel = duckdb.sql("SELECT * from tmp_df")
        else:
            rel = data

        assert src_schema.pivoted_dimension_name is not None
        rel2 = ddbf.unpivot(
            rel,
            src_schema.value_columns,
            src_schema.pivoted_dimension_name,
            dst_schema.value_column,
        )
        return self._ingest_table(conn, rel2, dst_schema)

    def ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
        connection: Optional[Connection] = None,
        **kwargs: Any,
    ) -> bool:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it.

        Parameters
        ----------
        data
            Input data to ingest into the database.
        schema
            Defines the destination table in the database.
        connection
            Optional connection to reuse. If adding many tables at once, it is significantly
            faster to use one connection. Refer to :meth:`ingest_tables` for built-in support.
            If connection is not set, chronify will commit the database changes
            or perform a rollback on error. If it is set, the caller must perform those actions.
            If you peform a rollback, you must call :meth:`rebuild_schema_cache` because the
            Store will cache all table names in memory.

        Returns
        -------
        bool
            Return True if a table was created.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        Examples
        --------
        >>> store = Store()
        >>> resolution = timedelta(hours=1)
        >>> df = pd.DataFrame(
        ...     {
        ...         "timestamp": pd.date_range(
        ...             "2020-01-01", "2020-12-31 23:00:00", freq=resolution
        ...         ),
        ...         "value": np.random.random(8784),
        ...     }
        ... )
        >>> df["id"] = 1
        >>> store.ingest_table(
        ...     df,
        ...     TableSchema(
        ...         name="devices",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ... )

        See Also
        --------
        ingest_tables
        """
        return self.ingest_tables((data,), schema, connection=connection, **kwargs)

    def ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
        connection: Optional[Connection] = None,
        **kwargs: Any,
    ) -> bool:
        """Ingest multiple input tables to the same database table.
        All tables must have the same schema.
        This offers significant performance advantages over calling :meth:`ingest_table` many
        times.

        Parameters
        ----------
        data
            Input tables to ingest into one database table.
        schema
            Defines the destination table.
        conn
            Optional connection to reuse. Refer to :meth:`ingest_table` for notes.

        Returns
        -------
        bool
            Return True if a table was created.

        Raises
        ------
        InvalidTable
            Raised if the data does not match the schema.

        See Also
        --------
        ingest_table
        """
        created_table = False
        if not data:
            return created_table

        try:
            if connection is None:
                with self._engine.begin() as conn:
                    created_table = self._ingest_tables(conn, data, schema, **kwargs)
            else:
                created_table = self._ingest_tables(connection, data, schema, **kwargs)
        except Exception:
            self._handle_sqlite_error_case(schema.name, connection)
            if schema.name in self._metadata.tables:
                self._metadata.remove(Table(schema.name, self._metadata))
            raise

        return created_table

    def _ingest_tables(
        self,
        conn: Connection,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
        skip_time_checks: bool = False,
    ) -> bool:
        created_table = False
        for table in data:
            if self._ingest_table(conn, table, schema):
                created_table = True
        if not skip_time_checks:
            check_timestamps(conn, Table(schema.name, self._metadata), schema)
        return created_table

    def _ingest_table(
        self,
        conn: Connection,
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
    ) -> bool:
        if self._engine.name == "hive":
            msg = "Data ingestion through Hive is not supported"
            raise NotImplementedError(msg)
        df = data.to_df() if isinstance(data, DuckDBPyRelation) else data
        check_columns(df.columns, schema.list_columns())

        table = self.try_get_table(schema.name)
        if table is None:
            duckdb_types = get_duckdb_types_from_pandas(df)
            dtypes = [get_sqlalchemy_type_from_duckdb(x) for x in duckdb_types]
            table = Table(
                schema.name,
                self._metadata,
                *[Column(x, y) for x, y in zip(df.columns, dtypes)],
            )
            self._metadata.create_all(conn)
            created_table = True
        else:
            created_table = False

        write_database(df, conn, schema.name, [schema.time_config])

        if created_table:
            self._schema_mgr.add_schema(conn, schema)

        return created_table

    def map_table_time_config(
        self,
        src_name: str,
        dst_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Map the existing table represented by src_name to a new table represented by
        dst_schema with a different time configuration.

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        dst_schema
            Defines the table to create in the database. Must not already exist.
        data_adjustment
            Defines how the dataframe may need to be adjusted with respect to time.
            Data is only adjusted when the conditions apply.
        wrap_time_allowed
            Defines whether the time column is allowed to be wrapped according to the time
            config in dst_schema when it does not line up with the time config
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Raises
        ------
        InvalidTable
            Raised if the schemas are incompatible.
        TableAlreadyExists
            Raised if the dst_schema name already exists.

        Examples
        --------
        >>> store = Store()
        >>> hours_per_year = 12 * 7 * 24
        >>> num_time_arrays = 3
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concatenate(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "month": np.tile(np.repeat(range(1, 13), 7 * 24), num_time_arrays),
        ...         "day_of_week": np.tile(np.tile(np.repeat(range(7), 24), 12), num_time_arrays),
        ...         "hour": np.tile(np.tile(range(24), 12 * 7), num_time_arrays),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="devices_by_representative_time",
        ...     value_column="value",
        ...     time_config=RepresentativePeriodTimeNTZ(
        ...         time_format=RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR,
        ...     ),
        ...     time_array_id_columns=["id"],
        ... )
        >>> store.ingest_table(df, schema)
        >>> store.map_table_time_config(
        ...     "devices_by_representative_time",
        ...     TableSchema(
        ...         name="devices_by_datetime",
        ...         value_column="value",
        ...         time_config=DatetimeRange(
        ...             time_column="timestamp",
        ...             start=datetime(2020, 1, 1, 0),
        ...             length=8784,
        ...             resolution=timedelta(hours=1),
        ...         ),
        ...         time_array_id_columns=["id"],
        ...     ),
        ... )
        """
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        src_schema = self._schema_mgr.get_schema(src_name)
        map_time(
            self._engine,
            self._metadata,
            src_schema,
            dst_schema,
            data_adjustment=data_adjustment,
            wrap_time_allowed=wrap_time_allowed,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)

    def convert_time_zone(
        self,
        src_name: str,
        time_zone: tzinfo | None,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> TableSchema:
        """
        Convert the time zone of the existing table represented by src_name to a new time zone

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        time_zone
            Time zone to convert to.
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Raises
        ------
        TableAlreadyExists
            Raised if the dst_schema name already exists.

        Examples
        --------
        >>> store = Store()
        >>> start = datetime(year=2018, month=1, day=1, tzinfo=ZoneInfo("EST"))
        >>> freq = timedelta(hours=1)
        >>> hours_per_year = 8760
        >>> num_time_arrays = 1
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concatenate(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "timestamp": np.tile(
        ...             pd.date_range(start, periods=hours_per_year, freq="h"), num_time_arrays
        ...         ),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="some_data",
        ...     time_config=DatetimeRange(
        ...         time_column="timestamp",
        ...         start=start,
        ...         length=hours_per_year,
        ...         resolution=freq,
        ...     ),
        ...     time_array_id_columns=["id"],
        ...     value_column="value",
        ... )
        >>> store.ingest_table(df, schema)
        >>> to_time_zone = ZoneInfo("US/Mountain")
        >>> dst_schema = store.convert_time_zone(
        ...     schema.name, to_time_zone, check_mapped_timestamps=True
        ... )
        """

        src_schema = self._schema_mgr.get_schema(src_name)
        tzc = TimeZoneConverter(self._engine, self._metadata, src_schema, time_zone)

        dst_schema = tzc.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzc.convert_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)

        return dst_schema

    def convert_time_zone_by_column(
        self,
        src_name: str,
        time_zone_column: str,
        wrap_time_allowed: bool = False,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> TableSchema:
        """
        Convert the time zone of the existing table represented by src_name to new time zone(s) defined by a column

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        time_zone_column
            Name of the time zone column for conversion.
        wrap_time_allowed
            Defines whether the time column is allowed to be wrapped to reflect the same time
            range as the src_name schema in tz-naive clock time
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Raises
        ------
        TableAlreadyExists
            Raised if the dst_schema name already exists.

        Examples
        --------
        >>> store = Store()
        >>> start = datetime(year=2018, month=1, day=1, tzinfo=ZoneInfo("EST"))
        >>> freq = timedelta(hours=1)
        >>> hours_per_year = 8760
        >>> num_time_arrays = 3
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concatenate(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "timestamp": np.tile(
        ...             pd.date_range(start, periods=hours_per_year, freq="h"), num_time_arrays
        ...         ),
        ...         "time_zone": np.repeat(["US/Eastern", "US/Mountain", "None"], hours_per_year),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="some_data",
        ...     time_config=DatetimeRange(
        ...         time_column="timestamp",
        ...         start=start,
        ...         length=hours_per_year,
        ...         resolution=freq,
        ...     ),
        ...     time_array_id_columns=["id"],
        ...     value_column="value",
        ... )
        >>> store.ingest_table(df, schema)
        >>> time_zone_column = "time_zone"
        >>> dst_schema = store.convert_time_zone_by_column(
        ...     schema.name,
        ...     time_zone_column,
        ...     wrap_time_allowed=False,
        ...     check_mapped_timestamps=True,
        ... )
        """

        src_schema = self._schema_mgr.get_schema(src_name)
        tzc = TimeZoneConverterByColumn(
            self._engine, self._metadata, src_schema, time_zone_column, wrap_time_allowed
        )

        dst_schema = tzc.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzc.convert_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)

        return dst_schema

    def localize_time_zone(
        self,
        src_name: str,
        time_zone: tzinfo | None,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> TableSchema:
        """
        Localize the time zone of the existing table represented by src_name to a specified time zone

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        time_zone
            Standard time zone to localize to. If None, keep as tz-naive.
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Raises
        ------
        TableAlreadyExists
            Raised if the dst_schema name already exists.

        Returns
        -------
        TableSchema
            The schema of the newly created table.

        Examples
        --------
        >>> store = Store()
        >>> start = datetime(year=2018, month=1, day=1)  # tz-naive
        >>> freq = timedelta(hours=1)
        >>> hours_per_year = 8760
        >>> num_time_arrays = 1
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concatenate(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "timestamp": np.tile(
        ...             pd.date_range(start, periods=hours_per_year, freq="h"), num_time_arrays
        ...         ),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="some_data",
        ...     time_config=DatetimeRange(
        ...         time_column="timestamp",
        ...         start=start,
        ...         length=hours_per_year,
        ...         resolution=freq,
        ...     ),
        ...     time_array_id_columns=["id"],
        ...     value_column="value",
        ... )
        >>> store.ingest_table(df, schema)
        >>> to_time_zone = ZoneInfo("EST")
        >>> dst_schema = store.localize_time_zone(
        ...     schema.name, to_time_zone, check_mapped_timestamps=True
        ... )
        """

        src_schema = self._schema_mgr.get_schema(src_name)
        tzl = TimeZoneLocalizer(self._engine, self._metadata, src_schema, time_zone)

        dst_schema = tzl.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzl.localize_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)

        return dst_schema

    def localize_time_zone_by_column(
        self,
        src_name: str,
        time_zone_column: str,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> TableSchema:
        """
        Localize the time zone of the existing table represented by src_name to time zones defined by a column

        Parameters
        ----------
        src_name
            Refers to the table name of the source data.
        time_zone_column
            Name of the time zone column for localization.
        scratch_dir
            Directory to use for temporary writes. Default to the system's tmp filesystem.
        output_file
            If set, write the mapped table to this Parquet file.
        check_mapped_timestamps
            Perform time checks on the result of the mapping operation. This can be slow and
            is not required.

        Raises
        ------
        TableAlreadyExists
            Raised if the dst_schema name already exists.

        Returns
        -------
        TableSchema
            The schema of the newly created table.

        Examples
        --------
        >>> store = Store()
        >>> start = datetime(year=2018, month=1, day=1)  # tz-naive
        >>> freq = timedelta(hours=1)
        >>> hours_per_year = 8760
        >>> num_time_arrays = 3
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": np.concatenate(
        ...             [np.repeat(i, hours_per_year) for i in range(1, 1 + num_time_arrays)]
        ...         ),
        ...         "timestamp": np.tile(
        ...             pd.date_range(start, periods=hours_per_year, freq="h"), num_time_arrays
        ...         ),
        ...         "time_zone": np.repeat(["EST", "CST", "MST"], hours_per_year),
        ...         "value": np.random.random(hours_per_year * num_time_arrays),
        ...     }
        ... )
        >>> schema = TableSchema(
        ...     name="some_data",
        ...     time_config=DatetimeRange(
        ...         time_column="timestamp",
        ...         start=start,
        ...         length=hours_per_year,
        ...         resolution=freq,
        ...     ),
        ...     time_array_id_columns=["id"],
        ...     value_column="value",
        ... )
        >>> store.ingest_table(df, schema)
        >>> time_zone_column = "time_zone"
        >>> dst_schema = store.localize_time_zone_by_column(
        ...     schema.name,
        ...     time_zone_column,
        ...     check_mapped_timestamps=True,
        ... )
        """

        src_schema = self._schema_mgr.get_schema(src_name)
        tzl = TimeZoneLocalizerByColumn(self._engine, self._metadata, src_schema, time_zone_column)

        dst_schema = tzl.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzl.localize_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, dst_schema)

        return dst_schema

    def read_query(
        self,
        name: str,
        query: Selectable | str,
        params: Any = None,
        connection: Optional[Connection] = None,
    ) -> pd.DataFrame:
        """Return the query result as a pandas DataFrame.

        Parameters
        ----------
        name
            Table or view name
        query
            SQL query expressed as a string or salqlchemy Selectable
        params
            Parameters for SQL query if expressed as a string

        Examples
        --------
        >>> df = store.read_query("SELECT * FROM devices")
        >>> df = store.read_query("SELECT * FROM devices WHERE id = ?", params=(3,))

        >>> from sqlalchemy import select
        >>> table = store.schemas.get_table("devices")
        >>> df = store.read_query(select(table).where(table.c.id == 3)
        """
        schema = self._schema_mgr.get_schema(name, conn=connection)
        if connection is None:
            with self._engine.begin() as conn:
                return read_database(query, conn, schema.time_config, params=params)
        else:
            return read_database(query, connection, schema.time_config, params=params)

    def read_table(self, name: str, connection: Optional[Connection] = None) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        table = self.get_table(name)
        stmt = select(table)
        return self.read_query(name, stmt, connection=connection)

    def read_raw_query(
        self, query: str, params: Any = None, connection: Optional[Connection] = None
    ) -> pd.DataFrame:
        """Execute a query directly on the backend database connection, bypassing sqlalchemy, and
        return the results as a DataFrame.

        Note: Unlike :meth:`read_query`, no conversion of timestamps is performed.
        Timestamps will be in the format of the underlying database. SQLite backends will return
        strings instead of datetime.

        Parameters
        ----------
        query
            SQL query to execute
        params
            Optional parameters for SQL query
        conn
            Optional sqlalchemy connection returned by `Store.engine.connect()`. This can
            improve performance when performing many reads. If used for database modifications,
            it is the caller's responsibility to perform a commit and ensure that the connection
            is closed correctly. Use of sqlalchemy's context manager is recommended.

        Examples
        --------
        >>> store = Store()
        >>> query1 = "SELECT * from my_table WHERE column = ?"
        >>> params1 = ("value1",)
        >>> query2 = "SELECT * from my_table WHERE column = ?'"
        >>> params2 = ("value2",)

        >>> df = store.read_raw_query(query1, params=params1)

        >>> with store.engine.connect() as conn:
        ...     df1 = store.read_raw_query(query1, params=params1, connection=conn)
        ...     df2 = store.read_raw_query(query2, params=params2, connection=conn)
        """
        if connection is None:
            with self._engine.connect() as conn:
                return self._read_raw_query(query, params, conn)
        else:
            return self._read_raw_query(query, params, connection)

    def _read_raw_query(self, query: str, params: Any, conn: Connection) -> pd.DataFrame:
        assert conn._dbapi_connection is not None
        assert conn._dbapi_connection.driver_connection is not None
        match self._engine.name:
            case "duckdb":
                df = conn._dbapi_connection.driver_connection.sql(query, params=params).to_df()
                assert isinstance(df, pd.DataFrame)
                return df
            case "sqlite":
                return pd.read_sql(query, conn._dbapi_connection.driver_connection, params=params)
            case _:
                msg = self._engine.name
                raise NotImplementedError(msg)

    def write_query_to_parquet(
        self,
        stmt: Selectable,
        file_path: Path | str,
        overwrite: bool = False,
        partition_columns: Optional[list[str]] = None,
    ) -> None:
        """Write the result of a query to a Parquet file."""
        # We could add a separate path where the query is a string and skip the intermediate
        # view if we passed parameters through the call stack.
        view_name = make_temp_view_name()
        create_view(view_name, stmt, self._engine, self._metadata)
        try:
            self.write_table_to_parquet(
                view_name, file_path, overwrite=overwrite, partition_columns=partition_columns
            )
        finally:
            with self._engine.connect() as conn:
                conn.execute(text(f"DROP VIEW {view_name}"))
            self._metadata.remove(Table(view_name, self._metadata))

    def write_table_to_parquet(
        self,
        name: str,
        file_path: Path | str,
        partition_columns: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Write a table or view to a Parquet file."""
        if not self.has_table(name):
            msg = f"table {name=} is not stored"
            raise TableNotStored(msg)

        write_query_to_parquet(
            self._engine,
            f"SELECT * FROM {name}",
            to_path(file_path),
            overwrite=overwrite,
            partition_columns=partition_columns,
        )
        logger.info("Wrote table or view to {}", file_path)

    def delete_rows(
        self,
        name: str,
        time_array_id_values: dict[str, Any],
        connection: Optional[Connection] = None,
    ) -> int:
        """Delete all rows matching the time_array_id_values.

        Parameters
        ----------
        name
            Name of table
        time_array_id_values
            Values for the time_array_id_values. Keys must match the columns in the schema.
        connnection
            Optional connection to the database. Refer :meth:`ingest_table` for notes.

        Returns
        -------
        int
            Number of deleted rows

        Examples
        --------
        >>> store.delete_rows("devices", {"id": 47})
        """
        # TODO: consider supporting a user-defined query. Would need to check consistency
        # afterwards.
        # The current approach doesn't need to check because only one single complete time
        # array can be deleted on each call.
        table = self.get_table(name)
        if not time_array_id_values:
            msg = "time_array_id_values cannot be empty"
            raise InvalidParameter(msg)

        schema = self._schema_mgr.get_schema(name, conn=connection)
        if sorted(time_array_id_values.keys()) != sorted(schema.time_array_id_columns):
            msg = (
                "The keys of time_array_id_values must match the schema columns. "
                f"Passed = {sorted(time_array_id_values.keys())} "
                f"Schema = {sorted(schema.time_array_id_columns)}"
            )
            raise InvalidParameter(msg)

        assert time_array_id_values
        stmt = delete(table)

        # duckdb does not offer a way to retrieve the number of deleted rows, so we must
        # compute it manually.
        # Deletions are not common. We are trading accuracy for peformance.
        count_stmt = select(func.count()).select_from(table)

        for column, value in time_array_id_values.items():
            stmt = stmt.where(table.c[column] == value)
            count_stmt = count_stmt.where(table.c[column] == value)

        if connection is None:
            with self._engine.begin() as conn:
                num_deleted = self._run_delete(conn, stmt, count_stmt)
        else:
            num_deleted = self._run_delete(connection, stmt, count_stmt)
            # Let the caller commit or rollback when ready.

        if num_deleted < 1:
            msg = f"Failed to delete rows: {stmt=} {num_deleted=}"
            raise InvalidParameter(msg)

        logger.info(
            "Delete all rows from table {} with time_array_id_values {}",
            name,
            time_array_id_values,
        )

        stmt2 = select(table).limit(1)
        is_empty = False
        if connection is None:
            with self._engine.connect() as conn:
                res = conn.execute(stmt2).fetchall()
        else:
            res = connection.execute(stmt2).fetchall()

        if not res:
            is_empty = True

        if is_empty:
            logger.info("Delete empty table {}", name)
            self.drop_table(name, connection=connection)

        return num_deleted

    def _run_delete(self, conn: Connection, stmt: Any, count_stmt: Any) -> int:
        count1: int | None = None
        if self._engine.name == "duckdb":
            res1 = conn.execute(count_stmt).fetchone()
            assert res1 is not None
            count1 = res1[0]
        res = conn.execute(stmt)
        if self._engine.name == "duckdb":
            res2 = conn.execute(count_stmt).fetchone()
            assert res2 is not None
            count2 = res2[0]
            assert count1 is not None
            num_deleted = count1 - count2
        else:
            num_deleted = res.rowcount
        return num_deleted  # type: ignore

    def drop_table(
        self,
        name: str,
        connection: Optional[Connection] = None,
        if_exists: bool = False,
    ) -> None:
        """Drop a table from the database."""
        self._drop_table_or_view(name, "TABLE", connection, if_exists)

    def create_view(self, schema: TableSchema, stmt: Selectable) -> None:
        """Create a view in the database."""
        create_view(schema.name, stmt, self._engine, self._metadata)
        with self._engine.begin() as conn:
            self._schema_mgr.add_schema(conn, schema)

    def drop_view(
        self,
        name: str,
        connection: Optional[Connection] = None,
        if_exists: bool = False,
    ) -> None:
        """Drop a view from the database."""
        self._drop_table_or_view(name, "VIEW", connection, if_exists)

    def _drop_table_or_view(
        self,
        name: str,
        table_type: str,
        connection: Optional[Connection],
        if_exists: bool,
    ) -> None:
        table = self.get_table(name)
        if_exists_str = " IF EXISTS" if if_exists else ""
        if connection is None:
            with self._engine.begin() as conn:
                conn.execute(text(f"DROP {table_type} {if_exists_str} {name}"))
                self._schema_mgr.remove_schema(conn, name)
        else:
            connection.execute(text(f"DROP {table_type} {if_exists_str} {name}"))
            self._schema_mgr.remove_schema(connection, name)

        self._metadata.remove(table)
        logger.info("Dropped {} {}", table_type.lower(), name)

    def _handle_sqlite_error_case(self, name: str, connection: Optional[Connection]) -> None:
        if connection is None and self._engine.name == "sqlite":
            with self._engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {name}"))


def check_columns(
    table_columns: Iterable[str],
    schema_columns: Iterable[str],
) -> None:
    """Check if the columns match the schema.

    Raises
    ------
    InvalidTable
        Raised if the columns don't match the schema.
    """
    columns_to_inspect = set(table_columns)
    expected_columns = schema_columns if isinstance(schema_columns, set) else set(schema_columns)
    if diff := expected_columns.difference(columns_to_inspect):
        cols = " ".join(sorted(diff))
        msg = f"These columns are defined in the schema but not present in the table: {cols}"
        raise InvalidTable(msg)
