from collections.abc import Iterable
from pathlib import Path
import shutil
from typing import Any, Optional
from datetime import tzinfo

import duckdb
import pandas as pd
from duckdb import DuckDBPyRelation
from loguru import logger
import ibis.expr.types as ir
import pyarrow as pa

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
from chronify.ibis.backend import IbisBackend, make_backend
from chronify.ibis.functions import (
    create_view_from_parquet,
    read_query,
    write_parquet,
    write_table,
)
from chronify.models import (
    CsvTableSchema,
    PivotedTableSchema,
    TableSchema,
)
from chronify.schema_manager import SchemaManager
from chronify.time_configs import DatetimeRange, IndexTimeRangeBase, TimeBasedDataAdjustment
from chronify.time_series_checker import check_timestamps
from chronify.time_series_mapper import map_time
from chronify.time_zone_converter import TimeZoneConverter, TimeZoneConverterByColumn
from chronify.time_zone_localizer import TimeZoneLocalizer, TimeZoneLocalizerByColumn
from chronify.utils.path_utils import check_overwrite, to_path


class Store:
    """Data store for time series data"""

    def __init__(
        self,
        backend: Optional[IbisBackend] = None,
        backend_name: Optional[str] = None,
        file_path: Optional[Path | str] = None,
        **connect_kwargs: Any,
    ) -> None:
        """Construct the Store.

        Parameters
        ----------
        backend
            Optional, an IbisBackend instance. Defaults to a DuckDB in-memory backend.
        backend_name
            Optional, name of backend to use ('duckdb', 'sqlite', 'spark').
            Mutually exclusive with backend.
        file_path
            Optional, use this file for the database. If the file does not exist, create a new
            database. If the file exists, load that existing database.
            Defaults to a new in-memory database.

        Examples
        --------
        >>> store1 = Store()
        >>> store2 = Store(backend_name="duckdb", file_path="time_series.db")
        >>> store3 = Store(backend_name="sqlite", file_path="time_series.db")
        >>> store4 = Store(backend=make_backend("duckdb"))
        """
        if backend and backend_name:
            msg = f"{backend=} and {backend_name=} cannot both be set"
            raise ConflictingInputsError(msg)

        if backend is None:
            name = backend_name or "duckdb"
            self._backend = make_backend(name, file_path=file_path, **connect_kwargs)
        else:
            self._backend = backend

        self._schema_mgr = SchemaManager(self._backend)

    @classmethod
    def create_in_memory_db(
        cls,
        backend_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with an in-memory database."""
        return Store(backend=make_backend(backend_name, **connect_kwargs))

    @classmethod
    def create_file_db(
        cls,
        file_path: Path | str = "time_series.db",
        backend_name: str = "duckdb",
        overwrite: bool = False,
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a Store with a file-based database."""
        path = to_path(file_path)
        check_overwrite(path, overwrite)
        return Store(backend=make_backend(backend_name, file_path=path, **connect_kwargs))

    @classmethod
    def create_new_spark_store(
        cls,
        session: Any = None,
        drop_schema: bool = True,
        **connect_kwargs: Any,
    ) -> "Store":
        """Create a new Store with Spark backend.

        Recommended usage is to create views from Parquet files. Ingesting data into tables
        from files or DataFrames is not supported.

        Parameters
        ----------
        session
            Optional SparkSession. If None, a new one will be created.
        drop_schema
            If True, drop the schema table if it's already there.

        Examples
        --------
        >>> store = Store.create_new_spark_store()

        See also
        --------
        create_view_from_parquet
        """
        backend = make_backend("spark", session=session, **connect_kwargs)

        if drop_schema and SchemaManager.SCHEMAS_TABLE in backend.list_tables():
            backend.drop_table(SchemaManager.SCHEMAS_TABLE)

        return cls(backend=backend)

    @classmethod
    def load_from_file(
        cls,
        file_path: Path | str,
        backend_name: str = "duckdb",
        **connect_kwargs: Any,
    ) -> "Store":
        """Load an existing store from a database."""
        path = to_path(file_path)
        if not path.exists():
            msg = str(path)
            raise FileNotFoundError(msg)
        return Store(backend=make_backend(backend_name, file_path=path, **connect_kwargs))

    def dispose(self) -> None:
        """Clean up the backend connection."""
        self._backend.dispose()

    def get_table(self, name: str) -> ir.Table:
        """Return the Ibis table reference."""
        if not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)

        return self._backend.table(name)

    def has_table(self, name: str) -> bool:
        """Return True if the database has a table with the given name."""
        return self._backend.has_table(name)

    def list_tables(self) -> list[str]:
        """Return a list of user tables in the database."""
        return [x for x in self._backend.list_tables() if x != SchemaManager.SCHEMAS_TABLE]

    def try_get_table(self, name: str) -> ir.Table | None:
        """Return the Ibis table object or None if it is not stored."""
        if not self.has_table(name):
            return None
        return self._backend.table(name)

    def backup(self, dst: Path | str, overwrite: bool = False) -> None:
        """Copy the database to a new location. Not yet supported for in-memory databases."""
        self._backend.dispose()
        path = to_path(dst)
        check_overwrite(path, overwrite)
        try:
            match self._backend.name:
                case "duckdb" | "sqlite":
                    if self._backend.database is None:
                        msg = "backup is only supported with a database backed by a file"
                        raise InvalidOperation(msg)
                    src_file = Path(self._backend.database)
                    shutil.copyfile(src_file, path)
                    logger.info("Copied database to {}", path)
                case _:
                    msg = self._backend.name
                    raise NotImplementedError(msg)
        finally:
            self._backend.reconnect()

    @property
    def backend(self) -> IbisBackend:
        """Return the Ibis backend."""
        return self._backend

    @property
    def schema_manager(self) -> SchemaManager:
        """Return the store's schema manager."""
        return self._schema_mgr

    def check_timestamps(self, name: str) -> None:
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
        check_timestamps(self._backend, table, schema)

    def create_view_from_parquet(
        self, path: Path, schema: TableSchema, bypass_checks: bool = False
    ) -> None:
        """Load a table into the database from a Parquet file."""
        self._create_view_from_parquet(path, schema)
        try:
            table = self.get_table(schema.name)
            if not bypass_checks:
                check_timestamps(self._backend, table, schema)
        except InvalidTable:
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
        create_view_from_parquet(self._backend, schema.name, to_path(path))
        self._schema_mgr.add_schema(schema)

    def ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
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
        return self.ingest_from_csvs((path,), src_schema, dst_schema)

    def ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        """Ingest data into the table specifed by schema. If the table does not exist,
        create it. This is faster than calling :meth:`ingest_from_csv` many times.
        Each file is loaded into memory one at a time.
        If any error occurs, all added data will be removed and the state of the database will
        be the same as the original state.

        Parameters
        ----------
        paths
            Source data files
        src_schema
            Defines the schema of the source files.
        dst_schema
            Defines the destination table in the database.

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
            created_table = self._ingest_from_csvs(paths, src_schema, dst_schema)
        except Exception:
            self._handle_error_case(dst_schema.name)
            raise

        return created_table

    def _ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        created_table = False
        paths_list = list(paths)
        if not paths_list:
            return created_table

        for path in paths_list:
            if self._ingest_from_csv(path, src_schema, dst_schema):
                created_table = True
        table = self._backend.table(dst_schema.name)
        check_timestamps(self._backend, table, dst_schema)
        return created_table

    def _ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        rel = read_csv(path, src_schema)
        columns = set(src_schema.list_columns())
        check_columns(rel.columns, columns)

        if isinstance(src_schema.time_config, IndexTimeRangeBase):
            if isinstance(dst_schema.time_config, DatetimeRange):
                raise NotImplementedError
            else:
                cls_name = dst_schema.time_config.__class__.__name__
                msg = f"{src_schema.time_config.__class__.__name__} cannot be converted to {cls_name}"
                raise NotImplementedError(msg)

        if src_schema.pivoted_dimension_name is not None:
            return self._ingest_pivoted_table(rel, src_schema, dst_schema)

        return self._ingest_table(rel, dst_schema)

    def ingest_pivoted_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
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
        return self.ingest_pivoted_tables((data,), src_schema, dst_schema)

    def ingest_pivoted_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
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

        Returns
        -------
        bool
            Return True if a table was created.

        See Also
        --------
        ingest_pivoted_table
        """
        try:
            created_table = self._ingest_pivoted_tables(data, src_schema, dst_schema)
        except Exception:
            self._handle_error_case(dst_schema.name)
            raise

        return created_table

    def _ingest_pivoted_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        created_table = False
        for table in data:
            if self._ingest_pivoted_table(table, src_schema, dst_schema):
                created_table = True
        check_timestamps(self._backend, self._backend.table(dst_schema.name), dst_schema)
        return created_table

    def _ingest_pivoted_table(
        self,
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
        return self._ingest_table(rel2, dst_schema)

    def ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation | ir.Table | pa.Table,
        schema: TableSchema,
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
        return self.ingest_tables((data,), schema, **kwargs)

    def ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation | ir.Table | pa.Table],
        schema: TableSchema,
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
        data_list = list(data)
        if not data_list:
            return created_table

        # Track if table existed before ingestion to avoid dropping existing data on error
        table_existed_before = self.has_table(schema.name)

        try:
            created_table = self._ingest_tables(data_list, schema, **kwargs)
        except Exception:
            # Only drop the table if we created it; don't destroy existing data
            if not table_existed_before:
                self._handle_error_case(schema.name)
            raise

        return created_table

    def _ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation | ir.Table | pa.Table],
        schema: TableSchema,
        skip_time_checks: bool = False,
    ) -> bool:
        created_table = False
        for table in data:
            if self._ingest_table(table, schema):
                created_table = True
        if not skip_time_checks:
            check_timestamps(self._backend, self._backend.table(schema.name), schema)
        return created_table

    def _ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation | ir.Table | pa.Table,
        schema: TableSchema,
    ) -> bool:
        if self._backend.name == "spark":
            msg = "Data ingestion through Spark is not supported"
            raise NotImplementedError(msg)

        # Convert to DataFrame/PyArrow for consistent handling
        if isinstance(data, ir.Table):
            df = data.to_pyarrow()
        elif isinstance(data, DuckDBPyRelation):
            df = data.to_df()
        else:
            df = data

        if isinstance(df, pa.Table):
            columns = df.column_names
        else:
            columns = df.columns

        check_columns(columns, schema.list_columns())

        table_exists = self.has_table(schema.name)
        if not table_exists:
            # Use write_table to ensure proper datetime handling for all backends
            write_table(self._backend, df, schema.name, [schema.time_config], if_exists="fail")
            created_table = True
        else:
            write_table(self._backend, df, schema.name, [schema.time_config], if_exists="append")
            created_table = False

        if created_table:
            self._schema_mgr.add_schema(schema)

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
            self._backend,
            src_schema,
            dst_schema,
            data_adjustment=data_adjustment,
            wrap_time_allowed=wrap_time_allowed,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        self._schema_mgr.add_schema(dst_schema)

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
        tzc = TimeZoneConverter(self._backend, src_schema, time_zone)

        dst_schema = tzc.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzc.convert_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        self._schema_mgr.add_schema(dst_schema)

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
            self._backend, src_schema, time_zone_column, wrap_time_allowed
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

        self._schema_mgr.add_schema(dst_schema)

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
        tzl = TimeZoneLocalizer(self._backend, src_schema, time_zone)

        dst_schema = tzl.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzl.localize_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        self._schema_mgr.add_schema(dst_schema)

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
        tzl = TimeZoneLocalizerByColumn(self._backend, src_schema, time_zone_column)

        dst_schema = tzl.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzl.localize_time_zone(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        self._schema_mgr.add_schema(dst_schema)

        return dst_schema

    def read_query(
        self,
        name: str,
        query: ir.Table | str,
    ) -> pd.DataFrame:
        """Return the query result as a pandas DataFrame.

        Parameters
        ----------
        name
            Table or view name
        query
            SQL query as string or Ibis table expression

        Examples
        --------
        >>> df = store.read_query("devices", "SELECT * FROM devices")
        >>> df = store.read_query("devices", store.get_table("devices").filter(...))
        """
        schema = self._schema_mgr.get_schema(name)
        if isinstance(query, str):
            expr = self._backend.sql(query)
        else:
            expr = query
        return read_query(self._backend, expr, schema.time_config)

    def read_table(self, name: str) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        table = self.get_table(name)
        return self.read_query(name, table)

    def read_raw_query(
        self,
        query: str,
    ) -> pd.DataFrame:
        """Execute a query directly on the backend and return results as DataFrame.

        Note: Unlike :meth:`read_query`, no conversion of timestamps is performed.
        Timestamps will be in the format of the underlying database. SQLite backends will return
        strings instead of datetime.

        Parameters
        ----------
        query
            SQL query to execute

        Examples
        --------
        >>> store = Store()
        >>> df = store.read_raw_query("SELECT * from my_table WHERE column = 'value'")
        """
        expr = self._backend.sql(query)
        return self._backend.execute(expr)

    def write_query_to_parquet(
        self,
        query: ir.Table | str,
        file_path: Path | str,
        overwrite: bool = False,
        partition_columns: Optional[list[str]] = None,
    ) -> None:
        """Write the result of a query to a Parquet file."""
        if isinstance(query, str):
            expr = self._backend.sql(query)
        else:
            expr = query

        write_parquet(
            self._backend,
            expr,
            to_path(file_path),
            overwrite=overwrite,
            partition_columns=partition_columns,
        )

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

        table = self.get_table(name)
        write_parquet(
            self._backend,
            table,
            to_path(file_path),
            overwrite=overwrite,
            partition_columns=partition_columns,
        )
        logger.info("Wrote table or view to {}", file_path)

    def delete_rows(
        self,
        name: str,
        time_array_id_values: dict[str, Any],
    ) -> int:
        """Delete all rows matching the time_array_id_values.

        Parameters
        ----------
        name
            Name of table
        time_array_id_values
            Values for the time_array_id_values. Keys must match the columns in the schema.

        Returns
        -------
        int
            Number of deleted rows

        Examples
        --------
        >>> store.delete_rows("devices", {"id": 47})
        """
        table = self.get_table(name)
        if not time_array_id_values:
            msg = "time_array_id_values cannot be empty"
            raise InvalidParameter(msg)

        schema = self._schema_mgr.get_schema(name)
        if sorted(time_array_id_values.keys()) != sorted(schema.time_array_id_columns):
            msg = (
                "The keys of time_array_id_values must match the schema columns. "
                f"Passed = {sorted(time_array_id_values.keys())} "
                f"Schema = {sorted(schema.time_array_id_columns)}"
            )
            raise InvalidParameter(msg)

        # Build filter condition
        filter_expr = None
        for column, value in time_array_id_values.items():
            condition = table[column] == value
            filter_expr = condition if filter_expr is None else (filter_expr & condition)

        # Count rows before deletion
        count_result = self._backend.execute(table.filter(filter_expr).count())
        if isinstance(count_result, pd.DataFrame):
            count_before = int(count_result.iloc[0, 0])  # type: ignore[arg-type]
        else:
            count_before = int(count_result)

        # Build and execute DELETE statement
        where_clauses = [
            f"{col} = {_escape_sql_value(val)}" for col, val in time_array_id_values.items()
        ]
        where_str = " AND ".join(where_clauses)
        self._backend.execute_sql(f"DELETE FROM {name} WHERE {where_str}")

        num_deleted = count_before

        if num_deleted < 1:
            msg = f"Failed to delete rows: {time_array_id_values=} {num_deleted=}"
            raise InvalidParameter(msg)

        logger.info(
            "Delete all rows from table {} with time_array_id_values {}",
            name,
            time_array_id_values,
        )

        # Check if table is empty
        remaining_result = self._backend.execute(table.count())
        if isinstance(remaining_result, pd.DataFrame):
            remaining_count = int(remaining_result.iloc[0, 0])  # type: ignore[arg-type]
        else:
            remaining_count = int(remaining_result)
        is_empty = remaining_count == 0

        if is_empty:
            logger.info("Delete empty table {}", name)
            self.drop_table(name)

        return num_deleted

    def drop_table(
        self,
        name: str,
        if_exists: bool = False,
    ) -> None:
        """Drop a table from the database."""
        if not if_exists and not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)

        self._backend.drop_table(name, if_exists=if_exists)
        if name in self._schema_mgr._cache:
            self._schema_mgr.remove_schema(name)
        logger.info("Dropped table {}", name)

    def create_view(
        self,
        schema: TableSchema,
        table: ir.Table,
        bypass_checks: bool = False,
    ) -> None:
        """Register an Ibis table as a view with time validation.

        Creates a view and persists the schema. Both the view and schema
        will survive process restarts (for backends that support persistent storage).

        When using Spark, enabling the Hive metastore is required to persist restarts.

        Parameters
        ----------
        schema
            Defines the schema of the view. The name in the schema will be used as
            the view name.
        table
            Ibis table expression to register. This can be a table loaded from a
            Parquet file, a query result, or any other Ibis table expression.
        bypass_checks
            If True, skip time validation checks. Defaults to False.

        Raises
        ------
        InvalidTable
            Raised if time validation fails and bypass_checks is False.
        """
        self._backend.create_view(schema.name, table)
        self._schema_mgr.add_schema(schema)

        if not bypass_checks:
            try:
                registered_table = self.get_table(schema.name)
                check_timestamps(self._backend, registered_table, schema)
            except InvalidTable:
                self._backend.drop_view(schema.name, if_exists=True)
                self._backend.drop_table(f"_backing_{schema.name}", if_exists=True)
                self._schema_mgr.remove_schema(schema.name)
                raise

        logger.info("Registered permanent view {} with schema", schema.name)

    def drop_view(
        self,
        name: str,
        if_exists: bool = False,
    ) -> None:
        """Drop a view from the database."""
        if not if_exists and not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)

        self._backend.drop_view(name, if_exists=if_exists)
        if name in self._schema_mgr._cache:
            self._schema_mgr.remove_schema(name)
        logger.info("Dropped view {}", name)

    def _handle_error_case(self, name: str) -> None:
        """Clean up after an error during ingestion."""
        # Ibis backends don't support transactions for DDL, so manually drop the table
        self._backend.drop_table(name, if_exists=True)


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


def _escape_sql_value(value: Any) -> str:
    """Escape a value for safe inclusion in SQL statements.

    Parameters
    ----------
    value
        The value to escape

    Returns
    -------
    str
        SQL-safe string representation of the value
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    # For other types, convert to string and escape
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"
