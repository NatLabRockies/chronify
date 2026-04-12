from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional
from datetime import tzinfo

import duckdb
import ibis.expr.types as ir
import pandas as pd
from duckdb import DuckDBPyRelation
from loguru import logger

import chronify.duckdb.functions as ddbf
from chronify.exceptions import (
    ConflictingInputsError,
    InvalidParameter,
    InvalidTable,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.csv_io import read_csv
from chronify.ibis import IbisBackend, ObjectType, make_backend
from chronify.ibis.functions import (
    create_view_from_parquet,
    read_query,
    read_table,
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
    ) -> None:
        """Construct the Store.

        Parameters
        ----------
        backend
            Optional, defaults to a DuckDB in-memory backend.
        backend_name
            Optional, name of backend to use ('duckdb', 'sqlite'). Mutually exclusive with backend.
        file_path
            Optional, use this file for the database. If the file does not exist, create a new
            database. If the file exists, load that existing database.
            Defaults to a new in-memory database.
        """
        if backend and backend_name:
            msg = f"{backend=} and {backend_name=} cannot both be set"
            raise ConflictingInputsError(msg)
        if backend is not None:
            self._backend = backend
        else:
            name = backend_name or "duckdb"
            database = str(file_path) if file_path else None
            self._backend = make_backend(name, database=database)

        self._schema_mgr = SchemaManager(self._backend)

    @classmethod
    def create_in_memory_db(
        cls,
        backend_name: str = "duckdb",
    ) -> "Store":
        """Create a Store with an in-memory database."""
        return Store(backend=make_backend(backend_name))

    @classmethod
    def create_file_db(
        cls,
        file_path: Path | str = "time_series.db",
        backend_name: str = "duckdb",
        overwrite: bool = False,
    ) -> "Store":
        """Create a Store with a file-based database."""
        path = to_path(file_path)
        check_overwrite(path, overwrite)
        return Store(backend=make_backend(backend_name, database=str(path)))

    @classmethod
    def load_from_file(
        cls,
        file_path: Path | str,
        backend_name: str = "duckdb",
    ) -> "Store":
        """Load an existing store from a database."""
        path = to_path(file_path)
        if not path.exists():
            msg = str(path)
            raise FileNotFoundError(msg)
        return Store(backend=make_backend(backend_name, database=str(path)))

    def dispose(self) -> None:
        """Dispose of the current connections."""
        self._backend.dispose()

    def get_table(self, name: str) -> ir.Table:
        """Return the ibis Table expression."""
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
        """Return the ibis Table expression or None if it is not stored."""
        if not self.has_table(name):
            return None
        return self._backend.table(name)

    def backup(self, dst: Path | str, overwrite: bool = False) -> None:
        """Copy the database to a new location. Not yet supported for in-memory databases."""
        path = to_path(dst)
        check_overwrite(path, overwrite)
        self._backend.backup(str(path))
        logger.info("Copied database to {}", path)

    @property
    def backend(self) -> IbisBackend:
        """Return the ibis backend."""
        return self._backend

    @property
    def schema_manager(self) -> SchemaManager:
        """Return the store's schema manager."""
        return self._schema_mgr

    def check_timestamps(self, name: str) -> None:
        """Check the timestamps in the table.

        Parameters
        ----------
        name
            Name of the table to check.

        Raises
        ------
        InvalidTable
            Raised if the timestamps do not match the schema.
        """
        schema = self._schema_mgr.get_schema(name)
        check_timestamps(self._backend, name, schema)

    def create_view_from_parquet(
        self, path: Path, schema: TableSchema, bypass_checks: bool = False
    ) -> None:
        """Load a table into the database from a Parquet file."""
        obj_type = self._create_view_from_parquet(path, schema)
        try:
            if not bypass_checks:
                check_timestamps(self._backend, schema.name, schema)
        except InvalidTable:
            if obj_type == ObjectType.TABLE:
                self._backend.drop_table(schema.name)
            else:
                self._backend.drop_view(schema.name)
            raise

    def _create_view_from_parquet(self, path: Path | str, schema: TableSchema) -> "ObjectType":
        """Create a view in the database from a Parquet file."""
        obj_type = create_view_from_parquet(self._backend, to_path(path), schema.name)
        self._schema_mgr.add_schema(schema)
        return obj_type

    def ingest_from_csv(
        self,
        path: Path | str,
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        """Ingest data from a CSV file."""
        return self.ingest_from_csvs((path,), src_schema, dst_schema)

    def ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        """Ingest data from multiple CSV files into the table specified by schema."""
        table_existed = self._backend.has_table(dst_schema.name)
        try:
            created_table = self._ingest_from_csvs(paths, src_schema, dst_schema)
        except Exception:
            if not table_existed and self._backend.has_table(dst_schema.name):
                self._backend.drop_table(dst_schema.name)
                self._schema_mgr.remove_schema(dst_schema.name)
            raise
        return created_table

    def _ingest_from_csvs(
        self,
        paths: Iterable[Path | str],
        src_schema: CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        created_table = False
        if not paths:
            return created_table

        for path in paths:
            if self._ingest_from_csv(path, src_schema, dst_schema):
                created_table = True
        check_timestamps(self._backend, dst_schema.name, dst_schema)
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
        """Ingest pivoted data into the table specified by schema."""
        return self.ingest_pivoted_tables((data,), src_schema, dst_schema)

    def ingest_pivoted_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        """Ingest pivoted data from multiple tables. Unpivot before ingesting."""
        table_existed = self._backend.has_table(dst_schema.name)
        try:
            created_table = self._ingest_pivoted_tables(data, src_schema, dst_schema)
        except Exception:
            if not table_existed and self._backend.has_table(dst_schema.name):
                self._backend.drop_table(dst_schema.name)
                self._schema_mgr.remove_schema(dst_schema.name)
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
        check_timestamps(self._backend, dst_schema.name, dst_schema)
        return created_table

    def _ingest_pivoted_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        src_schema: PivotedTableSchema | CsvTableSchema,
        dst_schema: TableSchema,
    ) -> bool:
        if isinstance(data, pd.DataFrame):
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
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
        **kwargs: Any,
    ) -> bool:
        """Ingest data into the table specified by schema. If the table does not exist,
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
        """
        return self.ingest_tables((data,), schema, **kwargs)

    def ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
        **kwargs: Any,
    ) -> bool:
        """Ingest multiple input tables to the same database table."""
        created_table = False
        if not data:
            return created_table

        table_existed = self._backend.has_table(schema.name)
        try:
            created_table = self._ingest_tables(data, schema, **kwargs)
        except Exception:
            if not table_existed and self._backend.has_table(schema.name):
                self._backend.drop_table(schema.name)
                self._schema_mgr.remove_schema(schema.name)
            raise
        return created_table

    def _ingest_tables(
        self,
        data: Iterable[pd.DataFrame | DuckDBPyRelation],
        schema: TableSchema,
        skip_time_checks: bool = False,
    ) -> bool:
        created_table = False
        for table in data:
            if self._ingest_table(table, schema):
                created_table = True
        if not skip_time_checks:
            check_timestamps(self._backend, schema.name, schema)
        return created_table

    def _ingest_table(
        self,
        data: pd.DataFrame | DuckDBPyRelation,
        schema: TableSchema,
    ) -> bool:
        df = data.to_df() if isinstance(data, DuckDBPyRelation) else data
        check_columns(df.columns, schema.list_columns())

        if not self._backend.has_table(schema.name):
            write_table(
                self._backend,
                df,
                schema.name,
                [schema.time_config],
                if_exists="fail",
            )
            self._schema_mgr.add_schema(schema)
            return True
        else:
            write_table(
                self._backend,
                df,
                schema.name,
                [schema.time_config],
                if_exists="append",
            )
            return False

    def map_table_time_config(
        self,
        src_name: str,
        dst_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Map the existing table represented by src_name to a new table represented by
        dst_schema with a different time configuration."""
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
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        self._schema_mgr.add_schema(dst_schema)

    def convert_time_zone(
        self,
        src_name: str,
        time_zone: tzinfo | None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> TableSchema:
        """Convert the time zone of the existing table represented by src_name to a new time zone."""
        src_schema = self._schema_mgr.get_schema(src_name)
        tzc = TimeZoneConverter(self._backend, src_schema, time_zone)

        dst_schema = tzc.generate_to_schema()
        if self.has_table(dst_schema.name):
            msg = dst_schema.name
            raise TableAlreadyExists(msg)

        tzc.convert_time_zone(
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
        >>> start = datetime(year=2018, month=1, day=1, tzinfo=ZoneInfo("Etc/GMT+5"))
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
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        self._schema_mgr.add_schema(dst_schema)
        return dst_schema

    def localize_time_zone(
        self,
        src_name: str,
        time_zone: tzinfo | None,
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
        >>> to_time_zone = ZoneInfo("Etc/GMT+5")
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
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
        self._schema_mgr.add_schema(dst_schema)
        return dst_schema

    def localize_time_zone_by_column(
        self,
        src_name: str,
        time_zone_column: Optional[str] = None,
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
            Name of the time zone column for localization, default to None
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
        ...         "time_zone": np.repeat(
        ...             ["Etc/GMT+5", "Etc/GMT+6", "Etc/GMT+7"], hours_per_year
        ...         ),  # EST, CST, MST
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
            SQL query as a string or ibis Table expression
        """
        schema = self._schema_mgr.get_schema(name)
        if isinstance(query, str):
            expr = self._backend.sql(query)
        else:
            expr = query
        return read_query(self._backend, expr, schema.time_config)

    def read_table(self, name: str) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        schema = self._schema_mgr.get_schema(name)
        return read_table(self._backend, name, schema.time_config)

    def read_raw_query(self, query: str) -> pd.DataFrame:
        """Execute a query directly on the backend and return the results as a DataFrame.

        Note: Unlike :meth:`read_query`, no conversion of timestamps is performed.
        Timestamps will be in the format of the underlying database. SQLite backends will return
        strings instead of datetime.

        Parameters
        ----------
        query
            SQL query to execute.

        Examples
        --------
        >>> store = Store()
        >>> df = store.read_raw_query("SELECT * from my_table WHERE column = 'value1'")
        """
        return self._backend.execute_sql_to_df(query)

    def write_query_to_parquet(
        self,
        stmt: ir.Table | str,
        file_path: Path | str,
        overwrite: bool = False,
        partition_columns: Optional[list[str]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Write the result of a query to a Parquet file.

        Parameters
        ----------
        stmt
            SQL query or ibis Table expression.
        file_path
            Output Parquet file path.
        overwrite
            Whether to overwrite an existing file.
        partition_columns
            Optional columns to partition by.
        name
            Optional table/view name used to look up the time config for
            backend-specific timestamp normalization (e.g. Spark).
        """
        config = self._schema_mgr.get_schema(name).time_config if name else None
        write_parquet(
            self._backend,
            stmt,
            to_path(file_path),
            overwrite=overwrite,
            partition_columns=partition_columns,
            config=config,
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

        expr = self._backend.table(name)
        write_parquet(
            self._backend,
            expr,
            to_path(file_path),
            overwrite=overwrite,
            partition_columns=partition_columns,
            config=self._schema_mgr.get_schema(name).time_config,
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
        """
        if not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)
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

        # Build the predicate using ibis (safe -- no string interpolation).
        table = self._backend.table(name)
        predicates = [table[column] == value for column, value in time_array_id_values.items()]
        filtered = table.filter(predicates)
        num_to_delete = int(filtered.count().execute())

        self._backend.delete_rows(name, time_array_id_values)

        if num_to_delete < 1:
            msg = f"Failed to delete rows: {time_array_id_values} {num_to_delete=}"
            raise InvalidParameter(msg)

        logger.info(
            "Delete all rows from table {} with time_array_id_values {}",
            name,
            time_array_id_values,
        )

        # Check if table is now empty
        remaining = int(self._backend.table(name).count().execute())
        if remaining == 0:
            logger.info("Delete empty table {}", name)
            self.drop_table(name)

        return num_to_delete

    def drop_table(self, name: str, if_exists: bool = False) -> None:
        """Drop a table from the database."""
        if not if_exists and not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)
        self._backend.drop_table(name)
        self._schema_mgr.remove_schema(name)
        logger.info("Dropped table {}", name)

    def create_view(self, schema: TableSchema, stmt: ir.Table) -> None:
        """Create a view in the database."""
        self._backend.create_view(schema.name, stmt)
        self._schema_mgr.add_schema(schema)

    def drop_view(self, name: str, if_exists: bool = False) -> None:
        """Drop a view from the database."""
        if not if_exists and not self.has_table(name):
            msg = f"{name=}"
            raise TableNotStored(msg)
        self._backend.drop_view(name)
        self._schema_mgr.remove_schema(name)
        logger.info("Dropped view {}", name)


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
