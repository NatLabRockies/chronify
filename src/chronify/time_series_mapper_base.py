import abc
from pathlib import Path
from typing import Any, Optional

import ibis.expr.datatypes as dt
import pandas as pd
from loguru import logger

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import (
    create_view_from_parquet,
    write_parquet,
    write_table,
)
from chronify.spark_functions import create_materialized_view
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import ConflictingInputsError, InvalidOperation
from chronify.time_series_checker import check_timestamps
from chronify.time import (
    TimeIntervalType,
    ResamplingOperationType,
    AggregationType,
    TimeDataType,
)
from chronify.time_configs import TimeBasedDataAdjustment
from chronify.utils.path_utils import to_path


class TimeSeriesMapperBase(abc.ABC):
    """Maps time series data from one configuration to another."""

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
        resampling_operation: Optional[ResamplingOperationType] = None,
    ) -> None:
        self._backend = backend
        self._from_schema = from_schema
        self._to_schema = to_schema
        # data_adjustment is used in mapping creation and time check of mapped time
        self._data_adjustment = data_adjustment or TimeBasedDataAdjustment()
        self._wrap_time_allowed = wrap_time_allowed
        self._adjust_interval = (
            self._from_schema.time_config.interval_type
            != self._to_schema.time_config.interval_type
        )
        self._resampling_operation = resampling_operation

    @abc.abstractmethod
    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""

    def _check_table_columns_producibility(self) -> None:
        """Check columns in destination table can be produced by source table."""
        available_cols = (
            self._from_schema.list_columns() + self._to_schema.time_config.list_time_columns()
        )
        final_cols = self._to_schema.list_columns()  # does not include pass-thru columns
        if diff := set(final_cols) - set(available_cols):
            msg = f"Source table {self._from_schema.name} cannot produce the columns: {diff}"
            raise ConflictingInputsError(msg)

    def _check_measurement_type_consistency(self) -> None:
        """Check that measurement_type is the same between schema."""
        from_mt = self._from_schema.time_config.measurement_type
        to_mt = self._to_schema.time_config.measurement_type
        if from_mt != to_mt:
            msg = f"Inconsistent measurement_types {from_mt=} vs. {to_mt=}"
            raise ConflictingInputsError(msg)

    def _check_time_interval_type(self) -> None:
        """Check time interval type consistency."""
        from_interval = self._from_schema.time_config.interval_type
        to_interval = self._to_schema.time_config.interval_type
        if TimeIntervalType.INSTANTANEOUS in (from_interval, to_interval) and (
            from_interval != to_interval
        ):
            msg = "If instantaneous time interval is used, it must exist in both from_scheme and to_schema."
            raise ConflictingInputsError(msg)

    @abc.abstractmethod
    def map_time(self) -> None:
        """Convert time columns with from_schema to to_schema configuration."""


def _ensure_mapping_types_match_source(
    df_mapping: pd.DataFrame,
    from_schema: TableSchema,
    backend: IbisBackend,
) -> pd.DataFrame:
    """Ensure mapping DataFrame 'from_*' column types match source table column types.

    When unpivoting, column names become string values. The mapping DataFrame may have
    integer types for columns like 'from_hour'. This function casts those columns
    to match the source table's types.
    """
    source_table = backend.table(from_schema.name)
    source_schema = source_table.schema()

    df = df_mapping.copy()
    for col in df.columns:
        if col.startswith("from_"):
            source_col = col.removeprefix("from_")
            if source_col in source_schema:
                source_type = source_schema[source_col]
                # Only coerce types when Ibis can determine the source type
                # Skip if source type is unknown (e.g., SQLite datetime)
                try:
                    if source_type.is_string() and not pd.api.types.is_string_dtype(df[col]):
                        df[col] = df[col].astype(str)
                    elif source_type.is_integer() and not pd.api.types.is_integer_dtype(df[col]):
                        df[col] = df[col].astype(int)
                except AttributeError:
                    # Unknown type - skip coercion
                    pass

    return df


def apply_mapping(
    df_mapping: pd.DataFrame,
    mapping_schema: MappingTableSchema,
    from_schema: TableSchema,
    to_schema: TableSchema,
    backend: IbisBackend,
    data_adjustment: TimeBasedDataAdjustment,
    resampling_operation: Optional[ResamplingOperationType] = None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> None:
    """
    Apply mapping to create result table with process to clean up and roll back if checks fail
    """
    # Ensure mapping DataFrame column types match source table column types
    df_mapping = _ensure_mapping_types_match_source(df_mapping, from_schema, backend)

    # Debug: Print mapping DF around target time
    try:
        debug_ts = pd.Timestamp("2020-11-01 08:00:00", tz="EST")
        from_cols = [c for c in df_mapping.columns if c.startswith("from_")]
        if from_cols:
            debug_row = df_mapping[df_mapping[from_cols[0]] == debug_ts]
            print(f"DEBUG: Mapping DF row for {debug_ts}:\n{debug_row}")
        else:
            print("DEBUG: No from_ columns in mapping DF")
    except Exception as e:
        print(f"DEBUG: Error printing mapping DF: {e}")

    # Create the mapping table
    write_table(
        backend,
        df_mapping,
        mapping_schema.name,
        mapping_schema.time_configs,
        if_exists="fail",
        scratch_dir=scratch_dir,
    )

    created_tmp_view = False
    try:
        _apply_mapping(
            mapping_schema.name,
            from_schema,
            to_schema,
            backend,
            resampling_operation=resampling_operation,
            scratch_dir=scratch_dir,
            output_file=output_file,
        )
        if check_mapped_timestamps:
            if output_file is not None:
                output_file = to_path(output_file)
                create_view_from_parquet(backend, to_schema.name, output_file)
                created_tmp_view = True
            mapped_table = backend.table(to_schema.name)
            try:
                check_timestamps(
                    backend,
                    mapped_table,
                    to_schema,
                    leap_day_adjustment=data_adjustment.leap_day_adjustment,
                )
            except Exception:
                logger.exception(
                    "check_timestamps failed on mapped table {}. Drop it",
                    to_schema.name,
                )
                if output_file is None:
                    backend.drop_table(to_schema.name, if_exists=True)
                raise
    finally:
        table_type = "view" if backend.name == "spark" else "table"
        backend.execute_sql(f"DROP {table_type} IF EXISTS {mapping_schema.name}")

        if created_tmp_view:
            backend.drop_view(to_schema.name, if_exists=True)


def _build_join_predicate(
    left_table: Any,
    right_table: Any,
    key: str,
    left_type: Any,
    right_type: Any,
) -> Any:
    """Build a join predicate with appropriate type casting."""
    left_col = left_table[key]
    right_col = right_table[f"from_{key}"]

    if left_type is None or right_type is None:
        return left_col == right_col

    left_is_unknown = not hasattr(left_type, "is_timestamp") or str(left_type).startswith(
        "unknown"
    )
    right_is_timestamp = hasattr(right_type, "is_timestamp") and right_type.is_timestamp()
    left_is_timestamp = hasattr(left_type, "is_timestamp") and left_type.is_timestamp()
    right_is_string = hasattr(right_type, "is_string") and right_type.is_string()
    left_is_string = hasattr(left_type, "is_string") and left_type.is_string()

    if left_is_unknown and right_is_timestamp:
        left_col = left_col.cast("timestamp")
    elif left_is_timestamp and right_is_string:
        right_col = right_col.cast("timestamp")
    elif left_is_string and right_is_timestamp:
        left_col = left_col.cast("timestamp")

    return left_col == right_col


def _get_timestamp_target_dtype(to_schema: TableSchema, backend_name: str) -> Any:
    """Get the target dtype for timestamp column casting."""
    time_config = to_schema.time_config
    if time_config.dtype == TimeDataType.TIMESTAMP_TZ:
        tz_obj = time_config.start.tzinfo if hasattr(time_config, "start") else None
        if backend_name == "sqlite":
            tz_str = "UTC"
        elif hasattr(tz_obj, "key"):
            tz_str = tz_obj.key
        elif tz_obj:
            tz_str = str(tz_obj)
        else:
            tz_str = "UTC"
        return dt.Timestamp(timezone=tz_str)
    return dt.timestamp


def _build_column_expr(
    joined: Any,
    col: str,
    left_table_columns: list[str],
    joined_columns: list[str],
) -> Any:
    """Build a column expression, handling name conflicts from join."""
    if col in left_table_columns and f"{col}_right" in joined_columns:
        return joined[f"{col}_right"].name(col)
    return joined[col]


def _apply_mapping(
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    backend: IbisBackend,
    resampling_operation: Optional[ResamplingOperationType] = None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Apply mapping to create result as a table according to_schema.

    Columns used to join the from_table are prefixed with "from_" in the mapping table.
    """
    left_table = backend.table(from_schema.name)
    right_table = backend.table(mapping_table_name)

    left_table_columns = list(left_table.columns)
    right_table_columns = list(right_table.columns)
    left_table_pass_thru_columns = set(left_table_columns).difference(
        set(from_schema.list_columns())
    )

    val_col = to_schema.value_column
    final_cols = set(to_schema.list_columns()).union(left_table_pass_thru_columns)
    right_cols = set(right_table_columns).intersection(final_cols)
    left_cols = final_cols - right_cols - {val_col}

    # Build join predicates
    from_keys = [x for x in right_table_columns if x.startswith("from_")]
    keys = [x.removeprefix("from_") for x in from_keys]
    assert set(keys).issubset(
        set(left_table_columns)
    ), f"Keys {keys} not in table={from_schema.name}"

    predicates = _build_join_predicates(left_table, right_table, keys)
    joined = left_table.join(right_table, predicates)
    joined_columns = list(joined.columns)

    # Build select columns
    select_cols = _build_select_columns(
        joined, left_cols, right_cols, left_table_columns, joined_columns, to_schema, backend.name
    )

    # Handle value column with optional factor multiplication
    tval_col = joined[val_col]
    if "factor" in right_table_columns:
        tval_col = tval_col * joined["factor"]

    query = _build_query(
        joined,
        select_cols,
        tval_col,
        val_col,
        left_cols,
        right_cols,
        left_table_columns,
        joined_columns,
        resampling_operation,
    )

    _write_result(query, to_schema.name, backend, scratch_dir, output_file)


def _build_join_predicates(left_table: Any, right_table: Any, keys: list[str]) -> list[Any]:
    """Build join predicates with type mismatch handling."""
    left_schema = left_table.schema()
    right_schema = right_table.schema()
    predicates = []

    for k in keys:
        left_type = left_schema.get(k)
        right_type = right_schema.get(f"from_{k}")
        try:
            pred = _build_join_predicate(left_table, right_table, k, left_type, right_type)
            predicates.append(pred)
        except Exception:
            # Fallback: cast both to string
            predicates.append(
                left_table[k].cast("string") == right_table[f"from_{k}"].cast("string")
            )

    return predicates


def _build_select_columns(
    joined: Any,
    left_cols: set[str],
    right_cols: set[str],
    left_table_columns: list[str],
    joined_columns: list[str],
    to_schema: TableSchema,
    backend_name: str,
) -> list[Any]:
    """Build the list of columns to select from the joined table."""
    select_cols: list[Any] = []

    # Left table columns
    for col in left_cols:
        select_cols.append(joined[col])

    # Right table columns with potential name conflict handling
    time_column = to_schema.time_config.time_column
    needs_timestamp_cast = (
        hasattr(to_schema.time_config, "dtype") and to_schema.time_config.dtype.is_timestamp()
    )

    for col in right_cols:
        col_expr = _build_column_expr(joined, col, left_table_columns, joined_columns)

        if col == time_column and needs_timestamp_cast:
            target_dtype = _get_timestamp_target_dtype(to_schema, backend_name)
            col_expr = col_expr.cast(target_dtype).name(col)

        select_cols.append(col_expr)

    return select_cols


def _build_query(
    joined: Any,
    select_cols: list[Any],
    tval_col: Any,
    val_col: str,
    left_cols: set[str],
    right_cols: set[str],
    left_table_columns: list[str],
    joined_columns: list[str],
    resampling_operation: Optional[ResamplingOperationType],
) -> Any:
    """Build the final query with optional aggregation."""
    if not resampling_operation:
        select_cols.append(tval_col.name(val_col))
        return joined.select(select_cols)

    # Aggregation case
    groupby_cols = [joined[col] for col in left_cols]
    for col in right_cols:
        groupby_cols.append(_build_column_expr(joined, col, left_table_columns, joined_columns))

    match resampling_operation:
        case AggregationType.SUM:
            agg_col = tval_col.sum().name(val_col)
        case _:
            msg = f"Unsupported {resampling_operation=}"
            raise InvalidOperation(msg)

    return joined.group_by(groupby_cols).aggregate(agg_col)


def _write_result(
    query: Any,
    table_name: str,
    backend: IbisBackend,
    scratch_dir: Optional[Path],
    output_file: Optional[Path],
) -> None:
    """Write the query result to the appropriate destination."""
    if output_file is not None:
        output_file = to_path(output_file)
        write_parquet(backend, query, output_file, overwrite=True)
        return

    if backend.name == "spark":
        create_materialized_view(
            str(query.compile()), table_name, backend, scratch_dir=scratch_dir
        )
    else:
        backend.create_table(table_name, query)
