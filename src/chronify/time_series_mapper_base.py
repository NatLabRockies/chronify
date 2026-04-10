import abc
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from chronify.ibis.base import IbisBackend, ObjectType
from chronify.ibis.functions import write_parquet, write_table, create_view_from_parquet
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import ConflictingInputsError, InvalidOperation
from chronify.time_series_checker import check_timestamps
from chronify.time import TimeIntervalType, ResamplingOperationType, AggregationType
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
        final_cols = self._to_schema.list_columns()
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


def apply_mapping(
    df_mapping: pd.DataFrame,
    mapping_schema: MappingTableSchema,
    from_schema: TableSchema,
    to_schema: TableSchema,
    backend: IbisBackend,
    data_adjustment: TimeBasedDataAdjustment,
    resampling_operation: Optional[ResamplingOperationType] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> None:
    """Apply mapping to create result table with process to clean up and roll back if checks fail."""
    write_table(
        backend,
        df_mapping,
        mapping_schema.name,
        mapping_schema.time_configs,
        if_exists="fail",
    )
    created_tmp_obj: Optional[ObjectType] = None
    try:
        _apply_mapping(
            mapping_schema.name,
            from_schema,
            to_schema,
            backend,
            resampling_operation=resampling_operation,
            output_file=output_file,
        )
        if check_mapped_timestamps:
            if output_file is not None:
                output_file = to_path(output_file)
                created_tmp_obj = create_view_from_parquet(backend, output_file, to_schema.name)
            try:
                check_timestamps(
                    backend,
                    to_schema.name,
                    to_schema,
                    leap_day_adjustment=data_adjustment.leap_day_adjustment,
                )
            except Exception:
                logger.exception(
                    "check_timestamps failed on mapped table {}. Drop it",
                    to_schema.name,
                )
                if output_file is None:
                    backend.drop_table(to_schema.name)
                raise
    finally:
        if backend.has_table(mapping_schema.name):
            backend.drop_table(mapping_schema.name)
        if created_tmp_obj is not None:
            if created_tmp_obj == ObjectType.TABLE:
                backend.drop_table(to_schema.name)
            else:
                backend.drop_view(to_schema.name)


def _apply_mapping(  # noqa: C901
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    backend: IbisBackend,
    resampling_operation: Optional[ResamplingOperationType] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Apply mapping to create result as a table according to_schema.
    Columns used to join the from_table are prefixed with "from_" in the mapping table.
    """
    left = backend.table(from_schema.name)
    right = backend.table(mapping_table_name)
    left_columns = left.columns
    right_columns = right.columns
    left_pass_thru_columns = set(left_columns) - set(from_schema.list_columns())

    val_col = to_schema.value_column
    final_cols = set(to_schema.list_columns()) | left_pass_thru_columns
    right_cols = set(right_columns) & final_cols
    left_cols = final_cols - right_cols - {val_col}

    # Build join predicates
    from_keys = [x for x in right_columns if x.startswith("from_")]
    keys = [x.removeprefix("from_") for x in from_keys]
    if not set(keys).issubset(set(left_columns)):
        msg = f"Mapping keys {keys} not found in source table {from_schema.name}"
        raise ConflictingInputsError(msg)
    predicates = []
    for k in keys:
        left_col = left[k]
        right_col = right["from_" + k]
        # Cast to match types if needed (e.g., string vs int from pivoted columns)
        if left_col.type() != right_col.type():
            right_col = right_col.cast(left_col.type())
        predicates.append(left_col == right_col)

    # Perform the join
    joined = left.join(right, predicates)

    # In ibis joins, conflicting right-side columns get a "_right" suffix.
    # Left-side columns keep their original names.
    def _left_col(col: str) -> Any:
        """Access a left-table column (keeps original name after join)."""
        return joined[col]

    def _right_col(col: str) -> Any:
        """Access a right-table column, handling disambiguation."""
        if col in left_columns and col in right_columns:
            return joined[col + "_right"]
        return joined[col]

    # Build value expression (always from the left/source table)
    val_expr: Any = _left_col(val_col)
    if "factor" in right_columns:
        val_expr = val_expr * _right_col("factor")

    # Build select columns
    select_exprs: list[Any] = []
    for col in left_cols:
        select_exprs.append(_left_col(col).name(col))
    for col in right_cols:
        select_exprs.append(_right_col(col).name(col))

    if not resampling_operation:
        select_exprs.append(val_expr.name(val_col))
        result = joined.select(select_exprs)
    else:
        group_exprs = select_exprs.copy()
        match resampling_operation:
            case AggregationType.SUM:
                agg_expr = val_expr.sum().name(val_col)
            case _:
                msg = f"Unsupported {resampling_operation=}"
                raise InvalidOperation(msg)
        result = joined.group_by(group_exprs).aggregate(agg_expr)

    if output_file is not None:
        output_file = to_path(output_file)
        write_parquet(backend, result, output_file, overwrite=True, config=to_schema.time_config)
        return

    backend.create_table(to_schema.name, result)
