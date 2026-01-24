import abc
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import Engine, MetaData, Table, select, text, func
from chronify.hive_functions import create_materialized_view

from chronify.sqlalchemy.functions import (
    create_view_from_parquet,
    write_database,
    write_query_to_parquet,
)
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import ConflictingInputsError, InvalidOperation
from chronify.utils.sqlalchemy_table import create_table
from chronify.time_series_checker import check_timestamps
from chronify.time import TimeIntervalType, ResamplingOperationType, AggregationType
from chronify.time_configs import TimeBasedDataAdjustment
from chronify.utils.path_utils import to_path


class TimeSeriesMapperBase(abc.ABC):
    """Maps time series data from one configuration to another."""

    def __init__(
        self,
        engine: Engine,
        metadata: MetaData,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
        resampling_operation: Optional[ResamplingOperationType] = None,
    ) -> None:
        self._engine = engine
        self._metadata = metadata
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


def apply_mapping(
    df_mapping: pd.DataFrame,
    mapping_schema: MappingTableSchema,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
    data_adjustment: TimeBasedDataAdjustment,
    resampling_operation: Optional[ResamplingOperationType] = None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> None:
    """
    Apply mapping to create result table with process to clean up and roll back if checks fail
    """
    with engine.begin() as conn:
        write_database(
            df_mapping,
            conn,
            mapping_schema.name,
            mapping_schema.time_configs,
            if_table_exists="fail",
            scratch_dir=scratch_dir,
        )
    metadata.reflect(engine, views=True)
    created_tmp_view = False
    try:
        _apply_mapping(
            mapping_schema.name,
            from_schema,
            to_schema,
            engine,
            metadata,
            resampling_operation=resampling_operation,
            scratch_dir=scratch_dir,
            output_file=output_file,
        )
        if check_mapped_timestamps:
            if output_file is not None:
                output_file = to_path(output_file)
                with engine.begin() as conn:
                    create_view_from_parquet(conn, to_schema.name, output_file)
                metadata.reflect(engine, views=True)
                created_tmp_view = True
            mapped_table = Table(to_schema.name, metadata)
            with engine.connect() as conn:
                try:
                    check_timestamps(
                        conn,
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
                        if engine.name == "hive":
                            conn.execute(text(f"DROP VIEW {to_schema.name}"))
                        else:
                            conn.execute(text(f"DROP TABLE {to_schema.name}"))
                    raise
    finally:
        with engine.begin() as conn:
            table_type = "view" if engine.name == "hive" else "table"
            conn.execute(text(f"DROP {table_type} IF EXISTS {mapping_schema.name}"))

            if created_tmp_view:
                conn.execute(text(f"DROP VIEW IF EXISTS {to_schema.name}"))
                metadata.remove(Table(to_schema.name, metadata))

        metadata.remove(Table(mapping_schema.name, metadata))
        metadata.reflect(engine, views=True)


def _apply_mapping(
    mapping_table_name: str,
    from_schema: TableSchema,
    to_schema: TableSchema,
    engine: Engine,
    metadata: MetaData,
    resampling_operation: Optional[ResamplingOperationType] = None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
) -> None:
    """Apply mapping to create result as a table according to_schema
    - Columns used to join the from_table are prefixed with "from_" in the mapping table
    """
    left_table = Table(from_schema.name, metadata)
    right_table = Table(mapping_table_name, metadata)
    left_table_columns = [x.name for x in left_table.columns]
    right_table_columns = [x.name for x in right_table.columns]
    left_table_pass_thru_columns = set(left_table_columns).difference(
        set(from_schema.list_columns())
    )

    val_col = to_schema.value_column  # from left_table
    final_cols = set(to_schema.list_columns()).union(left_table_pass_thru_columns)
    right_cols = set(right_table_columns).intersection(final_cols)
    left_cols = final_cols - right_cols - {val_col}

    select_stmt: list[Any] = [left_table.c[x] for x in left_cols]
    select_stmt += [right_table.c[x] for x in right_cols]

    tval_col = left_table.c[val_col]
    if "factor" in right_table_columns:
        tval_col *= right_table.c["factor"]  # type: ignore
    if not resampling_operation:
        select_stmt.append(tval_col)
    else:
        groupby_stmt = select_stmt.copy()
        match resampling_operation:
            case AggregationType.SUM:
                select_stmt.append(func.sum(tval_col).label(val_col))
            # case AggregationType.AVG:
            #     select_stmt.append(func.avg(tval_col).label(val_col))
            # case AggregationType.MIN:
            #     select_stmt.append(func.min(tval_col).label(val_col))
            # case AggregationType.MAX:
            #     select_stmt.append(func.max(tval_col).label(val_col))
            case _:
                msg = f"Unsupported {resampling_operation=}"
                raise InvalidOperation(msg)

    from_keys = [x for x in right_table_columns if x.startswith("from_")]
    keys = [x.removeprefix("from_") for x in from_keys]
    assert set(keys).issubset(
        set(left_table_columns)
    ), f"Keys {keys} not in table={from_schema.name}"
    on_stmt = reduce(and_, (left_table.c[x] == right_table.c["from_" + x] for x in keys))

    query = select(*select_stmt).select_from(left_table).join(right_table, on_stmt)
    if resampling_operation:
        query = query.group_by(*groupby_stmt)

    if output_file is not None:
        output_file = to_path(output_file)
        write_query_to_parquet(engine, str(query), output_file, overwrite=True)
        return

    if engine.name == "hive":
        create_materialized_view(
            str(query), to_schema.name, engine, metadata, scratch_dir=scratch_dir
        )
    else:
        create_table(to_schema.name, query, engine, metadata)
