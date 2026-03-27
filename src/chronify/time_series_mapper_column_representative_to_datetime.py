from typing import Optional, Generator
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query, write_table
from chronify.exceptions import InvalidParameter, InvalidValue
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import (
    YearMonthDayHourTimeNTZ,
    YearMonthDayPeriodTimeNTZ,
    MonthDayHourTimeNTZ,
    DatetimeRange,
    ColumnRepresentativeBase,
    ColumnRepresentativeTimes,
    TimeBasedDataAdjustment,
)
from chronify.datetime_range_generator import DatetimeRangeGenerator
from chronify.models import MappingTableSchema, TableSchema


class MapperColumnRepresentativeToDatetime(TimeSeriesMapperBase):

    """
    Mapper class to convert column representative time to datetime.

    Example
    -------

    Given the input row:

    | year | month | day | hour | value |
    |------|-------|-----|------|-------|
    | 2024 |   2   |  15 |  10  |  123  |

    Convert the row to datetime:

    |      timestamp      | value |
    |---------------------|-------|
    | 2024-02-15T10:00:00 |  123  |

    Methods
    -------

    map_time
        maps the column representative time to datetime in the provided sql engine.
    """

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
    ) -> None:
        super().__init__(backend, from_schema, to_schema, data_adjustment, wrap_time_allowed)

        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "Target schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(from_schema.time_config, ColumnRepresentativeBase):
            msg = "Source schema does not have a ColumnRepresentative time config. Use a different mapper."
            raise InvalidParameter(msg)

        self._to_time_config: DatetimeRange = to_schema.time_config
        self._from_time_config: ColumnRepresentativeTimes = from_schema.time_config

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        from_schema = self._from_schema
        drop_table = None
        if isinstance(self._from_time_config, YearMonthDayHourTimeNTZ):
            df_mapping, mapping_schema = self._create_ymdh_mapping(
                col_names=self._from_time_config.list_time_columns()
            )
        elif isinstance(self._from_time_config, MonthDayHourTimeNTZ):
            df_mapping, mapping_schema = self._create_mdh_mapping()
        elif isinstance(self._from_time_config, YearMonthDayPeriodTimeNTZ):
            int_mapping = self._intermediate_mapping_ymdp_to_ymdh(scratch_dir)
            from_schema = int_mapping
            drop_table = int_mapping.name
            df_mapping, mapping_schema = self._create_ymdh_mapping(
                col_names=int_mapping.time_config.list_time_columns()
            )
        else:
            msg = f"No mapping available for {type(self._from_time_config)}"
            raise InvalidParameter(msg)

        apply_mapping(
            df_mapping,
            mapping_schema,
            from_schema,
            self._to_schema,
            self._backend,
            self._data_adjustment,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

        if drop_table:
            table_type = "view" if self._backend.name == "spark" else "table"
            self._backend.execute_sql(f"DROP {table_type} IF EXISTS {drop_table}")

    def check_schema_consistency(self) -> None:
        if isinstance(self._from_time_config, MonthDayHourTimeNTZ):
            self._validate_mdh_time_config()

    def _validate_length_and_resolution(self) -> None:
        # not true for all input time config types
        if self._from_time_config.length != self._to_time_config.length:
            msg = "Length of time series arrays must match."
            raise InvalidParameter(msg)
        if self._to_time_config.resolution.total_seconds() != 3600:
            msg = "Resolution of destination schema must be 1 hour."
            raise InvalidParameter(msg)

    def _validate_mdh_time_config(self) -> None:
        if self._from_time_config.year is None:
            msg = "Year is required for mdh time range to be converter to DatetimeRange."
            raise InvalidParameter(msg)

    def _intermediate_mapping_ymdp_to_ymdh(self, scratch_dir: Path | None) -> TableSchema:
        """Convert ymdp to ymdh for intermediate mapping."""
        mapping_table_name = "intermediate_ymdp_to_ymdh"
        period_col = self._from_time_config.hour_columns[0]

        # Read distinct periods
        table = self._backend.table(self._from_schema.name)
        query = table.select(table[period_col]).distinct()
        periods = read_query(self._backend, query, self._from_time_config)
        df_mapping = generate_period_mapping(periods.iloc[:, 0])

        write_table(
            self._backend,
            df_mapping,
            mapping_table_name,
            [self._from_time_config],
            if_exists="replace",
            scratch_dir=scratch_dir,
        )

        # Build join query using Ibis
        ymdp_table = self._backend.table(self._from_schema.name)
        mapping_table = self._backend.table(mapping_table_name)

        # Select all columns from ymdp_table except period_col, plus hour from mapping
        ymdp_cols = list(ymdp_table.columns)
        select_cols = [ymdp_table[col] for col in ymdp_cols if col != period_col]
        select_cols.append(mapping_table["hour"])

        # Join and select
        joined = ymdp_table.join(
            mapping_table, ymdp_table[period_col] == mapping_table["from_period"]
        )
        query = joined.select(select_cols)

        intermediate_ymdh_table_name = "intermediate_Ymdh"
        self._backend.create_table(intermediate_ymdh_table_name, query)

        assert isinstance(
            self._from_time_config, YearMonthDayPeriodTimeNTZ
        ), "Intermediate mapping only valid for YearMonthDayPeriodNTZ time config"
        return self._create_intermediate_ymdh_schema(
            intermediate_ymdh_table_name, self._from_schema, self._from_time_config
        )

    def _create_intermediate_ymdh_schema(
        self, table_name: str, schema: TableSchema, time_config: YearMonthDayPeriodTimeNTZ
    ) -> TableSchema:
        """Create intermediate schema for ymdh time range from ymdp schema."""
        return TableSchema(
            name=table_name,
            value_column=schema.value_column,
            time_config=YearMonthDayHourTimeNTZ(
                length=time_config.length * 24,
                year=time_config.year,
                year_column=time_config.year_column,
                month_column=time_config.month_column,
                day_column=time_config.day_column,
                hour_columns=["hour"],
            ),
            time_array_id_columns=schema.time_array_id_columns,
        )

    def _iter_datetime(self) -> Generator[datetime, None, None]:
        datetime_generator = DatetimeRangeGenerator(self._to_time_config)
        yield from datetime_generator._iter_timestamps()

    def _create_ymdh_mapping(
        self, col_names: list[str] = ["year", "month", "day", "hour"]
    ) -> tuple[pd.DataFrame, MappingTableSchema]:
        """create mapping table and schema for ymdh time range."""
        row_generator = map(lambda ts: (ts, *ymdh_from_datetime(ts)), self._iter_datetime())
        df_mapping = pd.DataFrame(
            row_generator, columns=["timestamp", *["from_" + col for col in col_names]]
        )
        mapping_schema = MappingTableSchema(
            name="ymdh_mapping_table",
            time_configs=[self._to_time_config, self._from_time_config],
        )
        return df_mapping, mapping_schema

    def _create_mdh_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """create mapping table and schema for mdh time range."""
        row_generator = map(lambda ts: (ts, *mdh_from_datetime(ts)), self._iter_datetime())
        df_mapping = pd.DataFrame(
            row_generator, columns=["timestamp", "from_month", "from_day", "from_hour"]
        )
        mapping_schema = MappingTableSchema(
            name="mdh_mapping_table",
            time_configs=[self._to_time_config, self._from_time_config],
        )
        return df_mapping, mapping_schema


def ymdh_from_datetime(timestamp: datetime) -> tuple[int, int, int, int]:
    return timestamp.year, timestamp.month, timestamp.day, timestamp.hour + 1


def mdh_from_datetime(timestamp: datetime) -> tuple[int, int, int]:
    return timestamp.month, timestamp.day, timestamp.hour + 1


def generate_period_mapping(periods: "pd.Series[str]") -> pd.DataFrame:
    unique_periods = periods.unique()
    mappings = []
    for period_str in unique_periods:
        period_type, period_vals = parse_period(period_str)
        mappings.append(pd.DataFrame({"from_period": period_str, period_type: period_vals}))

    return pd.concat(mappings)


def parse_period(period: str) -> tuple[str, list[int]]:
    match = re.match(r"([HMD])(\d{1,2}-\d{1,2})", period)
    if match is None:
        msg = f"Cannot parse Period: {period}, expecting format like H1-4, D10-12, or M1-3"
        raise InvalidValue(msg)
    match_map = {"H": "hour", "M": "month", "D": "day"}
    val_type, val = match.groups()
    start, end = (int(v) for v in val.split("-"))
    vals = list(range(start, end + 1))
    return match_map[val_type], vals
