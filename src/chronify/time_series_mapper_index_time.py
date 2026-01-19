import logging
from pathlib import Path
from typing import Optional
import numpy as np
from datetime import datetime, timedelta

import pandas as pd

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRanges,
    IndexTimeRangeBase,
    IndexTimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time import TimeDataType, TimeType, DaylightSavingAdjustmentType, AggregationType

logger = logging.getLogger(__name__)


class MapperIndexTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
    ) -> None:
        # TODO: refactor to use new time configs
        super().__init__(backend, from_schema, to_schema, data_adjustment, wrap_time_allowed)
        self._dst_adjustment = self._data_adjustment.daylight_saving_adjustment
        if not isinstance(self._from_schema.time_config, IndexTimeRangeBase):
            msg = "Source schema does not have IndexTimeRangeBase time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(self._to_schema.time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: IndexTimeRanges = self._from_schema.time_config
        self._to_time_config: DatetimeRange = self._to_schema.time_config

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()
        self._check_time_resolution()
        self._check_time_length()

    def _check_time_resolution(self) -> None:
        if self._from_time_config.resolution != self._to_time_config.resolution:
            msg = "Handling of changing time resolution is not supported yet."
            raise NotImplementedError(msg)

    def _check_time_length(self) -> None:
        flen, tlen = self._from_time_config.length, self._to_time_config.length
        if flen != tlen and not self._wrap_time_allowed:
            msg = f"Length must match between {self._from_schema.__class__} from_schema and {self._to_schema.__class__} to_schema. {flen} vs. {tlen} OR wrap_time_allowed must be set to True"
            raise ConflictingInputsError(msg)

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert from index time to its represented datetime"""
        self.check_schema_consistency()

        # Convert from index time to its represented datetime
        if self._from_time_config.time_type == TimeType.INDEX_TZ_COL:
            if (
                self._dst_adjustment
                == DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_DUPLICATE_FALLBACK
            ):
                (
                    df,
                    mapping_schema,
                    mapped_schema,
                ) = self._create_interm_map_with_time_zone_and_dst_adjustment(
                    interpolate_fallback=False
                )
                resampling_operation = None
            elif (
                self._dst_adjustment
                == DaylightSavingAdjustmentType.DROP_SPRING_FORWARD_INTERPOLATE_FALLBACK
            ):
                (
                    df,
                    mapping_schema,
                    mapped_schema,
                ) = self._create_interm_map_with_time_zone_and_dst_adjustment(
                    interpolate_fallback=True
                )
                resampling_operation = AggregationType.SUM
            else:
                (
                    df,
                    mapping_schema,
                    mapped_schema,
                ) = self._create_interm_map_with_time_zone()
                resampling_operation = None
        else:
            df, mapping_schema, mapped_schema = self._create_interm_map()
            resampling_operation = None

        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            mapped_schema,
            self._backend,
            TimeBasedDataAdjustment(),
            resampling_operation=resampling_operation,
            scratch_dir=scratch_dir,
            check_mapped_timestamps=False,
        )

        # Convert from represented datetime to dst time_config
        MapperDatetimeToDatetime(
            self._backend,
            mapped_schema,
            self._to_schema,
            self._data_adjustment,
            self._wrap_time_allowed,
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_intermediate_schema(self) -> TableSchema:
        """Create the intermediate table schema for converting index time to its represented datetime"""
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["time_type"] = TimeType.DATETIME
        time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_column"] = "represented_time"

        schema_kwargs = self._from_schema.model_dump()
        schema_kwargs["name"] += "_intermediate"

        schema_kwargs["time_config"] = DatetimeRange(**time_kwargs)
        schema = TableSchema(**schema_kwargs)

        return schema

    def _create_local_time_config(self, time_zone: str) -> DatetimeRange:
        """Create the intermediate time config for converting index local time to its represented datetime"""
        time_kwargs = self._from_time_config.model_dump()
        time_kwargs = dict(
            filter(lambda k_v: k_v[0] in DatetimeRange.model_fields, time_kwargs.items())
        )
        time_kwargs["time_type"] = TimeType.DATETIME
        if isinstance(self._from_time_config.start_timestamp, datetime):
            # TODO: this is a hack. datetime is correct but only is present when Hive is used.
            # The code requires pandas Timestamps.
            time_kwargs["start"] = pd.Timestamp(self._from_time_config.start_timestamp)
        else:
            time_kwargs["start"] = self._from_time_config.start_timestamp
        time_kwargs["time_column"] = "represented_time"
        time_config = DatetimeRange(**time_kwargs)
        assert (
            time_config.start.tzinfo is None
        ), "Start time must be tz-naive for local time config"
        time_config.start = time_config.start.tz_localize(time_zone)  # type: ignore[attr-defined]

        return time_config

    def _create_interm_map(self) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX time to its represented datetime"""
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column
        mapped_time_data = make_time_range_generator(mapped_schema.time_config).list_timestamps()

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        from_time_config = self._from_time_config.model_copy(update={"time_column": from_time_col})
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                mapped_time_col: mapped_time_data,
            }
        )
        return df, mapping_schema, mapped_schema

    def _create_interm_map_with_time_zone(
        self,
    ) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX_TZ_COL time to its represented datetime"""
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        tz_col = self._from_time_config.get_time_zone_column()
        assert tz_col is not None, "Expecting a time zone column for INDEX_TZ_COL"
        from_tz_col = "from_" + tz_col

        table = self._backend.table(self._from_schema.name)
        query = table.select(table[tz_col]).distinct().filter(table[tz_col].notnull())
        time_zones = read_query(self._backend, query, self._from_time_config)[tz_col].to_list()

        from_time_config = self._from_time_config.model_copy(
            update={"time_column": from_time_col, "time_zone_column": from_tz_col}
        )
        assert isinstance(from_time_config, IndexTimeRangeWithTZColumn)

        df_tz = []
        to_tz = self._to_time_config.start.tzinfo
        for time_zone in time_zones:
            config_tz = self._create_local_time_config(time_zone)
            time_data = make_time_range_generator(config_tz).list_timestamps()
            # Preemptively convert to dst time tzinfo, otherwise pandas treats the col,
            # which consists of the timeseries of different time zones, as an object col
            mapped_time_data = [x.tz_convert(to_tz) for x in time_data]
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: time_zone,
                        mapped_time_col: mapped_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        # Update mapped_schema
        mapped_schema.time_config.start = df[mapped_time_col].min()
        mapped_schema.time_config.length = df[mapped_time_col].nunique()
        mapped_schema.time_config.dtype = TimeDataType.TIMESTAMP_TZ

        mapping_schema = MappingTableSchema(
            name="mapping_table_index_time",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        return df, mapping_schema, mapped_schema

    def _create_interm_map_with_time_zone_and_dst_adjustment(
        self,
        interpolate_fallback: bool = False,
    ) -> tuple[pd.DataFrame, MappingTableSchema, TableSchema]:
        """Create mapping dataframe for converting INDEX_TZ_COL time to its represented datetime
        with time-based daylight_saving adjustment that
        drops the spring-forward hour and, per user input,
        interpolates or duplicates the fall-back hour
        """
        mapped_schema = self._create_intermediate_schema()
        assert isinstance(mapped_schema.time_config, DatetimeRange)
        mapped_time_col = mapped_schema.time_config.time_column
        mapped_time_data_ntz = make_time_range_generator(
            mapped_schema.time_config
        ).list_timestamps()

        from_time_col = "from_" + self._from_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()

        df_ntz = pd.DataFrame(
            {
                from_time_col: from_time_data,
                "clock_time": mapped_time_data_ntz,
            }
        )
        df_ntz["clock_time"] = df_ntz["clock_time"].astype(str)

        tz_col = self._from_time_config.get_time_zone_column()
        assert tz_col is not None, "Expecting a time zone column for INDEX_TZ_COL"
        from_tz_col = "from_" + tz_col

        table = self._backend.table(self._from_schema.name)
        query = table.select(table[tz_col]).distinct().filter(table[tz_col].notnull())
        time_zones = read_query(self._backend, query, self._from_time_config)[tz_col].to_list()

        from_time_config = self._from_time_config.model_copy(
            update={"time_column": from_time_col, "time_zone_column": from_tz_col}
        )
        assert isinstance(from_time_config, IndexTimeRangeWithTZColumn)

        df_tz = []
        for time_zone in time_zones:
            if interpolate_fallback:
                df_map = self._create_fallback_interpolation_map(
                    time_zone, from_tz_col, mapped_time_col
                )
            else:
                df_map = self._create_fallback_duplication_map(
                    time_zone, from_tz_col, mapped_time_col
                )
            df_tz.append(df_map)
        df = pd.concat(df_tz, ignore_index=True)
        df = df.merge(df_ntz, on="clock_time").drop(columns=["clock_time"])

        # Update mapped_schema
        mapped_schema.time_config.start = df[mapped_time_col].min()
        mapped_schema.time_config.length = df[mapped_time_col].nunique()
        mapped_schema.time_config.dtype = TimeDataType.TIMESTAMP_TZ

        mapping_schema = MappingTableSchema(
            name="mapping_table_index_time_with_dst_adjustment",
            time_configs=[from_time_config, mapped_schema.time_config],
        )
        return df, mapping_schema, mapped_schema

    def _create_fallback_duplication_map(
        self, time_zone: str, from_tz_col: str, mapped_time_col: str
    ) -> pd.DataFrame:
        config_tz = self._create_local_time_config(time_zone)
        time_data = make_time_range_generator(config_tz).list_timestamps()
        # Extract clock time
        clock_time_data = [x.strftime("%Y-%m-%d %H:%M:%S") for x in time_data]
        # Preemptively convert to dst time tzinfo, otherwise pandas treats the col,
        # which consists of the timeseries of different time zones, as an object col
        to_tz = self._to_time_config.start.tzinfo
        mapped_time_data = [x.tz_convert(to_tz) for x in time_data]

        df_map = pd.DataFrame(
            {
                from_tz_col: time_zone,
                "clock_time": clock_time_data,  # str, mapping key
                mapped_time_col: mapped_time_data,
            }
        )
        return df_map

    def _create_fallback_interpolation_map(
        self, time_zone: str, from_tz_col: str, mapped_time_col: str
    ) -> pd.DataFrame:
        config_tz = self._create_local_time_config(time_zone)
        time_data = make_time_range_generator(config_tz).list_timestamps()
        # Extract clock time
        clock_time_data = [x.strftime("%Y-%m-%d %H:%M:%S") for x in time_data]
        # Preemptively convert to dst time tzinfo, otherwise pandas treats the col,
        # which consists of the timeseries of different time zones, as an object col
        to_tz = self._to_time_config.start.tzinfo
        mapped_time_data = [x.tz_convert(to_tz) for x in time_data]

        df_map = pd.DataFrame(
            {
                "clock_time": clock_time_data,  # str, mapping key
                mapped_time_col: mapped_time_data,
                "factor": 1,
            }
        )
        limit = timedelta(hours=1) / self._from_time_config.resolution
        assert (limit % 1 == 0) and (limit > 0), f"limit must be an integer, {limit}"
        limit = int(limit)

        # create interpolation map by locating where timestamp is duplicated
        cond = df_map["clock_time"].duplicated()
        df_map.loc[cond, "clock_time"] = np.nan
        df_map.loc[cond, "factor"] = np.nan
        df_map["lb"] = df_map["clock_time"].ffill(limit=limit).where(df_map["clock_time"].isna())
        df_map["ub"] = df_map["clock_time"].bfill(limit=limit).where(df_map["clock_time"].isna())

        # calculate ub_factor by counting consecutive values in ub
        x = ~df_map["ub"].isna()
        consecutive_count = x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        df_map["ub_factor"] = consecutive_count.replace(0, np.nan) / (limit + 1)
        df_map["lb_factor"] = 1 - df_map["ub_factor"]

        # capping: if a row do not have both lb and ub, cannot interpolate, set factor to 1
        for fact_col in ["lb_factor", "ub_factor"]:
            cond = ~(df_map[fact_col].where(df_map["lb"].isna() | df_map["ub"].isna()).isna())
            df_map.loc[cond, fact_col] = 1

        # finalize table by reducing columns
        lst = []
        for ts_col, fact_col in zip(
            ["clock_time", "lb", "ub"], ["factor", "lb_factor", "ub_factor"]
        ):
            lst.append(
                df_map.loc[~df_map[ts_col].isna(), [mapped_time_col, ts_col, fact_col]].rename(
                    columns={ts_col: "clock_time", fact_col: "factor"}
                )
            )
        df_map2 = pd.concat(lst).sort_values(by=[mapped_time_col], ignore_index=True)
        assert df_map2.groupby(mapped_time_col)["factor"].sum().unique().round(3) == np.array([1])

        df_map2[from_tz_col] = time_zone
        return df_map2
