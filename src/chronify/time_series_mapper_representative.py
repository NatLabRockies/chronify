import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import (
    InvalidParameter,
)
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.representative_time_range_generator import RepresentativePeriodTimeGenerator
from chronify.time_configs import (
    DatetimeRange,
    RepresentativePeriodTimes,
    RepresentativePeriodTimeBase,
    TimeBasedDataAdjustment,
)
from chronify.time_utils import shifted_interval_timestamps

logger = logging.getLogger(__name__)


class MapperRepresentativeTimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
    ) -> None:
        super().__init__(backend, from_schema, to_schema, data_adjustment, wrap_time_allowed)
        if not isinstance(from_schema.time_config, RepresentativePeriodTimeBase):
            msg = "source schema does not have RepresentativePeriodTimeBase time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(to_schema.time_config, DatetimeRange):
            msg = "destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: RepresentativePeriodTimes = from_schema.time_config
        self._to_time_config: DatetimeRange = to_schema.time_config
        self._generator = RepresentativePeriodTimeGenerator(self._from_time_config)

    def check_schema_consistency(self) -> None:
        """Check that from_schema can produce to_schema."""
        self._check_table_columns_producibility()
        self._check_measurement_type_consistency()
        self._check_time_interval_type()

    def map_time(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert time columns with from_schema to to_schema configuration."""
        is_tz_naive = self._to_time_config.start_time_is_tz_naive()
        self.check_schema_consistency()

        df, mapping_schema = self._create_mapping(is_tz_naive)
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._backend,
            self._data_adjustment,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_mapping(self, is_tz_naive: bool) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe
        - Handles time interval type adjustment
        - Columns used to join the from_table are prefixed with "from_"
        """
        timestamp_generator = make_time_range_generator(
            self._to_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        )

        to_time_col = self._to_time_config.time_column
        dft = pd.Series(timestamp_generator.list_timestamps()).rename(to_time_col).to_frame()

        if self._adjust_interval:
            time_col = "to_" + to_time_col
            # Mapping works backward for representative time by shifting interval type of
            # to_time_config to match from_time_config before extracting time info
            dft[time_col] = shifted_interval_timestamps(
                dft[to_time_col].tolist(),
                self._to_time_config.interval_type,
                self._from_time_config.interval_type,
            )
        else:
            time_col = to_time_col

        from_columns = self._from_time_config.list_time_columns()
        if is_tz_naive:
            df = self._generator.create_tz_naive_mapping_dataframe(dft, time_col)
        else:
            tz_col = self._from_time_config.get_time_zone_column()
            assert tz_col is not None, "Expecting a time zone column for REPRESENTATIVE time"
            table = self._backend.table(self._from_schema.name)
            query = table.select(table[tz_col]).distinct().filter(table[tz_col].notnull())
            time_zones = read_query(self._backend, query, self._from_time_config)[tz_col].to_list()
            df = self._generator.create_tz_aware_mapping_dataframe(
                dft, time_col, time_zones, tz_col
            )
            from_columns.append(tz_col)

        prefixed_from_columns = ["from_" + x for x in from_columns]
        df = df.rename(columns=dict(zip(from_columns, prefixed_from_columns)))

        if time_col != to_time_col:
            df.drop(time_col, axis=1, inplace=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                self._to_time_config,  # only DatetimeRange
            ],
        )
        return df, mapping_schema
