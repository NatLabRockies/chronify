import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from chronify.ibis.backend import IbisBackend
from chronify.models import TableSchema, MappingTableSchema
from chronify.exceptions import InvalidParameter, ConflictingInputsError
from chronify.time_series_mapper_base import TimeSeriesMapperBase, apply_mapping
from chronify.time_configs import DatetimeRange, TimeBasedDataAdjustment
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time_utils import (
    rolled_interval_timestamps,
    wrapped_time_timestamps,
    get_standard_time_zone,
)

logger = logging.getLogger(__name__)


class MapperDatetimeToDatetime(TimeSeriesMapperBase):
    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_schema: TableSchema,
        data_adjustment: Optional[TimeBasedDataAdjustment] = None,
        wrap_time_allowed: bool = False,
    ) -> None:
        super().__init__(backend, from_schema, to_schema, data_adjustment, wrap_time_allowed)
        if self._from_schema == self._to_schema and self._data_adjustment is None:
            msg = f"from_schema is the same as to_schema and no data_adjustment, nothing to do.\n{self._from_schema}"
            logger.info(msg)
        if not isinstance(self._from_schema.time_config, DatetimeRange):
            msg = "Source schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        if not isinstance(self._to_schema.time_config, DatetimeRange):
            msg = "Destination schema does not have DatetimeRange time config. Use a different mapper."
            raise InvalidParameter(msg)
        self._from_time_config: DatetimeRange = self._from_schema.time_config
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
        """Convert time columns with from_schema to to_schema configuration."""
        self.check_schema_consistency()
        df, mapping_schema = self._create_mapping()
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
        # TODO - add handling for changing resolution - Issue #30

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe
        Handles time interval type
        """
        from_time_col = "from_" + self._from_time_config.time_column
        to_time_col = self._to_time_config.time_column
        from_time_data = make_time_range_generator(self._from_time_config).list_timestamps()
        to_time_data = make_time_range_generator(
            self._to_time_config, leap_day_adjustment=self._data_adjustment.leap_day_adjustment
        ).list_timestamps()

        ser_from = pd.Series(from_time_data)
        # If from_tz or to_tz is naive, use tz_localize
        fm_tz = self._from_time_config.start.tzinfo
        to_tz = self._to_time_config.start.tzinfo
        match (fm_tz is None, to_tz is None):
            case (True, False):
                # get standard time zone of to_tz
                to_tz_std = get_standard_time_zone(to_tz)
                # tz-naive time does not have skips/dups, so always localize in std tz first
                ser_from = ser_from.dt.tz_localize(to_tz_std).dt.tz_convert(to_tz)
            case (False, True):
                # get standard time zone of fm_tz
                fm_tz_std = get_standard_time_zone(fm_tz)
                # convert to standard time zone of fm_tz before making it tz-naive
                ser_from = ser_from.dt.tz_convert(fm_tz_std).dt.tz_localize(to_tz)
        match (self._adjust_interval, self._wrap_time_allowed):
            case (True, _):
                ser = pd.Series(
                    rolled_interval_timestamps(
                        ser_from.tolist(),
                        self._from_time_config.interval_type,
                        self._to_time_config.interval_type,
                        to_time_data,
                    ),
                    index=ser_from.index,
                )
            case (False, True):
                ser = pd.Series(
                    wrapped_time_timestamps(ser_from.tolist(), to_time_data), index=ser_from.index
                )
            case (False, False):
                ser = ser_from

        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                to_time_col: ser,
            }
        )

        assert (
            df[to_time_col].nunique() == self._to_time_config.length
        ), "to_time_col does not have the right number of timestamps"
        from_time_config = self._from_time_config.model_copy(update={"time_column": from_time_col})
        mapping_schema = MappingTableSchema(
            name="mapping_table",
            time_configs=[
                from_time_config,
                self._to_time_config,
            ],
        )
        return df, mapping_schema
