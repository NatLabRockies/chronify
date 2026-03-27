from pathlib import Path
from typing import Optional

from chronify.ibis.backend import IbisBackend
from chronify.models import TableSchema

from chronify.time_series_mapper_representative import MapperRepresentativeTimeToDatetime
from chronify.time_series_mapper_datetime import MapperDatetimeToDatetime
from chronify.time_series_mapper_index_time import MapperIndexTimeToDatetime
from chronify.time_series_mapper_column_representative_to_datetime import (
    MapperColumnRepresentativeToDatetime,
)
from chronify.time_configs import (
    DatetimeRange,
    IndexTimeRangeBase,
    RepresentativePeriodTimeBase,
    TimeBasedDataAdjustment,
    ColumnRepresentativeBase,
)


def map_time(
    backend: IbisBackend,
    from_schema: TableSchema,
    to_schema: TableSchema,
    data_adjustment: Optional[TimeBasedDataAdjustment] = None,
    wrap_time_allowed: bool = False,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> None:
    """Function to map time using the appropriate TimeSeriesMapper model."""
    if isinstance(from_schema.time_config, RepresentativePeriodTimeBase) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperRepresentativeTimeToDatetime(
            backend, from_schema, to_schema, data_adjustment, wrap_time_allowed
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    elif isinstance(from_schema.time_config, DatetimeRange) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperDatetimeToDatetime(
            backend, from_schema, to_schema, data_adjustment, wrap_time_allowed
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    elif isinstance(from_schema.time_config, IndexTimeRangeBase) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        MapperIndexTimeToDatetime(
            backend, from_schema, to_schema, data_adjustment, wrap_time_allowed
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )
    elif isinstance(from_schema.time_config, ColumnRepresentativeBase) and isinstance(
        to_schema.time_config, DatetimeRange
    ):
        # No way to generate expected timestamps for YearMonthDayPeriodTimeNTZ
        # Is there a way to only check the output datetime timestamps?
        MapperColumnRepresentativeToDatetime(
            backend, from_schema, to_schema, data_adjustment, wrap_time_allowed
        ).map_time(
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=from_schema.time_config.check_timestamps,
        )
    else:
        msg = f"No mapping function for {from_schema.time_config.__class__=} >> {to_schema.time_config.__class__=}"
        raise NotImplementedError(msg)
