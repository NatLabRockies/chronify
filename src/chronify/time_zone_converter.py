import abc
from zoneinfo import ZoneInfo
from datetime import tzinfo
from typing import Optional
from pathlib import Path
import pandas as pd

from chronify.ibis.base import IbisBackend
from chronify.models import TableSchema, MappingTableSchema
from chronify.time_configs import (
    DatetimeRangeBase,
    DatetimeRange,
    DatetimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.datetime_range_generator import (
    DatetimeRangeGeneratorExternalTimeZone,
)
from chronify.exceptions import InvalidParameter, MissingValue
from chronify.time_series_mapper_base import apply_mapping
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time import TimeDataType, TimeType
from chronify.time_utils import wrapped_time_timestamps, get_tzname


def convert_time_zone(
    backend: IbisBackend,
    src_schema: TableSchema,
    to_time_zone: tzinfo | None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Convert time zone of a table to a specified time zone.

    Output timestamp is tz-naive with a new time_zone column added.
    """
    tzc = TimeZoneConverter(backend, src_schema, to_time_zone)
    tzc.convert_time_zone(
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )
    return tzc._to_schema


def convert_time_zone_by_column(
    backend: IbisBackend,
    src_schema: TableSchema,
    time_zone_column: str,
    wrap_time_allowed: Optional[bool] = False,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Convert time zone of a table to multiple time zones specified by a column.
    Output timestamp is tz-naive, reflecting the local time relative to the time_zone_column.
    """
    tzc = TimeZoneConverterByColumn(backend, src_schema, time_zone_column, wrap_time_allowed)
    tzc.convert_time_zone(
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )
    return tzc._to_schema


class TimeZoneConverterBase(abc.ABC):
    """Base class for time zone conversion of time series data."""

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
    ):
        self._backend = backend
        self._check_from_schema(from_schema)
        self._from_schema = from_schema

    def _check_from_schema(self, from_schema: TableSchema) -> None:
        msg = ""
        if not isinstance(from_schema.time_config, DatetimeRange):
            msg += "Source schema must have DatetimeRange time config. "
        if (
            isinstance(from_schema.time_config, DatetimeRange)
            and from_schema.time_config.dtype != TimeDataType.TIMESTAMP_TZ
        ):
            msg += "Source schema time config dtype must be Timestamp_TZ. "
        if (
            isinstance(from_schema.time_config, DatetimeRange)
            and from_schema.time_config.start_time_is_tz_naive()
        ):
            msg += (
                "Source schema time config start time must be timezone-aware. "
                "This converter will convert time zones and return timestamps as tz-naive "
                "along with time zone information in a column. "
                "To localize timestamps from timezone-naive to timezone-aware, "
                "use TimeZoneLocalizer() or TimeZoneLocalizerByColumn() instead. "
            )
        if msg != "":
            raise InvalidParameter(msg)

    @abc.abstractmethod
    def generate_to_schema(self) -> TableSchema:
        """Generate to_schema based on from_schema"""

    @abc.abstractmethod
    def convert_time_zone(
        self,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Convert time zone of the from_schema"""


class TimeZoneConverter(TimeZoneConverterBase):
    """Convert tz-aware timestamps to a specified time zone (tz-naive output)."""

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_time_zone: tzinfo | None,
    ):
        super().__init__(backend, from_schema)
        self._to_time_zone = to_time_zone
        self._to_schema = self.generate_to_schema()

    def generate_to_time_config(self) -> DatetimeRangeWithTZColumn:
        assert isinstance(self._from_schema.time_config, DatetimeRange)
        time_kwargs = self._from_schema.time_config.model_dump()
        time_kwargs = dict(
            filter(
                lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                time_kwargs.items(),
            )
        )
        if self._to_time_zone:
            time_kwargs["start"] = (
                time_kwargs["start"].astimezone(self._to_time_zone).replace(tzinfo=None)
            )
        else:
            time_kwargs["start"] = time_kwargs["start"].replace(tzinfo=None)
        time_kwargs["dtype"] = TimeDataType.TIMESTAMP_NTZ
        time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
        time_kwargs["time_zone_column"] = "time_zone"
        time_kwargs["time_zones"] = [self._to_time_zone]
        return DatetimeRangeWithTZColumn(**time_kwargs)

    def generate_to_schema(self) -> TableSchema:
        to_time_config = self.generate_to_time_config()
        id_cols = self._from_schema.time_array_id_columns
        if to_time_config.time_zone_column not in id_cols:
            id_cols.append(to_time_config.time_zone_column)
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": to_time_config,
                "time_array_id_columns": id_cols,
            }
        )
        return to_schema

    def convert_time_zone(
        self,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        df, mapping_schema = self._create_mapping()
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._backend,
            TimeBasedDataAdjustment(),
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        assert isinstance(self._from_schema.time_config, DatetimeRange)
        time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()
        to_time_generator = make_time_range_generator(self._to_schema.time_config)
        assert isinstance(to_time_generator, DatetimeRangeGeneratorExternalTimeZone)
        to_time_data_dct = to_time_generator.list_timestamps_by_time_zone()

        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self._to_schema.time_config
        assert isinstance(to_time_config, DatetimeRangeWithTZColumn)
        tz_col = to_time_config.time_zone_column
        tz_name = get_tzname(self._to_time_zone)
        to_time_data = to_time_data_dct[tz_name]
        df = pd.DataFrame(
            {
                from_time_col: from_time_data,
                tz_col: tz_name,
                time_col: to_time_data,
            }
        )

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema


class TimeZoneConverterByColumn(TimeZoneConverterBase):
    """Convert tz-aware timestamps to multiple time zones specified by a column."""

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        time_zone_column: str,
        wrap_time_allowed: Optional[bool] = False,
    ):
        if time_zone_column not in from_schema.time_array_id_columns:
            msg = f"{time_zone_column=} is missing from {from_schema.time_array_id_columns=}"
            raise MissingValue(msg)
        super().__init__(backend, from_schema)
        self.time_zone_column = time_zone_column
        self._wrap_time_allowed = wrap_time_allowed
        self._to_schema = self.generate_to_schema()

    def generate_to_time_config(self) -> DatetimeRangeBase:
        assert isinstance(self._from_schema.time_config, DatetimeRange)
        time_kwargs = self._from_schema.time_config.model_dump()
        time_kwargs = dict(
            filter(
                lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                time_kwargs.items(),
            )
        )
        if self._wrap_time_allowed:
            time_kwargs["start"] = time_kwargs["start"].replace(tzinfo=None)

        time_kwargs["dtype"] = TimeDataType.TIMESTAMP_NTZ
        time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
        time_kwargs["time_zone_column"] = self.time_zone_column
        time_kwargs["time_zones"] = self._get_time_zones()
        return DatetimeRangeWithTZColumn(**time_kwargs)

    def generate_to_schema(self) -> TableSchema:
        id_cols = self._from_schema.time_array_id_columns
        if "time_zone" not in id_cols:
            id_cols.append("time_zone")
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self.generate_to_time_config(),
                "time_array_id_columns": id_cols,
            }
        )
        return to_schema

    def convert_time_zone(
        self,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        df, mapping_schema = self._create_mapping()
        apply_mapping(
            df,
            mapping_schema,
            self._from_schema,
            self._to_schema,
            self._backend,
            TimeBasedDataAdjustment(),
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _get_time_zones(self) -> list[tzinfo | None]:
        table = self._backend.table(self._from_schema.name)
        expr = (
            table.select(self.time_zone_column)
            .distinct()
            .filter(table[self.time_zone_column].notnull())
        )
        time_zones = self._backend.read_query(expr, self._from_schema.time_config)[
            self.time_zone_column
        ].to_list()
        time_zones = [None if tz == "None" else ZoneInfo(tz) for tz in time_zones]
        return time_zones

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        assert isinstance(self._from_schema.time_config, DatetimeRange)
        time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + time_col
        from_time_data = make_time_range_generator(self._from_schema.time_config).list_timestamps()
        to_time_generator = make_time_range_generator(self._to_schema.time_config)
        assert isinstance(to_time_generator, DatetimeRangeGeneratorExternalTimeZone)
        to_time_data_dct = to_time_generator.list_timestamps_by_time_zone()

        from_tz_col = "from_" + self.time_zone_column
        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self._to_schema.time_config

        df_tz = []
        for tz_name, time_data in to_time_data_dct.items():
            to_time_data: list[pd.Timestamp]
            if self._wrap_time_allowed:
                final_time_data = [x.tz_localize(None) for x in from_time_data]
                to_time_data = wrapped_time_timestamps(time_data, final_time_data)
            else:
                to_time_data = time_data
            df_tz.append(
                pd.DataFrame(
                    {
                        from_time_col: from_time_data,
                        from_tz_col: tz_name,
                        time_col: to_time_data,
                    }
                )
            )
        df = pd.concat(df_tz, ignore_index=True)

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema
