import abc
import warnings
from zoneinfo import ZoneInfo
from datetime import tzinfo
from typing import Optional
from pathlib import Path
import pandas as pd
from pandas import DatetimeTZDtype

from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query
from chronify.models import TableSchema, MappingTableSchema
from chronify.time_configs import (
    DatetimeRangeBase,
    DatetimeRange,
    DatetimeRangeWithTZColumn,
    TimeBasedDataAdjustment,
)
from chronify.datetime_range_generator import (
    DatetimeRangeGenerator,
    DatetimeRangeGeneratorExternalTimeZone,
)
from chronify.exceptions import InvalidParameter, MissingValue
from chronify.time_series_mapper_base import apply_mapping
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.time import TimeDataType, TimeType
from chronify.time_series_mapper import map_time
from chronify.time_utils import get_standard_time_zone, is_standard_time_zone


def localize_time_zone(
    backend: IbisBackend,
    src_schema: TableSchema,
    to_time_zone: tzinfo | None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Localize TIMESTAMP_NTZ time column in a table to a specified standard time zone.
    Input data must be in a standard time zone (without DST) because it's ambiguous to localize
    tz-naive timestamps with skips and duplicates to a prevailing time zone.

    Updates table to TIMESTAMP_TZ time column and returns a new time config.

    Parameters
    ----------
    backend : IbisBackend
        Ibis backend.
    src_schema : TableSchema
        Defines the source table in the database.
    to_time_zone : tzinfo or None
        Standard time zone to convert to. If None, convert to tz-naive.
    scratch_dir : pathlib.Path, optional
        Directory to use for temporary writes. Defaults to the system's tmp filesystem.
    output_file : pathlib.Path, optional
        If set, write the mapped table to this Parquet file.
    check_mapped_timestamps : bool, optional
        Perform time checks on the result of the mapping operation. This can be slow and
        is not required.

    Returns
    -------
    TableSchema
        Schema of output table with converted timestamps.
    """
    tzl = TimeZoneLocalizer(backend, src_schema, to_time_zone)
    tzl.localize_time_zone(
        scratch_dir=scratch_dir,
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )

    return tzl._to_schema


def localize_time_zone_by_column(
    backend: IbisBackend,
    src_schema: TableSchema,
    time_zone_column: Optional[str] = None,
    scratch_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    check_mapped_timestamps: bool = False,
) -> TableSchema:
    """Localize TIMESTAMP_NTZ time column in a table to multiple time zones specified by a column.
    Updates table to TIMESTAMP_TZ time column and returns a new time config.

    Parameters
    ----------
    backend : IbisBackend
        Ibis backend.
    src_schema : TableSchema
        Defines the source table in the database.
    time_zone_column : Optional[str]
        Column name in the source table that contains the time zone information.
         - Required if src_schema.time_config is of type DatetimeRange.
         - Ignored if src_schema.time_config is of type DatetimeRangeWithTZColumn.
    scratch_dir : pathlib.Path, optional
        Directory to use for temporary writes. Default to the system's tmp filesystem.
    output_file : pathlib.Path, optional
        If set, write the mapped table to this Parquet file.
    check_mapped_timestamps : bool, optional
        Perform time checks on the result of the mapping operation. This can be slow and
        is not required.

    Returns
    -------
    dst_schema : TableSchema
        schema of output table with converted timestamps
    """
    if isinstance(src_schema.time_config, DatetimeRange) and time_zone_column is None:
        msg = (
            "time_zone_column must be provided when localizing time zones "
            "by column for DatetimeRange time config."
        )
        raise MissingValue(msg)

    tzl = TimeZoneLocalizerByColumn(backend, src_schema, time_zone_column=time_zone_column)
    tzl.localize_time_zone(
        scratch_dir=scratch_dir,
        output_file=output_file,
        check_mapped_timestamps=check_mapped_timestamps,
    )
    return tzl._to_schema


class TimeZoneLocalizerBase(abc.ABC):
    """Base class for time zone localization of time series data."""

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
    ):
        self._backend = backend
        self._from_schema = from_schema

    @staticmethod
    @abc.abstractmethod
    def _check_from_schema(from_schema: TableSchema) -> None:
        """Check that from_schema is valid for time zone localization"""

    @abc.abstractmethod
    def generate_to_schema(self) -> TableSchema:
        """Generate to_schema based on from_schema"""

    @abc.abstractmethod
    def localize_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        """Localize tz-naive timestamps to the time zone of the from_schema"""


class TimeZoneLocalizer(TimeZoneLocalizerBase):
    """Class for time zone localization of tz-naive time series data to a specified time zone.

    Input data table must contain tz-naive timestamps.
    Input time config must be of type DatetimeRange with Timestamp_NTZ dtype and tz-naive start time.
    to_time_zone must be a standard time zone (without DST) or None.
    Output data table will contain tz-aware timestamps.
    Output time config will be of type DatetimeRange with Timestamp_TZ dtype and tz-aware start time.
    """

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        to_time_zone: tzinfo | None,
    ):
        self._check_from_schema(from_schema)
        super().__init__(backend, from_schema)
        self._to_time_zone = self._check_standard_time_zone(to_time_zone)
        self._to_schema = self.generate_to_schema()

    @staticmethod
    def _check_from_schema(from_schema: TableSchema) -> None:
        if isinstance(from_schema.time_config, DatetimeRangeWithTZColumn):
            msg = (
                "Source schema time config is of type DatetimeRangeWithTZColumn, "
                f"try using TimeZoneLocalizerByColumn(). {from_schema.time_config}"
            )
            raise InvalidParameter(msg)
        if not isinstance(from_schema.time_config, DatetimeRange):
            msg = f"Source schema does not have DatetimeRange time config.\n{from_schema.time_config}"
            raise InvalidParameter(msg)
        msg = ""
        if from_schema.time_config.dtype != TimeDataType.TIMESTAMP_NTZ:
            msg += "Source schema time config dtype must be TIMESTAMP_NTZ. "
        if from_schema.time_config.start_time_is_tz_naive() is False:
            msg += (
                "Source schema time config start time must be tz-naive."
                "To convert between time zones for tz-aware timestamps, "
                "try using TimeZoneConverter() "
            )
        if msg != "":
            msg += f"\n{from_schema.time_config}"
            raise InvalidParameter(msg)

    @staticmethod
    def _check_standard_time_zone(to_time_zone: tzinfo | None) -> tzinfo | None:
        if to_time_zone is None:
            return None
        if not is_standard_time_zone(to_time_zone):
            std_tz = get_standard_time_zone(to_time_zone)
            msg = (
                "TimeZoneLocalizer only supports standard time zones (without DST). "
                f"{to_time_zone=} is not a standard time zone. Try instead: {std_tz}"
            )
            raise InvalidParameter(msg)
        return to_time_zone

    def generate_to_time_config(self) -> DatetimeRange:
        assert isinstance(self._from_schema.time_config, DatetimeRange)  # mypy
        to_time_config: DatetimeRange = self._from_schema.time_config.model_copy(
            update={
                "dtype": TimeDataType.TIMESTAMP_TZ
                if self._to_time_zone
                else TimeDataType.TIMESTAMP_NTZ,
                "start": self._from_schema.time_config.start.replace(tzinfo=self._to_time_zone),
            }
        )

        return to_time_config

    def generate_to_schema(self) -> TableSchema:
        to_time_config = self.generate_to_time_config()
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": to_time_config,
            }
        )
        return to_schema

    def localize_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
        output_file: Optional[Path] = None,
        check_mapped_timestamps: bool = False,
    ) -> None:
        map_time(
            backend=self._backend,
            from_schema=self._from_schema,
            to_schema=self._to_schema,
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )


class TimeZoneLocalizerByColumn(TimeZoneLocalizerBase):
    """Class for time zone localization of tz-naive time series data based on a time zone column.

    Input data table must contain tz-naive timestamps and a time zone column.
    Time zones in the time zone column must be standard time zones (without DST).
    Input time config must be of type DatetimeRangeWithTZColumn or DatetimeRange with Timestamp_NTZ dtype.
     - If DatetimeRangeWithTZColumn is used, time_zone_column, if provided, is ignored.
     - If DatetimeRange is used, time_zone_column must be provided. It is then converted to
       DatetimeRangeWithTZColumn internally.
    Output data table will contain tz-aware timestamps and the original time zone column.
    Output time config can be of type DatetimeRange or DatetimeRangeWithTZColumn with Timestamp_TZ dtype (see scenarios).

    I/O Time config scenarios:
    --------------------------------
    To localize tz-naive timestamps aligned_in_local_standard_time to multiple time zones specified in a column:
     - Input time config: DatetimeRangeWithTZColumn with tz-naive start time, Timestamp_NTZ dtype
     - Output time config: DatetimeRangeWithTZColumn with tz-naive start time, Timestamp_TZ dtype

    To localize tz-naive timestamps aligned_in_absolute_time to multiple time zones specified in a column:
     - Input time config: DatetimeRangeWithTZColumn with tz-aware start time, Timestamp_NTZ dtype
     - Output time config: DatetimeRange with tz-aware start time, Timestamp_TZ dtype
     Note: output time config is reduced to DatetimeRange (from DatetimeRangeWithTZColumn)
     since all timestamps are tz-aware and aligned in absolute time.
    --------------------------------
    """

    def __init__(
        self,
        backend: IbisBackend,
        from_schema: TableSchema,
        time_zone_column: Optional[str] = None,
    ):
        self._check_from_schema(from_schema)
        self._check_time_zone_column(from_schema, time_zone_column)
        # Make a deep copy to avoid mutating the original schema
        from_schema_copy = from_schema.model_copy(deep=True)
        super().__init__(backend, from_schema_copy)
        if isinstance(self._from_schema.time_config, DatetimeRange):
            self.time_zone_column = time_zone_column
            self._convert_from_time_config_to_datetime_range_with_tz_column()
        else:
            assert isinstance(self._from_schema.time_config, DatetimeRangeWithTZColumn)  # mypy
            self.time_zone_column = self._from_schema.time_config.time_zone_column
        self._check_standard_time_zones()
        self._to_schema = self.generate_to_schema()

    @staticmethod
    def _check_from_schema(from_schema: TableSchema) -> None:
        msg = ""
        if not isinstance(from_schema.time_config, (DatetimeRange, DatetimeRangeWithTZColumn)):
            msg += (
                "Source schema must have DatetimeRange or DatetimeRangeWithTZColumn time config. "
            )
            msg += f"\n{from_schema.time_config}"
            raise InvalidParameter(msg)
        if from_schema.time_config.dtype != TimeDataType.TIMESTAMP_NTZ:
            msg += "Source schema time config dtype must be TIMESTAMP_NTZ. "
        if msg != "":
            msg += f"\n{from_schema.time_config}"
            raise InvalidParameter(msg)

    @staticmethod
    def _check_time_zone_column(from_schema: TableSchema, time_zone_column: Optional[str]) -> None:
        if (
            isinstance(from_schema.time_config, DatetimeRangeWithTZColumn)
            and time_zone_column is not None
        ):
            msg = f"Input {time_zone_column=} will be ignored. time_zone_column is already defined in the time_config."
            warnings.warn(msg, stacklevel=3)

        msg = ""
        if isinstance(from_schema.time_config, DatetimeRange) and time_zone_column is None:
            msg += "time_zone_column must be provided when source schema time config is of type DatetimeRange. "
        # if time_zone_column not in from_schema.time_array_id_columns:
        #     msg = f"{time_zone_column=} must be in source schema time_array_id_columns."
        if msg != "":
            msg += f"\n{from_schema}"
            raise InvalidParameter(msg)

    def _check_standard_time_zones(self) -> None:
        """Check that all time zones in the time_zone_column are valid standard time zones."""
        assert isinstance(self._from_schema.time_config, DatetimeRangeWithTZColumn)  # mypy
        msg = ""
        time_zones = self._from_schema.time_config.time_zones
        for tz in time_zones:
            if tz is None:
                msg += "\nChronify does not support None time zone in time_zone_column. "
                raise InvalidParameter(msg)
            if not is_standard_time_zone(tz):
                std_tz = get_standard_time_zone(tz)
                msg = f"\n{tz} is not a standard time zone. Try instead: {std_tz}. "
        if msg != "":
            msg = (
                f"TimeZoneLocalizerByColumn only supports standard time zones (without DST). {time_zones}"
                + msg
            )
            raise InvalidParameter(msg)

    def _convert_from_time_config_to_datetime_range_with_tz_column(self) -> None:
        """Convert DatetimeRange from_schema time config to DatetimeRangeWithTZColumn time config
        for the rest of the workflow
        """
        assert isinstance(self._from_schema.time_config, DatetimeRange)
        time_kwargs = self._from_schema.time_config.model_dump()
        time_kwargs = dict(
            filter(
                lambda k_v: k_v[0] in DatetimeRangeWithTZColumn.model_fields,
                time_kwargs.items(),
            )
        )
        time_kwargs["time_type"] = TimeType.DATETIME_TZ_COL
        time_kwargs["time_zone_column"] = self.time_zone_column
        time_kwargs["time_zones"] = self._get_time_zones()

        self._from_schema.time_config = DatetimeRangeWithTZColumn(**time_kwargs)

    def generate_to_time_config(self) -> DatetimeRangeBase:
        assert isinstance(self._from_schema.time_config, DatetimeRangeWithTZColumn)  # mypy
        match self._from_schema.time_config.start_time_is_tz_naive():
            case True:
                # tz-naive start, aligned_in_local_time of the time zones
                to_time_config: DatetimeRangeWithTZColumn = (
                    self._from_schema.time_config.model_copy(
                        update={
                            "dtype": TimeDataType.TIMESTAMP_TZ,
                        }
                    )
                )
                return to_time_config
            case False:
                # tz-aware start, aligned_in_absolute_time, convert to DatetimeRange config
                time_kwargs = self._from_schema.time_config.model_dump()
                time_kwargs = dict(
                    filter(
                        lambda k_v: k_v[0] in DatetimeRange.model_fields,
                        time_kwargs.items(),
                    )
                )
                time_kwargs["dtype"] = TimeDataType.TIMESTAMP_TZ
                time_kwargs["time_type"] = TimeType.DATETIME
                return DatetimeRange(**time_kwargs)
            case _:
                msg = (
                    "Unable to determine if start time is tz-naive or tz-aware "
                    f"from time config: {self._from_schema.time_config}"
                )
                raise InvalidParameter(msg)

    def generate_to_schema(self) -> TableSchema:
        id_cols = self._from_schema.time_array_id_columns.copy()
        assert self.time_zone_column is not None  # Validated in __init__
        if self.time_zone_column not in id_cols:
            id_cols.append(self.time_zone_column)
        to_schema: TableSchema = self._from_schema.model_copy(
            update={
                "name": f"{self._from_schema.name}_tz_converted",
                "time_config": self.generate_to_time_config(),
                "time_array_id_columns": id_cols,
            }
        )
        return to_schema

    def localize_time_zone(
        self,
        scratch_dir: Optional[Path] = None,
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
            scratch_dir=scratch_dir,
            output_file=output_file,
            check_mapped_timestamps=check_mapped_timestamps,
        )

    def _get_time_zones(self) -> list[tzinfo | None]:
        table = self._backend.table(self._from_schema.name)
        query = (
            table.select(table[self.time_zone_column])
            .distinct()
            .filter(table[self.time_zone_column].notnull())
        )
        df = read_query(self._backend, query, self._from_schema.time_config)
        time_zones = df[self.time_zone_column].to_list()

        if "None" in time_zones and len(time_zones) > 1:
            msg = (
                "Chronify does not support mix of None and time zones in time_zone_column."
                "This is because databases do not support tz-aware and tz-naive timestamps "
                f"in the same column: {time_zones}"
            )
            raise InvalidParameter(msg)

        time_zones = [ZoneInfo(tz) for tz in time_zones]
        return time_zones

    def _create_mapping(self) -> tuple[pd.DataFrame, MappingTableSchema]:
        """Create mapping dataframe for localizing tz-naive datetime to column time zones"""
        assert isinstance(
            self._from_schema.time_config, DatetimeRangeWithTZColumn
        )  # Ensured by __init__
        time_col = self._from_schema.time_config.time_column
        from_time_col = "from_" + time_col
        from_time_generator = make_time_range_generator(self._from_schema.time_config)
        assert isinstance(from_time_generator, DatetimeRangeGeneratorExternalTimeZone)  # mypy
        from_time_data_dct = from_time_generator.list_timestamps_by_time_zone()

        to_time_generator = make_time_range_generator(self._to_schema.time_config)
        match to_time_generator:
            case DatetimeRangeGeneratorExternalTimeZone():  # mypy
                to_time_data_dct = to_time_generator.list_timestamps_by_time_zone()
            case DatetimeRangeGenerator():  # mypy
                to_time_data = to_time_generator.list_timestamps()
                to_time_data_dct = {tz_name: to_time_data for tz_name in from_time_data_dct.keys()}
            case _:
                msg = (
                    "to_time_generator must be of type "
                    "DatetimeRangeGeneratorExternalTimeZone or DatetimeRangeGenerator. "
                    f"Got {type(to_time_generator)}"
                )
                raise InvalidParameter(msg)

        assert self.time_zone_column is not None  # Validated in __init__
        from_tz_col = "from_" + self.time_zone_column
        from_time_config = self._from_schema.time_config.model_copy(
            update={"time_column": from_time_col}
        )
        to_time_config = self._to_schema.time_config

        df_tz = []
        primary_tz = ZoneInfo(list(from_time_data_dct.keys())[0])
        for tz_name, from_time_data in from_time_data_dct.items():
            # convert tz-aware timestamps to a single time zone for mapping
            # this is because pandas coerces tz-aware timestamps with mixed time zones to object dtype otherwise
            to_time_data = [ts.astimezone(primary_tz) for ts in to_time_data_dct[tz_name]]
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
        if not isinstance(df[time_col].dtype, DatetimeTZDtype):
            msg = (
                "Mapped time column is expected to be of "
                f"DatetimeTZDtype but got {df[time_col].dtype}"
            )
            raise InvalidParameter(msg)

        mapping_schema = MappingTableSchema(
            name="mapping_table_gtz_conversion",
            time_configs=[from_time_config, to_time_config],
        )
        return df, mapping_schema
