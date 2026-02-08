"""Definitions related to time"""

from enum import StrEnum
from typing import NamedTuple, Union


class TimeType(StrEnum):
    """Defines the formats of time config / representation."""

    DATETIME = "datetime"
    DATETIME_TZ_COL = "datetime_tz_col"
    ANNUAL = "annual"
    INDEX = "index"
    INDEX_TZ_COL = "index_tz_col"
    REPRESENTATIVE_PERIOD_NTZ = "representative_period_ntz"
    REPRESENTATIVE_PERIOD_TZ = "representative_period_tz"
    YEAR_MONTH_DAY_HOUR_NTZ = "year_month_day_hour"
    MONTH_DAY_HOUR_NTZ = "month_day_hour"
    YEAR_MONTH_DAY_PERIOD_NTZ = "year_month_day_period"


class TimeDataType(StrEnum):
    """Defines the data types of datetime columns in load_data."""

    TIMESTAMP_TZ = "timestamp_tz"
    TIMESTAMP_NTZ = "timestamp_ntz"
    STRING = "string"
    TIMESTAMP_IN_PARTS = "timestamp_in_parts"


class RepresentativePeriodFormat(StrEnum):
    """Defines the supported formats for representative period data."""

    # All instances of this Enum must declare frequency.
    # This Enum may be replaced by a generic implementation in order to support a large
    # number of permutations (seasons, weekend day vs week day, sub-hour time, etc).

    ONE_WEEK_PER_MONTH_BY_HOUR = "one_week_per_month_by_hour"
    ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR = (
        "one_weekday_day_and_one_weekend_day_per_month_by_hour",
    )


class OneWeekPerMonthByHour(NamedTuple):
    month: int
    day_of_week: int
    hour: int


class OneWeekdayDayOneWeekendDayPerMonthByHour(NamedTuple):
    month: int
    is_weekday: bool
    hour: int


def list_representative_time_columns(format_type: RepresentativePeriodFormat) -> list[str]:
    """Return the time columns for the format."""
    match format_type:
        case RepresentativePeriodFormat.ONE_WEEK_PER_MONTH_BY_HOUR:
            columns = list(OneWeekPerMonthByHour._fields)
        case RepresentativePeriodFormat.ONE_WEEKDAY_DAY_AND_ONE_WEEKEND_DAY_PER_MONTH_BY_HOUR:
            columns = list(OneWeekdayDayOneWeekendDayPerMonthByHour._fields)
        case _:
            msg = str(format_type)
            raise NotImplementedError(msg)
    return list(columns)


class LeapDayAdjustmentType(StrEnum):
    """Leap day adjustment enum types"""

    DROP_DEC31 = "drop_dec31"
    DROP_FEB29 = "drop_feb29"
    DROP_JAN1 = "drop_jan1"
    NONE = "none"


class DaylightSavingAdjustmentType(StrEnum):
    """Daylight saving data adjustment enum types"""

    DROP_SPRING_FORWARD_DUPLICATE_FALLBACK = "drop_spring_forward_duplicate_fallback"
    DROP_SPRING_FORWARD_INTERPOLATE_FALLBACK = "drop_spring_forward_interpolate_fallback"
    NONE = "none"


class TimeIntervalType(StrEnum):
    """Time interval enum types"""

    # TODO: R2PD uses a different set; do we want to align?
    # https://github.com/Smart-DS/R2PD/blob/master/R2PD/tshelpers.py#L15

    PERIOD_ENDING = "period_ending"
    # description="A time interval that is period ending is coded by the end time. E.g., 2pm (with"
    # " freq=1h) represents a period of time between 1-2pm.",
    PERIOD_BEGINNING = "period_beginning"
    # description="A time interval that is period beginning is coded by the beginning time. E.g.,"
    # " 2pm (with freq=01:00:00) represents a period of time between 2-3pm. This is the dsgrid"
    # " default.",
    INSTANTANEOUS = "instantaneous"
    # description="The time record value represents measured, instantaneous time",


class MeasurementType(StrEnum):
    """Time value measurement enum types"""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    MEASURED = "measured"
    # description="Data values represent the measured value at that reported time",
    TOTAL = "total"
    # description="Data values represent the sum of values in a time range",


class AggregationType(StrEnum):
    """Operation types for resampling / aggregation"""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class DisaggregationType(StrEnum):
    """Operation types for resampling / disaggregation"""

    INTERPOLATE = "interpolate"
    DUPLICATE_FFILL = "duplicate_ffill"
    DUPLICATE_BFILL = "duplicate_bfill"
    UNIFORM_DISAGGREGATE = "uniform_disaggregate"


ResamplingOperationType = Union[AggregationType, DisaggregationType]
