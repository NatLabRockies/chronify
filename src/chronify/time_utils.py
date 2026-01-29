"""Functions related to time"""

from calendar import isleap
import logging
from numpy.typing import NDArray
import numpy as np
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd

from chronify.time import (
    LeapDayAdjustmentType,
    TimeIntervalType,
)
from chronify.exceptions import InvalidParameter

logger = logging.getLogger(__name__)


def adjust_timestamp_by_dst_offset(timestamp: datetime, resolution: timedelta) -> datetime:
    """Reduce the timestamps within the daylight saving range by 1 hour.
    Used to ensure that a time series at daily (or lower) resolution returns each day at the
    same timestamp in prevailing time, an expected behavior in most standard libraries.
    (e.g., ensure a time series can return 2018-03-11 00:00, 2018-03-12 00:00, 2018-03-13 00:00...
    instead of 2018-03-11 00:00, 2018-03-12 01:00, 2018-03-13 01:00...)
    """
    if resolution < timedelta(hours=24):
        return timestamp

    offset = timestamp.dst() or timedelta(hours=0)
    return timestamp - offset


def shifted_interval_timestamps(
    ts_list: list[datetime],
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
) -> list[datetime]:
    """Shift ts_list by ONE time interval based on interval type.

    Example:
    >>> ts_list = pd.date_range("2018-12-31 23:00", periods=3, freq="h").tolist()
    [Timestamp('2018-12-31 23:00:00'), Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00')]

    >>> ts_list2 = shifted_interval_timestamps(
    ...     ts_list, TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING
    ... )
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00')]

    >>> ts_list2 = shifted_interval_timestamps(
    ...     ts_list, TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING
    ... )
    [Timestamp('2018-12-31 22:00:00'), Timestamp('2018-12-31 23:00:00'), Timestamp('2019-01-01 00:00:00')]
    """
    assert (
        from_interval_type != to_interval_type
    ), f"from_ and to_interval_type are the same: {from_interval_type}"
    arr: NDArray[np.datetime64] = np.sort(ts_list)  # type: ignore
    freqs = set((np.roll(arr, -1) - arr)[:-1])
    assert len(freqs) == 1, f"Timeseries must have exactly one frequency, found: {freqs}"
    freq: np.timedelta64 = next(iter(freqs))

    match (from_interval_type, to_interval_type):
        case (TimeIntervalType.PERIOD_BEGINNING, TimeIntervalType.PERIOD_ENDING):
            # shift time forward, 1am pd-beginning >> 2am pd-ending
            mult = 1
        case (TimeIntervalType.PERIOD_ENDING, TimeIntervalType.PERIOD_BEGINNING):
            # shift time backward
            mult = -1
        case _:
            msg = f"Cannot handle from {from_interval_type} to {to_interval_type}"
            raise InvalidParameter(msg)
    ts_list2 = (arr + freq * mult).tolist()
    return ts_list2  # type: ignore


def wrapped_time_timestamps(
    ts_list: list[datetime],
    to_timestamps: list[datetime],
) -> list[datetime]:
    """Returns the replacement timestamps in order to wrap the ts_list into the to_timestamps range.

    Example:
    >>> ts_list = pd.date_range("2018-12-31 23:00", periods=3, freq="h").tolist()
    [Timestamp('2018-12-31 23:00:00'), Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00')]

    >>> to_timestamps = pd.date_range("2019-01-01 00:00", periods=3, freq="h").tolist()
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00')]

    >>> ts_list2 = wrapped_time_timestamps(ts_list, to_timestamps)
    [Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00')]
    """
    to_arr = np.sort(np.array(to_timestamps))
    freqs = set((np.roll(to_arr, -1) - to_arr)[:-1])
    assert len(freqs) == 1, f"Timeseries must have exactly one frequency, found: {freqs}"
    freq = next(iter(freqs))
    tmin, tmax = to_arr[0], to_arr[-1]
    tdelta = tmax - tmin + freq

    arr = pd.Series(ts_list)  # np.array is not as robust as pd.Series here
    arr2 = arr.copy()
    lower_cond = arr < tmin
    if lower_cond.sum() > 0:
        arr2.loc[lower_cond] += tdelta
    upper_cond = arr > tmax
    if upper_cond.sum() > 0:
        arr2.loc[upper_cond] -= tdelta
    ts_list2 = arr2.tolist()
    return ts_list2  # type: ignore


def rolled_interval_timestamps(
    ts_list: list[datetime],
    from_interval_type: TimeIntervalType,
    to_interval_type: TimeIntervalType,
    to_timestamps: list[datetime],
) -> list[datetime]:
    """Roll ts_list by shifting time interval based on interval type and then
    wrapping timestamps according to to_timestamps.

    Example:
    >>> ts_list = pd.date_range("2019-01-01 00:00", periods=3, freq="h").tolist()  # period-ending
    [Timestamp('2018-12-31 23:00:00'), Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00')]

    >>> to_timestamps = pd.date_range(
    ...     "2019-01-01 00:00", periods=3, freq="h"
    ... ).tolist()  # period-beginning
    [Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00'), Timestamp('2019-01-01 02:00:00')]

    >>> ts_list2 = rolled_interval_timestamps(
    ...     ts_list,
    ...     TimeIntervalType.PERIOD_ENDING,
    ...     TimeIntervalType.PERIOD_BEGINNING,
    ...     to_timestamps,
    ... )
    [Timestamp('2019-01-01 02:00:00'), Timestamp('2019-01-01 00:00:00'), Timestamp('2019-01-01 01:00:00')]
    """
    ts_list2 = shifted_interval_timestamps(ts_list, from_interval_type, to_interval_type)
    ts_list3 = wrapped_time_timestamps(ts_list2, to_timestamps)
    return ts_list3


def is_prevailing_time_zone(tz: tzinfo | None) -> bool:
    """Check that tz is a prevailing time zone"""
    if not tz:
        return False
    ts1 = datetime(year=2020, month=1, day=1, tzinfo=tz)
    ts2 = datetime(year=2020, month=6, day=1, tzinfo=tz)

    return ts1.utcoffset() != ts2.utcoffset()


def is_standard_time_zone(tz: tzinfo | None) -> bool:
    """Check that tz is a standard time zone"""
    if not tz:
        return False
    ts1 = datetime(year=2020, month=1, day=1, tzinfo=tz)
    ts2 = datetime(year=2020, month=6, day=1, tzinfo=tz)

    return ts1.utcoffset() == ts2.utcoffset()


def get_standard_time_zone(tz: tzinfo | None) -> tzinfo | None:
    """Get the standard time zone counterpart of tz"""
    ts = datetime(year=2020, month=1, day=1, tzinfo=tz)
    std_tz_name = ts.tzname()
    if not std_tz_name:
        return None
    try:
        return ZoneInfo(std_tz_name)
    except ZoneInfoNotFoundError:
        utcoffset = ts.utcoffset()
        if not utcoffset:
            return None
        return timezone(utcoffset)


def get_tzname(tz: tzinfo | None) -> str:
    """Get the time zone name of tz
    Note: except for the tzname extracted from ZoneInfo,
    tzname may not be reinstantiated into a tzinfo object
    """
    if not tz:
        return "None"
    if isinstance(tz, ZoneInfo):
        return tz.key
    ts = datetime(year=2020, month=1, day=1, tzinfo=tz)
    return tz.tzname(ts)  # type: ignore


def list_timestamps(
    start: datetime, length: int, resolution: timedelta, adjustment: LeapDayAdjustmentType
) -> list[pd.Timestamp]:
    """Generate timestamps with adjustment using pd.date_range().
    Established time library can handle historical changes in time zone conversion to UTC.
    (e.g. Algeria (Africa/Algiers) changed from UTC+0 to UTC+1 on April 25, 1980)
    """
    pd_timestamps = pd.date_range(
        start=start,
        periods=length,
        freq=resolution,
    )

    match adjustment:
        case LeapDayAdjustmentType.DROP_FEB29:
            timestamps = [
                ts
                for ts in pd_timestamps
                if not (isleap(ts.year) and ts.month == 2 and ts.day == 29)
            ]
        case LeapDayAdjustmentType.DROP_DEC31:
            timestamps = [
                ts
                for ts in pd_timestamps
                if not (isleap(ts.year) and ts.month == 12 and ts.day == 31)
            ]
        case LeapDayAdjustmentType.DROP_JAN1:
            timestamps = [
                ts
                for ts in pd_timestamps
                if not (isleap(ts.year) and ts.month == 1 and ts.day == 1)
            ]
        case _:
            timestamps = pd_timestamps.tolist()
    return timestamps
