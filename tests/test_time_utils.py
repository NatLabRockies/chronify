import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from chronify.time_utils import (
    adjust_timestamp_by_dst_offset,
    shifted_interval_timestamps,
    wrapped_time_timestamps,
    rolled_interval_timestamps,
    is_prevailing_time_zone,
    is_standard_time_zone,
    get_standard_time_zone,
    get_tzname,
)
from chronify.time import TimeIntervalType


def test_adjust_timestamp_by_dst_offset() -> None:
    # DST-aware datetime vs standard time zone
    tzs = [ZoneInfo("America/New_York"), ZoneInfo("Etc/GMT+5")]
    hours = [23, 0]
    for tz, hour in zip(tzs, hours):
        dt = datetime(2020, 7, 1, 0, 0, tzinfo=tz)
        res = adjust_timestamp_by_dst_offset(dt, timedelta(days=1))
        assert res.hour == hour


def test_shifted_interval_timestamps_period_beginning_to_ending() -> None:
    ser = pd.date_range("2018-12-31 22:00", periods=4, freq="h").tolist()
    shifted = shifted_interval_timestamps(
        ser,
        TimeIntervalType.PERIOD_BEGINNING,
        TimeIntervalType.PERIOD_ENDING,
    )
    assert all(np.array(shifted) == np.array(ser) + pd.Timedelta(hours=1))


def test_shifted_interval_timestamps_period_ending_to_beginning() -> None:
    ser = pd.date_range("2018-12-31 22:00", periods=4, freq="h").tolist()
    shifted = shifted_interval_timestamps(
        ser,
        TimeIntervalType.PERIOD_ENDING,
        TimeIntervalType.PERIOD_BEGINNING,
    )
    assert all(np.array(shifted) == np.array(ser) - pd.Timedelta(hours=1))


def test_shifted_interval_timestamps_invalid() -> None:
    ser = pd.date_range("2018-12-31 22:00", periods=4, freq="h").tolist()
    with pytest.raises(Exception):
        shifted_interval_timestamps(
            ser,
            TimeIntervalType.PERIOD_BEGINNING,
            TimeIntervalType.PERIOD_BEGINNING,
        )


def test_wrapped_time_timestamps() -> None:
    ser = pd.date_range("2018-12-31 22:00", periods=4, freq="h").tolist()
    to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    wrapped = wrapped_time_timestamps(ser, to_timestamps)
    assert set(wrapped) <= set(to_timestamps)


def test_rolled_interval_timestamps() -> None:
    ser = pd.date_range("2018-12-31 22:00", periods=4, freq="h").tolist()
    to_timestamps = pd.date_range("2019-01-01 00:00", periods=4, freq="h").tolist()
    rolled = rolled_interval_timestamps(
        ser,
        TimeIntervalType.PERIOD_BEGINNING,
        TimeIntervalType.PERIOD_ENDING,
        to_timestamps,
    )
    assert set(rolled) <= set(to_timestamps)


def test_is_prevailing_time_zone() -> None:
    tz = ZoneInfo("America/New_York")
    assert is_prevailing_time_zone(tz) is True
    assert is_prevailing_time_zone(None) is False


def test_is_standard_time_zone() -> None:
    tz = timezone(timedelta(hours=0))
    assert is_standard_time_zone(tz) is True
    assert is_standard_time_zone(None) is False


def test_get_standard_time_zone() -> None:
    tzs = [
        ZoneInfo("America/New_York"),
        ZoneInfo("US/Eastern"),
        ZoneInfo("EST"),
        timezone(timedelta(hours=-5)),
        None,
    ]
    stzs = [
        ZoneInfo("Etc/GMT+5"),
        ZoneInfo("Etc/GMT+5"),
        ZoneInfo("Etc/GMT+5"),
        ZoneInfo("Etc/GMT+5"),
        None,
    ]
    for tz, stz in zip(tzs, stzs):
        std_tz = get_standard_time_zone(tz)
        if tz is None:
            assert std_tz is None
        else:
            assert is_standard_time_zone(stz) is True
            assert is_standard_time_zone(std_tz) is True
            ts = datetime(2020, 6, 1, 0, 0)
            assert stz.utcoffset(ts) == std_tz.utcoffset(ts)


def test_get_tzname() -> None:
    tzs = [
        ZoneInfo("America/New_York"),
        ZoneInfo("Etc/GMT+5"),
        timezone(timedelta(hours=-5)),
        None,
    ]
    etzs = [
        "America/New_York",
        "Etc/GMT+5",
        "UTC-05:00",
        "None",
    ]

    for tz, etz in zip(tzs, etzs):
        name = get_tzname(tz)
        assert isinstance(name, str)
        assert name == etz
