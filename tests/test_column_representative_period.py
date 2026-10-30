"""Tests for ColumnRepresentativeTimeGenerator with Period handler."""

from datetime import tzinfo

import pandas as pd
import pytest

from chronify.column_representative_time_range_generator import (
    ColumnRepresentativeHandlerPeriod,
    ColumnRepresentativeTimeGenerator,
)
from chronify.exceptions import InvalidOperation, InvalidValue
from chronify.time_configs import (
    ColumnRepresentativeBase,
    MonthDayHourTimeNTZ,
    YearMonthDayPeriodTimeNTZ,
)


def _make_period_config(year: int = 2024, length: int = 8760) -> YearMonthDayPeriodTimeNTZ:
    return YearMonthDayPeriodTimeNTZ(
        hour_columns=["period"],
        day_column="day",
        month_column="month",
        year_column="year",
        year=year,
        length=length,
    )


class _UnsupportedColumnRepresentative(ColumnRepresentativeBase):
    """ColumnRepresentativeBase subclass with no registered handler."""

    @classmethod
    def default_config(cls, length: int, year: int) -> "_UnsupportedColumnRepresentative":
        return cls(year=year, length=length, month_column="month", day_column="day")

    def list_time_columns(self) -> list[str]:
        return [self.month_column, self.day_column, *self.hour_columns]

    def get_time_zone_column(self) -> None:
        return None

    def get_time_zones(self) -> list[tzinfo | None]:
        return []


class TestColumnRepresentativeTimeGeneratorPeriod:
    def test_list_timestamps(self):
        config = _make_period_config(year=2024, length=8784)  # 366 days * 24 hours
        gen = ColumnRepresentativeTimeGenerator(config)
        timestamps = gen.list_timestamps()
        # 366 days in 2024 (leap year)
        assert len(timestamps) == 366
        assert timestamps[0] == (2024, 1, 1)
        assert timestamps[-1] == (2024, 12, 31)

    def test_list_time_columns(self):
        config = _make_period_config()
        gen = ColumnRepresentativeTimeGenerator(config)
        assert gen.list_time_columns() == ["year", "month", "day", "period"]


class TestColumnRepresentativeHandlerPeriod:
    def test_iter_timestamps(self):
        config = _make_period_config(year=2023, length=24 * 3)  # 3 days
        handler = ColumnRepresentativeHandlerPeriod(config, 2023)
        timestamps = list(handler._iter_timestamps())
        assert len(timestamps) == 3
        assert timestamps[0] == (2023, 1, 1)
        assert timestamps[1] == (2023, 1, 2)
        assert timestamps[2] == (2023, 1, 3)

    def test_list_distinct_timestamps_from_dataframe(self):
        config = _make_period_config(year=2023, length=24 * 3)
        handler = ColumnRepresentativeHandlerPeriod(config, 2023)
        df = pd.DataFrame(
            {
                "year": [2023, 2023, 2023, 2023],
                "month": [1, 1, 1, 1],
                "day": [1, 1, 2, 2],
                "period": ["H1-5", "H6-12", "H1-5", "H6-12"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = handler.list_distinct_timestamps_from_dataframe(df)
        assert result == [(2023, 1, 1), (2023, 1, 2)]


class TestColumnRepresentativeErrors:
    def test_no_year_raises(self):
        config = MonthDayHourTimeNTZ(
            day_column="day",
            month_column="month",
            hour_columns=["hour"],
            year=None,
            length=8760,
        )
        with pytest.raises(InvalidValue, match="without year"):
            ColumnRepresentativeTimeGenerator(config)

    def test_unsupported_config_raises(self):
        """ColumnRepresentativeBase subclasses not matching known handlers should raise."""
        config = _UnsupportedColumnRepresentative.default_config(length=8760, year=2024)
        with pytest.raises(InvalidOperation, match="No time generator"):
            ColumnRepresentativeTimeGenerator(config)
