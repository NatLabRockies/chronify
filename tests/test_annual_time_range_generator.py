"""Tests for AnnualTimeRangeGenerator."""

import pytest

from chronify.annual_time_range_generator import AnnualTimeRangeGenerator
from chronify.time_configs import AnnualTimeRange


def _make_config(start: int = 1, length: int = 5) -> AnnualTimeRange:
    return AnnualTimeRange(
        time_column="year",
        start=start,
        length=length,
    )


def test_list_timestamps():
    gen = AnnualTimeRangeGenerator(_make_config(start=1, length=5))
    assert gen.list_timestamps() == [1, 2, 3, 4, 5]


def test_list_timestamps_single():
    gen = AnnualTimeRangeGenerator(_make_config(start=1, length=1))
    assert gen.list_timestamps() == [1]


def test_list_time_columns():
    gen = AnnualTimeRangeGenerator(_make_config())
    assert gen.list_time_columns() == ["year"]


def test_list_distinct_timestamps_from_dataframe_not_implemented():
    import pandas as pd

    gen = AnnualTimeRangeGenerator(_make_config())
    with pytest.raises(NotImplementedError):
        gen.list_distinct_timestamps_from_dataframe(pd.DataFrame())
