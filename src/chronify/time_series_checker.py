from typing import Any, Optional, cast
from datetime import datetime, tzinfo

import ibis
import pandas as pd

from chronify.exceptions import InvalidTable
from chronify.ibis.base import IbisBackend
from chronify.models import TableSchema
from chronify.time_configs import DatetimeRangeWithTZColumn
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.datetime_range_generator import DatetimeRangeGeneratorExternalTimeZone
from chronify.time import LeapDayAdjustmentType
from chronify.time_utils import is_prevailing_time_zone


def check_timestamps(
    backend: IbisBackend,
    table_name: str,
    schema: TableSchema,
    leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
) -> None:
    """Performs checks on time series arrays in a table."""
    TimeSeriesChecker(
        backend, table_name, schema, leap_day_adjustment=leap_day_adjustment
    ).check_timestamps()


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table.
    Timestamps in the table will be checked against expected timestamps generated from the
    TableSchema's time_config. TZ-awareness of the generated timestamps will match that of thetable.
    """

    def __init__(
        self,
        backend: IbisBackend,
        table_name: str,
        schema: TableSchema,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        self._backend = backend
        self._schema = schema
        self._table_name = table_name
        self._time_generator = make_time_range_generator(
            schema.time_config, leap_day_adjustment=leap_day_adjustment
        )

    def check_timestamps(self) -> None:
        count = self._check_expected_timestamps()
        self._check_null_consistency()
        self._check_expected_timestamps_by_time_array(count)

    @staticmethod
    def _has_prevailing_time_zone(lst: list[tzinfo | None]) -> bool:
        for tz in lst:
            if is_prevailing_time_zone(tz):
                return True
        return False

    def _check_expected_timestamps(self) -> int:
        """Check that the timestamps in the table match the expected timestamps."""
        if isinstance(self._time_generator, DatetimeRangeGeneratorExternalTimeZone):
            return self._check_expected_timestamps_with_external_time_zone()
        return self._check_expected_timestamps_datetime()

    def _check_expected_timestamps_datetime(self) -> int:
        """For tz-naive or tz-aware time without external time zone column"""
        expected = self._time_generator.list_timestamps()
        time_columns = self._time_generator.list_time_columns()
        table = self._backend.table(self._table_name)
        expr = table.select(time_columns).distinct()
        for col in time_columns:
            expr = expr.filter(table[col].notnull())
        df = self._backend.read_query(expr, self._schema.time_config)
        actual = self._time_generator.list_distinct_timestamps_from_dataframe(df)
        expected = sorted(set(expected))  # drop duplicates for tz-naive prevailing time
        check_timestamp_lists(actual, expected)
        return len(expected)

    def _check_expected_timestamps_with_external_time_zone(self) -> int:
        """For tz-naive or tz-aware time with external time zone column
        tz-awareness is based on self._schema.time_config.dtype
        """
        assert isinstance(self._time_generator, DatetimeRangeGeneratorExternalTimeZone)  # for mypy
        expected_dct = self._time_generator.list_timestamps_by_time_zone()
        time_columns = self._time_generator.list_time_columns()
        assert isinstance(self._schema.time_config, DatetimeRangeWithTZColumn)  # for mypy
        time_columns.append(self._schema.time_config.get_time_zone_column())
        table = self._backend.table(self._table_name)
        expr = table.select(time_columns).distinct()
        for col in time_columns:
            expr = expr.filter(table[col].notnull())
        df = self._backend.read_query(expr, self._schema.time_config)
        actual_dct = self._time_generator.list_distinct_timestamps_by_time_zone_from_dataframe(df)
        if sorted(expected_dct.keys()) != sorted(actual_dct.keys()):
            msg = (
                "Time zone records do not match between expected and actual from table "
                f"\nexpected: {sorted(expected_dct.keys())} vs. \nactual: {sorted(actual_dct.keys())}"
                f"\ntime_config: {self._schema.time_config}"
            )
            raise InvalidTable(msg)

        assert len(expected_dct) > 0  # for mypy
        count = set()
        for tz_name in expected_dct.keys():
            count.add(len(expected_dct[tz_name]))
            # drops duplicates for tz-naive prevailing time
            expected = sorted(set(expected_dct[tz_name]))
            actual = actual_dct[tz_name]
            check_timestamp_lists(actual, expected, msg_prefix=f"For {tz_name}\n")
        # return len by preserving duplicates for tz-naive prevailing time
        assert len(count) == 1, "Mismatch in counts among time zones"
        return count.pop()

    def _check_null_consistency(self) -> None:
        # If any time column has a NULL, all time columns must have a NULL.
        time_columns = self._time_generator.list_time_columns()
        if len(time_columns) == 1:
            return

        table = self._backend.table(self._table_name)
        null_exprs = [table[col].isnull() for col in time_columns]
        count_all = int(cast(Any, table.filter(ibis.and_(*null_exprs)).count().execute()))
        count_any = int(cast(Any, table.filter(ibis.or_(*null_exprs)).count().execute()))
        if count_all != count_any:
            msg = (
                "If any time columns have a NULL value for a row, all time columns in that "
                "row must be NULL. "
                f"Row count where all time values are NULL: {count_all}. "
                f"Row count where any time values are NULL: {count_any}. "
            )
            raise InvalidTable(msg)

    def _check_expected_timestamps_by_time_array(self, count: int) -> None:
        if isinstance(
            self._time_generator, DatetimeRangeGeneratorExternalTimeZone
        ) and self._has_prevailing_time_zone(self._schema.time_config.get_time_zones()):
            has_tz_naive_prevailing = True
        else:
            has_tz_naive_prevailing = False

        id_cols = self._schema.time_array_id_columns
        time_cols = self._schema.time_config.list_time_columns()
        first_time_col = self._time_generator.list_time_columns()[0]

        table = self._backend.table(self._table_name)
        filtered = table.filter(table[first_time_col].notnull())

        if not id_cols:
            distinct_count_by_ta = int(
                cast(Any, filtered.select(time_cols).distinct().count().execute())
            )
            count_by_ta = int(cast(Any, filtered.count().execute()))
            df = pd.DataFrame(
                [{"distinct_count_by_ta": distinct_count_by_ta, "count_by_ta": count_by_ta}]
            )
        else:
            counts = filtered.group_by(id_cols).aggregate(count_by_ta=filtered.count())
            distinct_rows = filtered.select(id_cols + time_cols).distinct()
            distinct = distinct_rows.group_by(id_cols).aggregate(
                distinct_count_by_ta=distinct_rows.count()
            )
            df = counts.join(distinct, id_cols).execute()

        for _, result in df.iterrows():
            distinct_count_by_ta = result["distinct_count_by_ta"]
            count_by_ta = result["count_by_ta"]

            if has_tz_naive_prevailing and not count_by_ta == count:
                values = ", ".join(f"{x}={result[x]}" for x in id_cols)
                msg = (
                    f"The count of time values in each time array must be {count}."
                    f"Time array identifiers: {values}. "
                    f"count = {count_by_ta}"
                )
                raise InvalidTable(msg)

            if not has_tz_naive_prevailing and not count_by_ta == count == distinct_count_by_ta:
                values = ", ".join(f"{x}={result[x]}" for x in id_cols)
                msg = (
                    f"The count of time values in each time array must be {count}, and each "
                    "value must be distinct. "
                    f"Time array identifiers: {values}. "
                    f"count = {count_by_ta}, distinct count = {distinct_count_by_ta}. "
                )
                raise InvalidTable(msg)


def check_timestamp_lists(
    actual: list[pd.Timestamp] | list[datetime],
    expected: list[pd.Timestamp] | list[datetime],
    msg_prefix: str = "",
) -> None:
    match = actual == expected
    msg = msg_prefix
    if not match:
        if len(actual) != len(expected):
            msg += f"Mismatch number of timestamps: actual: {len(actual)} vs. expected: {len(expected)}\n"
        missing = set(expected).difference(set(actual))
        extra = set(actual).difference(set(expected))
        msg += "Actual timestamps do not match expected timestamps. \n"
        msg += f"Missing: {missing} \n"
        msg += f"Extra: {extra}"
        raise InvalidTable(msg)
