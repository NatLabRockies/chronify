from sqlalchemy import Connection, Table, select, text
from typing import Optional
from datetime import datetime, tzinfo

import pandas as pd

from chronify.exceptions import InvalidTable
from chronify.models import TableSchema
from chronify.time_configs import DatetimeRangeWithTZColumn
from chronify.sqlalchemy.functions import read_database
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.datetime_range_generator import DatetimeRangeGeneratorExternalTimeZone
from chronify.time import LeapDayAdjustmentType
from chronify.time_utils import is_prevailing_time_zone


def check_timestamps(
    conn: Connection,
    table: Table,
    schema: TableSchema,
    leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
) -> None:
    """Performs checks on time series arrays in a table."""
    TimeSeriesChecker(
        conn, table, schema, leap_day_adjustment=leap_day_adjustment
    ).check_timestamps()


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table.
    Timestamps in the table will be checked against expected timestamps generated from the
    TableSchema's time_config. TZ-awareness of the generated timestamps will match that of thetable.
    """

    def __init__(
        self,
        conn: Connection,
        table: Table,
        schema: TableSchema,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        self._conn = conn
        self._schema = schema
        self._table = table
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
        stmt = select(*(self._table.c[x] for x in time_columns)).distinct()
        for col in time_columns:
            stmt = stmt.where(self._table.c[col].is_not(None))
        df = read_database(stmt, self._conn, self._schema.time_config)
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
        stmt = select(*(self._table.c[x] for x in time_columns)).distinct()
        for col in time_columns:
            stmt = stmt.where(self._table.c[col].is_not(None))
        df = read_database(stmt, self._conn, self._schema.time_config)
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

        all_are_null = " AND ".join((f"{x} IS NULL" for x in time_columns))
        any_are_null = " OR ".join((f"{x} IS NULL" for x in time_columns))
        query_all = f"SELECT COUNT(*) FROM {self._schema.name} WHERE {all_are_null}"
        query_any = f"SELECT COUNT(*) FROM {self._schema.name} WHERE {any_are_null}"
        res_all = self._conn.execute(text(query_all)).fetchone()
        assert res_all is not None
        res_any = self._conn.execute(text(query_any)).fetchone()
        assert res_any is not None
        if res_all[0] != res_any[0]:
            msg = (
                "If any time columns have a NULL value for a row, all time columns in that "
                "row must be NULL. "
                f"Row count where all time values are NULL: {res_all[0]}. "
                f"Row count where any time values are NULL: {res_any[0]}. "
            )
            raise InvalidTable(msg)

    def _check_expected_timestamps_by_time_array(self, count: int) -> None:
        if isinstance(
            self._time_generator, DatetimeRangeGeneratorExternalTimeZone
        ) and self._has_prevailing_time_zone(self._schema.time_config.get_time_zones()):
            # cannot check counts by timestamps when tz-naive prevailing time zones are present
            has_tz_naive_prevailing = True
        else:
            has_tz_naive_prevailing = False

        id_cols = ",".join(self._schema.time_array_id_columns)
        time_cols = ",".join(self._schema.time_config.list_time_columns())
        # NULL consistency was checked above.
        where_clause = f"{self._time_generator.list_time_columns()[0]} IS NOT NULL"
        on_expr = " AND ".join([f"t1.{x} = t2.{x}" for x in self._schema.time_array_id_columns])
        t1_id_cols = ",".join((f"t1.{x}" for x in self._schema.time_array_id_columns))

        if not self._schema.time_array_id_columns:
            query = f"""
                WITH distinct_time_values_by_array AS (
                    SELECT DISTINCT {time_cols}
                    FROM {self._schema.name}
                    WHERE {where_clause}
                ),
                t1 AS (
                    SELECT COUNT(*) AS distinct_count_by_ta
                    FROM distinct_time_values_by_array
                ),
                t2 AS (
                    SELECT COUNT(*) AS count_by_ta
                    FROM {self._schema.name}
                    WHERE {where_clause}
                )
                SELECT
                    t1.distinct_count_by_ta
                    ,t2.count_by_ta
                FROM t1
                CROSS JOIN t2
            """
        else:
            query = f"""
                WITH distinct_time_values_by_array AS (
                    SELECT DISTINCT {id_cols}, {time_cols}
                    FROM {self._schema.name}
                    WHERE {where_clause}
                ),
                t1 AS (
                    SELECT {id_cols}, COUNT(*) AS distinct_count_by_ta
                    FROM distinct_time_values_by_array
                    GROUP BY {id_cols}
                ),
                t2 AS (
                    SELECT {id_cols}, COUNT(*) AS count_by_ta
                    FROM {self._schema.name}
                    WHERE {where_clause}
                    GROUP BY {id_cols}
                )
                SELECT
                    t1.distinct_count_by_ta
                    ,t2.count_by_ta
                    ,{t1_id_cols}
                FROM t1
                JOIN t2
                ON {on_expr}
            """

        for result in self._conn.execute(text(query)).fetchall():
            distinct_count_by_ta = result[0]
            count_by_ta = result[1]

            if has_tz_naive_prevailing and not count_by_ta == count:
                id_vals = result[2:]
                values = ", ".join(
                    f"{x}={y}" for x, y in zip(self._schema.time_array_id_columns, id_vals)
                )
                msg = (
                    f"The count of time values in each time array must be {count}."
                    f"Time array identifiers: {values}. "
                    f"count = {count_by_ta}"
                )
                raise InvalidTable(msg)

            if not has_tz_naive_prevailing and not count_by_ta == count == distinct_count_by_ta:
                id_vals = result[2:]
                values = ", ".join(
                    f"{x}={y}" for x, y in zip(self._schema.time_array_id_columns, id_vals)
                )
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
