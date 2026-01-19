from typing import Optional
from datetime import datetime, tzinfo

import ibis.expr.types as ir
import pandas as pd

from chronify.exceptions import InvalidTable
from chronify.ibis.backend import IbisBackend
from chronify.ibis.functions import read_query
from chronify.models import TableSchema
from chronify.time_configs import DatetimeRangeWithTZColumn
from chronify.time_range_generator_factory import make_time_range_generator
from chronify.datetime_range_generator import DatetimeRangeGeneratorExternalTimeZone
from chronify.time import LeapDayAdjustmentType
from chronify.time_utils import is_prevailing_time_zone


def check_timestamps(
    backend: IbisBackend,
    table: ir.Table,
    schema: TableSchema,
    leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
) -> None:
    """Performs checks on time series arrays in a table."""
    TimeSeriesChecker(
        backend, table, schema, leap_day_adjustment=leap_day_adjustment
    ).check_timestamps()


class TimeSeriesChecker:
    """Performs checks on time series arrays in a table.
    Timestamps in the table will be checked against expected timestamps generated from the
    TableSchema's time_config. TZ-awareness of the generated timestamps will match that of the table.
    """

    def __init__(
        self,
        backend: IbisBackend,
        table: ir.Table,
        schema: TableSchema,
        leap_day_adjustment: Optional[LeapDayAdjustmentType] = None,
    ) -> None:
        self._backend = backend
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

        # Build query with distinct and not null filters
        query = self._table.select([self._table[x] for x in time_columns]).distinct()
        for col in time_columns:
            query = query.filter(self._table[col].notnull())

        df = read_query(self._backend, query, self._schema.time_config)
        actual = self._time_generator.list_distinct_timestamps_from_dataframe(df)
        expected = sorted(set(expected))  # drop duplicates for tz-naive prevailing time
        check_timestamp_lists(actual, expected)
        return len(expected)

    def _check_expected_timestamps_with_external_time_zone(self) -> int:
        """For tz-naive time with external time zone column"""
        assert isinstance(self._time_generator, DatetimeRangeGeneratorExternalTimeZone)  # for mypy
        expected_dct = self._time_generator.list_timestamps_by_time_zone()
        time_columns = self._time_generator.list_time_columns()
        assert isinstance(self._schema.time_config, DatetimeRangeWithTZColumn)  # for mypy
        time_columns.append(self._schema.time_config.get_time_zone_column())

        # Build query with distinct and not null filters
        query = self._table.select([self._table[x] for x in time_columns]).distinct()
        for col in time_columns:
            query = query.filter(self._table[col].notnull())

        df = read_query(self._backend, query, self._schema.time_config)
        actual_dct = self._time_generator.list_distinct_timestamps_by_time_zone_from_dataframe(df)
        if sorted(expected_dct.keys()) != sorted(actual_dct.keys()):
            msg = (
                "Time zone records do not match between expected and actual from table "
                f"\nexpected: {sorted(expected_dct.keys())} vs. \nactual: {sorted(actual_dct.keys())}"
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

        res_all = self._backend.execute_sql_to_df(query_all)
        count_all = res_all.iloc[0, 0]
        res_any = self._backend.execute_sql_to_df(query_any)
        count_any = res_any.iloc[0, 0]

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

        results = self._backend.execute_sql_to_df(query)
        for _, row in results.iterrows():
            distinct_count_by_ta = row.iloc[0]
            count_by_ta = row.iloc[1]

            if has_tz_naive_prevailing and not count_by_ta == count:
                id_vals = row.iloc[2:].tolist()
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
                id_vals = row.iloc[2:].tolist()
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
