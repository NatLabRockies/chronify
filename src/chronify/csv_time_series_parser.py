from collections.abc import Iterable
from enum import Enum
from pathlib import Path
import pandas as pd


from chronify.exceptions import InvalidValue
from chronify.time_configs import (
    ColumnRepresentativeTimes,
    YearMonthDayHourTimeNTZ,
    MonthDayHourTimeNTZ,
    YearMonthDayPeriodTimeNTZ,
)
from chronify.models import TableSchema, PivotedTableSchema
from chronify.store import Store


INT_COLS = ["year", "month", "day"] + list(str(x) for x in range(1, 25))
COLUMN_DTYPES: dict[str, str] = {
    "name": "str",
    "period": "str",
    "value": "float64",
    **{name: "int32" for name in INT_COLS},
}


class CsvTimeSeriesFormats(Enum):
    TS_NYMDPV = ("name", "year", "month", "day", "period", "value")
    TS_NYMDH = (
        "name",
        "year",
        "month",
        "day",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    )
    TS_NMDH = (
        "name",
        "month",
        "day",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
    )

    @classmethod
    def from_columns(cls, columns: Iterable[str]) -> "CsvTimeSeriesFormats":
        column_set = set(columns)
        for selected_enum in cls.__members__.values():
            if column_set == set(selected_enum.value):
                return selected_enum

        msg = f"No format for columns: {columns}"
        raise InvalidValue(msg)


PIVOTED_TABLES = {CsvTimeSeriesFormats.TS_NMDH, CsvTimeSeriesFormats.TS_NYMDH}
UNPIVOTED_TABLES = {CsvTimeSeriesFormats.TS_NYMDPV}


class CsvTimeSeriesParser:
    def __init__(self, store: Store) -> None:
        self._store = store

    @staticmethod
    def _check_input_format(data_file: Path) -> None:
        valid_extensions = [".csv"]
        if data_file.suffix not in valid_extensions:
            msg = (
                f"{data_file.name} does not have a file extension in the "
                f"supported extensions: {valid_extensions}"
            )
            raise InvalidValue(msg)

    @staticmethod
    def _read_data_file(data_file: Path) -> pd.DataFrame:
        return pd.read_csv(data_file, header=0, dtype=COLUMN_DTYPES)  # type: ignore

    def _ingest_data(self, data: pd.DataFrame, table_name: str, year: int, length: int) -> None:
        csv_fmt = CsvTimeSeriesFormats.from_columns(data.columns)
        src_schema, dst_schema = self._create_schemas(csv_fmt, table_name, year, length)
        match csv_fmt:
            case fmt if fmt in PIVOTED_TABLES:
                assert src_schema is not None
                self._store.ingest_pivoted_table(data, src_schema, dst_schema)
            case fmt if fmt in UNPIVOTED_TABLES:
                self._store.ingest_table(data, dst_schema, skip_time_checks=True)

    @staticmethod
    def _create_schemas(
        csv_fmt: CsvTimeSeriesFormats, name: str, year: int, length: int
    ) -> tuple[PivotedTableSchema | None, TableSchema]:
        """
        Create a PivotedTableSchema if necessary, and a TableSchema for both
        the time format and datetime format.
        """
        create_pivoted_schema = True
        pivoted_dimension_name = "hour"
        value_columns = [str(x) for x in range(1, 25)]

        time_config: ColumnRepresentativeTimes
        match csv_fmt:
            case CsvTimeSeriesFormats.TS_NMDH:
                time_config = MonthDayHourTimeNTZ.default_config(length, year)
            case CsvTimeSeriesFormats.TS_NYMDH:
                time_config = YearMonthDayHourTimeNTZ.default_config(length, year)
            case CsvTimeSeriesFormats.TS_NYMDPV:
                create_pivoted_schema = False
                time_config = YearMonthDayPeriodTimeNTZ.default_config(length, year)

        table_schema = TableSchema(
            name=name,
            value_column="value",
            time_config=time_config,
            time_array_id_columns=["name"],
        )

        pivoted_schema = None
        if create_pivoted_schema:
            pivoted_schema = PivotedTableSchema(
                pivoted_dimension_name=pivoted_dimension_name,
                value_columns=value_columns,
                time_config=time_config,
            )

        return pivoted_schema, table_schema

    def ingest_to_datetime(
        self, data_file: Path, table_name: str, data_year: int, length: int
    ) -> None:
        """
        Given a file of csv time series data, convert the time format to datetime
        timestamps
        and ingest into database
        """
        self._check_input_format(data_file)
        df = self._read_data_file(data_file)
        self._ingest_data(df, table_name, data_year, length)
