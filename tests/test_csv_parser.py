from pathlib import Path
from chronify.csv_time_series_parser import CsvTimeSeriesParser
from tempfile import NamedTemporaryFile
from chronify.store import Store
import pytest


def temp_csv_file(data: str):
    with NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        return Path(tmp_file.name)


@pytest.fixture
def time_series_NMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    return temp_csv_file(f"name,month,day,{hours}\nGeneration,1,1,{load1}\nGeneration,1,2,{load2}")


@pytest.fixture
def time_series_NYMDH():
    hours = ",".join((str(x) for x in range(1, 25)))
    load1 = ",".join((str(x) for x in range(25, 49)))
    load2 = ",".join((str(x) for x in range(49, 73)))
    return temp_csv_file(
        f"name,year,month,day,{hours}\ntest_generator,2025,1,1,{load1}\ntest_generator,2025,1,2,{load2}"
    )


@pytest.fixture
def time_series_NYMDPV():
    header = "name,year,month,day,period,value\n"
    data = "test_generator,2025,1,1,H1-5,100\ntest_generator,2025,1,1,H6-12,200\ntest_generator,2025,1,1,H13-24,300"
    return temp_csv_file(header + data)


def test_NMDH_parser(time_series_NMDH, iter_backends):
    store = Store(backend=iter_backends)
    parser = CsvTimeSeriesParser(store)
    parser.ingest_to_datetime(time_series_NMDH, "test_NMDH", 2023, 48)


def test_NYMDH_parser(time_series_NYMDH, iter_backends):
    store = Store(backend=iter_backends)
    parser = CsvTimeSeriesParser(store)
    parser.ingest_to_datetime(time_series_NYMDH, "test_NYMDH", 2025, 48)


def test_NYMDPV_parser(time_series_NYMDPV, iter_backends):
    store = Store(backend=iter_backends)
    parser = CsvTimeSeriesParser(store)
    parser.ingest_to_datetime(time_series_NYMDPV, "test_NYMDPV", 2025, 24)
