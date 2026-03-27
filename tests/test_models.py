import pytest
import ibis.expr.datatypes as dt

from chronify.models import ColumnDType, _check_name
from chronify.exceptions import InvalidValue


def test_column_dtypes() -> None:
    ColumnDType(name="col1", dtype=dt.Int64())
    for dtype in (dt.Int64, dt.Boolean, dt.Timestamp, dt.Float64, dt.String):
        ColumnDType(name="col1", dtype=dtype())

    for string_type in ("int", "bigint", "bool", "datetime", "float", "str"):
        ColumnDType(name="col1", dtype=string_type)

    with pytest.raises(InvalidValue):
        ColumnDType(name="col1", dtype="invalid")


def test_invalid_column_name() -> None:
    with pytest.raises(InvalidValue):
        _check_name(name="invalid - name")
