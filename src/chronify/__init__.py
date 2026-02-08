import importlib.metadata as metadata
import sys

from chronify._vendor.kyuubi import TCLIService
from chronify._vendor.kyuubi import pyhive
from chronify.exceptions import (
    ChronifyExceptionBase,
    ConflictingInputsError,
    InvalidTable,
    InvalidOperation,
    InvalidParameter,
    MissingParameter,
    TableAlreadyExists,
    TableNotStored,
)
from chronify.models import (
    ColumnDType,
    CsvTableSchema,
    PivotedTableSchema,
    TableSchema,
)
from chronify.store import Store
from chronify.time import RepresentativePeriodFormat, TimeDataType
from chronify.time_configs import (
    AnnualTimeRange,
    DatetimeRange,
    DatetimeRangeWithTZColumn,
    IndexTimeRange,
    IndexTimeRangeWithTZColumn,
    RepresentativePeriodTimeNTZ,
    RepresentativePeriodTimeTZ,
    TimeBaseModel,
    TimeBasedDataAdjustment,
)

__all__ = (
    "AnnualTimeRange",
    "ChronifyExceptionBase",
    "ColumnDType",
    "ConflictingInputsError",
    "CsvTableSchema",
    "DatetimeRange",
    "DatetimeRangeWithTZColumn",
    "IndexTimeRange",
    "IndexTimeRangeWithTZColumn",
    "InvalidOperation",
    "InvalidParameter",
    "InvalidTable",
    "MissingParameter",
    "PivotedTableSchema",
    "RepresentativePeriodFormat",
    "RepresentativePeriodTimeNTZ",
    "RepresentativePeriodTimeTZ",
    "Store",
    "TableAlreadyExists",
    "TableNotStored",
    "TableSchema",
    "TimeBaseModel",
    "TimeBasedDataAdjustment",
    "TimeDataType",
)

__version__ = metadata.metadata("chronify")["Version"]


# Make pyhive importable as if it were installed separately.
sys.modules["pyhive"] = pyhive
sys.modules["TCLIService"] = TCLIService
