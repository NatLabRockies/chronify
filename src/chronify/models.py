import re
from typing import Any, Optional

import ibis.expr.datatypes as dt
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from chronify.base_models import ChronifyBaseModel
from chronify.exceptions import InvalidParameter, InvalidValue
from chronify.time_configs import TimeConfig
from chronify.ibis.types import (
    get_ibis_type_from_string,
    _COLUMN_TYPES,
)


REGEX_NAME_REQUIREMENT = re.compile(r"^\w+$")


class TableSchemaBase(ChronifyBaseModel):
    """Base model for table schema."""

    time_config: TimeConfig
    time_array_id_columns: Annotated[
        list[str],
        Field(
            description="Columns in the table that uniquely identify time arrays. "
            "These could be geographic identifiers, such as county and state, or an integer ID. "
            "Can be None if the table is pivoted and each pivoted column denotes a time array. "
            "Should not include time columns."
        ),
    ]

    @field_validator("time_config")
    @classmethod
    def check_time_config(cls, time_config: TimeConfig) -> TimeConfig:
        for column in time_config.list_time_columns():
            _check_name(column)
        tz_column = time_config.get_time_zone_column()
        if tz_column is not None:
            _check_name(tz_column)
        return time_config

    @field_validator("time_array_id_columns")
    @classmethod
    def check_columns(cls, columns: list[str]) -> list[str]:
        for column in columns:
            _check_name(column)
        return columns

    def list_columns(self) -> list[str]:
        """Return the column names in the schema."""
        return self.time_array_id_columns + self.time_config.list_time_columns()


class TableSchema(TableSchemaBase):
    """Defines the schema for a time series table stored in the database."""

    name: Annotated[
        str,
        Field(description="Name of the table or view in the database.", frozen=True),
    ]
    value_column: Annotated[str, Field(description="Column in the table that contain values.")]

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        _check_name(name)
        if name.lower() == "table":
            msg = f"Table schema cannot use {name=}."
            raise InvalidValue(msg)
        return name

    @field_validator("value_column")
    @classmethod
    def check_column(cls, column: str) -> str:
        _check_name(column)
        return column

    def list_columns(self) -> list[str]:
        return super().list_columns() + [self.value_column]


class PivotedTableSchema(TableSchemaBase):
    """Defines the schema for an input table with pivoted format."""

    pivoted_dimension_name: str = Field(
        description="Use this name for the column representing the pivoted dimension during "
        "an unpivot operation.",
    )
    value_columns: list[str] = Field(description="Columns in the table that contain values.")
    time_array_id_columns: list[str] = []

    @field_validator("value_columns")
    @classmethod
    def check_column(cls, value_columns: str) -> str:
        for column in value_columns:
            _check_name(column)
        return value_columns

    @field_validator("time_array_id_columns")
    @classmethod
    def check_time_array_id_columns(cls, value: list[str]) -> list[str]:
        if value:
            msg = f"PivotedTableSchema doesn't yet support time_array_id_columns: {value}"
            raise InvalidValue(msg)
        return value

    def list_columns(self) -> list[str]:
        return super().list_columns() + self.value_columns


class MappingTableSchema(ChronifyBaseModel):
    """Defines the schema for a mapping table for time conversion."""

    name: Annotated[
        str,
        Field(description="Name of the table or view in the database.", frozen=True),
    ]
    time_configs: list[TimeConfig]

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        _check_name(name)
        if name.lower() == "table":
            msg = f"Table schema cannot use {name=}."
            raise InvalidValue(msg)
        return name

    @field_validator("time_configs")
    @classmethod
    def check_time_configs(cls, time_configs: list[TimeConfig]) -> list[TimeConfig]:
        for config in time_configs:
            for column in config.list_time_columns():
                _check_name(column)
        return time_configs

    def list_columns(self) -> list[str]:
        time_columns = []
        for config in self.time_configs:
            time_columns += config.list_time_columns()
        return time_columns


# Set of valid Ibis dtype instances for validation
_IBIS_DTYPES = {type(v) for v in _COLUMN_TYPES.values()}


class ColumnDType(ChronifyBaseModel):
    """Defines the dtype of a column."""

    name: str
    dtype: Any

    @model_validator(mode="before")
    @classmethod
    def fix_data_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        dtype = data.get("dtype")
        if dtype is None:
            return data

        # If already an Ibis dtype instance, accept it
        if isinstance(dtype, dt.DataType):
            return data

        if isinstance(dtype, str):
            try:
                data["dtype"] = get_ibis_type_from_string(dtype)
            except InvalidParameter:
                options = sorted(_COLUMN_TYPES.keys())
                msg = f"{dtype=} must be one of {options}"
                raise InvalidValue(msg)
        else:
            msg = (
                f"dtype is an unsupported type: {type(dtype)}. It must be a str or Ibis DataType."
            )
            raise InvalidValue(msg)
        return data


class CsvTableSchema(TableSchemaBase):
    """Defines the schema of data in a CSV file."""

    pivoted_dimension_name: Optional[str] = Field(
        default=None,
        description="Only set if the table is pivoted. Use this name for the column "
        "representing that dimension when unpivoting.",
    )
    column_dtypes: Optional[list[ColumnDType]] = Field(
        default=None,
        description="Column types. Will try to infer types of any column not listed.",
    )
    value_columns: list[str] = Field(description="Columns in the table that contain values.")
    time_array_id_columns: list[str] = Field(
        default=[],
        description="Columns in the table that uniquely identify time arrays. "
        "These could be geographic identifiers, such as county and state, or an integer ID. "
        "Can be empty if the table is pivoted and each pivoted column denotes a time array. "
        "Should not include time columns.",
    )

    @field_validator("value_columns")
    @classmethod
    def check_columns(cls, columns: list[str]) -> list[str]:
        for column in columns:
            _check_name(column)
        return columns

    def list_columns(self) -> list[str]:
        return super().list_columns() + self.value_columns


class CsvTableSchemaSingleTimeArrayPivotedByComponent(CsvTableSchema):
    """Defines the schema of data in a CSV file where there is a single time array for each of
    multiple components and those components are pivoted columns."""


def _check_name(name: str) -> None:
    if not REGEX_NAME_REQUIREMENT.search(name):
        msg = f"A name can only have alphanumeric characters: {name=}"
        raise InvalidValue(msg)
