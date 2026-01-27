# Copied this code from https://github.com/sqlalchemy/sqlalchemy/wiki/Views

from typing import Any

import sqlalchemy as sa
from sqlalchemy import Engine, MetaData, Selectable, TableClause
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table
from sqlalchemy.sql.sqltypes import DATETIME, TIMESTAMP, TEXT


class CreateTable(DDLElement):
    def __init__(self, name: str, selectable: Selectable) -> None:
        self.name = name
        self.selectable = selectable


class DropTable(DDLElement):
    def __init__(self, name: str) -> None:
        self.name = name


@compiler.compiles(CreateTable)
def _create_table(element: Any, compiler: Any, **kw: Any) -> str:
    return "CREATE TABLE %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropTable)
def _drop_table(element: Any, compiler: Any, **kw: Any) -> str:
    return "DROP TABLE %s" % (element.name)


def _table_exists(ddl: Any, target: Any, connection: Any, **kw: Any) -> Any:
    return ddl.name in sa.inspect(connection).get_table_names()


def _table_doesnt_exist(ddl: Any, target: Any, connection: Any, **kw: Any) -> bool:
    return not _table_exists(ddl, target, connection, **kw)


def create_table(
    name: str, selectable: Selectable, engine: Engine, metadata: MetaData
) -> TableClause:
    """Create a table from a selectable."""
    table_ = table(name)
    table_._columns._populate_separate_keys(
        col._make_proxy(table_)
        for col in selectable.selected_columns  # type: ignore
    )
    sa.event.listen(
        metadata,
        "after_create",
        CreateTable(name, selectable).execute_if(callable_=_table_doesnt_exist),  # type: ignore
    )
    sa.event.listen(metadata, "before_drop", DropTable(name).execute_if(callable_=_table_exists))  # type: ignore
    metadata.create_all(engine)
    metadata.reflect(engine, views=True)
    mtable = metadata.tables[name]
    if engine.name == "sqlite":
        # This is a workaround for a case we don't understand.
        # In some cases the datetime column schema is set to NUMERIC when the real values
        # are strings.
        for col in table_._columns:
            mcol = mtable.columns[col.name]
            if (
                isinstance(col.type, TIMESTAMP) or isinstance(col.type, DATETIME)
            ) and not isinstance(mcol.type, TEXT):
                mcol.type = TEXT()
    return table_
