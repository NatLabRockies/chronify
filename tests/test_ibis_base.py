"""Tests for the IbisBackend base class (transaction, execute_sql, etc.)."""

import pytest

from chronify.ibis import make_backend
from chronify.ibis.base import ObjectType


def test_execute_sql(create_duckdb_backend):
    backend = create_duckdb_backend
    backend.execute_sql("CREATE TABLE test_exec_sql (id INTEGER, val DOUBLE)")
    assert backend.has_table("test_exec_sql")


def test_execute_sql_to_df(create_duckdb_backend):
    backend = create_duckdb_backend
    backend.execute_sql("CREATE TABLE test_sql_df (id INTEGER, val DOUBLE)")
    backend.execute_sql("INSERT INTO test_sql_df VALUES (1, 2.5)")
    df = backend.execute_sql_to_df("SELECT * FROM test_sql_df")
    assert len(df) == 1
    assert df["id"].iloc[0] == 1


def test_dispose():
    backend = make_backend("duckdb")
    backend.dispose()


def test_transaction_success(create_duckdb_backend):
    backend = create_duckdb_backend
    with backend.transaction() as created:
        backend.create_table(
            "txn_table",
            obj=None,
            schema={"id": "int64", "val": "float64"},
        )
        created.append(("txn_table", ObjectType.TABLE))

    # Table should still exist after successful transaction
    assert backend.has_table("txn_table")


def test_transaction_rollback_on_exception(create_duckdb_backend):
    import pandas as pd

    backend = create_duckdb_backend
    df = pd.DataFrame({"id": [1], "val": [2.0]})

    with pytest.raises(ValueError, match="test error"):
        with backend.transaction() as created:
            backend.create_table("txn_rollback", obj=df)
            created.append(("txn_rollback", ObjectType.TABLE))
            msg = "test error"
            raise ValueError(msg)

    # Table should have been cleaned up
    assert not backend.has_table("txn_rollback")


def test_transaction_rollback_view(create_duckdb_backend):
    import pandas as pd

    backend = create_duckdb_backend
    df = pd.DataFrame({"id": [1], "val": [2.0]})
    backend.create_table("base_for_view", obj=df)
    expr = backend.table("base_for_view")

    with pytest.raises(ValueError, match="test error"):
        with backend.transaction() as created:
            backend.create_view("txn_view", expr)
            created.append(("txn_view", ObjectType.VIEW))
            msg = "test error"
            raise ValueError(msg)

    assert not backend.has_table("txn_view")
