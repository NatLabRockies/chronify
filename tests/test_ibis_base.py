"""Tests for the IbisBackend base class (transaction, execute_sql, etc.)."""

import ibis
import pandas as pd
import pytest

from chronify.ibis import make_backend


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
    with backend.transaction():
        backend.create_table(
            "txn_table",
            obj=None,
            schema={"id": "int64", "val": "float64"},
        )

    assert backend.has_table("txn_table")


def test_transaction_rollback_on_exception(create_duckdb_backend):
    backend = create_duckdb_backend
    df = pd.DataFrame({"id": [1], "val": [2.0]})

    with pytest.raises(ValueError, match="test error"):
        with backend.transaction():
            backend.create_table("txn_rollback", obj=df)
            msg = "test error"
            raise ValueError(msg)

    assert not backend.has_table("txn_rollback")


def test_transaction_rollback_view(create_duckdb_backend):
    backend = create_duckdb_backend
    df = pd.DataFrame({"id": [1], "val": [2.0]})
    backend.create_table("base_for_view", obj=df)
    expr = backend.table("base_for_view")

    with pytest.raises(ValueError, match="test error"):
        with backend.transaction():
            backend.create_view("txn_view", expr)
            msg = "test error"
            raise ValueError(msg)

    assert not backend.has_table("txn_view")


def test_transaction_rolls_back_ddl_and_dml(iter_backends):
    """A failing transaction must roll back both CREATE TABLE and INSERTs.

    Regression for an ibis-sqlite quirk: ``Backend.create_table`` runs its work
    inside ``with self.begin() as cur:`` which calls ``con.commit()`` on
    success, terminating any outer BEGIN. The chronify SQLiteBackend swaps in
    a no-commit connection proxy for the duration of ``transaction()`` so the
    commit is suppressed and the rollback covers DDL.
    """
    backend = iter_backends
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="boom"):
        with backend.transaction():
            backend.create_table("ddl_rollback", obj=df)
            assert backend.has_table("ddl_rollback")
            backend.insert("ddl_rollback", df)
            msg = "boom"
            raise ValueError(msg)
    assert not backend.has_table("ddl_rollback")


def test_transaction_commit_persists_ddl_and_dml(iter_backends):
    backend = iter_backends
    df = pd.DataFrame({"x": [1, 2, 3]})
    with backend.transaction():
        backend.create_table("ddl_commit", schema=ibis.schema({"x": "int64"}))
        backend.insert("ddl_commit", df)
    assert backend.has_table("ddl_commit")
    rows = backend.execute(backend.table("ddl_commit"))
    assert len(rows) == 3


def test_sqlite_inner_commit_does_not_leak_proxy(tmp_path):
    """Reviewer regression: if user code finalizes the SQLite transaction
    inside the block (e.g. ``execute_sql("COMMIT")``), the wrapper's exit
    must still restore the connection. Previously the wrapper's COMMIT raised
    before the proxy was put back, leaving the backend silently pointed at
    ``_NoCommitProxy`` so later writes never committed.
    """
    import sqlite3

    from chronify.ibis.sqlite_backend import SQLiteBackend, _NoCommitProxy

    db_path = tmp_path / "leak.db"
    backend = SQLiteBackend(database=str(db_path))

    with backend.transaction():
        backend.create_table("leak_t", schema=ibis.schema({"x": "int64"}))
        backend.execute_sql("COMMIT")

    # Backend must be back on the real connection.
    assert not isinstance(backend._connection.con, _NoCommitProxy)
    assert backend._real_con is None
    assert not backend._in_transaction

    # And later writes must actually commit — visible from a fresh connection.
    backend.insert("leak_t", pd.DataFrame({"x": [1, 2, 3]}))
    other = sqlite3.connect(str(db_path))
    try:
        rows = other.execute("SELECT COUNT(*) FROM leak_t").fetchone()
        assert rows == (3,)
    finally:
        other.close()


def test_sqlite_failed_begin_does_not_corrupt_backend():
    """A failed BEGIN must not leave SQLite pointed at the no-commit proxy.

    Regression for a reviewer-reported issue: the proxy was installed before
    BEGIN, so any BEGIN failure (e.g. an outer transaction already active)
    would leak the proxy and silently swallow all subsequent commits.
    """
    from chronify.ibis.sqlite_backend import SQLiteBackend, _NoCommitProxy

    backend = SQLiteBackend(database=":memory:")
    # Simulate an outer transaction already in progress.
    backend._connection.con.execute("BEGIN")
    with pytest.raises(Exception, match="transaction"):
        backend._begin_transaction()
    # Backend must still be operating on the real sqlite3 connection.
    assert not isinstance(backend._connection.con, _NoCommitProxy)
    assert not backend._in_transaction

    # Recovery: clean up the stranded outer BEGIN, then a normal transaction
    # block should work end-to-end.
    backend._connection.con.rollback()
    with backend.transaction():
        backend.create_table("recover", schema=ibis.schema({"x": "int64"}))
    assert backend.has_table("recover")


def test_transaction_nesting_commits_as_a_unit(iter_backends):
    """Nested transaction() calls compose: the outermost block governs."""
    backend = iter_backends
    with backend.transaction():
        backend.create_table("nest_outer", schema=ibis.schema({"x": "int64"}))
        with backend.transaction():
            backend.create_table("nest_inner", schema=ibis.schema({"x": "int64"}))
        # Inner block exited successfully, but the outer transaction is
        # still in progress — both tables remain pending until the outer
        # commit.
        assert backend._in_transaction
    assert backend.has_table("nest_outer")
    assert backend.has_table("nest_inner")
    assert not backend._in_transaction


def test_transaction_nesting_outer_rollback_includes_inner(iter_backends):
    """An exception in the outer block must roll back work done inside an
    inner (nested) transaction() block too."""
    backend = iter_backends
    with pytest.raises(ValueError, match="boom"):
        with backend.transaction():
            backend.create_table("nest_outer_r", schema=ibis.schema({"x": "int64"}))
            with backend.transaction():
                backend.create_table("nest_inner_r", schema=ibis.schema({"x": "int64"}))
            msg = "boom"
            raise ValueError(msg)
    assert not backend.has_table("nest_outer_r")
    assert not backend.has_table("nest_inner_r")
    assert not backend._in_transaction
