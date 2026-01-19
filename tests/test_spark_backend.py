"""Tests for the Spark backend."""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from chronify import DatetimeRange, Store, TableSchema
from chronify.exceptions import InvalidTable


pytestmark = pytest.mark.skipif(
    os.getenv("CHRONIFY_SPARK_TEST") is None,
    reason="Spark tests require CHRONIFY_SPARK_TEST environment variable",
)


@pytest.fixture
def spark_store(tmp_path):
    """Create a Spark store for testing."""
    from pyspark.sql import SparkSession

    # Use a temp directory for the spark warehouse to avoid conflicts
    warehouse_dir = tmp_path / "spark-warehouse"
    spark = (
        SparkSession.builder.appName("chronify_test")
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .getOrCreate()
    )

    store = Store.create_new_spark_store(session=spark)
    orig_tables = set(store._backend.list_tables())
    yield store
    # Cleanup views and tables created during test
    for name in store._backend.list_tables():
        if name not in orig_tables:
            try:
                store._backend.drop_view(name, if_exists=True)
            except Exception:
                pass
            try:
                store._backend.drop_table(name, if_exists=True)
            except Exception:
                pass


@pytest.fixture
def sample_parquet_file(tmp_path):
    """Create a sample Parquet file for testing."""
    initial_time = datetime(2020, 1, 1)
    resolution = timedelta(hours=1)
    length = 8784  # leap year
    timestamps = pd.date_range(initial_time, periods=length, freq=resolution, unit="us")

    dfs = []
    for i in range(1, 4):
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "id": i,
                "value": np.random.random(length),
            }
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    schema = TableSchema(
        name="devices",
        value_column="value",
        time_config=DatetimeRange(
            time_column="timestamp",
            start=initial_time,
            length=length,
            resolution=resolution,
        ),
        time_array_id_columns=["id"],
    )
    return parquet_path, schema, df


class TestCreateViewFromParquet:
    """Tests for Store.create_view_from_parquet with Spark backend."""

    def test_create_view_from_parquet_basic(self, spark_store, sample_parquet_file):
        """Test basic view creation from a Parquet file."""
        parquet_path, schema, original_df = sample_parquet_file

        spark_store.create_view_from_parquet(parquet_path, schema)

        assert spark_store.has_table(schema.name)
        assert schema.name in spark_store.list_tables()

        df = spark_store.read_table(schema.name)
        assert len(df) == len(original_df)
        assert set(df.columns) == set(original_df.columns)
        assert len(df["id"].unique()) == 3

    def test_create_view_from_parquet_time_validation(self, spark_store, tmp_path):
        """Test that time validation catches invalid data."""
        initial_time = datetime(2020, 1, 1)
        resolution = timedelta(hours=1)
        length = 100

        # Create data with missing timestamps (only 90 rows instead of 100)
        timestamps = pd.date_range(initial_time, periods=90, freq=resolution, unit="us")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "id": 1,
                "value": np.random.random(90),
            }
        )
        parquet_path = tmp_path / "invalid_data.parquet"
        df.to_parquet(parquet_path, index=False)

        schema = TableSchema(
            name="invalid_devices",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=length,
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )

        with pytest.raises(InvalidTable):
            spark_store.create_view_from_parquet(parquet_path, schema)

        # View should not exist after validation failure
        assert not spark_store.has_table(schema.name)

    def test_create_view_from_parquet_bypass_checks(self, spark_store, tmp_path):
        """Test creating a view with bypass_checks=True."""
        initial_time = datetime(2020, 1, 1)
        resolution = timedelta(hours=1)
        length = 100

        # Create incomplete data
        timestamps = pd.date_range(initial_time, periods=50, freq=resolution, unit="us")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "id": 1,
                "value": np.random.random(50),
            }
        )
        parquet_path = tmp_path / "partial_data.parquet"
        df.to_parquet(parquet_path, index=False)

        schema = TableSchema(
            name="partial_devices",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=length,
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )

        # Should succeed with bypass_checks=True
        spark_store.create_view_from_parquet(parquet_path, schema, bypass_checks=True)
        assert spark_store.has_table(schema.name)

    def test_drop_view(self, spark_store, sample_parquet_file):
        """Test dropping a view created from Parquet."""
        parquet_path, schema, _ = sample_parquet_file

        spark_store.create_view_from_parquet(parquet_path, schema)
        assert spark_store.has_table(schema.name)

        spark_store.drop_view(schema.name)
        assert not spark_store.has_table(schema.name)


class TestCreateView:
    """Tests for Store.create_view with Spark backend."""

    def test_create_view_from_existing_table(self, spark_store, sample_parquet_file):
        """Test registering an Ibis table as a view from an existing persistent table."""
        parquet_path, base_schema, original_df = sample_parquet_file

        # First create a persistent view from parquet
        spark_store.create_view_from_parquet(parquet_path, base_schema)

        # Now get the table and create another view from it
        table = spark_store.get_table(base_schema.name)

        # Create a new schema for the derived view
        schema = TableSchema(
            name="ibis_view",
            value_column="value",
            time_config=base_schema.time_config,
            time_array_id_columns=["id"],
        )

        spark_store.create_view(schema, table)

        assert spark_store.has_table(schema.name)
        df = spark_store.read_table(schema.name)
        assert len(df) == len(original_df)

    def test_create_view_with_filter(self, spark_store, sample_parquet_file):
        """Test registering a filtered Ibis table as a view."""
        parquet_path, base_schema, _ = sample_parquet_file

        # First create a persistent view from parquet
        spark_store.create_view_from_parquet(parquet_path, base_schema)

        # Get table and apply a filter
        table = spark_store.get_table(base_schema.name)
        filtered_table = table.filter(table["id"] == 1)

        # Schema for single id
        schema = TableSchema(
            name="filtered_view",
            value_column="value",
            time_config=base_schema.time_config,
            time_array_id_columns=["id"],
        )

        spark_store.create_view(schema, filtered_table)

        df = spark_store.read_table(schema.name)
        assert len(df["id"].unique()) == 1
        assert df["id"].iloc[0] == 1

    def test_create_view_time_validation(self, spark_store, tmp_path):
        """Test that create_view validates timestamps."""
        initial_time = datetime(2020, 1, 1)
        resolution = timedelta(hours=1)
        length = 100

        # Create incomplete data
        timestamps = pd.date_range(initial_time, periods=50, freq=resolution, unit="us")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "id": 1,
                "value": np.random.random(50),
            }
        )
        parquet_path = tmp_path / "incomplete.parquet"
        df.to_parquet(parquet_path, index=False)

        # First create a view from parquet with bypass_checks to get a persistent table
        base_schema = TableSchema(
            name="incomplete_base",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=50,  # Match actual data
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )
        spark_store.create_view_from_parquet(parquet_path, base_schema)

        # Now try to create a view with stricter schema
        table = spark_store.get_table(base_schema.name)

        schema = TableSchema(
            name="incomplete_view",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=length,  # Expects 100 rows but only 50 exist
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )

        with pytest.raises(InvalidTable):
            spark_store.create_view(schema, table)

        assert not spark_store.has_table(schema.name)

    def test_create_view_bypass_checks(self, spark_store, tmp_path):
        """Test create_view with bypass_checks=True."""
        initial_time = datetime(2020, 1, 1)
        resolution = timedelta(hours=1)

        # Create minimal data
        timestamps = pd.date_range(initial_time, periods=10, freq=resolution, unit="us")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "id": 1,
                "value": np.random.random(10),
            }
        )
        parquet_path = tmp_path / "minimal.parquet"
        df.to_parquet(parquet_path, index=False)

        # First create a view from parquet with bypass_checks
        base_schema = TableSchema(
            name="minimal_base",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=10,
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )
        spark_store.create_view_from_parquet(parquet_path, base_schema)

        table = spark_store.get_table(base_schema.name)

        schema = TableSchema(
            name="minimal_view",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=initial_time,
                length=1000,  # Much larger than actual data
                resolution=resolution,
            ),
            time_array_id_columns=["id"],
        )

        # Should succeed with bypass_checks=True
        spark_store.create_view(schema, table, bypass_checks=True)
        assert spark_store.has_table(schema.name)

    def test_create_view_with_transformation(self, spark_store, sample_parquet_file):
        """Test create_view with a transformed Ibis table."""
        parquet_path, base_schema, _ = sample_parquet_file

        # First create a persistent view from parquet
        spark_store.create_view_from_parquet(parquet_path, base_schema)

        table = spark_store.get_table(base_schema.name)

        # Apply transformation: multiply value by 2
        transformed = table.mutate(value=table["value"] * 2)

        schema = TableSchema(
            name="transformed_view",
            value_column="value",
            time_config=base_schema.time_config,
            time_array_id_columns=["id"],
        )

        spark_store.create_view(schema, transformed)

        df = spark_store.read_table(schema.name)
        assert spark_store.has_table(schema.name)
        assert len(df) == 8784 * 3


class TestSparkStoreCreation:
    """Tests for Store.create_new_spark_store."""

    def test_create_new_spark_store(self):
        """Test creating a new Spark store."""
        store = Store.create_new_spark_store()
        assert store._backend.name == "spark"

    def test_create_spark_store_with_existing_session(self):
        """Test creating a Spark store with an existing SparkSession."""
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("test_chronify").getOrCreate()

        store = Store.create_new_spark_store(session=spark)
        assert store._backend.name == "spark"
        assert store._backend._session is spark

    def test_spark_data_ingestion_not_supported(self, spark_store):
        """Test that data ingestion raises NotImplementedError for Spark."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=10, freq="h"),
                "id": 1,
                "value": range(10),
            }
        )
        schema = TableSchema(
            name="test_table",
            value_column="value",
            time_config=DatetimeRange(
                time_column="timestamp",
                start=datetime(2020, 1, 1),
                length=10,
                resolution=timedelta(hours=1),
            ),
            time_array_id_columns=["id"],
        )

        with pytest.raises(NotImplementedError, match="Data ingestion through Spark"):
            spark_store.ingest_table(df, schema)


class TestSparkReadOperations:
    """Tests for read operations with Spark backend."""

    def test_read_table(self, spark_store, sample_parquet_file):
        """Test reading a table."""
        parquet_path, schema, original_df = sample_parquet_file
        spark_store.create_view_from_parquet(parquet_path, schema)

        df = spark_store.read_table(schema.name)
        assert len(df) == len(original_df)

    def test_read_query(self, spark_store, sample_parquet_file):
        """Test reading with a query."""
        parquet_path, schema, _ = sample_parquet_file
        spark_store.create_view_from_parquet(parquet_path, schema)

        query = f"SELECT * FROM {schema.name} WHERE id = 1"
        df = spark_store.read_query(schema.name, query)
        assert len(df["id"].unique()) == 1
        assert df["id"].iloc[0] == 1

    def test_read_query_with_ibis_expression(self, spark_store, sample_parquet_file):
        """Test reading with an Ibis expression."""
        parquet_path, schema, _ = sample_parquet_file
        spark_store.create_view_from_parquet(parquet_path, schema)

        table = spark_store.get_table(schema.name)
        filtered = table.filter(table["id"] == 2)

        df = spark_store.read_query(schema.name, filtered)
        assert len(df["id"].unique()) == 1
        assert df["id"].iloc[0] == 2


class TestSparkParquetOutput:
    """Tests for writing Parquet with Spark backend."""

    def test_write_table_to_parquet(self, spark_store, sample_parquet_file, tmp_path):
        """Test writing a table to Parquet."""
        parquet_path, schema, _ = sample_parquet_file
        spark_store.create_view_from_parquet(parquet_path, schema)

        output_path = tmp_path / "output.parquet"
        spark_store.write_table_to_parquet(schema.name, output_path)

        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 8784 * 3

    def test_write_query_to_parquet(self, spark_store, sample_parquet_file, tmp_path):
        """Test writing a query result to Parquet."""
        parquet_path, schema, _ = sample_parquet_file
        spark_store.create_view_from_parquet(parquet_path, schema)

        table = spark_store.get_table(schema.name)
        filtered = table.filter(table["id"] == 1)

        output_path = tmp_path / "filtered_output.parquet"
        spark_store.write_query_to_parquet(filtered, output_path)

        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 8784
        assert df["id"].iloc[0] == 1
