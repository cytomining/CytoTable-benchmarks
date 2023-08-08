"""
An example to test duckdb execution without closing connections
using parquet files.
"""

import duckdb

def _duckdb_reader() -> duckdb.DuckDBPyConnection:
    """
    Creates a DuckDB connection with the
    sqlite_scanner installed and loaded.

    Returns:
        duckdb.DuckDBPyConnection
    """

    return duckdb.connect()


for parquet_file in [
    "./examples/data/random_number_data.parquet",
    "./examples/data/random_number_data-copy.parquet",
]:
    sql_stmt = f"""
    SELECT * from '{parquet_file}';
    """
    _duckdb_reader().execute(sql_stmt)
