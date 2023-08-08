"""
An example to test duckdb execution with closed connections
using parquet files.
"""

import duckdb

def _duckdb_reader(sql_stmt: str) -> duckdb.DuckDBPyConnection:
    """
    Creates a DuckDB connection with the
    sqlite_scanner installed and loaded.

    Returns:
        duckdb.DuckDBPyConnection
    """

    # setup a duckdb client with various configuration options
    duckdb_client = duckdb.connect()

    # gather the result from the duckdb connection
    result = duckdb_client.execute(sql_stmt).arrow()

    # explicitly close the connection before returning result
    duckdb_client.close()

    return result


for parquet_file in [
    "./examples/data/random_number_data.parquet",
    "./examples/data/random_number_data-copy.parquet",
]:
    sql_stmt = f"""
    SELECT * from '{parquet_file}';
    """
    _duckdb_reader(sql_stmt)
