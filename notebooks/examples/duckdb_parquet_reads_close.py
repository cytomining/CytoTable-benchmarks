"""
An example to test duckdb execution with closed connections
using parquet files.
"""
import multiprocessing

import duckdb

# set max threads for duckdb
MAX_THREADS = multiprocessing.cpu_count()


def _duckdb_reader(sql_stmt: str) -> duckdb.DuckDBPyConnection:
    """
    Creates a DuckDB connection with the
    sqlite_scanner installed and loaded.

    Returns:
        duckdb.DuckDBPyConnection
    """

    # setup a duckdb client with various configuration options
    duckdb_client = duckdb.connect().execute(
        # note: we use an f-string here to
        # dynamically configure threads as appropriate
        f"""
        /* Install and load sqlite plugin for duckdb */
        INSTALL sqlite_scanner;
        LOAD sqlite_scanner;

        /*
        Set threads available to duckdb
        See the following for more information:
        https://duckdb.org/docs/sql/pragmas#memory_limit-threads
        */
        PRAGMA threads={MAX_THREADS};

        /*
        Allow unordered results for performance increase possibilities
        See the following for more information:
        https://duckdb.org/docs/sql/configuration#configuration-reference
        */
        PRAGMA preserve_insertion_order=FALSE;

        /*
        Allow parallel csv reads for performance increase possibilities
        See the following for more information:
        https://duckdb.org/docs/sql/configuration#configuration-reference
        */
        PRAGMA experimental_parallel_csv=TRUE;
        """,
    )

    # gather the result from the duckdb connection
    result = duckdb_client.execute(sql_stmt).arrow()

    # explicitly close the connection before returning result
    duckdb_client.close()

    return result


for parquet_file in [
    "./examples/data/all_cellprofiler.sqlite.nuclei.parquet",
    "./examples/data/all_cellprofiler.sqlite.nuclei-copy.parquet",
]:
    sql_stmt = f"""
    SELECT * from '{parquet_file}';
    """
    _duckdb_reader(sql_stmt)
