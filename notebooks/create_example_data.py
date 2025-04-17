# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Create Example Data
#
# Create example data for use in other work within this repo.

# +
import pathlib
import shutil
import sqlite3

import duckdb
import numpy as np
import pyarrow as pa
import requests
from pyarrow import csv, parquet
from utilities import download_file
# -

url = "https://github.com/cytomining/CytoTable/blob/main/tests/data/cellprofiler/NF1_SchwannCell_data/all_cellprofiler.sqlite?raw=true"
orig_filepath = "./examples/data/all_cellprofiler.sqlite"

# create a data dir
pathlib.Path(orig_filepath).parent.mkdir(exist_ok=True)

# download the original file
download_file(url, orig_filepath)

# create a duplicate file for use in looped testing
shutil.copy(
    orig_filepath,
    orig_filepath.replace("all_cellprofiler", "all_cellprofiler_duplicate"),
)
shutil.copy(
    orig_filepath,
    orig_filepath.replace("all_cellprofiler", "all_cellprofiler_duplicate_two"),
)
shutil.copy(
    orig_filepath,
    orig_filepath.replace("all_cellprofiler", "all_cellprofiler_duplicate_three"),
)


def multiply_database_size(filename: str, multiplier: int = 2):
    """
    A function for doubling the size of the database given a filename.
    Note: unique to CellProfiler SQLite output and accounts for
    various unique keys.
    """
    print(filename)

    # Connect to the SQLite database
    with sqlite3.connect(filename) as conn:
        # Use sqlite3.Row to access columns by name
        conn.row_factory = sqlite3.Row

        # Create a cursor
        cur = conn.cursor()

        for tablename in ["Per_Image", "Per_Cytoplasm", "Per_Nuclei", "Per_Cells"]:
            print(
                f"Start count {tablename}: {dict(cur.execute(f'SELECT count(*) FROM {tablename}').fetchall()[0])}"
            )
            # Select all rows from the table
            cur.execute(f"SELECT * FROM {tablename}")
            rows = cur.fetchall()

            # Find the maximum id in the existing data
            max_id = max(row["ImageNumber"] for row in rows)

            new_rows = []
            # use a mutliplier to control how many times the data is multiplied
            for loop_multiply in range(1, multiplier):
                # Copy the rows and increment the id values
                for row in rows:
                    new_row = dict(row)
                    new_row["ImageNumber"] += max_id * loop_multiply
                    new_rows.append(new_row)

            # Insert the new rows into the table
            for row in new_rows:
                placeholders = ", ".join("?" * len(row))
                columns = ", ".join(row.keys())
                cur.execute(
                    f"INSERT INTO {tablename} ({columns}) VALUES ({placeholders})",
                    list(row.values()),
                )

            print(
                f"End count {tablename}: {dict(cur.execute(f'SELECT count(*) FROM {tablename}').fetchall()[0])}"
            )


# loop for copying the database and
# doubling the database size each time
number = 2
previous_filepath = orig_filepath
for _ in range(0, 9):
    new_filepath = orig_filepath.replace(".sqlite", f"-x{number}.sqlite")
    shutil.copy(previous_filepath, new_filepath)
    multiply_database_size(filename=new_filepath, multiplier=2)
    previous_filepath = new_filepath
    number *= 2

# add example parquet file
duckdb.connect().execute(
    f"""
    /* Install and load sqlite plugin for duckdb */
    INSTALL sqlite_scanner;
    LOAD sqlite_scanner;

    /* Copy content from nuclei table to parquet file */
    COPY (select * from sqlite_scan('{orig_filepath}', 'Per_Nuclei')) 
    TO '{orig_filepath + '.nuclei.parquet'}'
    (FORMAT PARQUET);
    """,
).close()

# create a duplicate file for use in looped testing
shutil.copy(
    orig_filepath + ".nuclei.parquet",
    orig_filepath + ".nuclei-copy.parquet",
)

# create randomized number data and related pyarrow table
tbl_numeric = pa.Table.from_arrays(
    [pa.array(np.random.rand(1000, 100)[:, i]) for i in range(100)],
    names=[f"Column_{i}" for i in range(100)],
)
# Create a table and write it to file
parquet.write_table(
    table=tbl_numeric,
    where="./examples/data/random_number_data.parquet",
)
csv.write_csv(data=tbl_numeric, output_file="./examples/data/random_number_data.csv")

# create a duplicate file for use in looped testing
shutil.copy(
    "./examples/data/random_number_data.parquet",
    "./examples/data/random_number_data-copy.parquet",
)
shutil.copy(
    "./examples/data/random_number_data.csv",
    "./examples/data/random_number_data-copy.csv",
)
