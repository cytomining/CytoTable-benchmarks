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
import pandas as pd
import pyarrow as pa
from pyarrow import csv, parquet
from utilities import download_file

# -

url = "https://github.com/cytomining/CytoTable/blob/main/tests/data/cellprofiler/NF1_SchwannCell_data/all_cellprofiler.sqlite?raw=true"
orig_filepath_sqlite = "./examples/data/all_cellprofiler.sqlite"
orig_filepath_csv = "./examples/data/examplehuman_cellprofiler_features_csv"

# create a data dir
pathlib.Path(orig_filepath_sqlite).parent.mkdir(exist_ok=True)

# download the original file
download_file(url, orig_filepath_sqlite)

# create a duplicate file for use in looped testing
shutil.copy(
    orig_filepath_sqlite,
    orig_filepath_sqlite.replace("all_cellprofiler", "all_cellprofiler_duplicate"),
)
shutil.copy(
    orig_filepath_sqlite,
    orig_filepath_sqlite.replace("all_cellprofiler", "all_cellprofiler_duplicate_two"),
)
shutil.copy(
    orig_filepath_sqlite,
    orig_filepath_sqlite.replace(
        "all_cellprofiler", "all_cellprofiler_duplicate_three"
    ),
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
            # use a multiplier to control how many times the data is multiplied
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
previous_filepath = orig_filepath_sqlite
for _ in range(0, 9):
    new_filepath = orig_filepath_sqlite.replace(".sqlite", f"-x{number}.sqlite")
    shutil.copy(previous_filepath, new_filepath)
    multiply_database_size(filename=new_filepath, multiplier=2)
    previous_filepath = new_filepath
    number *= 2


def multiply_csv_dataset_size(directory: str, multiplier: int = 2):
    """
    A function for multiplying the size of CSV datasets by duplicating rows and
    incrementing ImageNumber identifiers to simulate larger datasets.

    Assumes CSV files are named like: Per_Image.csv, Per_Cytoplasm.csv, etc.

    Parameters:
    - directory: str or Path to the folder containing the CSV files.
    - multiplier: how many total multiples of the data to generate (default is 2x).
    """
    directory = pathlib.Path(directory)
    tables = ["Image", "Cytoplasm", "Nuclei", "Cells"]

    for tablename in tables:
        filepath = directory / f"{tablename}.csv"
        if not filepath.exists():
            print(f"Skipping missing file: {filepath}")
            continue

        df = pd.read_csv(filepath)
        print(f"Start count {tablename}: {len(df)}")

        if "ImageNumber" not in df.columns:
            print(f"Skipping {tablename} — missing 'ImageNumber' column.")
            continue

        max_id = df["ImageNumber"].max()
        new_dfs = [df]  # original data

        for i in range(1, multiplier):
            df_copy = df.copy()
            df_copy["ImageNumber"] += max_id * i
            new_dfs.append(df_copy)

        full_df = pd.concat(new_dfs, ignore_index=True)
        full_df.to_csv(filepath, index=False)

        print(f"End count {tablename}: {len(full_df)}")


# loop for copying sets of csv's and
# doubling the size each time
number = 2
previous_dir = pathlib.Path(orig_filepath_csv).resolve()
for _ in range(0, 9):
    new_dir = orig_filepath_csv.replace("_csv", f"_csv-x{number}")
    if pathlib.Path(new_dir).is_dir():
        shutil.rmtree(new_dir)
    shutil.copytree(previous_dir, new_dir)
    multiply_csv_dataset_size(new_dir, multiplier=2)
    previous_dir = new_dir
    number *= 2

# add example parquet file
duckdb.connect().execute(
    f"""
    /* Install and load sqlite plugin for duckdb */
    INSTALL sqlite_scanner;
    LOAD sqlite_scanner;

    /* Copy content from nuclei table to parquet file */
    COPY (select * from sqlite_scan('{orig_filepath_sqlite}', 'Per_Nuclei'))
    TO '{orig_filepath_sqlite + ".nuclei.parquet"}'
    (FORMAT PARQUET);
    """,
).close()

# create a duplicate file for use in looped testing
shutil.copy(
    orig_filepath_sqlite + ".nuclei.parquet",
    orig_filepath_sqlite + ".nuclei-copy.parquet",
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
