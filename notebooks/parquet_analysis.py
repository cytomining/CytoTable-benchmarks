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

# + [markdown] papermill={"duration": 0.005861, "end_time": "2025-05-14T17:08:18.714587", "exception": false, "start_time": "2025-05-14T17:08:18.708726", "status": "completed"}
# # Why Parquet?
#
# This notebook explores the benefits or drawbacks of using the [parquet](https://parquet.apache.org/docs/) file format relative to other formats such as CSV or SQLite.

# + papermill={"duration": 0.583058, "end_time": "2025-05-14T17:08:19.303101", "exception": false, "start_time": "2025-05-14T17:08:18.720043", "status": "completed"}
import os
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
from utilities import get_system_info, timer

# + papermill={"duration": 0.013558, "end_time": "2025-05-14T17:08:19.322066", "exception": false, "start_time": "2025-05-14T17:08:19.308508", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.011987, "end_time": "2025-05-14T17:08:19.339250", "exception": false, "start_time": "2025-05-14T17:08:19.327263", "status": "completed"}
# target file or table names
image_dir = "images"
csv_name = "example.csv.gz"
parquet_name = "example.parquet"
sqlite_name = "example.sqlite"
sqlite_tbl_name = "tbl_example"
file_write_time_image = f"{image_dir}/parquet-comparisons-file-write-time.png"
file_storage_size_image = f"{image_dir}/parquet-comparisons-file-storage-size.png"
file_read_time_all_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-all-columns.png"
)
file_read_time_one_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-one-column.png"
)

# + papermill={"duration": 0.261715, "end_time": "2025-05-14T17:08:19.605977", "exception": false, "start_time": "2025-05-14T17:08:19.344262", "status": "completed"}
# avoid a "cold start" for tested packages by using them before benchmarks
df = pd.DataFrame(np.random.rand(2, 2), columns=[f"col_{num}" for num in range(0, 2)])
# export and read using various methods
df.to_csv(path_or_buf=csv_name, compression="gzip")
pd.read_csv(filepath_or_buffer=csv_name, compression="gzip")
df.to_sql(name=sqlite_tbl_name, con=f"sqlite:///{sqlite_name}")
pd.read_sql(sql=f"SELECT * FROM {sqlite_tbl_name}", con=f"sqlite:///{sqlite_name}")
df.to_parquet(path=parquet_name, compression="gzip")
pd.read_parquet(path=parquet_name)

# + papermill={"duration": 0.012958, "end_time": "2025-05-14T17:08:19.624817", "exception": false, "start_time": "2025-05-14T17:08:19.611859", "status": "completed"}
# remove any existing prior work
for filename in [csv_name, parquet_name, sqlite_name]:
    pathlib.Path(filename).unlink(missing_ok=True)

# + papermill={"duration": 24.982215, "end_time": "2025-05-14T17:08:44.612490", "exception": false, "start_time": "2025-05-14T17:08:19.630275", "status": "completed"}
# starting rowcount and col count
nrows = 320
ncols = 160

# result list for storing data
results = []

# loop for iterating over increasingly large dataframes
# and gathering data about operations on them
for _ in range(1, 4):
    # increase the size of the dataframe
    nrows *= 2
    ncols *= 2

    # form a dataframe using randomized data
    df = pd.DataFrame(
        np.random.rand(nrows, ncols), columns=[f"col_{num}" for num in range(0, ncols)]
    )

    # append data to the result list
    results.append(
        {
            # general information about the dataframe
            "dataframe_shape (rows, cols)": str(df.shape),
            # information about CSV
            "csv_write_time (secs)": timer(
                df.to_csv, path_or_buf=csv_name, compression="gzip"
            ),
            "csv_size (bytes)": os.stat(csv_name).st_size,
            "csv_read_time_all (secs)": timer(
                pd.read_csv, filepath_or_buffer=csv_name, compression="gzip"
            ),
            "csv_read_time_one (secs)": timer(
                pd.read_csv,
                filepath_or_buffer=csv_name,
                compression="gzip",
                usecols=["col_2"],
            ),
            # information about SQLite
            "sqlite_write_time (secs)": timer(
                df.to_sql,
                name=sqlite_tbl_name,
                con=f"sqlite:///{sqlite_name}",
            ),
            "sqlite_size (bytes)": os.stat(sqlite_name).st_size,
            "sqlite_read_time_all (secs)": timer(
                pd.read_sql,
                sql=f"SELECT * FROM {sqlite_tbl_name}",
                con=f"sqlite:///{sqlite_name}",
            ),
            "sqlite_read_time_one (secs)": timer(
                pd.read_sql,
                sql=f"SELECT col_2 FROM {sqlite_tbl_name}",
                con=f"sqlite:///{sqlite_name}",
            ),
            # information about Parquet
            "parquet_write_time (secs)": timer(
                df.to_parquet, path=parquet_name, compression="gzip"
            ),
            "parquet_size (bytes)": os.stat(parquet_name).st_size,
            "parquet_read_time_all (secs)": timer(pd.read_parquet, path=parquet_name),
            "parquet_read_time_one (secs)": timer(
                pd.read_parquet, path=parquet_name, columns=["col_2"]
            ),
        }
    )

    # remove any existing files in preparation for next steps
    for filename in [csv_name, parquet_name, sqlite_name]:
        pathlib.Path(filename).unlink(missing_ok=True)


df_results = pd.DataFrame(results)
df_results

# + papermill={"duration": 1.252307, "end_time": "2025-05-14T17:08:45.871726", "exception": false, "start_time": "2025-05-14T17:08:44.619419", "status": "completed"}
# write times barchart
fig = px.bar(
    df_results,
    x=[
        "csv_write_time (secs)",
        "sqlite_write_time (secs)",
        "parquet_write_time (secs)",
    ],
    y="dataframe_shape (rows, cols)",
    orientation="h",
    barmode="group",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Seconds"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    legend_title_text="File Write Duration",
    legend=dict(x=0.68, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)

pio.write_image(fig, file_write_time_image)
Image(url=file_write_time_image)

# + papermill={"duration": 0.257468, "end_time": "2025-05-14T17:08:46.135855", "exception": false, "start_time": "2025-05-14T17:08:45.878387", "status": "completed"}
# filesize barchart
fig = px.bar(
    df_results,
    x=[
        "csv_size (bytes)",
        "sqlite_size (bytes)",
        "parquet_size (bytes)",
    ],
    y="dataframe_shape (rows, cols)",
    orientation="h",
    barmode="group",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Bytes"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    legend_title_text="File Size",
    legend=dict(x=0.72, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)

pio.write_image(fig, file_storage_size_image)
Image(url=file_storage_size_image)

# + papermill={"duration": 0.159619, "end_time": "2025-05-14T17:08:46.301590", "exception": false, "start_time": "2025-05-14T17:08:46.141971", "status": "completed"}
# read time barchart (all columns)
fig = px.line(
    df_results,
    y=[
        "csv_read_time_all (secs)",
        "sqlite_read_time_all (secs)",
        "parquet_read_time_all (secs)",
    ],
    x="dataframe_shape (rows, cols)",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Seconds"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    legend_title_text="File Read Duration (all columns)",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)
fig.update_xaxes(range=[0, 2.13])
fig.update_traces(mode="lines+markers")

pio.write_image(fig, file_read_time_all_image)
Image(url=file_read_time_all_image)

# + papermill={"duration": 0.146281, "end_time": "2025-05-14T17:08:46.454748", "exception": false, "start_time": "2025-05-14T17:08:46.308467", "status": "completed"}
# read time barchart (one column)
fig = px.line(
    df_results,
    y=[
        "csv_read_time_one (secs)",
        "sqlite_read_time_one (secs)",
        "parquet_read_time_one (secs)",
    ],
    x="dataframe_shape (rows, cols)",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Seconds"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    legend_title_text="File Read Duration (one column)",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)
fig.update_xaxes(range=[0, 2.13])
fig.update_traces(mode="lines+markers")

pio.write_image(fig, file_read_time_one_image)
Image(url=file_read_time_one_image)
