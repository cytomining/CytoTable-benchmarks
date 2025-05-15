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

# + [markdown] papermill={"duration": 0.002917, "end_time": "2025-05-14T17:08:09.837556", "exception": false, "start_time": "2025-05-14T17:08:09.834639", "status": "completed"}
# # Why Arrow?
#
# This notebook explores the benefits or drawbacks of using the [Arrow](https://arrow.apache.org) in-memory data format relative to other formats such as Pandas DataFrames.

# + papermill={"duration": 0.652163, "end_time": "2025-05-14T17:08:10.495039", "exception": false, "start_time": "2025-05-14T17:08:09.842876", "status": "completed"}
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import polars as pl
from IPython.display import Image
from pyarrow import parquet
from pympler.asizeof import asizeof
from utilities import get_system_info, timer

# + papermill={"duration": 0.013927, "end_time": "2025-05-14T17:08:10.511606", "exception": false, "start_time": "2025-05-14T17:08:10.497679", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.008306, "end_time": "2025-05-14T17:08:10.522475", "exception": false, "start_time": "2025-05-14T17:08:10.514169", "status": "completed"}
# target file or table names
image_dir = "images"
parquet_name = "example.parquet"
mem_times_image = f"{image_dir}/arrow-comparisons-mem-read-times.png"
mem_read_size_image = f"{image_dir}/arrow-comparisons-mem-read-size.png"

# + papermill={"duration": 0.007631, "end_time": "2025-05-14T17:08:10.532742", "exception": false, "start_time": "2025-05-14T17:08:10.525111", "status": "completed"}
# remove any existing prior work
pathlib.Path(parquet_name).unlink(missing_ok=True)

# + papermill={"duration": 0.088766, "end_time": "2025-05-14T17:08:10.624040", "exception": false, "start_time": "2025-05-14T17:08:10.535274", "status": "completed"}
# avoid a "cold start" for tested packages by using them before benchmarks
df = pd.DataFrame(np.random.rand(2, 2), columns=[f"col_{num}" for num in range(0, 2)])
# write to parquet for tests below
df.to_parquet(path=(coldstart_file := "coldstart.parquet"), compression="snappy")

# read the file using the benchmarked packages
pd.read_parquet(path=coldstart_file)
parquet.read_table(source=coldstart_file)
pl.scan_parquet(source=coldstart_file).collect()

# remove the coldstart file
pathlib.Path(coldstart_file).unlink(missing_ok=True)

# + papermill={"duration": 1.879187, "end_time": "2025-05-14T17:08:12.506121", "exception": false, "start_time": "2025-05-14T17:08:10.626934", "status": "completed"}
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
    # write to parquet for tests below
    df.to_parquet(path=parquet_name, compression="snappy")

    # append data to the result list
    results.append(
        {
            # general information about the dataframe
            "dataframe_shape (rows, cols)": str(df.shape),
            # information about pandas
            "pandas_read_time (secs)": timer(pd.read_parquet, path=parquet_name),
            "pandas_size (bytes)": asizeof(pd.read_parquet(path=parquet_name)),
            # information about pyarrow
            "pyarrow_read_time (secs)": timer(parquet.read_table, source=parquet_name),
            "pyarrow_size (bytes)": asizeof(parquet.read_table(source=parquet_name)),
            # information about polars
            "polars_read_time (secs)": timer(
                pl.scan_parquet, source=parquet_name, method_chain="collect"
            ),
            "polars_size (bytes)": pl.scan_parquet(source=parquet_name)
            .collect()
            .estimated_size(),
        }
    )

    # remove any existing files in preparation for next steps
    pathlib.Path(parquet_name).unlink(missing_ok=True)


df_results = pd.DataFrame(results)
df_results
# + papermill={"duration": 1.327157, "end_time": "2025-05-14T17:08:13.838450", "exception": false, "start_time": "2025-05-14T17:08:12.511293", "status": "completed"}
# write times barchart
fig = px.bar(
    df_results,
    x=[
        "pandas_read_time (secs)",
        "pyarrow_read_time (secs)",
        "polars_read_time (secs)",
    ],
    y="dataframe_shape (rows, cols)",
    orientation="h",
    barmode="group",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Seconds"},
    width=1300,
)
fig.update_layout(
    legend_title_text="In-memory Read Duration",
    legend=dict(x=0.72, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=17.5,  # global font size
    ),
)

pio.write_image(fig, mem_times_image)
Image(url=mem_times_image)


# + papermill={"duration": 0.210952, "end_time": "2025-05-14T17:08:14.055171", "exception": false, "start_time": "2025-05-14T17:08:13.844219", "status": "completed"}
# write times barchart
fig = px.bar(
    df_results,
    x=[
        "pandas_size (bytes)",
        "pyarrow_size (bytes)",
        "polars_size (bytes)",
    ],
    y="dataframe_shape (rows, cols)",
    orientation="h",
    barmode="group",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Bytes"},
    width=1300,
)
fig.update_layout(
    legend_title_text="In-memory Data Size",
    legend=dict(x=0.72, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)

pio.write_image(fig, mem_read_size_image)
Image(url=mem_read_size_image)

# + papermill={"duration": 0.005354, "end_time": "2025-05-14T17:08:14.066558", "exception": false, "start_time": "2025-05-14T17:08:14.061204", "status": "completed"}
