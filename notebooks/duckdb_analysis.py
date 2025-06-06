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

# + [markdown] papermill={"duration": 0.004932, "end_time": "2025-05-14T17:08:50.925134", "exception": false, "start_time": "2025-05-14T17:08:50.920202", "status": "completed"}
# # Why DuckDB?
#
# This notebook explores the benefits or drawbacks of using the [DuckDB](https://duckdb.org/) data joins relative to other methods such as Pandas DataFrames.

# + papermill={"duration": 0.580047, "end_time": "2025-05-14T17:08:51.510527", "exception": false, "start_time": "2025-05-14T17:08:50.930480", "status": "completed"}
import itertools
import json
import pathlib
import subprocess
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
from utilities import get_system_info

# + papermill={"duration": 0.013888, "end_time": "2025-05-14T17:08:51.529064", "exception": false, "start_time": "2025-05-14T17:08:51.515176", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.012347, "end_time": "2025-05-14T17:08:51.546019", "exception": false, "start_time": "2025-05-14T17:08:51.533672", "status": "completed"}
# target file or table names
image_dir = "images"
examples_dir = "examples"
join_read_time_image = f"{image_dir}/duckdb-comparisons-join-read-time.png"
join_mem_size_image = f"{image_dir}/duckdb-comparisons-join-memory-size.png"
example_files_list = [
    f"{examples_dir}/join_pandas.py",
    f"{examples_dir}/join_duckdb.py",
]
example_data_list = [
    f"{examples_dir}/data/all_cellprofiler.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x2.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x4.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x8.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x16.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x32.sqlite",
]
# format for memray time strings
tformat = "%Y-%m-%d %H:%M:%S.%f%z"

# + papermill={"duration": 1.741537, "end_time": "2025-05-14T17:08:53.291136", "exception": false, "start_time": "2025-05-14T17:08:51.549599", "status": "completed"}
# avoid a "cold start" for tested packages by using them before benchmarks
for example_file in example_files_list:
    run = subprocess.run(
        ["python", example_file, example_data_list[0]],
        capture_output=True,
    )

# + papermill={"duration": 52.024412, "end_time": "2025-05-14T17:09:45.318911", "exception": false, "start_time": "2025-05-14T17:08:53.294499", "status": "completed"}
# result list for storing data
results = []

# loop for iterating over examples and example data
# and gathering data about operations on them
for example_file, example_data in itertools.product(
    example_files_list,
    example_data_list,
):
    target_bin = f"{example_file}_with_{example_data.replace(f'{examples_dir}/data/', '')}.memray.bin"
    target_json = f"{target_bin}.json"
    memray_run = subprocess.run(
        [
            "memray",
            "run",
            "--output",
            target_bin,
            "--force",
            example_file,
            example_data,
        ],
        capture_output=True,
        check=True,
    )

    memray_stats = subprocess.run(
        [
            "memray",
            "stats",
            "--json",
            "--output",
            target_json,
            "--force",
            target_bin,
        ],
        capture_output=True,
        check=True,
    )

    # open the json data
    with open(target_json) as memray_json_file:
        memray_data = json.load(memray_json_file)

    # append data to the result list
    results.append(
        {
            # general information about the dataframe
            "file_input": example_file.replace(f"{examples_dir}/", ""),
            "data_input": example_data.replace(f"{examples_dir}/data/", ""),
            # information about pandas
            "time_duration (secs)": (
                datetime.strptime(memray_data["metadata"]["end_time"], tformat)
                - datetime.strptime(memray_data["metadata"]["start_time"], tformat)
            ).total_seconds(),
            "total_memory (bytes)": memray_data["total_bytes_allocated"],
        }
    )

    # cleanup
    pathlib.Path(target_bin).unlink(missing_ok=True)
    pathlib.Path(target_json).unlink(missing_ok=True)

df_results = pd.DataFrame(results)
df_results

# + papermill={"duration": 0.041861, "end_time": "2025-05-14T17:09:45.365371", "exception": false, "start_time": "2025-05-14T17:09:45.323510", "status": "completed"}
df_results["data_input_renamed"] = df_results["data_input"].str.replace(
    "all_cellprofiler", "input"
)
df_results["pandas_time_duration (secs)"] = df_results[
    df_results["file_input"] == "join_pandas.py"
]["time_duration (secs)"]
df_results["pandas_total_memory (bytes)"] = df_results[
    df_results["file_input"] == "join_pandas.py"
]["total_memory (bytes)"]
df_results["duckdb_time_duration (secs)"] = df_results[
    df_results["file_input"] == "join_duckdb.py"
]["time_duration (secs)"]
df_results["duckdb_total_memory (bytes)"] = df_results[
    df_results["file_input"] == "join_duckdb.py"
]["total_memory (bytes)"]
df_results = (
    df_results.apply(lambda x: pd.Series(x.dropna().values))
    .drop(["file_input", "time_duration (secs)", "total_memory (bytes)"], axis=1)
    .dropna()
)
df_results

# + papermill={"duration": 1.312852, "end_time": "2025-05-14T17:09:46.683082", "exception": false, "start_time": "2025-05-14T17:09:45.370230", "status": "completed"}
# read time chart
fig = px.line(
    df_results,
    y=[
        "pandas_time_duration (secs)",
        "duckdb_time_duration (secs)",
    ],
    x="data_input_renamed",
    labels={"data_input_renamed": "Input File", "value": "Seconds"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.T10,
)
fig.update_layout(
    legend_title_text="Read Time Duration",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)
fig.update_xaxes(range=[-0.03, 5.2])
fig.update_traces(mode="lines+markers")

pio.write_image(fig, join_read_time_image)
Image(url=join_read_time_image)

# + papermill={"duration": 0.163838, "end_time": "2025-05-14T17:09:46.850046", "exception": false, "start_time": "2025-05-14T17:09:46.686208", "status": "completed"}
# memory size
fig = px.bar(
    df_results,
    x=[
        "pandas_total_memory (bytes)",
        "duckdb_total_memory (bytes)",
    ],
    y="data_input_renamed",
    labels={"data_input_renamed": "Input File", "value": "Bytes"},
    orientation="h",
    barmode="group",
    width=1300,
    color_discrete_sequence=px.colors.qualitative.T10,
)
fig.update_layout(
    legend_title_text="In-memory Data Size",
    legend=dict(x=0.58, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)

pio.write_image(fig, join_mem_size_image)
Image(url=join_mem_size_image)
