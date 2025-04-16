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
#     display_name: Python 3 (ipyflow)
#     language: python
#     name: ipyflow
# ---

# # CytoTable (convert) and Pycytominer (SingleCells.merge_single_cells) Performance Comparisons
#
# This notebook explores CytoTable (convert) and Pycytominer (SingleCells.merge_single_cells) usage with datasets of varying size to help describe performance impacts.

# +
import io
import itertools
import json
import os
import pathlib
import subprocess
import tokenize
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import Image

# set plotly default theme
pio.templates.default = "simple_white"
# -

# observe the virtual env for dependency inheritance with memray
# from subprocedure calls
"/".join(
    subprocess.run(
        [
            "which",
            "memray",
        ],
        capture_output=True,
        check=True,
    )
    # decode bytestring as utf-8
    .stdout.decode("utf-8")
    # remove personal file structure
    .split("/")[6:]
    # replace final newline
).replace("\n", "")

# target file or table names
image_dir = "images"
examples_dir = "examples"
join_read_time_image = (
    f"{image_dir}/cytotable-and-pycytominer-comparisons-join-completion-time.png"
)
join_mem_size_image = (
    f"{image_dir}/cytotable-and-pycytominer-comparisons-join-memory-size.png"
)
example_files_list = [
    f"{examples_dir}/cytotable_convert_nf1.py",
    f"{examples_dir}/pycytominer_merge_nf1.py",
]
example_data_list = [
    f"{examples_dir}/data/all_cellprofiler.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x2.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x4.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x8.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x16.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x32.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x64.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x128.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x256.sqlite",
    f"{examples_dir}/data/all_cellprofiler-x512.sqlite",
]
# format for memray time strings
tformat = "%Y-%m-%d %H:%M:%S.%f"

# avoid a "cold start" for tested packages by using them before benchmarks
for example_file in example_files_list:
    run = subprocess.run(
        ["python", example_file, example_data_list[0]],
        capture_output=True,
    )

# +
# result list for storing data
results = []

# loop for iterating over examples and example data
# and gathering data about operations on them
for example_file, example_data in itertools.product(
    example_files_list,
    example_data_list,
):
    target_bin = f"{example_file}_with_{example_data.replace(f'{examples_dir}/data/','')}.memray.bin"
    target_json = f"{target_bin}.json"
    memray_run = subprocess.run(
        [
            "memray",
            "run",
            "--follow-fork",
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
            "data_input": example_data,
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

# +
# add columns for data understandability in plots


def get_file_size_mb(file_path):
    """
    Gather filesize given a file_path
    """
    try:
        return pathlib.Path(file_path).stat().st_size / 1024 / 1024
    except FileNotFoundError:
        return None


# memory usage in MB
df_results["total_memory (GB)"] = (
    df_results["total_memory (bytes)"] / 1024 / 1024 / 1024
)

# data input size additions for greater context
df_results["data_input_size_mb"] = df_results["data_input"].apply(get_file_size_mb)
df_results["data_input_with_size"] = (
    df_results["data_input"]
    + " ("
    + round(df_results["data_input_size_mb"]).astype("int64").astype("str")
    + " MB)"
)

# rename data input to simplify
df_results["data_input_renamed"] = (
    df_results["data_input_with_size"]
    .str.replace(f"{examples_dir}/data/", "")
    .str.replace("all_cellprofiler", "input")
)
df_results
# -

# build cols for split reference in the plot
df_results["cytotable_time_duration (secs)"] = df_results[
    df_results["file_input"] == "cytotable_convert_nf1.py"
]["time_duration (secs)"]
df_results["cytotable_total_memory (GB)"] = df_results[
    df_results["file_input"] == "cytotable_convert_nf1.py"
]["total_memory (GB)"]
df_results["pycytominer_time_duration (secs)"] = df_results[
    df_results["file_input"] == "pycytominer_merge_nf1.py"
]["time_duration (secs)"]
df_results["pycytominer_total_memory (GB)"] = df_results[
    df_results["file_input"] == "pycytominer_merge_nf1.py"
]["total_memory (GB)"]
df_results = (
    df_results.apply(lambda x: pd.Series(x.dropna().values))
    .drop(["file_input", "time_duration (secs)", "total_memory (GB)"], axis=1)
    .dropna()
)
df_results

# +
# read time chart
fig = px.line(
    df_results,
    y=[
        "cytotable_time_duration (secs)",
        "pycytominer_time_duration (secs)",
    ],
    x="data_input_renamed",
    title="CytoTable and Pycytominer SQLite Processing Time Comparison",
    labels={"data_input_renamed": "Input File", "value": "Seconds"},
    height=500,
    width=900,
    symbol_sequence=["diamond"],
    color_discrete_sequence=[
        px.colors.qualitative.Vivid[6],
        px.colors.qualitative.Vivid[4],
    ],
)

# rename the lines for the legend
newnames = {
    "cytotable_time_duration (secs)": "CytoTable",
    "pycytominer_time_duration (secs)": "Pycytominer",
}
# referenced from: https://stackoverflow.com/a/64378982
fig.for_each_trace(
    lambda t: t.update(
        name=newnames[t.name],
        legendgroup=newnames[t.name],
        hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
    )
)

# update the legend
fig.update_layout(
    legend_title_text="",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=16,  # global font size
    ),
)
# fig.update_xaxes(range=[-0.03, 5.2])
fig.update_traces(mode="lines+markers")

fig.write_image(join_read_time_image)
fig.write_image(join_read_time_image.replace(".png", ".svg"))
Image(url=join_read_time_image.replace(".png", ".svg"))

# +
# memory size

fig = px.line(
    df_results,
    y=[
        "cytotable_total_memory (GB)",
        "pycytominer_total_memory (GB)",
    ],
    x="data_input_renamed",
    title="CytoTable and Pycytominer SQLite Total Memory Consumption",
    labels={"data_input_renamed": "Input File", "value": "Gigabytes (GB)"},
    height=500,
    width=900,
    symbol_sequence=["diamond"],
    color_discrete_sequence=[
        px.colors.qualitative.Vivid[6],
        px.colors.qualitative.Vivid[4],
    ],
)

# rename the lines for the legend
newnames = {
    "cytotable_total_memory (GB)": "CytoTable",
    "pycytominer_total_memory (GB)": "Pycytominer",
}
# referenced from: https://stackoverflow.com/a/64378982
fig.for_each_trace(
    lambda t: t.update(
        name=newnames[t.name],
        legendgroup=newnames[t.name],
        hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
    )
)

# update the legend
fig.update_layout(
    legend_title_text="",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=16,  # global font size
    ),
)
fig.update_traces(mode="lines+markers")


fig.write_image(join_mem_size_image)
fig.write_image(join_mem_size_image.replace(".png", ".svg"))
Image(url=join_mem_size_image.replace(".png", ".svg"))
