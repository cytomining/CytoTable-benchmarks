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

# + [markdown] papermill={"duration": 0.005297, "end_time": "2025-05-14T18:14:03.265955", "exception": false, "start_time": "2025-05-14T18:14:03.260658", "status": "completed"}
# # CytoTable (convert) and Pandas (Merges) Performance Comparisons
#
# This notebook explores CytoTable (convert) and Pandas (DataFrame merges) usage with datasets of varying size to help describe performance impacts.

# + papermill={"duration": 0.78774, "end_time": "2025-05-14T18:14:04.060120", "exception": false, "start_time": "2025-05-14T18:14:03.272380", "status": "completed"}
import itertools
import json
import os
import pathlib
import re
import signal
import subprocess
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
import psutil
from IPython.display import Image
from utilities import (
    get_memory_peak_and_time_duration,
    get_parsl_peak_memory,
    get_system_info,
)

# set plotly default theme
pio.templates.default = "simple_white"
# monitoring database for parsl multiprocessing work
db_file = "runinfo/monitoring.db"

# + papermill={"duration": 0.013683, "end_time": "2025-05-14T18:14:04.078668", "exception": false, "start_time": "2025-05-14T18:14:04.064985", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.018972, "end_time": "2025-05-14T18:14:04.102214", "exception": false, "start_time": "2025-05-14T18:14:04.083242", "status": "completed"}
# observe the virtual env for dependency inheritance with memray
# from subprocedure calls
"/".join(
    subprocess.run(
        [
            "which",
            "python",
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

# + papermill={"duration": 0.012864, "end_time": "2025-05-14T18:14:04.120203", "exception": false, "start_time": "2025-05-14T18:14:04.107339", "status": "completed"}
# target file or table names
image_dir = "images"
examples_dir = "examples"
join_read_time_image = (
    f"{image_dir}/cytotable-and-pandas-comparisons-join-completion-time-csv.png"
)
join_mem_size_image = (
    f"{image_dir}/cytotable-and-pandas-comparisons-join-memory-size-csv.png"
)
example_files_list = [
    f"{examples_dir}/cytotable_convert_examplehuman_multiprocess_csv.py",
    f"{examples_dir}/cytotable_convert_examplehuman_multithread_csv.py",
    f"{examples_dir}/pandas_merge_examplehuman_csv.py",
]
example_data_list = [
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x2",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x4",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x8",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x16",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x32",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x64",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x128",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x256",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x512",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x1024",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x2048",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x4096",
    f"{examples_dir}/data/examplehuman_cellprofiler_features_csv-x8192",
]

# format for memray time strings
tformat = "%Y-%m-%d %H:%M:%S.%f%z"

# + papermill={"duration": 0.012649, "end_time": "2025-05-14T18:14:04.138079", "exception": false, "start_time": "2025-05-14T18:14:04.125430", "status": "completed"}
# Define the Parquet file path
results_parquet_file = "cytotable_pandas_results.parquet"

# Load existing results if available
if pathlib.Path(results_parquet_file).exists():
    df_results = pd.read_parquet(results_parquet_file)
    results = df_results.to_dict(orient="records")
else:
    results = []

# + papermill={"duration": 6155.02308, "end_time": "2025-05-14T19:56:39.166337", "exception": false, "start_time": "2025-05-14T18:14:04.143257", "status": "completed"}
# Number of iterations for each combination
num_iterations = 6

# Loop through each combination of example file and data file
for example_file, example_data in itertools.product(
    example_files_list, example_data_list
):
    for iteration in range(num_iterations):

        print(f"Starting {example_file} with {example_data}, iteration {iteration}.")
        # Skip if this combination and iteration are already processed
        if any(
            result["file_input"] == example_file
            and result["data_input"] == example_data
            and result["iteration"] == iteration
            for result in results
        ):
            print(
                f"Skipping already processed: {example_file} with {example_data}, iteration {iteration}"
            )
            continue

        try:
            # gather memory peak and time duration
            memory_peak, time_duration = get_memory_peak_and_time_duration(
                cmd=[
                    "python",
                    example_file,
                    example_data,
                ],
                polling_pause_seconds=0.1,
                # if we have a multiprocessed parsl process skip memory
                # (we will check this via parsl monitoring).
                skip_memory_check=("multiprocess" in example_file),
            )

            # Append the result
            results.append(
                {
                    "file_input": example_file.replace(f"{examples_dir}/", ""),
                    "data_input": example_data,
                    "iteration": iteration,
                    "time_duration (secs)": time_duration,
                    "peak_memory (bytes)": (
                        memory_peak
                        # if we have a multiprocessed parsl result we must
                        # gather the peak memory using parsl's monitoring
                        # database.
                        if "multiprocess" not in example_file
                        else get_parsl_peak_memory(db_file=db_file)
                    ),
                }
            )

            # Save intermediate results to Parquet
            df_results = pd.DataFrame(results)
            df_results.to_parquet(results_parquet_file, index=False)

        except Exception as e:
            print(
                f"Error processing {example_file} with {example_data}, iteration {iteration}: {e}"
            )

        finally:
            # remove monitoring database if present from parsl processing
            if pathlib.Path(db_file).is_file():
                pathlib.Path(db_file).unlink()
            print(
                f"Finished {example_file} with {example_data}, iteration {iteration}."
            )


# Final save to Parquet
df_results = pd.DataFrame(results)
df_results.to_parquet(results_parquet_file, index=False)

print(f"Processing complete. Results saved to '{results_parquet_file}'.")


# + papermill={"duration": 0.07414, "end_time": "2025-05-14T19:56:39.265283", "exception": false, "start_time": "2025-05-14T19:56:39.191143", "status": "completed"}
# add columns for data understandability in plots
def get_file_size_mb(path):
    """
    Returns the size in MB of a file or total size of all files in a directory.
    """
    p = pathlib.Path(path)
    try:
        if p.is_file():
            return p.stat().st_size / 1024 / 1024
        elif p.is_dir():
            return (
                sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
            )
        else:
            return None
    except FileNotFoundError:
        return None


# memory usage in MB
df_results["peak_memory (GB)"] = df_results["peak_memory (bytes)"] / 1024 / 1024 / 1024

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
    .str.replace("examplehuman_cellprofiler_features_", "input_")
)
df_results

# + papermill={"duration": 0.050608, "end_time": "2025-05-14T19:56:39.335582", "exception": false, "start_time": "2025-05-14T19:56:39.284974", "status": "completed"}
# build cols for split reference in the plot
df_results["cytotable_time_duration (multiprocess) (secs)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multiprocess_csv.py"
]["time_duration (secs)"]
df_results["cytotable_peak_memory (multiprocess) (GB)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multiprocess_csv.py"
]["peak_memory (GB)"]
df_results["cytotable_time_duration (multithread) (secs)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multithread_csv.py"
]["time_duration (secs)"]
df_results["cytotable_peak_memory (multithread) (GB)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multithread_csv.py"
]["peak_memory (GB)"]
df_results["pandas_time_duration (secs)"] = df_results[
    df_results["file_input"] == "pandas_merge_examplehuman_csv.py"
]["time_duration (secs)"]
df_results["pandas_peak_memory (GB)"] = df_results[
    df_results["file_input"] == "pandas_merge_examplehuman_csv.py"
]["peak_memory (GB)"]
df_results = (
    df_results.apply(lambda x: pd.Series(x.dropna().values))
    .drop(["file_input", "time_duration (secs)", "peak_memory (GB)"], axis=1)
    .dropna()
)
df_results

# + papermill={"duration": 0.05393, "end_time": "2025-05-14T19:56:39.409503", "exception": false, "start_time": "2025-05-14T19:56:39.355573", "status": "completed"}
# Group by data_input_renamed and calculate mean, min, and max
aggregated_results = df_results.groupby("data_input_renamed").agg(
    {
        "cytotable_time_duration (multiprocess) (secs)": ["mean", "min", "max"],
        "cytotable_time_duration (multithread) (secs)": ["mean", "min", "max"],
        "pandas_time_duration (secs)": ["mean", "min", "max"],
        "cytotable_peak_memory (multiprocess) (GB)": ["mean", "min", "max"],
        "cytotable_peak_memory (multithread) (GB)": ["mean", "min", "max"],
        "pandas_peak_memory (GB)": ["mean", "min", "max"],
    }
)
# Flatten the multi-level columns
aggregated_results.columns = [
    f"{col[0]} ({col[1]})" for col in aggregated_results.columns
]
aggregated_results.reset_index(inplace=True)


# Helper function to extract numeric value or None
def sort_key(s):
    match = re.search(r"\d+", s)
    if match:
        return (1, int(match.group()))  # numeric items: (1, number)
    else:
        return (0, s.lower())  # non-numeric items: (0, alphabetical)


# Sort using the custom key
aggregated_results = aggregated_results.sort_values(
    by="data_input_renamed", key=lambda col: col.map(sort_key)
)
aggregated_results

# + papermill={"duration": 1.388203, "end_time": "2025-05-14T19:56:40.818276", "exception": false, "start_time": "2025-05-14T19:56:39.430073", "status": "completed"}
# Time plot with min and max errors
fig = px.line(
    aggregated_results,
    x="data_input_renamed",
    y=[
        "cytotable_time_duration (multiprocess) (secs) (mean)",
        "cytotable_time_duration (multithread) (secs) (mean)",
        "pandas_time_duration (secs) (mean)",
    ],
    title="CytoTable and Pandas CSV Processing Time with Min/Max Errors",
    labels={"data_input_renamed": "Input File", "value": "Seconds"},
    height=500,
    width=900,
    symbol_sequence=["diamond"],
    color_discrete_sequence=[
        px.colors.qualitative.Vivid[6],
        px.colors.qualitative.Vivid[7],
        px.colors.qualitative.Vivid[4],
    ],
)

# Add error bars for each trace
for i, col in enumerate(
    [
        "cytotable_time_duration (multiprocess) (secs)",
        "cytotable_time_duration (multithread) (secs)",
        "pandas_time_duration (secs)",
    ]
):
    fig.data[i].update(
        error_y=dict(
            array=(
                aggregated_results[f"{col} (max)"] - aggregated_results[f"{col} (mean)"]
            ),
            arrayminus=(
                aggregated_results[f"{col} (mean)"] - aggregated_results[f"{col} (min)"]
            ),
        )
    )

# Rename the lines for the legend
newnames = {
    "cytotable_time_duration (multiprocess) (secs) (mean)": "CytoTable (multiprocess)",
    "cytotable_time_duration (multithread) (secs) (mean)": "CytoTable (multithread)",
    "pandas_time_duration (secs) (mean)": "Pandas",
}
fig.for_each_trace(
    lambda t: t.update(
        name=newnames[t.name],
        legendgroup=newnames[t.name],
        hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
    )
)

# Update the legend and layout
fig.update_layout(
    legend_title_text="",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(size=16),  # Global font size
)
fig.update_traces(mode="lines+markers")
fig.update_layout(yaxis=dict(rangemode="tozero", autorange=True))

# Save and display the plot
fig.write_image(join_read_time_image)
fig.write_image(join_read_time_image.replace(".png", ".svg"))
Image(url=join_read_time_image.replace(".png", ".svg"))

# + papermill={"duration": 0.351038, "end_time": "2025-05-14T19:56:41.201059", "exception": false, "start_time": "2025-05-14T19:56:40.850021", "status": "completed"}
# Memory plot with min and max errors
fig = px.line(
    aggregated_results,
    x="data_input_renamed",
    y=[
        "cytotable_peak_memory (multiprocess) (GB) (mean)",
        "cytotable_peak_memory (multithread) (GB) (mean)",
        "pandas_peak_memory (GB) (mean)",
    ],
    title="CytoTable and Pandas CSV Peak Memory with Min/Max Errors",
    labels={"data_input_renamed": "Input File", "value": "Gigabytes (GB)"},
    height=500,
    width=900,
    symbol_sequence=["diamond"],
    color_discrete_sequence=[
        px.colors.qualitative.Vivid[6],
        px.colors.qualitative.Vivid[7],
        px.colors.qualitative.Vivid[4],
    ],
)

# Add error bars for each trace
for i, col in enumerate(
    [
        "cytotable_peak_memory (multiprocess) (GB)",
        "cytotable_peak_memory (multithread) (GB)",
        "pandas_peak_memory (GB)",
    ]
):
    fig.data[i].update(
        error_y=dict(
            array=(
                aggregated_results[f"{col} (max)"] - aggregated_results[f"{col} (mean)"]
            ),
            arrayminus=(
                aggregated_results[f"{col} (mean)"] - aggregated_results[f"{col} (min)"]
            ),
        )
    )

# Rename the lines for the legend
newnames = {
    "cytotable_peak_memory (multiprocess) (GB) (mean)": "CytoTable (multiprocess)",
    "cytotable_peak_memory (multithread) (GB) (mean)": "CytoTable (multithread)",
    "pandas_peak_memory (GB) (mean)": "Pandas",
}
fig.for_each_trace(
    lambda t: t.update(
        name=newnames[t.name],
        legendgroup=newnames[t.name],
        hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
    )
)

# Update the legend and layout
fig.update_layout(
    legend_title_text="",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(size=16),  # Global font size
)
fig.update_traces(mode="lines+markers")
fig.update_layout(yaxis=dict(rangemode="tozero", autorange=True))

# Save and display the plot
fig.write_image(join_mem_size_image)
fig.write_image(join_mem_size_image.replace(".png", ".svg"))
Image(url=join_mem_size_image.replace(".png", ".svg"))
