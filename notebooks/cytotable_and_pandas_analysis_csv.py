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

# # CytoTable (convert) and Pandas (Merges) Performance Comparisons
#
# This notebook explores CytoTable (convert) and Pandas (DataFrame merges) usage with datasets of varying size to help describe performance impacts.

# +
import itertools
import json
import os
import pathlib
import signal
import subprocess
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.io as pio
import psutil
from IPython.display import Image
from utilities import get_system_info

# set plotly default theme
pio.templates.default = "simple_white"
# -

# show the system information
_ = get_system_info(show_output=True)

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

# +
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
]

# format for memray time strings
tformat = "%Y-%m-%d %H:%M:%S.%f%z"
# -

# avoid a "cold start" for tested packages by using them before benchmarks
for example_file in example_files_list:
    _ = subprocess.run(
        ["python", example_file, example_data_list[0]],
        check=True,
    )

# +
# Define the Parquet file path
results_parquet_file = "cytotable_pandas_results.parquet"

# Load existing results if available
if pathlib.Path(results_parquet_file).exists():
    df_results = pd.read_parquet(results_parquet_file)
    results = df_results.to_dict(orient="records")
else:
    results = []

# +
# Number of iterations for each combination
num_iterations = 6

# Loop through each combination of example file and data file
for example_file, example_data in itertools.product(
    example_files_list, example_data_list
):
    for iteration in range(num_iterations):
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

        target_bin = f"{example_file}_with_{example_data.replace(f'{examples_dir}/data/', '')}_iter_{iteration}.memray.bin"
        target_json = f"{target_bin}.json"

        try:
            # Run the example file with memray
            subprocess.run(
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Generate memray stats
            subprocess.run(
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

            # Load the JSON data
            with open(target_json) as memray_json_file:
                memray_data = json.load(memray_json_file)

            # Append the result
            results.append(
                {
                    "file_input": example_file.replace(f"{examples_dir}/", ""),
                    "data_input": example_data,
                    "iteration": iteration,
                    "time_duration (secs)": (
                        datetime.strptime(memray_data["metadata"]["end_time"], tformat)
                        - datetime.strptime(
                            memray_data["metadata"]["start_time"], tformat
                        )
                    ).total_seconds(),
                    "total_memory (bytes)": memray_data["total_bytes_allocated"],
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
            # Cleanup temporary files
            pathlib.Path(target_bin).unlink(missing_ok=True)
            pathlib.Path(target_json).unlink(missing_ok=True)
            # Cleanup any Parsl processes which may yet still exist
            for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline")
                    if isinstance(cmdline, list) and any(
                        "parsl" in part for part in cmdline
                    ):
                        os.kill(proc.info["pid"], signal.SIGKILL)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            print(
                f"Finished {example_file} with {example_data}, iteration {iteration}."
            )


# Final save to Parquet
df_results = pd.DataFrame(results)
df_results.to_parquet(results_parquet_file, index=False)

print(f"Processing complete. Results saved to '{results_parquet_file}'.")


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
    .str.replace("examplehuman_cellprofiler_features_", "input_")
)
df_results
# -

# build cols for split reference in the plot
df_results["cytotable_time_duration (multiprocess) (secs)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multiprocess_csv.py"
]["time_duration (secs)"]
df_results["cytotable_total_memory (multiprocess) (GB)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multiprocess_csv.py"
]["total_memory (GB)"]
df_results["cytotable_time_duration (multithread) (secs)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multithread_csv.py"
]["time_duration (secs)"]
df_results["cytotable_total_memory (multithread) (GB)"] = df_results[
    df_results["file_input"] == "cytotable_convert_examplehuman_multithread_csv.py"
]["total_memory (GB)"]
df_results["pandas_time_duration (secs)"] = df_results[
    df_results["file_input"] == "pandas_merge_examplehuman_csv.py"
]["time_duration (secs)"]
df_results["pandas_total_memory (GB)"] = df_results[
    df_results["file_input"] == "pandas_merge_examplehuman_csv.py"
]["total_memory (GB)"]
df_results = (
    df_results.apply(lambda x: pd.Series(x.dropna().values))
    .drop(["file_input", "time_duration (secs)", "total_memory (GB)"], axis=1)
    .dropna()
)
df_results

# Group by data_input_renamed and calculate mean, min, and max
aggregated_results = df_results.groupby("data_input_renamed").agg(
    {
        "cytotable_time_duration (multiprocess) (secs)": ["mean", "min", "max"],
        "cytotable_time_duration (multithread) (secs)": ["mean", "min", "max"],
        "pandas_time_duration (secs)": ["mean", "min", "max"],
        "cytotable_total_memory (multiprocess) (GB)": ["mean", "min", "max"],
        "cytotable_total_memory (multithread) (GB)": ["mean", "min", "max"],
        "pandas_total_memory (GB)": ["mean", "min", "max"],
    }
)
# Flatten the multi-level columns
aggregated_results.columns = [
    f"{col[0]} ({col[1]})" for col in aggregated_results.columns
]
aggregated_results.reset_index(inplace=True)
aggregated_results

# +
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

# +
# Memory plot with min and max errors
fig = px.line(
    aggregated_results,
    x="data_input_renamed",
    y=[
        "cytotable_total_memory (multiprocess) (GB) (mean)",
        "cytotable_total_memory (multithread) (GB) (mean)",
        "pandas_total_memory (GB) (mean)",
    ],
    title="CytoTable and Pandas CSV Total Memory Consumption with Min/Max Errors",
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
        "cytotable_total_memory (multiprocess) (GB)",
        "cytotable_total_memory (multithread) (GB)",
        "pandas_total_memory (GB)",
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
    "cytotable_total_memory (multiprocess) (GB) (mean)": "CytoTable (multiprocess)",
    "cytotable_total_memory (multithread) (GB) (mean)": "CytoTable (multithread)",
    "pandas_total_memory (GB) (mean)": "Pandas",
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
# -
