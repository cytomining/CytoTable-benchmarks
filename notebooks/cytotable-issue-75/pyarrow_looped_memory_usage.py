# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipyflow)
#     language: python
#     name: ipyflow
# ---

# # PyArrow looped memory usage analysis
#
# This notebook explores how PyArrow uses memory when reading parquet files implemented in a loop. The work is related to [CytoTable#75](https://github.com/cytomining/CytoTable/issues/75).

# +
import io
import itertools
import json
import pathlib
import subprocess

from IPython.display import IFrame
# -

# setup variables for use below
target_python_list = [
    "./examples/pyarrow_parquet_reads_malloc.py",
    "./examples/pyarrow_parquet_reads_mimalloc.py",
    "./examples/pyarrow_parquet_reads_jemalloc.py",
    "./examples/pyarrow_parquet_reads_malloc_memorymap.py",
    "./examples/pyarrow_parquet_reads_mimalloc_memorymap.py",
    "./examples/pyarrow_parquet_reads_jemalloc_memorymap.py",
]
target_bin_list = [
    f"{pathlib.Path(target).name}.memray.bin" for target in target_python_list
]
target_html_list = [f"{target_bin}.html" for target_bin in target_bin_list]

for target_python, target_bin, target_html in zip(
    target_python_list, target_bin_list, target_html_list
):
    # create memory profile
    memray_run = subprocess.run(
        [
            "memray",
            "run",
            "--output",
            target_bin,
            "--force",
            "--native",
            "--follow-fork",
            target_python,
        ],
        capture_output=True,
        check=True,
    )
    # create flamegraph data
    memray_flamegraph = subprocess.run(
        [
            "memray",
            "flamegraph",
            "--output",
            target_html,
            "--force",
            target_bin,
        ],
        capture_output=True,
        check=True,
    )

# display flamegraph results
print(target_html_list[0])
IFrame(target_html_list[0], width="100%", height="1000")

# display flamegraph results
print(target_html_list[1])
IFrame(target_html_list[1], width="100%", height="1000")

# display flamegraph results
print(target_html_list[2])
IFrame(target_html_list[2], width="100%", height="1000")

# display flamegraph results
print(target_html_list[3])
IFrame(target_html_list[3], width="100%", height="1000")

# display flamegraph results
print(target_html_list[4])
IFrame(target_html_list[4], width="100%", height="1000")

# display flamegraph results
print(target_html_list[5])
IFrame(target_html_list[5], width="100%", height="1000")
