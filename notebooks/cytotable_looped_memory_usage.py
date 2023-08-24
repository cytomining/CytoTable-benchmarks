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

# # CytoTable looped memory usage analysis
#
# This notebook explores how CytoTable uses memory when implemented in a loop. The work is related to [CytoTable#75](https://github.com/cytomining/CytoTable/issues/75).

# +
import io
import itertools
import json
import os
import pathlib
import subprocess

from IPython.display import IFrame
# -

# setup variables for use below
target_python_list = [
    "./examples/loop_cytotable_memory_one.py",
    "./examples/loop_cytotable_memory_two.py",
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
            target_python,
        ],
        capture_output=True,
        check=True,
        # env={**dict(os.environ), **{"ARROW_DEFAULT_MEMORY_POOL": "jemalloc"}},
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


