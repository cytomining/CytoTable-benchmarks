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

# # CytoTable object graph analysis
#
# This notebook explores how CytoTable objects operate. The work is related to [CytoTable#75](https://github.com/cytomining/CytoTable/issues/75).

# +
import gc
import logging
import os
import shutil
import sys
import tempfile

import cytotable
import objgraph
import parsl
from IPython.display import Image


class ExcludeCollectableMessages(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("gc: collectable")


logger = logging.getLogger(__name__)
logger.addFilter(ExcludeCollectableMessages)

gc.set_debug(gc.DEBUG_LEAK)
# -

# create a list of files to reference
list_of_sqlite_files = [
    "./examples/data/all_cellprofiler.sqlite",
]
graph_img = "cytotable-object-graph.png"

cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

objgraph.show_refs(cytotable, refcounts=True, filename=graph_img)

cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

print(objgraph.show_growth(limit=3), end="\n\n")

cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

print(objgraph.show_growth(limit=3), end="\n\n")

cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

print(objgraph.show_growth(limit=3), end="\n\n")

objgraph.show_refs(cytotable, refcounts=True, filename=graph_img)

roots = objgraph.get_leaking_objects()
objgraph.show_refs(roots[:3], refcounts=True, filename="roots.png")

objgraph.show_most_common_types(objects=roots, shortnames=False)

print(objgraph.by_type(typename="dict", objects=roots)[:5])

print(objgraph.by_type(typename="cell", objects=roots)[:5])

# +
# Explicitly collect garbage and check for memory leak warnings
import sys

import pandas as pd
from pympler import asizeof

collected = gc.collect()
if gc.garbage:
    print(f"Memory leak detected: {len(gc.garbage)} objects")
    df = pd.DataFrame(
        [
            {
                "id": id(obj),
                "type": type(obj),
                "refcount": sys.getrefcount(obj),
                "repr": repr(obj),
                "size": sys.getsizeof(obj),
            }
            for obj in gc.garbage
        ]
    )

df.head()
# -

df.sort_values(by=["size", "refcount"], ascending=False).drop_duplicates(
    subset="id"
).head(30)

df[
    ~df["repr"].str.contains("AppFuture") & ~df["repr"].str.contains("deque")
].sort_values(by=["size", "refcount"], ascending=False).drop_duplicates(
    subset="id"
).head(
    30
)

df.sort_values(by="refcount", ascending=False).drop_duplicates(subset="id")[
    "type"
].value_counts()

gc.collect()


