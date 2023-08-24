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
import shutil
import tempfile

import cytotable
import objgraph
from IPython.display import Image
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

print(objgraph.by_type(typename="dict", objects=roots)[:20])


