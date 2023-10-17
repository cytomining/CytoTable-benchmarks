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
import sys

import cytotable
import pandas as pd

gc.set_debug(gc.DEBUG_LEAK)

cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

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

# create a list of files to reference
list_of_sqlite_files = [
    "./examples/data/all_cellprofiler.sqlite",
]



cytotable.convert(
    source_path="./examples/data/all_cellprofiler.sqlite",
    dest_path="./examples/data/test-result.parquet",
    dest_datatype="parquet",
    preset="cellprofiler_sqlite_pycytominer",
)

# +
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
).to_csv(
    "leaks.csv"
)

{
    "source_group_name": "Per_image.sqlite",
    "source": {
        "source_path": PosixPath(
            "/Users/dabu5788/Documents/work/CytoTable-benchmarks-d33bs/notebooks/examples/data/all_cellprofiler.sqlite"
        ),
        "table_name": "Per_Image",
        "offsets": [0],
    },
    "chunk_size": 1000,
    "offset": 0,
    "dest_path": PosixPath(
        "/Users/dabu5788/Documents/work/CytoTable-benchmarks-d33bs/notebooks/examples/data/test-result.parquet"
    ),
    "data_type_cast_map": None,
}

df.sort_values(by="refcount", ascending=False).drop_duplicates(subset="id")[
    "type"
].value_counts()

df[df["id"] == 5304345792].sort_values(by="refcount", ascending=False).iloc[0]["repr"]

{
    "source_group_name": "Per_nuclei.sqlite",
    "source": {
        "source_path": PosixPath(
            "/Users/dabu5788/Documents/work/CytoTable-benchmarks-d33bs/notebooks/examples/data/all_cellprofiler.sqlite"
        ),
        "table_name": "Per_Nuclei",
        "offsets": [0],
    },
    "chunk_size": 1000,
    "offset": 0,
    "dest_path": PosixPath(
        "/Users/dabu5788/Documents/work/CytoTable-benchmarks-d33bs/notebooks/examples/data/test-result.parquet"
    ),
    "data_type_cast_map": None,
}


