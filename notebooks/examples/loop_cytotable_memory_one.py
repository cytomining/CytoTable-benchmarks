"""
A file to demonstrate use of CytoTable in a loop to analyze resource utilization.
Referenced with modifications from https://github.com/cytomining/CytoTable/issues/75
"""

import shutil
import tempfile

import cytotable

# create a list of files to reference
list_of_sqlite_files = [
    "./examples/data/all_cellprofiler.sqlite",
]

# create a list of temporary directories to use for destination paths
# note: we create isolated dirs here to help avoid overlapping results
list_of_dest_paths = [tempfile.mkdtemp() for _ in range(len(list_of_sqlite_files))]

for source_file, dest_path in zip(list_of_sqlite_files, list_of_dest_paths):
    cytotable.convert(
        source_path=source_file,
        dest_path=dest_path,
        dest_datatype="parquet",
        preset="cellprofiler_sqlite_pycytominer",
    )

# cleanup the temporary dirs
for dest_path in list_of_dest_paths:
    shutil.rmtree(path=dest_path, ignore_errors=True)
