"""
An example to test pyarrow reads using parquet files.
"""

import pyarrow as pa
from pyarrow import parquet

pa.set_memory_pool(pa.jemalloc_memory_pool())

for parquet_file in [
    "./examples/data/random_number_data.parquet",
    "./examples/data/random_number_data-copy.parquet",
]:
    parquet.read_table(source=parquet_file)
