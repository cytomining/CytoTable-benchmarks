import shutil

import numpy as np
import pyarrow as pa
from pyarrow import parquet

# create randomized number data and related pyarrow table
tbl_numeric = pa.Table.from_arrays(
    [pa.array(np.random.rand(1000, 100)[:, i]) for i in range(100)],
    names=[f"Column_{i}" for i in range(100)],
)
# Create a table and write it to file
parquet.write_table(
    table=tbl_numeric,
    where="./random_number_data.parquet",
)

# create a duplicate file for use in looped testing
shutil.copy(
    "./examples/data/random_number_data.parquet",
    "./examples/data/random_number_data-copy.parquet",
)
