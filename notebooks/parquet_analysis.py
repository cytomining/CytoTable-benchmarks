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

# + [markdown] papermill={"duration": 0.006888, "end_time": "2025-09-03T21:55:12.133407", "exception": false, "start_time": "2025-09-03T21:55:12.126519", "status": "completed"}
# # Why Parquet?
#
# This notebook explores the benefits or drawbacks of using the [parquet](https://parquet.apache.org/docs/) file format relative to other formats such as CSV or SQLite.

# + papermill={"duration": 1.269103, "end_time": "2025-09-03T21:55:13.408073", "exception": false, "start_time": "2025-09-03T21:55:12.138970", "status": "completed"}
import os
import pathlib
import shutil
import string
import warnings
from typing import Literal

import anndata as ad
import duckdb
import hdf5plugin
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from anndata import ImplicitModificationWarning
from IPython.display import Image
from utilities import get_system_info, timer

# ignore anndata warnings about index conversion
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

# + papermill={"duration": 0.015148, "end_time": "2025-09-03T21:55:13.429168", "exception": false, "start_time": "2025-09-03T21:55:13.414020", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.049587, "end_time": "2025-09-03T21:55:13.484421", "exception": false, "start_time": "2025-09-03T21:55:13.434834", "status": "completed"}
# target file or table names
image_dir = "images"
csv_name = "example.csv.gz"
parquet_noc_name = "example.parquet"
parquet_snappy_name = "example.snappy.parquet"
parquet_gzip_name = "example.gzip.parquet"
parquet_lz4_name = "example.lz4.parquet"
parquet_zstd_name = "example.zstd.parquet"
sqlite_name = "example.sqlite"
sqlite_tbl_name = "tbl_example"
anndata_h5_noc_name = "adata.noc.h5ad"
anndata_h5_gzip_name = "adata.gzip.h5ad"
anndata_h5_lz4_name = "adata.lz4.h5ad"
anndata_h5_zstd_name = "adata.zstd.h5ad"
anndata_zarr_name = "adata.zarr"
file_write_time_image = f"{image_dir}/parquet-comparisons-file-write-time.png"
file_storage_size_image = f"{image_dir}/parquet-comparisons-file-storage-size.png"
file_read_time_all_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-all-columns.png"
)
file_read_time_one_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-one-column.png"
)
file_read_time_write_and_read_time_image = (
    f"{image_dir}/parquet-comparisons-file-write-and-read-time.png"
)


def remove_files():
    """
    Utility function to remove files as needed.
    """
    for name in [
        csv_name,
        parquet_noc_name,
        parquet_snappy_name,
        parquet_gzip_name,
        parquet_lz4_name,
        parquet_zstd_name,
        sqlite_name,
        anndata_h5_noc_name,
        anndata_h5_gzip_name,
        anndata_h5_lz4_name,
        anndata_h5_zstd_name,
    ]:
        pathlib.Path(name).unlink(missing_ok=True)

    if pathlib.Path(anndata_zarr_name).is_dir():
        shutil.rmtree(anndata_zarr_name)


# remove all files just in case
remove_files()


# + papermill={"duration": 0.017819, "end_time": "2025-09-03T21:55:13.509355", "exception": false, "start_time": "2025-09-03T21:55:13.491536", "status": "completed"}
def write_anndata(
    df: pd.DataFrame,
    write_to: Literal["h5ad", "zarr"],
    compression: Literal["gzip", "lz4", "zstd", "none"],
    dest_path: str,
) -> str:
    """
    Serialize a DataFrame to AnnData (h5ad or zarr).

    Numeric columns are stored in ``X`` (observations × variables). All
    remaining columns are stored in ``.obs``. Variable (feature) names are taken
    from the numeric column labels, and observation names from the DataFrame
    index.

    Args:
        df:
            Input table with rows as observations and columns as features.
        write_to:
            Output format. Either ``"h5ad"`` or ``"zarr"``.
        compression:
            The type of compression to use with
        dest_path:
            Destination file (``.h5ad``) or directory (zarr store)
            to write to. Parent directories are created if missing.

    Returns:
        The path written to as a string.
    """
    dest = pathlib.Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns found to place in AnnData.X.")

    non_numeric = df.select_dtypes(exclude=["number"])

    adata = ad.AnnData(X=numeric)
    adata.obs_names = df.index.astype(str)
    adata.var_names = numeric.columns.astype(str)
    # Align non-numeric obs metadata to the same index
    adata.obs = non_numeric

    if write_to == "h5ad":
        # we default to use None for compression
        # meaning no compression.
        comp_arg = None
        if compression == "gzip":
            comp_arg = "gzip"
        elif compression == "zstd":
            comp_arg = hdf5plugin.FILTERS["zstd"]
        elif compression == "lz4":
            comp_arg = hdf5plugin.FILTERS["lz4"]

        adata.write_h5ad(filename=str(dest), compression=comp_arg)
    elif write_to == "zarr":
        # For zarr, the destination is a directory-like store
        adata.write_zarr(str(dest))
    else:
        raise ValueError('write_to must be "h5ad" or "zarr".')

    return str(dest)


def read_anndata(
    path: str,
    read_from: Literal["h5ad", "zarr"],
    read_one: bool = False,
) -> pd.DataFrame:
    """
    Load an AnnData file (h5ad or zarr) as a single pandas DataFrame.

    The returned DataFrame concatenates ``.obs`` (non-numeric metadata) with
    ``X`` converted to a DataFrame using the variable names.

    Args:
        path:
            Str path to the AnnData object. For zarr, this is a directory-like
            store; for h5ad, a file path.
        read_from:
            Input format. Either ``"h5ad"`` or ``"zarr"``.
        read_one:
            Whether to read just one column.

    Returns:
        A pandas DataFrame with ``.obs`` columns followed by the numeric
        columns from ``X`` (``adata.to_df()``), indexed from 0..n-1.
    """

    if read_from == "h5ad":
        adata = ad.read_h5ad(path)
    elif read_from == "zarr":
        adata = ad.read_zarr(path)
    else:
        raise ValueError('read_from must be "h5ad" or "zarr".')

    if read_one:
        return adata.to_df()["col_2"]

    return adata.obs.join(adata.to_df(), how="left").reset_index(drop=True)


# + papermill={"duration": 0.242991, "end_time": "2025-09-03T21:55:13.758506", "exception": false, "start_time": "2025-09-03T21:55:13.515515", "status": "completed"}
# avoid a "cold start" for tested packages by using them before benchmarks
df = pd.DataFrame(np.random.rand(2, 2), columns=[f"col_{num}" for num in range(0, 2)])
# export and read using various methods
df.to_csv(path_or_buf=csv_name, compression="gzip")
pd.read_csv(filepath_or_buffer=csv_name, compression="gzip")
df.to_sql(name=sqlite_tbl_name, con=f"sqlite:///{sqlite_name}")
pd.read_sql(sql=f"SELECT * FROM {sqlite_tbl_name}", con=f"sqlite:///{sqlite_name}")
df.to_parquet(path=parquet_gzip_name, compression="gzip")
pd.read_parquet(path=parquet_gzip_name)

# + papermill={"duration": 0.013525, "end_time": "2025-09-03T21:55:13.778550", "exception": false, "start_time": "2025-09-03T21:55:13.765025", "status": "completed"}
# remove any existing prior work
for filename in [csv_name, parquet_gzip_name, sqlite_name]:
    pathlib.Path(filename).unlink(missing_ok=True)

# + papermill={"duration": 126.463193, "end_time": "2025-09-03T21:57:20.248198", "exception": false, "start_time": "2025-09-03T21:55:13.785005", "status": "completed"}
# starting rowcount and col count
nrows = 320
ncols = 124

# result list for storing data
results = []

# loop for iterating over increasingly large dataframes
# and gathering data about operations on them
for _ in range(1, 6):
    # increase the size of the dataframe
    nrows *= 2
    ncols *= 2

    # form a dataframe using randomized data
    df = pd.DataFrame(
        np.random.rand(nrows, ncols), columns=[f"col_{num}" for num in range(0, ncols)]
    )

    # add some string data
    alphabet = np.array(list(string.ascii_lowercase + string.digits))
    df = df.assign(
        **{
            f"str_{i+1}": [
                "".join(np.random.default_rng(10).choice(alphabet, 10))
                for _ in range(len(df))
            ]
            for i in range(10)
        }
    )

    print(df.shape)

    # run multiple times for error and average
    for _ in range(1, 5):
        # remove any existing files in preparation for next steps
        remove_files()
        # append data to the result list
        results.append(
            {
                # general information about the dataframe
                "dataframe_shape (rows, cols)": str(df.shape),
                # information about CSV (uncompressed)
                "csv_write_time (secs)": timer(
                    df.to_csv, path_or_buf=csv_name, compression="gzip"
                ),
                "csv_size (bytes)": os.stat(csv_name).st_size,
                "csv_read_time_all (secs)": timer(
                    pd.read_csv, filepath_or_buffer=csv_name, compression="gzip"
                ),
                "csv_read_time_one (secs)": timer(
                    pd.read_csv,
                    filepath_or_buffer=csv_name,
                    compression="gzip",
                    usecols=["col_2"],
                ),
                # information about SQLite
                "sqlite_write_time (secs)": (
                    timer(
                        df.to_sql,
                        name=sqlite_tbl_name,
                        con=f"sqlite:///{sqlite_name}",
                    )
                    if ncols < 2000
                    else None
                ),
                "sqlite_size (bytes)": (
                    os.stat(sqlite_name).st_size if ncols < 2000 else None
                ),
                "sqlite_read_time_all (secs)": (
                    timer(
                        pd.read_sql,
                        sql=f"SELECT * FROM {sqlite_tbl_name}",
                        con=f"sqlite:///{sqlite_name}",
                    )
                    if ncols < 2000
                    else None
                ),
                "sqlite_read_time_one (secs)": (
                    timer(
                        pd.read_sql,
                        sql=f"SELECT col_2 FROM {sqlite_tbl_name}",
                        con=f"sqlite:///{sqlite_name}",
                    )
                    if ncols < 2000
                    else None
                ),
                # information about anndata h5ad (no compression)
                "anndata_h5ad_noc_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="h5ad",
                    compression="none",
                    dest_path=anndata_h5_noc_name,
                ),
                "anndata_h5ad_noc_size (bytes)": os.stat(anndata_h5_noc_name).st_size,
                "anndata_h5ad_noc_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_h5_noc_name,
                    read_from="h5ad",
                    read_one=False,
                ),
                "anndata_h5ad_noc_read_time_one (secs)": timer(
                    read_anndata,
                    path=anndata_h5_noc_name,
                    read_from="h5ad",
                    read_one=True,
                ),
                # information about anndata h5ad (gzip)
                "anndata_h5ad_gzip_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="h5ad",
                    compression="gzip",
                    dest_path=anndata_h5_gzip_name,
                ),
                "anndata_h5ad_gzip_size (bytes)": os.stat(anndata_h5_gzip_name).st_size,
                "anndata_h5ad_gzip_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_h5_gzip_name,
                    read_from="h5ad",
                    read_one=False,
                ),
                "anndata_h5ad_gzip_read_time_one (secs)": timer(
                    read_anndata,
                    path=anndata_h5_gzip_name,
                    read_from="h5ad",
                    read_one=True,
                ),
                # information about anndata h5ad (lz4)
                "anndata_h5ad_lz4_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="h5ad",
                    compression="lz4",
                    dest_path=anndata_h5_lz4_name,
                ),
                "anndata_h5ad_lz4_size (bytes)": os.stat(anndata_h5_lz4_name).st_size,
                "anndata_h5ad_lz4_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_h5_lz4_name,
                    read_from="h5ad",
                    read_one=False,
                ),
                "anndata_h5ad_lz4_read_time_one (secs)": timer(
                    read_anndata,
                    path=anndata_h5_lz4_name,
                    read_from="h5ad",
                    read_one=True,
                ),
                # information about anndata h5ad (zstd)
                "anndata_h5ad_zstd_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="h5ad",
                    compression="zstd",
                    dest_path=anndata_h5_zstd_name,
                ),
                "anndata_h5ad_zstd_size (bytes)": os.stat(anndata_h5_zstd_name).st_size,
                "anndata_h5ad_zstd_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_h5_zstd_name,
                    read_from="h5ad",
                    read_one=False,
                ),
                "anndata_h5ad_zstd_read_time_one (secs)": timer(
                    read_anndata,
                    path=anndata_h5_zstd_name,
                    read_from="h5ad",
                    read_one=True,
                ),
                # information about anndata zarr
                "anndata_zarr_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="zarr",
                    compression="none",
                    dest_path=anndata_zarr_name,
                ),
                # note: we use a comprehension below to recurse through
                # the zarr directory for a true estimate of size.
                "anndata_zarr_size (bytes)": sum(
                    f.stat().st_size
                    for f in pathlib.Path(anndata_zarr_name).rglob("**/*")
                    if f.is_file()
                ),
                "anndata_zarr_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_zarr_name,
                    read_from="zarr",
                    read_one=False,
                ),
                "anndata_zarr_read_time_one (secs)": timer(
                    read_anndata,
                    path=anndata_zarr_name,
                    read_from="zarr",
                    read_one=True,
                ),
                # information about Parquet with no compression
                "parquet_noc_write_time (secs)": timer(
                    df.to_parquet, path=parquet_noc_name, compression=None
                ),
                "parquet_noc_size (bytes)": os.stat(parquet_noc_name).st_size,
                "parquet_noc_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_noc_name
                ),
                "parquet_noc_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_noc_name, columns=["col_2"]
                ),
                # information about Parquet with snappy compression
                "parquet_snappy_write_time (secs)": timer(
                    df.to_parquet, path=parquet_snappy_name, compression="snappy"
                ),
                "parquet_snappy_size (bytes)": os.stat(parquet_snappy_name).st_size,
                "parquet_snappy_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_snappy_name
                ),
                "parquet_snappy_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_snappy_name, columns=["col_2"]
                ),
                # information about Parquet with gzip compression
                "parquet_gzip_write_time (secs)": timer(
                    df.to_parquet, path=parquet_gzip_name, compression="gzip"
                ),
                "parquet_gzip_size (bytes)": os.stat(parquet_gzip_name).st_size,
                "parquet_gzip_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_gzip_name
                ),
                "parquet_gzip_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_gzip_name, columns=["col_2"]
                ),
                # information about Parquet with zstd compression
                "parquet_zstd_write_time (secs)": timer(
                    df.to_parquet, path=parquet_zstd_name, compression="zstd"
                ),
                "parquet_zstd_size (bytes)": os.stat(parquet_zstd_name).st_size,
                "parquet_zstd_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_zstd_name
                ),
                "parquet_zstd_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_zstd_name, columns=["col_2"]
                ),
                # information about Parquet with lz4 compression
                "parquet_lz4_write_time (secs)": timer(
                    df.to_parquet, path=parquet_lz4_name, compression="lz4"
                ),
                "parquet_lz4_size (bytes)": os.stat(parquet_lz4_name).st_size,
                "parquet_lz4_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_lz4_name
                ),
                "parquet_lz4_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_lz4_name, columns=["col_2"]
                ),
            }
        )


df_results = pd.DataFrame(results)
df_results

# +
# calculate full write + read time for each format
df_results["csv_write_and_read_time (secs)"] = (
    df_results["csv_write_time (secs)"] + df_results["csv_read_time_all (secs)"]
)
df_results["sqlite_write_and_read_time (secs)"] = (
    df_results["sqlite_write_time (secs)"] + df_results["sqlite_read_time_all (secs)"]
)

df_results["anndata_h5ad_noc_write_and_read_time (secs)"] = (
    df_results["anndata_h5ad_noc_write_time (secs)"]
    + df_results["anndata_h5ad_noc_read_time_all (secs)"]
)
df_results["anndata_h5ad_gzip_write_and_read_time (secs)"] = (
    df_results["anndata_h5ad_gzip_write_time (secs)"]
    + df_results["anndata_h5ad_gzip_read_time_all (secs)"]
)
df_results["anndata_h5ad_lz4_write_and_read_time (secs)"] = (
    df_results["anndata_h5ad_lz4_write_time (secs)"]
    + df_results["anndata_h5ad_lz4_read_time_all (secs)"]
)

df_results["anndata_h5ad_zstd_write_and_read_time (secs)"] = (
    df_results["anndata_h5ad_zstd_write_time (secs)"]
    + df_results["anndata_h5ad_zstd_read_time_all (secs)"]
)
df_results["anndata_zarr_write_and_read_time (secs)"] = (
    df_results["anndata_zarr_write_time (secs)"]
    + df_results["anndata_zarr_read_time_all (secs)"]
)

df_results["parquet_noc_write_and_read_time (secs)"] = (
    df_results["parquet_noc_write_time (secs)"]
    + df_results["parquet_noc_read_time_all (secs)"]
)
df_results["parquet_snappy_write_and_read_time (secs)"] = (
    df_results["parquet_snappy_write_time (secs)"]
    + df_results["parquet_snappy_read_time_all (secs)"]
)
df_results["parquet_gzip_write_and_read_time (secs)"] = (
    df_results["parquet_gzip_write_time (secs)"]
    + df_results["parquet_gzip_read_time_all (secs)"]
)
df_results["parquet_zstd_write_and_read_time (secs)"] = (
    df_results["parquet_zstd_write_time (secs)"]
    + df_results["parquet_zstd_read_time_all (secs)"]
)
df_results["parquet_lz4_write_and_read_time (secs)"] = (
    df_results["parquet_lz4_write_time (secs)"]
    + df_results["parquet_lz4_read_time_all (secs)"]
)

# + papermill={"duration": 0.048407, "end_time": "2025-09-03T21:57:20.305775", "exception": false, "start_time": "2025-09-03T21:57:20.257368", "status": "completed"}
# gather average, min, and max for error bar implementation
average = (
    df_results.groupby("dataframe_shape (rows, cols)")
    .mean()
    .reset_index()
    .sort_values(by="csv_size (bytes)")
)
minimums = (
    df_results.groupby("dataframe_shape (rows, cols)")
    .min()
    .reset_index()
    .sort_values(by="csv_size (bytes)")
)
maximums = (
    df_results.groupby("dataframe_shape (rows, cols)")
    .max()
    .reset_index()
    .sort_values(by="csv_size (bytes)")
)

key = "dataframe_shape (rows, cols)"

result = (
    average.set_index(key)
    .add_suffix(" mean")
    .join(minimums.set_index(key).add_suffix(" min"))
    .join(maximums.set_index(key).add_suffix(" max"))
    .reset_index()
)

result

# + papermill={"duration": 1.365292, "end_time": "2025-09-03T21:57:21.680374", "exception": false, "start_time": "2025-09-03T21:57:20.315082", "status": "completed"}
# write time plot

key = "dataframe_shape (rows, cols)"

cols = {
    "CSV (GZIP)": (
        "csv_write_time (secs) mean",
        "csv_write_time (secs) min",
        "csv_write_time (secs) max",
    ),
    "SQLite": (
        "sqlite_write_time (secs) mean",
        "sqlite_write_time (secs) min",
        "sqlite_write_time (secs) max",
    ),
    "AnnData (H5AD - uncompressed)": (
        "anndata_h5ad_noc_write_time (secs) mean",
        "anndata_h5ad_noc_write_time (secs) min",
        "anndata_h5ad_noc_write_time (secs) max",
    ),
    "AnnData (H5AD - GZIP)": (
        "anndata_h5ad_gzip_write_time (secs) mean",
        "anndata_h5ad_gzip_write_time (secs) min",
        "anndata_h5ad_gzip_write_time (secs) max",
    ),
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_write_time (secs) mean",
        "anndata_h5ad_zstd_write_time (secs) min",
        "anndata_h5ad_zstd_write_time (secs) max",
    ),
    "AnnData (H5AD - LZ4) (": (
        "anndata_h5ad_lz4_write_time (secs) mean",
        "anndata_h5ad_lz4_write_time (secs) min",
        "anndata_h5ad_lz4_write_time (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_write_time (secs) mean",
        "anndata_zarr_write_time (secs) min",
        "anndata_zarr_write_time (secs) max",
    ),
    "Parquet (uncompressed)": (
        "parquet_noc_write_time (secs) mean",
        "parquet_noc_write_time (secs) min",
        "parquet_noc_write_time (secs) max",
    ),
    "Parquet (Snappy)": (
        "parquet_snappy_write_time (secs) mean",
        "parquet_snappy_write_time (secs) min",
        "parquet_snappy_write_time (secs) max",
    ),
    "Parquet (GZIP)": (
        "parquet_gzip_write_time (secs) mean",
        "parquet_gzip_write_time (secs) min",
        "parquet_gzip_write_time (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_write_time (secs) mean",
        "parquet_zstd_write_time (secs) min",
        "parquet_zstd_write_time (secs) max",
    ),
    "Parquet (LZ4)": (
        "parquet_lz4_write_time (secs) mean",
        "parquet_lz4_write_time (secs) min",
        "parquet_lz4_write_time (secs) max",
    ),
}

parts = []
for fmt, (mcol, mincol, maxcol) in cols.items():
    tmp = result[[key, mcol, mincol, maxcol]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mcol: "mean", mincol: "min", maxcol: "max"}, inplace=True)
    tmp["err_plus"] = tmp["max"] - tmp["mean"]
    tmp["err_minus"] = tmp["mean"] - tmp["min"]
    parts.append(tmp[[key, "format", "mean", "err_plus", "err_minus"]])


stats = pd.concat(parts, ignore_index=True)

x_order = result[key].tolist()  # not reversed; use iloc[::-1] to reverse
pos = {k: i for i, k in enumerate(x_order)}  # category → position index

# 2) give each row its x position and sort per-trace
stats = stats.assign(xpos=stats[key].map(pos)).sort_values(["format", "xpos"])

fig = px.line(
    stats,  # already trace-sorted by xpos
    x=key,
    y="mean",
    color="format",
    error_y="err_plus",
    error_y_minus="err_minus",
    markers=True,
    category_orders={key: x_order},  # sets axis order & legend hover categories
    labels={key: "Data Shape", "mean": "Seconds (log)"},
    width=1300,
    log_y=True,
    title="File format write time duration (seconds)",
)
fig.update_traces(mode="lines+markers")
fig.update_traces(marker_color=None, line_color=None).update_layout(
    colorway=px.colors.qualitative.Dark24
)
fig.update_layout(legend_title_text="Format")


pio.write_image(fig, file_write_time_image)
Image(url=file_write_time_image)

# + papermill={"duration": 0.313924, "end_time": "2025-09-03T21:57:22.005171", "exception": false, "start_time": "2025-09-03T21:57:21.691247", "status": "completed"}
# file size plot

key = "dataframe_shape (rows, cols)"

size_cols = {
    "csv_size (bytes)": "CSV (GZIP)",
    "sqlite_size (bytes)": "SQLite",
    "anndata_h5ad_noc_size (bytes)": "AnnData (H5AD - uncompressed)",
    "anndata_h5ad_gzip_size (bytes)": "AnnData (H5AD - GZIP)",
    "anndata_h5ad_lz4_size (bytes)": "AnnData (H5AD - LZ4)",
    "anndata_h5ad_zstd_size (bytes)": "AnnData (H5AD - ZSTD)",
    "anndata_zarr_size (bytes)": "AnnData (Zarr)",
    "parquet_noc_size (bytes)": "Parquet (uncompressed)",
    "parquet_snappy_size (bytes)": "Parquet (Snappy)",
    "parquet_gzip_size (bytes)": "Parquet (GZIP)",
    "parquet_zstd_size (bytes)": "Parquet (ZSTD)",
    "parquet_lz4_size (bytes)": "Parquet (LZ4)",
}

# Long-form + average across repeats
long = df_results.melt(
    id_vars=[key],
    value_vars=list(size_cols.keys()),
    var_name="col",
    value_name="bytes",
).dropna(subset=["bytes"])
long["format"] = long["col"].map(size_cols)

stats = long.groupby([key, "format"], as_index=False)["bytes"].mean()

# Choose x-axis category order (keep your current result order, reversed here).
x_order = result[key].iloc[::-1].tolist()

# Ensure each trace's points follow that order (pre-sort rows)
pos = {cat: i for i, cat in enumerate(x_order)}
stats_sorted = stats.assign(xpos=stats[key].map(pos)).sort_values(["format", "xpos"])

fig = px.line(
    stats_sorted,
    x=key,
    y="bytes",
    color="format",
    markers=True,
    category_orders={key: x_order},
    labels={key: "Data Shape", "bytes": "Bytes"},
    width=1300,
    title="File format size (bytes)",
)

fig.update_traces(mode="lines+markers")
fig.update_xaxes(autorange="reversed")
fig.update_traces(marker_color=None, line_color=None).update_layout(
    colorway=px.colors.qualitative.Dark24
)
fig.update_layout(
    legend=dict(
        x=1.02,
        y=1,  # just outside the plotting area
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    ),
    margin=dict(r=220),  # add right margin so legend fits
    font=dict(size=18),
)

pio.write_image(fig, file_storage_size_image)
Image(url=file_storage_size_image)

# + papermill={"duration": 0.250629, "end_time": "2025-09-03T21:57:22.266295", "exception": false, "start_time": "2025-09-03T21:57:22.015666", "status": "completed"}
# read time plot (all columns)
key = "dataframe_shape (rows, cols)"

cols = {
    "CSV (GZIP)": (
        "csv_read_time_all (secs) mean",
        "csv_read_time_all (secs) min",
        "csv_read_time_all (secs) max",
    ),
    "SQLite": (
        "sqlite_read_time_all (secs) mean",
        "sqlite_read_time_all (secs) min",
        "sqlite_read_time_all (secs) max",
    ),
    "AnnData (H5AD - uncompressed)": (
        "anndata_h5ad_noc_read_time_all (secs) mean",
        "anndata_h5ad_noc_read_time_all (secs) min",
        "anndata_h5ad_noc_read_time_all (secs) max",
    ),
    "AnnData (H5AD - GZIP)": (
        "anndata_h5ad_gzip_read_time_all (secs) mean",
        "anndata_h5ad_gzip_read_time_all (secs) min",
        "anndata_h5ad_gzip_read_time_all (secs) max",
    ),
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_read_time_all (secs) mean",
        "anndata_h5ad_zstd_read_time_all (secs) min",
        "anndata_h5ad_zstd_read_time_all (secs) max",
    ),
    "AnnData (H5AD - LZ4) (": (
        "anndata_h5ad_lz4_read_time_all (secs) mean",
        "anndata_h5ad_lz4_read_time_all (secs) min",
        "anndata_h5ad_lz4_read_time_all (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_read_time_all (secs) mean",
        "anndata_zarr_read_time_all (secs) min",
        "anndata_zarr_read_time_all (secs) max",
    ),
    "Parquet (uncompressed)": (
        "parquet_noc_read_time_all (secs) mean",
        "parquet_noc_read_time_all (secs) min",
        "parquet_noc_read_time_all (secs) max",
    ),
    "Parquet (Snappy)": (
        "parquet_snappy_read_time_all (secs) mean",
        "parquet_snappy_read_time_all (secs) min",
        "parquet_snappy_read_time_all (secs) max",
    ),
    "Parquet (GZIP)": (
        "parquet_gzip_read_time_all (secs) mean",
        "parquet_gzip_read_time_all (secs) min",
        "parquet_gzip_read_time_all (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_read_time_all (secs) mean",
        "parquet_zstd_read_time_all (secs) min",
        "parquet_zstd_read_time_all (secs) max",
    ),
    "Parquet (LZ4)": (
        "parquet_lz4_read_time_all (secs) mean",
        "parquet_lz4_read_time_all (secs) min",
        "parquet_lz4_read_time_all (secs) max",
    ),
}

parts = []
for fmt, (mcol, mincol, maxcol) in cols.items():
    tmp = result[[key, mcol, mincol, maxcol]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mcol: "mean", mincol: "min", maxcol: "max"}, inplace=True)
    tmp["err_plus"] = tmp["max"] - tmp["mean"]
    tmp["err_minus"] = tmp["mean"] - tmp["min"]
    parts.append(tmp[[key, "format", "mean", "err_plus", "err_minus"]])


stats = pd.concat(parts, ignore_index=True)

x_order = result[key].tolist()  # not reversed; use iloc[::-1] to reverse
pos = {k: i for i, k in enumerate(x_order)}  # category → position index

# 2) give each row its x position and sort per-trace
stats = stats.assign(xpos=stats[key].map(pos)).sort_values(["format", "xpos"])

fig = px.line(
    stats,  # already trace-sorted by xpos
    x=key,
    y="mean",
    color="format",
    error_y="err_plus",
    error_y_minus="err_minus",
    markers=True,
    category_orders={key: x_order},  # sets axis order & legend hover categories
    labels={key: "Data Shape", "mean": "Seconds"},
    width=1300,
    log_y=True,
    title="File format read time duration (full dataset) (seconds)",
)
fig.update_traces(mode="lines+markers")
fig.update_traces(marker_color=None, line_color=None).update_layout(
    colorway=px.colors.qualitative.Dark24
)
fig.update_layout(legend_title_text="Format")


pio.write_image(fig, file_read_time_all_image)
Image(url=file_read_time_all_image)

# + papermill={"duration": 0.260285, "end_time": "2025-09-03T21:57:22.537386", "exception": false, "start_time": "2025-09-03T21:57:22.277101", "status": "completed"}
# read time plot (one column)

key = "dataframe_shape (rows, cols)"

cols = {
    "CSV (GZIP)": (
        "csv_read_time_one (secs) mean",
        "csv_read_time_one (secs) min",
        "csv_read_time_one (secs) max",
    ),
    "SQLite": (
        "sqlite_read_time_one (secs) mean",
        "sqlite_read_time_one (secs) min",
        "sqlite_read_time_one (secs) max",
    ),
    "AnnData (H5AD - uncompressed)": (
        "anndata_h5ad_noc_read_time_one (secs) mean",
        "anndata_h5ad_noc_read_time_one (secs) min",
        "anndata_h5ad_noc_read_time_one (secs) max",
    ),
    "AnnData (H5AD - GZIP)": (
        "anndata_h5ad_gzip_read_time_one (secs) mean",
        "anndata_h5ad_gzip_read_time_one (secs) min",
        "anndata_h5ad_gzip_read_time_one (secs) max",
    ),
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_read_time_one (secs) mean",
        "anndata_h5ad_zstd_read_time_one (secs) min",
        "anndata_h5ad_zstd_read_time_one (secs) max",
    ),
    "AnnData (H5AD - LZ4) (": (
        "anndata_h5ad_lz4_read_time_one (secs) mean",
        "anndata_h5ad_lz4_read_time_one (secs) min",
        "anndata_h5ad_lz4_read_time_one (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_read_time_one (secs) mean",
        "anndata_zarr_read_time_one (secs) min",
        "anndata_zarr_read_time_one (secs) max",
    ),
    "Parquet (uncompressed)": (
        "parquet_noc_read_time_one (secs) mean",
        "parquet_noc_read_time_one (secs) min",
        "parquet_noc_read_time_one (secs) max",
    ),
    "Parquet (Snappy)": (
        "parquet_snappy_read_time_one (secs) mean",
        "parquet_snappy_read_time_one (secs) min",
        "parquet_snappy_read_time_one (secs) max",
    ),
    "Parquet (GZIP)": (
        "parquet_gzip_read_time_one (secs) mean",
        "parquet_gzip_read_time_one (secs) min",
        "parquet_gzip_read_time_one (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_read_time_one (secs) mean",
        "parquet_zstd_read_time_one (secs) min",
        "parquet_zstd_read_time_one (secs) max",
    ),
    "Parquet (LZ4)": (
        "parquet_lz4_read_time_one (secs) mean",
        "parquet_lz4_read_time_one (secs) min",
        "parquet_lz4_read_time_one (secs) max",
    ),
}


parts = []
for fmt, (mcol, mincol, maxcol) in cols.items():
    tmp = result[[key, mcol, mincol, maxcol]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mcol: "mean", mincol: "min", maxcol: "max"}, inplace=True)
    tmp["err_plus"] = tmp["max"] - tmp["mean"]
    tmp["err_minus"] = tmp["mean"] - tmp["min"]
    parts.append(tmp[[key, "format", "mean", "err_plus", "err_minus"]])


stats = pd.concat(parts, ignore_index=True)

x_order = result[key].tolist()  # not reversed; use iloc[::-1] to reverse
pos = {k: i for i, k in enumerate(x_order)}  # category → position index

# 2) give each row its x position and sort per-trace
stats = stats.assign(xpos=stats[key].map(pos)).sort_values(["format", "xpos"])

fig = px.line(
    stats,  # already trace-sorted by xpos
    x=key,
    y="mean",
    color="format",
    error_y="err_plus",
    error_y_minus="err_minus",
    markers=True,
    category_orders={key: x_order},  # sets axis order & legend hover categories
    labels={key: "Data Shape", "mean": "Seconds (log)"},
    width=1300,
    log_y=True,
    title="File format read time duration (one column) (seconds)",
)
fig.update_traces(mode="lines+markers")
fig.update_traces(marker_color=None, line_color=None).update_layout(
    colorway=px.colors.qualitative.Dark24
)
fig.update_layout(legend_title_text="Format")


pio.write_image(fig, file_read_time_one_image)
Image(url=file_read_time_one_image)

# +
# write and read time plot (combined times from write and read all)

key = "dataframe_shape (rows, cols)"

cols = {
    "CSV (GZIP)": (
        "csv_write_and_read_time (secs) mean",
        "csv_write_and_read_time (secs) min",
        "csv_write_and_read_time (secs) max",
    ),
    "SQLite": (
        "sqlite_write_and_read_time (secs) mean",
        "sqlite_write_and_read_time (secs) min",
        "sqlite_write_and_read_time (secs) max",
    ),
    "AnnData (H5AD - uncompressed)": (
        "anndata_h5ad_noc_write_and_read_time (secs) mean",
        "anndata_h5ad_noc_write_and_read_time (secs) min",
        "anndata_h5ad_noc_write_and_read_time (secs) max",
    ),
    "AnnData (H5AD - GZIP)": (
        "anndata_h5ad_gzip_write_and_read_time (secs) mean",
        "anndata_h5ad_gzip_write_and_read_time (secs) min",
        "anndata_h5ad_gzip_write_and_read_time (secs) max",
    ),
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_write_and_read_time (secs) mean",
        "anndata_h5ad_zstd_write_and_read_time (secs) min",
        "anndata_h5ad_zstd_write_and_read_time (secs) max",
    ),
    "AnnData (H5AD - LZ4) (": (
        "anndata_h5ad_lz4_write_and_read_time (secs) mean",
        "anndata_h5ad_lz4_write_and_read_time (secs) min",
        "anndata_h5ad_lz4_write_and_read_time (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_write_and_read_time (secs) mean",
        "anndata_zarr_write_and_read_time (secs) min",
        "anndata_zarr_write_and_read_time (secs) max",
    ),
    "Parquet (uncompressed)": (
        "parquet_noc_write_and_read_time (secs) mean",
        "parquet_noc_write_and_read_time (secs) min",
        "parquet_noc_write_and_read_time (secs) max",
    ),
    "Parquet (Snappy)": (
        "parquet_snappy_write_and_read_time (secs) mean",
        "parquet_snappy_write_and_read_time (secs) min",
        "parquet_snappy_write_and_read_time (secs) max",
    ),
    "Parquet (GZIP)": (
        "parquet_gzip_write_and_read_time (secs) mean",
        "parquet_gzip_write_and_read_time (secs) min",
        "parquet_gzip_write_and_read_time (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_write_and_read_time (secs) mean",
        "parquet_zstd_write_and_read_time (secs) min",
        "parquet_zstd_write_and_read_time (secs) max",
    ),
    "Parquet (LZ4)": (
        "parquet_lz4_write_and_read_time (secs) mean",
        "parquet_lz4_write_and_read_time (secs) min",
        "parquet_lz4_write_and_read_time (secs) max",
    ),
}


parts = []
for fmt, (mcol, mincol, maxcol) in cols.items():
    tmp = result[[key, mcol, mincol, maxcol]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mcol: "mean", mincol: "min", maxcol: "max"}, inplace=True)
    tmp["err_plus"] = tmp["max"] - tmp["mean"]
    tmp["err_minus"] = tmp["mean"] - tmp["min"]
    parts.append(tmp[[key, "format", "mean", "err_plus", "err_minus"]])


stats = pd.concat(parts, ignore_index=True)

x_order = result[key].tolist()  # not reversed; use iloc[::-1] to reverse
pos = {k: i for i, k in enumerate(x_order)}  # category → position index

# 2) give each row its x position and sort per-trace
stats = stats.assign(xpos=stats[key].map(pos)).sort_values(["format", "xpos"])

fig = px.line(
    stats,  # already trace-sorted by xpos
    x=key,
    y="mean",
    color="format",
    error_y="err_plus",
    error_y_minus="err_minus",
    markers=True,
    category_orders={key: x_order},  # sets axis order & legend hover categories
    labels={key: "Data Shape", "mean": "Seconds (log)"},
    width=1300,
    log_y=True,
    title="File format write and read time (full dataset) (seconds)",
)
fig.update_traces(mode="lines+markers")
fig.update_traces(marker_color=None, line_color=None).update_layout(
    colorway=px.colors.qualitative.Dark24
)
fig.update_layout(legend_title_text="Format")


pio.write_image(fig, file_read_time_write_and_read_time_image)
Image(url=file_read_time_write_and_read_time_image)
