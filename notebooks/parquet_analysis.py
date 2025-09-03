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
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IPython.display import Image
from utilities import get_system_info, timer

# + papermill={"duration": 0.015148, "end_time": "2025-09-03T21:55:13.429168", "exception": false, "start_time": "2025-09-03T21:55:13.414020", "status": "completed"}
# show the system information
_ = get_system_info(show_output=True)

# + papermill={"duration": 0.049587, "end_time": "2025-09-03T21:55:13.484421", "exception": false, "start_time": "2025-09-03T21:55:13.434834", "status": "completed"}
# target file or table names
image_dir = "images"
csv_name = "example.csv.gz"
parquet_name = "example.parquet"
sqlite_name = "example.sqlite"
sqlite_tbl_name = "tbl_example"
anndata_h5_name = "adata.h5ad"
anndata_zarr_name = "adata.zarr"
file_write_time_image = f"{image_dir}/parquet-comparisons-file-write-time.png"
file_storage_size_image = f"{image_dir}/parquet-comparisons-file-storage-size.png"
file_read_time_all_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-all-columns.png"
)
file_read_time_one_image = (
    f"{image_dir}/parquet-comparisons-file-read-time-one-column.png"
)

pathlib.Path(csv_name).unlink(missing_ok=True)
pathlib.Path(parquet_name).unlink(missing_ok=True)
pathlib.Path(sqlite_name).unlink(missing_ok=True)
pathlib.Path(anndata_h5_name).unlink(missing_ok=True)
if pathlib.Path(anndata_zarr_name).is_dir():
    shutil.rmtree(anndata_zarr_name)


# + papermill={"duration": 0.017819, "end_time": "2025-09-03T21:55:13.509355", "exception": false, "start_time": "2025-09-03T21:55:13.491536", "status": "completed"}
def write_anndata(
    df: pd.DataFrame,
    write_to: Literal["h5ad", "zarr"],
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

    adata = ad.AnnData(X=numeric.to_numpy())
    adata.obs_names = df.index.astype(str)
    adata.var_names = numeric.columns.astype(str)
    # Align non-numeric obs metadata to the same index
    adata.obs = non_numeric.copy()

    if write_to == "h5ad":
        adata.write_h5ad(str(dest))
    elif write_to == "zarr":
        # For zarr, the destination is a directory-like store
        adata.write_zarr(str(dest))
    else:
        raise ValueError('write_to must be "h5ad" or "zarr".')

    return str(dest)


def read_anndata(
    path: str,
    read_from: Literal["h5ad", "zarr"],
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

    return adata.obs.join(adata.to_df(), how="left").reset_index(drop=True)


# + papermill={"duration": 0.242991, "end_time": "2025-09-03T21:55:13.758506", "exception": false, "start_time": "2025-09-03T21:55:13.515515", "status": "completed"}
# avoid a "cold start" for tested packages by using them before benchmarks
df = pd.DataFrame(np.random.rand(2, 2), columns=[f"col_{num}" for num in range(0, 2)])
# export and read using various methods
df.to_csv(path_or_buf=csv_name, compression="gzip")
pd.read_csv(filepath_or_buffer=csv_name, compression="gzip")
df.to_sql(name=sqlite_tbl_name, con=f"sqlite:///{sqlite_name}")
pd.read_sql(sql=f"SELECT * FROM {sqlite_tbl_name}", con=f"sqlite:///{sqlite_name}")
df.to_parquet(path=parquet_name, compression="gzip")
pd.read_parquet(path=parquet_name)

# + papermill={"duration": 0.013525, "end_time": "2025-09-03T21:55:13.778550", "exception": false, "start_time": "2025-09-03T21:55:13.765025", "status": "completed"}
# remove any existing prior work
for filename in [csv_name, parquet_name, sqlite_name]:
    pathlib.Path(filename).unlink(missing_ok=True)

# + papermill={"duration": 126.463193, "end_time": "2025-09-03T21:57:20.248198", "exception": false, "start_time": "2025-09-03T21:55:13.785005", "status": "completed"}
# starting rowcount and col count
nrows = 320
ncols = 160

# result list for storing data
results = []

# loop for iterating over increasingly large dataframes
# and gathering data about operations on them
for _ in range(1, 4):
    # increase the size of the dataframe
    nrows *= 2
    ncols *= 2

    # form a dataframe using randomized data
    df = pd.DataFrame(
        np.random.rand(nrows, ncols), columns=[f"col_{num}" for num in range(0, ncols)]
    )

    # run multiple times for error and average
    for _ in range(1, 5):
        # remove any existing files in preparation for next steps
        for filename in [
            csv_name,
            parquet_name,
            sqlite_name,
            anndata_h5_name,
            anndata_zarr_name,
        ]:
            if pathlib.Path(filename).is_dir():
                shutil.rmtree(anndata_zarr_name)
            else:
                pathlib.Path(filename).unlink(missing_ok=True)
        # append data to the result list
        results.append(
            {
                # general information about the dataframe
                "dataframe_shape (rows, cols)": str(df.shape),
                # information about CSV
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
                "sqlite_write_time (secs)": timer(
                    df.to_sql,
                    name=sqlite_tbl_name,
                    con=f"sqlite:///{sqlite_name}",
                ),
                "sqlite_size (bytes)": os.stat(sqlite_name).st_size,
                "sqlite_read_time_all (secs)": timer(
                    pd.read_sql,
                    sql=f"SELECT * FROM {sqlite_tbl_name}",
                    con=f"sqlite:///{sqlite_name}",
                ),
                "sqlite_read_time_one (secs)": timer(
                    pd.read_sql,
                    sql=f"SELECT col_2 FROM {sqlite_tbl_name}",
                    con=f"sqlite:///{sqlite_name}",
                ),
                # information about anndata h5ad
                "anndata_h5ad_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="h5ad",
                    dest_path=anndata_h5_name,
                ),
                "anndata_h5ad_size (bytes)": os.stat(anndata_h5_name).st_size,
                "anndata_h5ad_read_time_all (secs)": timer(
                    read_anndata,
                    path=anndata_h5_name,
                    read_from="h5ad",
                ),
                # information about anndata zarr
                "anndata_zarr_write_time (secs)": timer(
                    write_anndata,
                    df=df,
                    write_to="zarr",
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
                ),
                # information about Parquet
                "parquet_write_time (secs)": timer(
                    df.to_parquet, path=parquet_name, compression="gzip"
                ),
                "parquet_size (bytes)": os.stat(parquet_name).st_size,
                "parquet_read_time_all (secs)": timer(
                    pd.read_parquet, path=parquet_name
                ),
                "parquet_read_time_one (secs)": timer(
                    pd.read_parquet, path=parquet_name, columns=["col_2"]
                ),
            }
        )


df_results = pd.DataFrame(results)
df_results

# + papermill={"duration": 0.048407, "end_time": "2025-09-03T21:57:20.305775", "exception": false, "start_time": "2025-09-03T21:57:20.257368", "status": "completed"}
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
key = "dataframe_shape (rows, cols)"

cols = {
    "CSV": (
        "csv_write_time (secs) mean",
        "csv_write_time (secs) min",
        "csv_write_time (secs) max",
    ),
    "SQLite": (
        "sqlite_write_time (secs) mean",
        "sqlite_write_time (secs) min",
        "sqlite_write_time (secs) max",
    ),
    "AnnData (H5AD)": (
        "anndata_h5ad_write_time (secs) mean",
        "anndata_h5ad_write_time (secs) min",
        "anndata_h5ad_write_time (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_write_time (secs) mean",
        "anndata_zarr_write_time (secs) min",
        "anndata_zarr_write_time (secs) max",
    ),
    "Parquet": (
        "parquet_write_time (secs) mean",
        "parquet_write_time (secs) min",
        "parquet_write_time (secs) max",
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

y_order = result[key].iloc[::-1].tolist()

fig = px.bar(
    stats.sort_values(by="format"),
    x="mean",
    y=key,
    color="format",
    error_x="err_plus",
    error_x_minus="err_minus",
    orientation="h",
    barmode="group",
    category_orders={key: y_order},
    labels={key: "Data Shape", "mean": "Seconds"},
    width=1300,
)
fig.update_layout(
    legend_title_text="File Write Duration",
    legend=dict(x=0.68, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(size=18),
)

pio.write_image(fig, file_write_time_image)
Image(url=file_write_time_image)

# + papermill={"duration": 0.313924, "end_time": "2025-09-03T21:57:22.005171", "exception": false, "start_time": "2025-09-03T21:57:21.691247", "status": "completed"}
key = "dataframe_shape (rows, cols)"

size_cols = {
    "csv_size (bytes)": "CSV",
    "sqlite_size (bytes)": "SQLite",
    "anndata_h5ad_size (bytes)": "AnnData (H5AD)",
    "anndata_zarr_size (bytes)": "AnnData (Zarr)",
    "parquet_size (bytes)": "Parquet",
}

# Long-form; if you have repeated runs per shape, we'll average them
long = df_results.melt(
    id_vars=[key],
    value_vars=list(size_cols.keys()),
    var_name="col",
    value_name="bytes",
)
long["format"] = long["col"].map(size_cols)

stats = long.groupby([key, "format"], as_index=False)["bytes"].mean()

# Descending y-axis by total size across formats (largest first).
# Swap this for a format-specific sort if you prefer (see below).
y_order = (
    stats.groupby(key, as_index=False)["bytes"]
    .sum()
    .sort_values("bytes", ascending=False)[key]
    .tolist()
)

fig = px.bar(
    stats.sort_values(by="format"),
    x="bytes",
    y=key,
    color="format",
    orientation="h",
    barmode="group",
    category_orders={key: result[key].iloc[::-1].tolist()},
    labels={key: "Data Shape", "bytes": "Bytes"},
    width=1300,
)

fig.update_layout(
    legend_title_text="File Size",
    legend=dict(x=0.72, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(size=18),
)

pio.write_image(fig, file_storage_size_image)
Image(url=file_storage_size_image)

# + papermill={"duration": 0.250629, "end_time": "2025-09-03T21:57:22.266295", "exception": false, "start_time": "2025-09-03T21:57:22.015666", "status": "completed"}
# read time barchart (all columns)
key = "dataframe_shape (rows, cols)"

cols = {
    "CSV": (
        "csv_read_time_all (secs) mean",
        "csv_read_time_all (secs) min",
        "csv_read_time_all (secs) max",
    ),
    "SQLite": (
        "sqlite_read_time_all (secs) mean",
        "sqlite_read_time_all (secs) min",
        "sqlite_read_time_all (secs) max",
    ),
    "AnnData (H5AD)": (
        "anndata_h5ad_read_time_all (secs) mean",
        "anndata_h5ad_read_time_all (secs) min",
        "anndata_h5ad_read_time_all (secs) max",
    ),
    "AnnData (Zarr)": (
        "anndata_zarr_read_time_all (secs) mean",
        "anndata_zarr_read_time_all (secs) min",
        "anndata_zarr_read_time_all (secs) max",
    ),
    "Parquet": (
        "parquet_read_time_all (secs) mean",
        "parquet_read_time_all (secs) min",
        "parquet_read_time_all (secs) max",
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

y_order = result[key].iloc[::-1].tolist()

fig = px.bar(
    stats.sort_values(by="format"),
    x="mean",
    y=key,
    color="format",
    error_x="err_plus",
    error_x_minus="err_minus",
    orientation="h",
    barmode="group",
    category_orders={key: y_order},
    labels={key: "Data Shape", "mean": "Seconds"},
    width=1300,
)
fig.update_layout(
    legend_title_text="File Read Duration",
    legend=dict(x=0.68, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(size=18),
)

pio.write_image(fig, file_read_time_all_image)
Image(url=file_read_time_all_image)

# + papermill={"duration": 0.260285, "end_time": "2025-09-03T21:57:22.537386", "exception": false, "start_time": "2025-09-03T21:57:22.277101", "status": "completed"}
# read time barchart (one column)
fig = px.line(
    df_results,
    y=[
        "csv_read_time_one (secs)",
        "sqlite_read_time_one (secs)",
        "parquet_read_time_one (secs)",
    ],
    x="dataframe_shape (rows, cols)",
    labels={"dataframe_shape (rows, cols)": "Data Shape", "value": "Seconds"},
    width=1300,
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    legend_title_text="File Read Duration (one column)",
    legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    font=dict(
        size=20,  # global font size
    ),
)
fig.update_xaxes(range=[0, 2.13])
fig.update_traces(mode="lines+markers")

pio.write_image(fig, file_read_time_one_image)
Image(url=file_read_time_one_image)
