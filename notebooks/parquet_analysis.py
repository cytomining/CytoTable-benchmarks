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
from typing import Dict, Literal

import anndata as ad
import duckdb
import hdf5plugin
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

color_palette = [
    "#4F58C7",
    "#BF442F",
    "#00A37A",
    "#8950C7",
    "#CC8148",
    "#14A9C2",
    "#CC5275",
    "#92BA66",
    "#CC79CC",
    "#CAA142",
]


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


def generate_format_comparison_plot(
    df: pd.DataFrame, cols: Dict[str, tuple], title: str, save_file: str
):
    """ """
    key = "dataframe_shape (rows, cols)"

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
        stats,
        x=key,
        y="mean",
        color="format",
        error_y="err_plus",
        error_y_minus="err_minus",
        markers=True,
        category_orders={key: x_order},
        labels={
            key: "Data Shape",
            "mean": "Seconds (log)",
        },  # we'll override titles later
        log_y=True,
        title=title,
        facet_col="format",
        facet_col_wrap=4,
        color_discrete_sequence=color_palette,
        render_mode="webgl",
        width=1200,
        height=500,
    )

    # Remove facet headers
    fig.for_each_annotation(lambda a: a.update(text=""))
    for k in list(fig.layout):
        if k.startswith("xaxis") or k.startswith("yaxis"):
            fig.layout[k].title = None

    # (Optional) if you still have facet headers, hide them
    fig.for_each_annotation(lambda a: a.update(text=""))

    # make sure main title is not center aligned
    # Remove the built-in title
    fig.update_layout(title_text=None)

    # Add a figure-level title annotation (left-aligned)
    fig.add_annotation(
        text="File format read time duration (one column) (seconds)",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=20),
        align="left",
        yshift=32,
    )

    def _axis_key(ref: str) -> str:
        # "x"   -> "xaxis",  "x2" -> "xaxis2"
        # "y"   -> "yaxis",  "y2" -> "yaxis2"
        return f"{ref[0]}axis" if len(ref) == 1 else f"{ref[0]}axis{ref[1:]}"

    def add_subplot_subtitles(
        fig, label_by_trace_name=True, rel_offset=0.02, font_size=12
    ):
        """
        Add subplot subtitles centered above each facet.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
        label_by_trace_name : bool, default=True
            If True, use trace.name for the text.
        rel_offset : float, default=0.02
            Fraction of subplot height to push the label above the top.
            Smaller values = closer to the plot.
        font_size : int, default=12
            Font size for subtitles.
        """
        fig.for_each_annotation(
            lambda a: a.update(text="")
        )  # clear default facet headers
        seen = set()

        def _axis_key(ref: str) -> str:
            return f"{ref[0]}axis" if len(ref) == 1 else f"{ref[0]}axis{ref[1:]}"

        for tr in fig.data:
            key = (tr.xaxis, tr.yaxis)
            if key in seen:
                continue
            seen.add(key)

            xaxis = _axis_key(tr.xaxis)
            yaxis = _axis_key(tr.yaxis)
            xdom = fig.layout[xaxis].domain
            ydom = fig.layout[yaxis].domain

            # Offset proportional to subplot height
            height = ydom[1] - ydom[0]
            y = ydom[1] + rel_offset * height
            x = (xdom[0] + xdom[1]) / 2

            text = tr.name if label_by_trace_name else "subplot"
            color = getattr(tr.line, "color", None) or getattr(
                getattr(tr, "marker", {}), "color", None
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                text=text,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=font_size),
            )

    add_subplot_subtitles(fig, label_by_trace_name=True)

    fig.add_annotation(
        text=title,
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.05,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=20),
        align="left",
        yshift=32,
    )

    fig.add_annotation(
        text="Data Shape",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(size=16),
        yshift=-60,
    )

    # Add ONE centered y-axis title for the whole figure
    fig.add_annotation(
        text="Seconds (log)",
        xref="paper",
        yref="paper",
        x=0,
        y=0.5,
        xanchor="right",
        yanchor="middle",
        textangle=-90,
        showarrow=False,
        font=dict(size=16),
        xshift=-50,
    )
    pio.write_image(fig, save_file)


# -

# remove all files just in case
remove_files()

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
# capture primary data metrics on file format performance

# avoid a rewrite of the data if we already have it
if not pathlib.Path("parquet_analysis.parquet").is_file():
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
            np.random.rand(nrows, ncols),
            columns=[f"col_{num}" for num in range(0, ncols)],
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
                    "anndata_h5ad_noc_size (bytes)": os.stat(
                        anndata_h5_noc_name
                    ).st_size,
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
                    "anndata_h5ad_gzip_size (bytes)": os.stat(
                        anndata_h5_gzip_name
                    ).st_size,
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
                    "anndata_h5ad_lz4_size (bytes)": os.stat(
                        anndata_h5_lz4_name
                    ).st_size,
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
                    "anndata_h5ad_zstd_size (bytes)": os.stat(
                        anndata_h5_zstd_name
                    ).st_size,
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
else:
    df_results = pd.read_parquet("parquet_analysis.parquet")
df_results
# -

# calculate full write + read time for each format
if not pathlib.Path("parquet_analysis.parquet").is_file():
    df_results["csv_write_and_read_time (secs)"] = (
        df_results["csv_write_time (secs)"] + df_results["csv_read_time_all (secs)"]
    )
    df_results["sqlite_write_and_read_time (secs)"] = (
        df_results["sqlite_write_time (secs)"]
        + df_results["sqlite_read_time_all (secs)"]
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

# export data to file
if not pathlib.Path("parquet_analysis.parquet").is_file():
    df_results.to_parquet("parquet_analysis.parquet")

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
    "AnnData (H5AD - LZ4)": (
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

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format write time duration (seconds)",
    save_file=file_write_time_image,
)
Image(url=file_write_time_image)

# +
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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_write_time (secs) mean",
        "anndata_h5ad_zstd_write_time (secs) min",
        "anndata_h5ad_zstd_write_time (secs) max",
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
    "Parquet (ZSTD)": (
        "parquet_zstd_write_time (secs) mean",
        "parquet_zstd_write_time (secs) min",
        "parquet_zstd_write_time (secs) max",
    ),
}

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format write time duration (seconds)",
    save_file=file_write_time_image.replace(".parquet", "-reduced.parquet"),
)
Image(url=file_write_time_image.replace(".parquet", "-reduced.parquet"))

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
    font=dict(size=16),
)

pio.write_image(fig, file_storage_size_image)
Image(url=file_storage_size_image)

# +
key = "dataframe_shape (rows, cols)"

# ↓ Reduced to only the formats you want
size_cols = {
    "csv_size (bytes)": "CSV (GZIP)",
    "sqlite_size (bytes)": "SQLite",
    "anndata_h5ad_zstd_size (bytes)": "AnnData (H5AD - ZSTD)",
    "parquet_zstd_size (bytes)": "Parquet (ZSTD)",
}

# (Optional) sanity check: fail fast if a requested column is missing
missing = [c for c in size_cols if c not in df_results.columns]
if missing:
    raise KeyError(f"Missing expected size columns: {missing}")

# Long-form + average across repeats
long = df_results.melt(
    id_vars=[key],
    value_vars=list(size_cols.keys()),
    var_name="col",
    value_name="bytes",
).dropna(subset=["bytes"])
long["format"] = long["col"].map(size_cols)

stats = long.groupby([key, "format"], as_index=False)["bytes"].mean()

# x-axis category order (groups on the axis) – keep your existing convention
x_order = result[key].iloc[::-1].tolist()

# Pre-sort so each trace follows x_order
pos = {cat: i for i, cat in enumerate(x_order)}
stats_sorted = stats.assign(xpos=stats[key].map(pos))

# Desired legend/color order
format_order = [
    "CSV (GZIP)",
    "SQLite",
    "AnnData (H5AD - ZSTD)",
    "Parquet (ZSTD)",
]

# Enforce categorical order then sort (xpos, format)
stats_sorted["format"] = pd.Categorical(
    stats_sorted["format"], categories=format_order, ordered=True
)
stats_sorted = stats_sorted.sort_values(["xpos", "format"]).drop(columns="xpos")

fig = px.bar(
    stats_sorted,
    x=key,
    y="bytes",
    color="format",
    barmode="group",  # side-by-side bars per data shape
    category_orders={key: x_order, "format": format_order},
    labels={key: "Data Shape", "bytes": "Output file size (bytes)"},
    width=1300,
    title="File format size (bytes)",
)

# ---- Flip bars within each group (reverse trace order) but keep legend order ----
# Reverse the actual trace list (controls bar order)
fig.data = tuple(fig.data[::-1])
# Show legend in the original (non-reversed) order
fig.update_layout(legend_traceorder="reversed")

# Inset legend + styling
fig.update_layout(
    legend_title_text=None,
    legend=dict(
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    ),
    margin=dict(r=80, t=80, l=80, b=80),
    font=dict(size=18),
    bargap=0.15,      # space between groups
    bargroupgap=0.05, # space within a group
)

# Keep the same left-to-right x ordering convention as before
fig.update_xaxes(autorange="reversed")

pio.write_image(
    fig, (img_file := file_storage_size_image.replace(".png", "-reduced.png"))
)
Image(url=img_file)

# + papermill={"duration": 0.250629, "end_time": "2025-09-03T21:57:22.266295", "exception": false, "start_time": "2025-09-03T21:57:22.015666", "status": "completed"}
# read time plot (all columns)
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
    "AnnData (H5AD - LZ4)": (
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

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read time duration (full dataset) (seconds)",
    save_file=file_read_time_all_image,
)
Image(url=file_read_time_all_image)

# +
# read time plot (all columns)
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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_read_time_all (secs) mean",
        "anndata_h5ad_zstd_read_time_all (secs) min",
        "anndata_h5ad_zstd_read_time_all (secs) max",
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
    "Parquet (ZSTD)": (
        "parquet_zstd_read_time_all (secs) mean",
        "parquet_zstd_read_time_all (secs) min",
        "parquet_zstd_read_time_all (secs) max",
    ),
}

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read time duration (full dataset) (seconds)",
    save_file=file_read_time_all_image.replace(".png", "-reduced.png"),
)
Image(url=file_read_time_all_image.replace(".png", "-reduced.png"))

# +
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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_read_time_all (secs) mean",
        "anndata_h5ad_zstd_read_time_all (secs) min",
        "anndata_h5ad_zstd_read_time_all (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_read_time_all (secs) mean",
        "parquet_zstd_read_time_all (secs) min",
        "parquet_zstd_read_time_all (secs) max",
    ),
}

# --- Build long dataframe with mean/min/max → error bars ---
records = []
for fmt, (mean_col, min_col, max_col) in cols.items():
    tmp = result[[key, mean_col, min_col, max_col]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mean_col: "mean", min_col: "min", max_col: "max"}, inplace=True)
    records.append(tmp)

long = pd.concat(records, ignore_index=True).dropna(subset=["mean"])
long["err_plus"] = (long["max"] - long["mean"]).clip(lower=0)
long["err_minus"] = (long["mean"] - long["min"]).clip(lower=0)

# --- Orders ---
# X groups in the order they appear in the data (keep as-is)
x_seen = list(pd.unique(long[key]))
# If you want reversed x later, switch to: x_seen = x_seen[::-1]

# Fixed bar/legend order inside each group
format_order = ["CSV (GZIP)", "SQLite", "AnnData (H5AD - ZSTD)", "Parquet (ZSTD)"]
long["format"] = pd.Categorical(long["format"], categories=format_order, ordered=True)

# Sort table to make traces stable (x then format)
long_sorted = long.sort_values([key, "format"])

# --- Build fig with explicit trace order (guarantees bar order) ---
fig = go.Figure()
for fmt in format_order:
    sub = long_sorted[long_sorted["format"] == fmt]
    # Plotly needs arrays (lists); ensure aligned per x order
    # Reindex each sub to x_seen to keep empty categories aligned if any missing
    sub = sub.set_index(key).reindex(x_seen)
    fig.add_trace(
        go.Bar(
            name=fmt,
            x=[str(x) for x in x_seen],  # safe labels (tuples → str)
            y=sub["mean"],
            error_y=dict(
                type="data",
                array=sub["err_plus"],
                arrayminus=sub["err_minus"],
                visible=True,
            ),
            # offsetgroup ensures grouped bars align even with missing categories
            offsetgroup=fmt,
        )
    )

fig.update_layout(
    barmode="group",
    title="File format read time duration (full dataset) with error bars",
    xaxis_title="Data Shape",
    yaxis_title="Read Time<br>(log(seconds))",
    yaxis_type="log",
    width=1300,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    ),
    margin=dict(r=80, t=80, l=80, b=80),
    bargap=0.15,
    bargroupgap=0.05,
    font=dict(
        size=18,
    )
)
pio.write_image(
    fig,
    (img_file := file_read_time_all_image.replace(".png", "-reduced-nonfacet-bar.png")),
)
Image(url=img_file)

# + papermill={"duration": 0.260285, "end_time": "2025-09-03T21:57:22.537386", "exception": false, "start_time": "2025-09-03T21:57:22.277101", "status": "completed"}
# read time plot (one column)
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
    "AnnData (H5AD - LZ4)": (
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

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read time duration (one column) (seconds)",
    save_file=file_read_time_one_image,
)
Image(url=file_read_time_one_image)

# +
# read time plot (one column)
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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_read_time_one (secs) mean",
        "anndata_h5ad_zstd_read_time_one (secs) min",
        "anndata_h5ad_zstd_read_time_one (secs) max",
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
    "Parquet (ZSTD)": (
        "parquet_zstd_read_time_one (secs) mean",
        "parquet_zstd_read_time_one (secs) min",
        "parquet_zstd_read_time_one (secs) max",
    ),
}

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read time duration (one column) (seconds)",
    save_file=file_read_time_one_image.replace(".png", "-reduced.png"),
)
Image(url=file_read_time_one_image.replace(".png", "-reduced.png"))

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
    "AnnData (H5AD - LZ4)": (
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

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read and write time duration (full dataset) (seconds)",
    save_file=file_read_time_write_and_read_time_image,
)
Image(url=file_read_time_write_and_read_time_image)

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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_write_and_read_time (secs) mean",
        "anndata_h5ad_zstd_write_and_read_time (secs) min",
        "anndata_h5ad_zstd_write_and_read_time (secs) max",
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
    "Parquet (ZSTD)": (
        "parquet_zstd_write_and_read_time (secs) mean",
        "parquet_zstd_write_and_read_time (secs) min",
        "parquet_zstd_write_and_read_time (secs) max",
    ),
}

generate_format_comparison_plot(
    df=result,
    cols=cols,
    title="File format read and write time duration (full dataset) (seconds)",
    save_file=file_read_time_write_and_read_time_image.replace(".png", "-reduced.png"),
)
Image(url=file_read_time_write_and_read_time_image.replace(".png", "-reduced.png"))

# +
key = "dataframe_shape (rows, cols)"

# Use your reduced set (adjust if you want more/less)
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
    "AnnData (H5AD - ZSTD)": (
        "anndata_h5ad_zstd_write_and_read_time (secs) mean",
        "anndata_h5ad_zstd_write_and_read_time (secs) min",
        "anndata_h5ad_zstd_write_and_read_time (secs) max",
    ),
    "Parquet (ZSTD)": (
        "parquet_zstd_write_and_read_time (secs) mean",
        "parquet_zstd_write_and_read_time (secs) min",
        "parquet_zstd_write_and_read_time (secs) max",
    ),
}

# --- Build long dataframe with mean/min/max → error bars ---
records = []
for fmt, (mean_col, min_col, max_col) in cols.items():
    tmp = result[[key, mean_col, min_col, max_col]].copy()
    tmp["format"] = fmt
    tmp.rename(columns={mean_col: "mean", min_col: "min", max_col: "max"}, inplace=True)
    records.append(tmp)

long = pd.concat(records, ignore_index=True).dropna(subset=["mean"])
long["err_plus"] = (long["max"] - long["mean"]).clip(lower=0)
long["err_minus"] = (long["mean"] - long["min"]).clip(lower=0)

# --- Orders ---
# X groups in the order they appear in the data (keep as-is)
x_seen = list(pd.unique(long[key]))
# If you want reversed x later, switch to: x_seen = x_seen[::-1]

# Fixed bar/legend order inside each group
format_order = ["CSV (GZIP)", "SQLite", "AnnData (H5AD - ZSTD)", "Parquet (ZSTD)"]
long["format"] = pd.Categorical(long["format"], categories=format_order, ordered=True)

# Sort table to make traces stable (x then format)
long_sorted = long.sort_values([key, "format"])

# --- Build fig with explicit trace order (guarantees bar order) ---
fig = go.Figure()
for fmt in format_order:
    sub = long_sorted[long_sorted["format"] == fmt]
    # Plotly needs arrays (lists); ensure aligned per x order
    # Reindex each sub to x_seen to keep empty categories aligned if any missing
    sub = sub.set_index(key).reindex(x_seen)
    fig.add_trace(
        go.Bar(
            name=fmt,
            x=[str(x) for x in x_seen],  # safe labels (tuples → str)
            y=sub["mean"],
            error_y=dict(
                type="data",
                array=sub["err_plus"],
                arrayminus=sub["err_minus"],
                visible=True,
            ),
            # offsetgroup ensures grouped bars align even with missing categories
            offsetgroup=fmt,
        )
    )

fig.update_layout(
    barmode="group",
    title="File format write and read time duration (full dataset) with error bars",
    xaxis_title="Data Shape",
    yaxis_title="Write and read time<br>(log(seconds))",
    yaxis_type="log",
    width=1300,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    ),
    margin=dict(r=80, t=80, l=80, b=80),
    bargap=0.15,
    bargroupgap=0.05,
    font=dict(
        size=18,
    )
)


pio.write_image(
    fig,
    (img_file := file_read_time_all_image.replace(".png", "-reduced-nonfacet-bar.png")),
)
Image(url=img_file)

# +
# summarize read/write performance data
key = "dataframe_shape (rows, cols)"
mean_cols = {
    "CSV (GZIP)": "csv_write_and_read_time (secs) mean",
    "SQLite": "sqlite_write_and_read_time (secs) mean",
    "AnnData (H5AD - ZSTD)": "anndata_h5ad_zstd_write_and_read_time (secs) mean",
    "Parquet (ZSTD)": "parquet_zstd_write_and_read_time (secs) mean",
}

# Build a wide table of the mean read times by data shape
wide = result[[key] + list(mean_cols.values())].rename(
    columns={v: k for k, v in mean_cols.items()}
)

# Compute "% faster" for Parquet vs each baseline, per shape
for baseline in ["CSV (GZIP)", "SQLite", "AnnData (H5AD - ZSTD)"]:
    wide[f"Parquet vs {baseline} (% faster)"] = (
        (wide[baseline] - wide["Parquet (ZSTD)"]) / wide[baseline] * 100
    )

cols = [
    key,
    "Parquet vs CSV (GZIP) (% faster)",
    "Parquet vs SQLite (% faster)",
    "Parquet vs AnnData (H5AD - ZSTD) (% faster)",
]
percent_by_shape = wide[cols].sort_values(key)
print(percent_by_shape)
print(
    "Parquet percent average performance increase over AnnData (zstd):",
    percent_by_shape["Parquet vs AnnData (H5AD - ZSTD) (% faster)"].mean(),
)

# +
# summarize filesize data
key = "dataframe_shape (rows, cols)"
target = "AnnData (H5AD - ZSTD)"  # the format we're evaluating

# Start from your 'stats' (mean bytes by [shape, format])
# -> make it wide: one column per format
wide = stats.pivot(index=key, columns="format", values="bytes").reset_index()

# Baselines are all formats except the key and target
baselines = [c for c in wide.columns if c not in [key, target]]

# Compute % less storage for AnnData vs each baseline (per shape)
for b in baselines:
    wide[f"{target} vs {b} (% less)"] = (wide[b] - wide[target]) / wide[b] * 100

# Keep only display columns (shape + percent columns)
percent_cols = [c for c in wide.columns if c.endswith("(% less)")]
table = wide[[key] + percent_cols].copy()

# (Optional) order rows to match your previous x_order, if you have it:
# table = table.set_index(key).loc[x_order].reset_index()

# Add an Average % row (simple arithmetic mean across shapes)
avg = table[percent_cols].mean(numeric_only=True)
avg_row = pd.DataFrame([{key: "Average (across shapes)"} | avg.to_dict()])
table_with_avg = pd.concat([table, avg_row], ignore_index=True)

# Nice formatting (1 decimal place + % sign)
display_table = table_with_avg.copy()
for c in percent_cols:
    display_table[c] = display_table[c].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "–"
    )

display_table
