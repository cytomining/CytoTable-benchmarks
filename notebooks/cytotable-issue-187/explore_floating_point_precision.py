# ---
# jupyter:
#   jupytext:
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

# # Exploring Python, NumPy, DuckDB, and Arrow floating point precision
#
# This notebook explores how floating point number precision is captured within Python, NumPy, DuckDB, and Arrow to better understand the implications of storing data within these formats.
# The work originally was inspired by [CytoTable issue #187](https://github.com/cytomining/CytoTable/issues/187).

# + colab={"base_uri": "https://localhost:8080/"} id="KIQ-xx3OHSTz" outputId="5b38cb9f-23bc-4769-c74d-554d2f0b395c"
import decimal
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as parquet

# add modules from a directory above this one
sys.path = sys.path + [".."]

from utilities import get_system_info

# + colab={"base_uri": "https://localhost:8080/"} id="nCM_5ihHHUpV" outputId="de641b02-3cdd-4497-961a-5fc88eb2283f"
# show the system information
_ = get_system_info(show_output=True)

# + colab={"base_uri": "https://localhost:8080/"} id="37vL3L2L9h4p" outputId="8bc37e1d-61ff-445a-94e4-3aff58667d19"
# default interpreted value in Python
float_value = 3.5215257120407011
float_value

# + colab={"base_uri": "https://localhost:8080/", "height": 36} id="mF7CpvMg-1oo" outputId="45650ef8-3d50-47f0-bc63-a23dbc598687"
# as a formatted string
"{0:.16f}".format(3.5215257120407011)

# + colab={"base_uri": "https://localhost:8080/"} id="Fxi-026I-DRY" outputId="54853fdf-d717-4503-85c8-ac2330a0733b"
# as a python Decimal
decimal.Decimal(3.5215257120407011)

# + colab={"base_uri": "https://localhost:8080/"} id="Rccc5GFA_eAU" outputId="7426e753-a5d3-4917-f2a2-07065fffcb01"
# as numpy value within np.float64 array
arr = np.array([3.5215257120407011], dtype=np.float64)
arr[0]

# + colab={"base_uri": "https://localhost:8080/"} id="pjLRZ673AZgQ" outputId="612fe5b7-0401-4a58-dab9-298bd8a12c5c"
# as numpy value within np.longdouble array
arr = np.array([3.5215257120407011], dtype=np.longdouble)
arr[0]

# + colab={"base_uri": "https://localhost:8080/", "height": 81} id="XA1tbb7dHJLj" outputId="cde6a81d-5985-4cd7-8346-319b9a4dfa1e"
# try to read with pandas
pd.DataFrame({"col_a": [3.5215257120407011]})

# + colab={"base_uri": "https://localhost:8080/"} id="6dJNxJwwHbfJ" outputId="5d49ca96-9256-47fe-ea7e-d54dee462739"
# try to read with pandas through pyarrow
# (referenced auto-inferred duckdb decimal settings, see below, which appear to align)
pd.DataFrame({"col_a": [3.5215257120407011]})["col_a"].astype(
    pd.ArrowDtype(pa.decimal128(17, 16))
)

# + colab={"base_uri": "https://localhost:8080/", "height": 217} id="8kpEQGZ19C5X" outputId="546f3294-4146-42c7-90cc-cc6cae531886"
# show results from pyarrow array
pa.array([decimal.Decimal("3.5215257120407011")], type=pa.decimal128(17, 16))

# + id="It1QyOsGNBf9"
# write the data to a parquet file to see how it's retained
pd.DataFrame({"col_a": [3.5215257120407011]}).astype(
    pd.ArrowDtype(pa.decimal128(17, 16))
).to_parquet("example.parquet")

# + colab={"base_uri": "https://localhost:8080/"} id="4RcWQg46NFYu" outputId="4cec0005-0921-44f8-f235-4cbd93a2a128"
# show what's inside the file from pandas and pyarrow's perspective
print(pd.read_parquet("example.parquet"), "\n")
print(parquet.read_table("example.parquet"), "\n")
print(parquet.read_schema("example.parquet"), "\n")

# + colab={"base_uri": "https://localhost:8080/"} id="okAp858D6JUp" outputId="59c9c01e-5626-4e1e-fc72-8f4449a6d1dd"
# show how number is interpreted without cast in duckdb
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT 3.5215257120407011;
  """
    ).arrow()
result

# + colab={"base_uri": "https://localhost:8080/"} id="XG3UvYvA3vwf" outputId="93f3d468-4e2f-48c1-d952-b0f0566dcc38"
# show how number is interpreted with cast to DOUBLE
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT CAST(3.5215257120407011 AS DOUBLE);
  """
    ).arrow()
result

# + colab={"base_uri": "https://localhost:8080/"} id="4kziAF6DBrZo" outputId="726694e3-d73d-4544-f9d1-741dbb47fe20"
# show how number is interpreted with cast to DECIMAL (inferenced size)
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT CAST(3.5215257120407011 AS DECIMAL);
  """
    ).arrow()
result

# + colab={"base_uri": "https://localhost:8080/"} id="9tCCvPFAPuMK" outputId="97f631ff-0d22-47ea-f5df-420b8c8efdbc"
# show how the data are read from Parquet
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT *
  FROM read_parquet('example.parquet');
  """
    ).arrow()
result

# + id="SCcyQCU532Gu"
# write a one column, one value csv with the floating point number
with open(file="example.csv", mode="w", encoding="utf-8") as file:
    file.write("col_a\n")
    file.write("3.5215257120407011")

# + colab={"base_uri": "https://localhost:8080/", "height": 81} id="uCjxRAyfRNN1" outputId="a28806dc-5660-4b9a-9821-e4ca613829c6"
# read the value from pandas
pd.read_csv("example.csv")

# + colab={"base_uri": "https://localhost:8080/"} id="KoWP20-c-L-Y" outputId="92b4b9c8-effd-487e-94e8-e221e57d3fbe"
# try to read the value from pyarrow's csv reader
pacsv.read_csv(input_file="example.csv")

# + colab={"base_uri": "https://localhost:8080/"} id="CORq3iO54EsW" outputId="4e541389-0dc5-4f35-f18d-338efd1461e5"
# show how the csv reader interprets the value by default (automatic settings)
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT *
  FROM read_csv('example.csv');
  """
    ).arrow()

result

# + colab={"base_uri": "https://localhost:8080/", "height": 332} id="bu8yVDCH4Nn8" outputId="51f0ecd9-91f5-437a-96d8-d80df16c95f6"
# try to modify the auto_type_candidates to incorporate the type we saw earlier
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
  SELECT *
  FROM read_csv('example.csv', auto_type_candidates=['DECIMAL(18,16)']);
  """
    ).arrow()

result
