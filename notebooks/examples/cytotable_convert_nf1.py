#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets.
Note: intended to be used for profiling via memray.
"""
import cytotable
import pathlib
import sys


def main():
    input_file = sys.argv[1]
    dest_path = (
        f"{pathlib.Path(__file__).parent.resolve()}/{pathlib.Path(input_file).name}"
    )

    result = cytotable.convert(
        source_path=input_file,
        dest_path=dest_path,
        dest_datatype="parquet",
        source_datatype="sqlite",
        preset="cellprofiler_sqlite_pycytominer",
        chunk_size=200000,
    )

    pathlib.Path(dest_path).unlink()


if __name__ == "__main__":
    main()
