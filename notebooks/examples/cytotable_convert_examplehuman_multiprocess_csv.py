#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets.
Note: intended to be used for profiling via memray.
"""

import pathlib
import sys

import parsl
from parsl.monitoring.monitoring import MonitoringHub
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_route
import pathlib
import cytotable

if __name__ == "__main__":
    input_file = sys.argv[1]
    dest_path = (
        f"{pathlib.Path(__file__).parent.resolve()}/"
        f"{pathlib.Path(input_file).name}.cytotable.parquet"
    )

    # clean up previous runs if they still exist
    if pathlib.Path(dest_path).exists():
        pathlib.Path(dest_path).unlink(missing_ok=True)

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                address=address_by_route(),
            )
        ],
        monitoring=MonitoringHub(
            hub_address=address_by_route(),
            resource_monitoring_interval=0.000001,
        ),
        strategy="none",
    )

    result = cytotable.convert(
        source_path=input_file,
        dest_path=dest_path,
        dest_datatype="parquet",
        source_datatype="csv",
        preset="cellprofiler_csv",
        chunk_size=200000,
    )

    # clear the parsl config
    # to help clean up.
    parsl.clear()

    # clean up file
    pathlib.Path(dest_path).unlink()
