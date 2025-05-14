#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets.
Note: intended to be used for benchmarking.
"""

import pathlib
import sys

import cytotable
import parsl
from parsl.addresses import address_by_route
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.monitoring.monitoring import MonitoringHub

if __name__ == "__main__":
    input_file = sys.argv[1]
    dest_path = (
        f"{pathlib.Path(__file__).parent.resolve()}/"
        f"{pathlib.Path(input_file).name}.cytotable.parquet"
    )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                address=address_by_route(),
            )
        ],
        monitoring=MonitoringHub(
            hub_address=address_by_route(),
            resource_monitoring_interval=1.0,
        ),
        strategy="none",
    )

    result = cytotable.convert(
        source_path=input_file,
        dest_path=dest_path,
        dest_datatype="parquet",
        source_datatype="sqlite",
        preset="cellprofiler_sqlite_pycytominer",
        chunk_size=200000,
        parsl_config=config,
    )

    # clear the parsl config
    # to help clean up.
    parsl.clear()

    # clean up file
    pathlib.Path(dest_path).unlink()
