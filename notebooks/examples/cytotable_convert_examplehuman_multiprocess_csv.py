#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets.
Note: intended to be used for profiling via memray.
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
            resource_monitoring_interval=0.1,
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
        parsl_config=config,
    )

    # clean up file
    pathlib.Path(dest_path).unlink()
