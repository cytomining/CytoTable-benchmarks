"""
Utilities for running various benchmarks.
"""

import pathlib
import platform
import sqlite3
import subprocess
import time
from contextlib import closing
from typing import Callable, List, Optional, Tuple, Union

import psutil
import requests


def timer(func: Callable, method_chain: Optional[str] = None, *args, **kwargs) -> float:
    """
    A timer function which runs a function and related arguments
    to return the total time in seconds which were taken for completion.
    """

    # find the start time
    start_time = time.time()

    # run the function with given args
    result = func(*args, **kwargs)

    # chain the result to the specified method
    if method_chain is not None:
        result = getattr(result, method_chain)()

    # return the current time minus the start time
    return time.time() - start_time


def download_file(urlstr, filename):
    """
    Download a file given a string url
    """
    if pathlib.Path(filename).exists():
        print("We already have downloaded the file!")
        return

    # Send a HTTP request to the URL of the file you want to access
    response = requests.get(urlstr, timeout=30)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, "wb") as file:
            # Write the contents of the response to a file
            file.write(response.content)
    else:
        print(f"Failed to download file, status code: {response.status_code}")


def get_system_info(show_output: bool = False) -> dict:
    """
    Retrieve system information such as
    OS, CPU, RAM, and Python version.

    Args:
        pretty_print (bool):
            If True, prints the system
            information in a readable format.
            Defaults to False.

    Returns:
        dict:
            A dictionary containing system information.
    """

    info = {
        "Operating System": platform.system(),
        "Machine Type": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores (Logical)": psutil.cpu_count(logical=True),
        "CPU Cores (Physical)": psutil.cpu_count(logical=False),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "Python Version": platform.python_version(),
    }

    if show_output:
        print("\nSystem Information:")
        for key, value in info.items():
            print(f"{key}: {value}")

    return info


def get_parsl_peak_memory(db_file: str) -> float:
    """Retrieves the maximum resident memory from the resource table.

    This function connects to the specified SQLite database, queries the
    'resource' table for the maximum value of the
    'psutil_process_memory_resident' column, and returns the result.

    Args:
        db_file: The path to the SQLite database file.

    Returns:
        The maximum resident memory value, or -1 if an error occurs
        or the value is not found.
    """
    try:
        with closing(sqlite3.connect(db_file)) as cx:
            result = cx.execute(
                "SELECT MAX(psutil_process_memory_resident) FROM resource;"
            ).fetchone()

        if result and result[0] is not None:
            return float(result[0])
        else:
            return -1.0
    except sqlite3.Error as e:
        print(f"Error: {e}")
        return -1.0


def get_memory_peak_and_time_duration(
    cmd: List[Union[str, bytes]],
    polling_pause_seconds: float = 0.2,
    skip_memory_check: bool = False,
) -> Tuple[float, float]:
    """
    Track peak memory usage and runtime of a subprocess and its process tree.

    Args:
        cmd:
            Command to run as a subprocess (e.g., ["python", "script.py"]).
        polling_pause_seconds:
            Time between memory checks (default: 0.2s).
        skip_memory_check:
            If True, skip memory check and only time execution.

    Returns:
        Tuple[float, float]: Peak memory in MB (or -1 if skipped), and total runtime in seconds.
    """
    start_time = time.time()
    proc = subprocess.Popen(cmd, start_new_session=True)
    peak = -1.0  # default if skipped or error

    if skip_memory_check:
        proc.wait()
        return -1.0, time.time() - start_time

    try:
        root = psutil.Process(proc.pid)
        peak = 0.0

        while proc.poll() is None:
            try:
                all_procs = [root] + root.children(recursive=True)
                mem = 0
                for p in all_procs:
                    try:
                        if p.is_running():
                            mem += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                peak = max(peak, mem)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # Root died while polling, break cleanly
                break
            time.sleep(polling_pause_seconds)

    finally:
        # Always wait to make sure process has ended
        proc.wait()
        duration = time.time() - start_time
        return peak, duration
