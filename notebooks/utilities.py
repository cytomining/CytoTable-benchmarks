"""
Utilities for running various benchmarks.
"""
import pathlib
import time
from typing import Callable, Optional
import platform
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
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Python Version": platform.python_version(),
    }

    if show_output:
        print("\nSystem Information:")
        for key, value in info.items():
            print(f"{key}: {value}")

    return info