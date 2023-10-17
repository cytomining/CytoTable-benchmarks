"""
Example demonstrating what happens when we use sleep after
allocating some memory in Python.
"""

import gc
import time

import random

# create randomized number data
example = [[random.random() for _ in range(100)] for _ in range(1000)]

# sleep for memory visibility
time.sleep(10)

# delete reference to example data
del example

# invoke garbage collection
gc.collect()

# sleep for memory visibility
time.sleep(10)
