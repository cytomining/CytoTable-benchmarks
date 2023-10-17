"""
Example demonstrating what happens when we use sleep after
allocating some memory in Python.
"""

import random
import time

# create randomized number data
example = [[random.random() for _ in range(100)] for _ in range(1000)]

# sleep for memory visibility
time.sleep(10)
