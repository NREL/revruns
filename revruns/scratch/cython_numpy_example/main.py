# cython: language_level=3
# -*- coding: utf-8 -*-
"""Module Name.

Module description.

Author: travis
Date: Fri Jan  6 06:47:48 PM MST 2023
"""
import timeit

import numpy as np

# from compute import compute
from revruns.cython_compute import ccompute
# from numba_compute import ncompute, ncompute2


ROOT = 20_000
SIZE = ROOT ** 2
A1 = np.random.randint(0, 100, SIZE).reshape((ROOT, ROOT))
A2 = np.random.randint(0, 100, SIZE).reshape((ROOT, ROOT))
A = 10
B = 20
C = 30


# def main():
#     """Run cython function."""
#     print("Running Python Version")
#     compute(A1, A2, A, B, C)


def cmain():
    """Run cython function."""
    print("Running Cython Version")
    ccompute(A1, A2, A, B, C)


# def nmain():
#     """Run cython function."""
#     print("Running Numba Version")
#     ncompute(A1, A2, A, B, C)


# def nmain2():
#     """Run cython function."""
#     print("Running Numba-Python Version")
#     ncompute2(A1, A2, A, B, C)


if __name__ == "__main__":
    # ptime = timeit.timeit("main()", number=1, globals=globals())
    # ntime2 = timeit.timeit("nmain2()", number=1, globals=globals())
    # ntime = timeit.timeit("nmain()", number=1, globals=globals())
    ctime = timeit.timeit("cmain()", number=1, globals=globals())

    # print(f"Python Time: {round(ptime, 2)} seconds")
    # print(f"Numba-Python Time: {round(ntime2, 2)} seconds")
    # print(f"Numba Time: {round(ntime, 2)} seconds")
    print(f"Cython Time: {round(ctime, 2)} seconds")
