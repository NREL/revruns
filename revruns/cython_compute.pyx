# -*- coding: utf-8 -*-
# cython: language_level=3, infer_types=True
"""Tutorial #1

IDG TECHtalk: https://www.youtube.com/watch?v=Vvd_CaHFXgg

- Almost all of the new syntax is simply to define c types.
- Move only the code that needs to be done in cython to cython
- Keep number of function calls between python and cython to a minimum
- Don't perform iteration on the python side, use the cython method

Author: travis
Date: Fri Jan  6 06:13:14 PM MST 2023
"""
import numpy as np
cimport cython

from tqdm import tqdm


DTYPE = np.intc


cdef clip(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)


@cython.boundscheck(False)  # Disable aboundary and negative index checking
@cython.wraparound(False)  # Not sure
def ccompute(long[:, ::1] a1, long[:, ::1] a2, int a, int b, int c):
    """Example Cython Numpy calculation."""
    cdef Py_ssize_t y_max = a1.shape[0]  # Index cannot go out of bounds
    cdef Py_ssize_t x_max = a1.shape[1]  # Or go negative

    result = np.zeros((x_max, y_max), dtype=DTYPE)
    cdef int[:, :] result_view = result

    cdef int tmp
    cdef Py_ssize_t x, y  # This is the suggested int type for array indexing

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip(a2[x, y], 2, 10)
            tmp = tmp * a + a2[x, y] * b
            result_view[y, x] = tmp + c

    return result
