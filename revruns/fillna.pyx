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
cimport cython

import numpy as np

from tqdm import tqdm


DTYPE = np.intc


@cython.boundscheck(False)
@cython.wraparound(False)
def fillna(int[:, ::1] array):
    """Fill nan values with mode of neighbors."""

    print("Made it to the beginning")
    cdef Py_ssize_t ymax = array.shape[0]
    cdef Py_ssize_t xmax = array.shape[1]
    cdef int[:, :] window
    cdef int[:] values
    cdef int mode
    cdef int zero
    cdef Py_ssize_t x, y

    print("Made it past declarations")

    for x in range(xmax):
        for y in range(ymax):
            window = array[y-1: y+1, x-1: x+1]
            values = [v for v in window if v > zero]
            mode = max(set(values), key=values.count)
            array[y, x] = mode
            print("Made it past first iteration")

    return array