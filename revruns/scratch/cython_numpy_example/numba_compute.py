# -*- coding: utf-8 -*-
# cython : language_level=3
"""Tutorial #1

Numba is much more limited to numerical calculations. It speeds up a "subset"
of Python code.

However, Numba is a lot easier to use to update code, requireing just a
decorator. It is a Just In Time Compiler (JIT), so there's a little of overhead
at first, but subsequent calls will be fast.

Where you have a single numerical function that is called a lot, this is good.
Obviously, since it's a JIT compiler, if you only call a function a few times,
this isn't ideal.

Author: travis
Date: Fri Jan  6 06:13:14 PM MST 2023
"""
import numba as nb
import numpy as np

from tqdm import tqdm


@nb.njit  # No Python JIT, faster, less flexible
def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)


@nb.njit
def ncompute(a1, a2, a, b, c):
    y_max = a1.shape[0]
    x_max = a1.shape[1]

    result = np.zeros((y_max, x_max), dtype=a1.dtype)

    for x in range(y_max):
        for y in range(x_max):
            tmp = clip(a2[y, x], 2, 10)
            tmp = tmp * a + a2[y, x] * b
            result[y, x] = tmp + c
    return result


@nb.jit
def clip2(a, min_value, max_value):
    return min(max(a, min_value), max_value)


@nb.jit
def ncompute2(a1, a2, a, b, c):
    y_max = a1.shape[0]
    x_max = a1.shape[1]

    result = np.zeros((y_max, x_max), dtype=a1.dtype)

    for x in range(y_max):
        for y in range(x_max):
            tmp = clip2(a2[y, x], 2, 10)
            tmp = tmp * a + a2[y, x] * b
            result[y, x] = tmp + c
    return result
