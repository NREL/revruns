# -*- coding: utf-8 -*-
# cython : language_level=3
"""Tutorial #1

IDG TECHtalk: https://www.youtube.com/watch?v=Vvd_CaHFXgg

Author: travis
Date: Fri Jan  6 06:13:14 PM MST 2023
"""
import numpy as np

from tqdm import tqdm


def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)


def compute(a1, a2, a, b, c):
    y_max = a1.shape[0]
    x_max = a1.shape[1]

    result = np.zeros((x_max, y_max), dtype=a1.dtype)

    for x in tqdm(range(x_max)):
        for y in range(y_max):
            tmp = clip(a2[x, y], 2, 10)
            tmp = tmp * a + a2[x, y] * b
            result[y, x] = tmp + c

    return result


def main():
    root = 7_000
    size = root ** 2
    a1 = np.random.randint(0, 100, size).reshape((root, root))
    a2 = np.random.randint(0, 100, size).reshape((root, root))
    a = 10
    b = 20
    c = 30
    compute(a1, a2, a, b, c)


if __name__ == "__main__":
    main()
