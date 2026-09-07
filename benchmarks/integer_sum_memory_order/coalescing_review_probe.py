"""Independent modular-sum checks for the stride-coalescing implementation.

Run with the interpreter and PYTHONPATH selecting the Numba build under review.
This prints a JSON record and does not write files or perform benchmarks.
"""

import json
import math
import time

import numba
import numpy as np
from numba import jit, types


@jit
def total(a):
    return np.sum(a)


@jit
def narrow(a):
    return np.sum(a, dtype=np.int8)


@jit
def unsigned(a):
    return np.sum(a, dtype=np.uint64)


def oracle(a, width=64, signed=True):
    value = sum(int(x) for x in a.flat) % (1 << width)
    if signed and value >= (1 << (width - 1)):
        value -= 1 << width
    return value


def run():
    start = time.monotonic()
    rng = np.random.default_rng(9819)
    counts = {}

    def check(a, category):
        before = a.copy()
        input_unsigned = a.dtype.kind == 'u' and a.itemsize == 8
        for func, width, signed in ((total, 64, not input_unsigned),
                                     (narrow, 8, True),
                                     (unsigned, 64, False)):
            actual = func(a)
            expected = oracle(a, width, signed)
            assert actual == expected, (
                category, a.dtype, a.shape, a.strides,
                func.py_func.__name__, actual, expected)
            counts[category] = counts.get(category, 0) + 1
        np.testing.assert_array_equal(a, before)

    for i in range(100):
        shape = tuple(int(x) for x in rng.integers(2, 7, size=3))
        a = rng.bit_generator.random_raw(math.prod(shape)).reshape(shape)
        if i % 2:
            a = a.view(np.int64)
        a = a.transpose(tuple(int(x) for x in rng.permutation(3)))
        steps = tuple(slice(None, None, int(x))
                      for x in rng.choice([-2, -1, 1, 2], size=3))
        a = a[steps]
        if i % 7 == 0:
            a = np.broadcast_to(a[:1], (3,) + a.shape[1:])
        check(a, 'seeded_rank3')

    rgb = rng.integers(0, 256, size=(19, 23, 3), dtype=np.uint8)
    for crop in (rgb, rgb[1:-1, 2:-2], rgb[::2, 1::2]):
        for axes in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            for steps in ((1, 1, 1), (-1, 1, -1), (-1, -1, -1)):
                a = crop.transpose(axes)
                check(a[tuple(slice(None, None, x) for x in steps)], 'rgb')

    for i in range(80):
        shape = tuple(int(x) for x in rng.integers(1, 5, size=4))
        strides = tuple(int(x) * 8 for x in rng.choice(
            [-97, -11, -3, -1, 0, 1, 3, 11, 97], size=4))
        low = sum(min(0, (n - 1) * s) for n, s in zip(shape, strides))
        high = sum(max(0, (n - 1) * s) for n, s in zip(shape, strides))
        data = rng.bit_generator.random_raw((high - low) // 8 + 1)
        dtype = np.int64 if i % 2 else np.uint64
        a = np.ndarray(shape, dtype=dtype, buffer=data,
                       offset=-low, strides=strides)
        check(a, 'rank4_valid_gap_overlap')

    raw = np.array([0, 2, 128, 255, 0, 4] * 4, dtype=np.uint8).view(np.bool_)
    for a in (raw.reshape(2, 3, 4),
              raw.reshape(2, 3, 4).transpose(2, 0, 1), raw[::-1]):
        check(a, 'noncanonical_bool_bytes')

    unaligned = jit(types.int64(types.Array(
        types.int64, 3, 'A', aligned=False)))(lambda a: np.sum(a))
    buffer = np.empty(24 * 8 + 1, dtype=np.uint8)
    a = np.ndarray((2, 3, 4), dtype=np.int64, buffer=buffer, offset=1)
    a[...] = np.arange(24).reshape(2, 3, 4)
    for b in (a, a.transpose(2, 0, 1), a[::-1, :, ::-1]):
        assert unaligned(b) == oracle(b)
        counts['explicit_unaligned'] = counts.get('explicit_unaligned', 0) + 1

    arbitrary4 = jit(types.int64(types.Array(
        types.int64, 4, 'A')))(lambda a: np.sum(a))
    limits = np.iinfo(np.intp)
    base = np.arange(24, dtype=np.int64)
    singleton_cases = (
        ((1, 2, 3, 4), (limits.min, 32, 64, 8)),
        ((1, 1, 2, 3), (limits.max, limits.min, 8, 16)),
        ((1, 1, 1, 1), (limits.min, limits.max, -8, 0)),
    )
    for shape, strides in singleton_cases:
        a = np.ndarray(shape, dtype=np.int64, buffer=base, strides=strides)
        assert arbitrary4(a) == oracle(a)
        counts['extreme_singleton_stride'] = counts.get(
            'extreme_singleton_stride', 0) + 1

    for axis in range(4):
        shape = [2, 3, 4, 5]
        shape[axis] = 0
        a = np.ndarray(tuple(shape), dtype=np.int64, buffer=base,
                       strides=(limits.min, limits.max, -8, 0))
        assert arbitrary4(a) == 0
        counts['forced_arbitrary_empty'] = counts.get(
            'forced_arbitrary_empty', 0) + 1

    return {
        'result': 'PASS', 'cases': sum(counts.values()), 'categories': counts,
        'seed': 9819, 'seconds': round(time.monotonic() - start, 3),
        'numba': numba.__version__, 'numpy': np.__version__,
        'source': numba.__file__,
        'oracle': 'Python integer logical-element sum modulo output width',
        'timing_note': 'Correctness diagnostic duration, not a benchmark',
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
