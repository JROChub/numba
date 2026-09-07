"""Read-only qualification of Numba issue 9819; no optimization is applied."""

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import time

import llvmlite
import numba
import numpy as np


@numba.jit
def compiled_sum(values):
    return np.sum(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.size <= 256 or not 3 <= args.repeats <= 100:
        parser.error("size must be 1..256 and repeats must be 3..100")

    values = np.random.default_rng(123).normal(size=(args.size,) * 3)
    layouts = {
        "c": values,
        "transpose_2_0_1": values.transpose(2, 0, 1),
        "fortran": np.asfortranarray(values),
        "negative_stride": values[::-1, :, :],
    }
    rows = []
    for name, array in layouts.items():
        expected = float(np.sum(array))
        actual = float(compiled_sum(array))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        functions = {"numpy": np.sum, "numba": compiled_sum}
        samples = {key: [] for key in functions}
        for repeat in range(args.repeats):
            order = ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy")
            for key in order:
                start = time.perf_counter_ns()
                result = float(functions[key](array))
                elapsed = time.perf_counter_ns() - start
                if result.hex() != (expected if key == "numpy" else actual).hex():
                    raise RuntimeError(f"{name}/{key}: result changed between runs")
                samples[key].append(elapsed)
        medians = {key: statistics.median(ns) for key, ns in samples.items()}
        rows.append({
            "layout": name,
            "shape": list(array.shape),
            "strides_bytes": list(array.strides),
            "numpy_result_hex": expected.hex(),
            "numba_result_hex": actual.hex(),
            "bit_identical": actual.hex() == expected.hex(),
            "allclose_rtol": 1e-12,
            "allclose_atol": 1e-12,
            "allclose_passed": True,
            "samples_ns": samples,
            "median_ns": medians,
            "numba_over_numpy_median_ratio": medians["numba"] / medians["numpy"],
        })
    print(json.dumps({
        "schema": "mfenx-opportunity-baseline-v1",
        "issue": "https://github.com/numba/numba/issues/9819",
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "unmodified installed-version baseline; not an achieved speedup",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
        },
        "seed": 123,
        "input_sha256_c_order": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "jit_compilation_excluded": True,
        "paired_order_alternated": True,
        "repeats": args.repeats,
        "measurements": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
