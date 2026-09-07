"""Synthetic integer sum qualification; run before/after in fresh processes.
Compilation is excluded. Small cases use a Python modular-integer oracle;
large cases use NumPy with the compiled accumulator dtype. No external data.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

os.environ["NUMBA_NUM_THREADS"] = "2"

import llvmlite
from llvmlite import binding as llvm
import numba
import numpy as np


def image_total(frames):
    roi = frames[:, 1:-1, ::-1].transpose(2, 0, 1)
    corrected = np.maximum(roi - np.int32(64), np.int32(0))
    return np.sum(corrected)


@numba.jit
def compiled_sum(values):
    return np.sum(values)


compiled_image = numba.jit(image_total)


def identity(array):
    description = json.dumps([array.dtype.str, list(array.shape)])
    digest = hashlib.sha256(description.encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def integer_record(value, dtype):
    width = dtype.itemsize * 8
    return {"dtype": dtype.str, "value": str(int(value)),
            "word_hex": f"{int(value) % (1 << width):0{width // 4}x}"}


def modular_oracle(values, dtype):
    width = dtype.itemsize * 8
    total = sum(map(int, values.flat)) % (1 << width)
    if dtype.kind == "i" and total >= 1 << (width - 1):
        total -= 1 << width
    return total


def measure(name, array, producer, compiled, repeats, image=False):
    before = identity(array)
    actual = compiled(array)
    signature = (numba.typeof(array),)
    return_type = compiled.overloads[signature].signature.return_type
    accumulator = np.dtype(str(return_type))
    expected = producer(array)
    reference_values = (np.maximum(
        array[:, 1:-1, ::-1].transpose(2, 0, 1) - np.int32(64), 0)
        if image else array)
    reference = np.sum(reference_values, dtype=accumulator)
    if int(actual) != int(reference):
        raise AssertionError(f"{name}: compiled result differs from reference")
    sample = array[tuple(slice(0, min(4, size)) for size in array.shape)]
    oracle_values = sample
    if image:
        oracle_values = np.maximum(
            sample[:, 1:-1, ::-1].transpose(2, 0, 1) - np.int32(64), 0)
    oracle = modular_oracle(oracle_values, accumulator)
    if int(compiled(sample)) != oracle:
        raise AssertionError(f"{name}: bounded integer oracle mismatch")
    functions = {"numpy": producer, "numba": compiled}
    samples = {key: [] for key in functions}
    for function in functions.values():
        function(array)
    for repeat in range(repeats):
        order = ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy")
        for key in order:
            start = time.perf_counter_ns()
            value = functions[key](array)
            elapsed = time.perf_counter_ns() - start
            target = expected if key == "numpy" else reference
            if int(value) != int(target):
                raise AssertionError(f"{name}/{key}: result changed")
            samples[key].append(elapsed)
    if identity(array) != before:
        raise AssertionError(f"{name}: input changed")
    return {
        "case": name, "synthetic_image_pipeline": image,
        "shape": list(array.shape), "strides_bytes": list(array.strides),
        "input_dtype": array.dtype.str, "logical_bytes": array.nbytes,
        "input_sha256": before, "input_unchanged": True,
        "numpy_default_result": integer_record(
            expected, np.asarray(expected).dtype),
        "numba_result": integer_record(actual, accumulator),
        "numpy_same_accumulator_reference": integer_record(
            reference, accumulator),
        "matches_numpy_default": int(actual) == int(expected),
        "small_oracle": {"elements": oracle_values.size,
                         "result": integer_record(oracle, accumulator),
                         "accepted": True},
        "samples_ns": samples,
        "median_ns": {key: statistics.median(value)
                      for key, value in samples.items()},
        "compiled_signature": str(compiled.overloads[signature].signature),
    }


def provenance():
    directory = Path(numba.__file__).resolve().parent
    def git(*args):
        result = subprocess.run(["git", "-C", str(directory), *args],
                                capture_output=True, timeout=10)
        return result.stdout if result.returncode == 0 else None
    revision, diff = git("rev-parse", "HEAD"), git("diff", "--binary", "HEAD")
    source = directory / "np" / "arraymath.py"
    return {"module_path": str(directory),
            "git_revision": revision.decode().strip() if revision else None,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()
            if diff is not None else None,
            "arraymath_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 4 <= args.size <= 256 or not 3 <= args.repeats <= 100:
        parser.error("size must be 4..256 and repeats must be 3..100")
    rows = []
    for dtype in (np.int64, np.uint64, np.int32):
        rng = np.random.default_rng(123)
        limits = np.iinfo(dtype)
        base = rng.integers(limits.min, limits.max, (args.size,) * 3,
                            dtype=dtype, endpoint=True)
        layouts = {"c": base, "fortran": np.asfortranarray(base),
                   "transpose_2_0_1": base.transpose(2, 0, 1),
                   "sliced": base[:, ::2, ::2],
                   "negative_stride": base[::-1, :, ::-1],
                   "zero_stride": np.broadcast_to(base[:1, :, :], base.shape)}
        for layout, array in layouts.items():
            rows.append(measure(f"{np.dtype(dtype).name}/{layout}", array,
                                np.sum, compiled_sum, args.repeats))
    frames = np.random.default_rng(123).integers(
        0, 4096, (args.size,) * 3, dtype=np.int32)
    rows.append(measure("synthetic_image/dark_corrected_roi_total", frames,
                        image_total, compiled_image, args.repeats, image=True))
    print(json.dumps({
        "schema": "numba-integer-reduction-benchmark.v1", "label": args.label,
        "measured_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "numba": numba.__version__,
                     "llvmlite": llvmlite.__version__},
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else None,
        "llvm_cpu": llvm.get_host_cpu_name(),
        "llvm_version": llvm.llvm_version_info,
        "numba_threads": numba.get_num_threads(), "parallel_jit": False,
        "provenance": provenance(), "seed": 123, "size": args.size,
        "repeats": args.repeats, "jit_compilation_excluded": True,
        "paired_order_alternated": True,
        "data_origin": "synthetic; not external",
        "image_workload": "int32 12-bit crop, dark subtraction, clamp, sum",
        "measurements": rows,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
