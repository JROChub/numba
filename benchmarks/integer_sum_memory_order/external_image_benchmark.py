"""Ad-hoc integer workloads on a pinned public image, not an upstream benchmark.

Run before/after in fresh processes. Decoding, hashing, compilation and oracle
checks are excluded from timing. The image is never resized, tiled or repeated.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform

from integer_benchmark import (
    compiled_image, compiled_sum, identity, image_total, integer_record,
    llvmlite, llvm, measure, numba, np, provenance,
)
from PIL import Image, __version__ as pillow_version


IMAGE_SHA256 = (
    "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
)
SOURCE = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.25.2/skimage/data/astronaut.png"
)
REGISTRY = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.25.2/skimage/data/_registry.py"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Pinned astronaut.png path")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error("repeats must be 3..100")
    file_hash = hashlib.sha256(args.image.read_bytes()).hexdigest()
    if file_hash != IMAGE_SHA256:
        parser.error("image SHA-256 does not match the pinned public dataset")
    with Image.open(args.image) as source:
        if source.mode != "RGB" or source.size != (512, 512):
            parser.error("expected the original 512 x 512 RGB image")
        pixels = np.array(source)
    if pixels.dtype != np.uint8 or pixels.shape != (512, 512, 3):
        raise AssertionError("unexpected decoded image representation")
    input_identity = identity(pixels)
    layouts = {
        "channel_first_total": pixels.transpose(2, 0, 1),
        "cropped_reversed_total":
            pixels[32:-32, 32:-32, ::-1].transpose(2, 0, 1),
        "dark_corrected_roi_total": pixels,
    }
    rows = []
    for name, array in layouts.items():
        pipeline = name == "dark_corrected_roi_total"
        producer = image_total if pipeline else np.sum
        compiled = compiled_image if pipeline else compiled_sum
        row = measure(name, array, producer, compiled, args.repeats,
                      image=pipeline)
        oracle_values = array[:, 1:-1, ::-1] if pipeline else array
        oracle = (sum(max(int(value) - 64, 0) for value in oracle_values.flat)
                  if pipeline else sum(map(int, oracle_values.flat)))
        if any(int(row[key]["value"]) != oracle for key in (
                "numba_result", "numpy_default_result",
                "numpy_same_accumulator_reference")):
            raise AssertionError(f"{name}: full Python-integer oracle failed")
        row.pop("synthetic_image_pipeline")
        row["integer_image_pipeline"] = pipeline
        row["input_is_view_of_decoded_image"] = bool(
            np.shares_memory(array, pixels))
        row["full_python_integer_oracle"] = {
            "elements": oracle_values.size,
            "result": integer_record(oracle, np.dtype("uint64")),
            "accepted": True, "timed": False,
        }
        rows.append(row)
    if identity(pixels) != input_identity:
        raise AssertionError("decoded image changed during execution")
    print(json.dumps({
        "schema": "numba-public-image-reduction-benchmark.v1",
        "label": args.label,
        "measured_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "dataset": {"name": "scikit-image astronaut, NASA public image",
                    "source_url": SOURCE, "checksum_registry_url": REGISTRY,
                    "file_sha256": file_hash, "decoded_sha256": input_identity,
                    "shape": list(pixels.shape), "dtype": pixels.dtype.str,
                    "public_domain_source": "https://scikit-image.org/docs/"
                    "stable/api/skimage.data.html#skimage.data.astronaut",
                    "resized_or_repeated": False},
        "workload_origin": "ad-hoc workloads on a real public dataset; "
        "not an upstream benchmark or application-level acceleration claim",
        "dark_correction": "crop columns 1:-1, reverse channels, transpose "
        "to channel-first, subtract int32(64), clamp at zero, integer sum",
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "numba": numba.__version__,
                     "llvmlite": llvmlite.__version__,
                     "pillow": pillow_version},
        "platform": platform.platform(), "machine": platform.machine(),
        "llvm_cpu": llvm.get_host_cpu_name(),
        "llvm_version": llvm.llvm_version_info,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity") else None,
        "numba_threads": numba.get_num_threads(), "parallel_jit": False,
        "provenance": provenance(), "repeats": args.repeats,
        "jit_compilation_excluded": True, "decoding_excluded": True,
        "paired_order_alternated": True, "input_unchanged": True,
        "measurements": rows,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
