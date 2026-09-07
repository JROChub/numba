# Reproduce the integer-sum comparison

The recorded experiment used Linux x86-64, CPython 3.13.12, NumPy 2.3.5,
llvmlite 0.50.0dev2 (LLVM 22.1.0), GCC 15.2.0 and Pillow 12.2.0. LLVM identified
the host CPU as `sandybridge`. These are historical environment details, not a
claim that another machine will reproduce the same timings.

The upstream base is `b6168b053d8d7f22a8c7986f073cceb3c124e838`; the implementation
commit is `8e4401c86d4c57484bc7efb7e82df4574bd1dcdb`. Measurements were
collected before committing, with both checkouts at the base revision and the
patch applied only to one. This explains the recorded `.dirty` version suffix.
The measured patched `numba/np/arraymath.py` SHA-256 is
`35b24b313abd518331c13912a27f17bec19815a86c5f436fcefde8fe51c101db`.
The baseline file SHA-256 is
`9267439fad1446287eab25e8fe1a0529c0d89c6852e40300b6415fed5b0b04d6`.

Publication copies standardize descriptive metadata and redact local filesystem
paths for privacy. Numerical samples, input identities, source digests, failures
and test outcomes are unchanged. Original records remain with the author.
The submitted implementation is byte-identical to the measured source.

## Isolated setup

The commands below use Bash on Linux, an installed CPython 3.13 interpreter,
Git, a C compiler, development headers, `curl`, `sha256sum` and `taskset`.
They create a new temporary workspace without changing system packages. Run the
blocks in order in the same Bash session; stop if a command fails.

```sh
set -euo pipefail
numba_work=$(mktemp -d)
git clone --branch evidence/integer-sum-memory-order \
  https://github.com/JROChub/numba.git "$numba_work/repository"
numba_evidence="$numba_work/repository/benchmarks/integer_sum_memory_order"
numba_base=b6168b053d8d7f22a8c7986f073cceb3c124e838
numba_patch=8e4401c86d4c57484bc7efb7e82df4574bd1dcdb
git -C "$numba_work/repository" worktree add --detach \
  "$numba_work/baseline" "$numba_base"
git -C "$numba_work/repository" worktree add --detach \
  "$numba_work/patched" "$numba_base"
git -C "$numba_work/repository" diff "$numba_base" "$numba_patch" | \
  git -C "$numba_work/patched" apply
python3.13 -m venv "$numba_work/venv"
numba_python="$numba_work/venv/bin/python"
"$numba_python" -m pip install \
  numpy==2.3.5 Pillow==12.2.0 setuptools wheel \
  flake8==7.3.0 pycodestyle==2.14.0 pyflakes==3.4.0 mccabe==0.7.0
"$numba_python" -m pip install \
  'https://api.anaconda.org/download/numba/llvmlite/0.50.0.dev2/llvmlite-0.50.0.dev2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
(cd "$numba_work/baseline" && \
  "$numba_python" setup.py build_ext --inplace -j 2)
(cd "$numba_work/patched" && \
  "$numba_python" setup.py build_ext --inplace -j 2)
sha256sum "$numba_work/baseline/numba/np/arraymath.py" \
  "$numba_work/patched/numba/np/arraymath.py"
```

That official Numba-channel llvmlite wheel is specific to CPython 3.13 and
Linux x86-64. Other architectures require a compatible build, and must be
reported as a different environment. The commands are a portable setup recipe,
not a fully locked reproducible build: setuptools/wheel and system tooling are
not pinned here. Historically the two checkouts shared identical built
C-extension binaries; the patch changes Python LLVM-generation code only.

## Correctness and lint

Run from the patched source so imports cannot silently select system Numba:

```sh
(cd "$numba_work/patched" && PYTHONPATH="$numba_work/patched" \
  "$numba_python" -m numba.runtests -m 2 \
  numba.tests.test_array_reductions numba.tests.test_array_methods -v)
(cd "$numba_work/patched" && "$numba_python" -m flake8 --jobs 1 \
  numba/np/arraymath.py numba/tests/test_array_reductions.py)
PYTHONPATH="$numba_work/patched" "$numba_python" -B \
  "$numba_evidence/coalescing_review_probe.py"
```

The recorded final run completed 539 tests: 536 passed and three existing
NumPy-2-incompatible tests skipped. The separate Python-integer oracle checked
640 cases. These are differential validation results, not an external audit.
Repository lint excludes the legacy reduction test file; the
recorded explicit comparison using `--exclude=` found 85 existing diagnostics
in both versions and zero new ones. See the published test and lint records.

## Fresh benchmark trials

Download the original public image and verify its identity. No resizing,
replication or tiling is part of this workload.

```sh
curl --fail --location \
  https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png \
  --output "$numba_work/astronaut.png"
sha256sum "$numba_work/astronaut.png"
```

Expected SHA-256:
`88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`.
The image harness also checks this before execution.

Wait for compilation and test jobs to finish before timing. Historical trials
used CPU 0; check available affinity with `taskset -pc $$`. If CPU 0 is unavailable,
select one available CPU consistently and record that difference. The historical
summary validator intentionally requires affinity `[0]`.

Keep fresh results separate from published history. Copy only the three
benchmark/comparison scripts into a new run directory; the image script imports
the integer harness from the same directory.

```sh
numba_runs="$numba_work/new-results"
mkdir "$numba_runs"
cp "$numba_evidence/integer_benchmark.py" \
  "$numba_evidence/external_image_benchmark.py" \
  "$numba_evidence/summarize_results.py" "$numba_runs/"
numba_cpu=0
for numba_group in integer image; do
  for numba_trial in before-1 after-2 after-3 before-4; do
    case "$numba_trial" in
      before-*) numba_source="$numba_work/baseline" ;;
      after-*) numba_source="$numba_work/patched" ;;
    esac
    if [ "$numba_group" = integer ]; then
      PYTHONPATH="$numba_source" taskset -c "$numba_cpu" \
        "$numba_python" -B "$numba_runs/integer_benchmark.py" \
        --label "$numba_trial" --size 128 --repeats 31 \
        > "$numba_runs/controlled-integer-$numba_trial.json"
    else
      PYTHONPATH="$numba_source" taskset -c "$numba_cpu" \
        "$numba_python" -B "$numba_runs/external_image_benchmark.py" \
        "$numba_work/astronaut.png" --label "$numba_trial" --repeats 31 \
        > "$numba_runs/controlled-image-$numba_trial.json"
    fi
  done
done
"$numba_python" "$numba_runs/summarize_results.py" \
  --source "$numba_work/patched/numba/np/arraymath.py" \
  > "$numba_runs/controlled-summary.json"
```

Each process records 31 paired samples per case, alternating NumPy/Numba order.
The four fresh processes alternate baseline, patched, patched, baseline. The
summary uses the median of the two process medians per build, and reports
**unpatched Numba time divided by patched Numba time**, not speedup over NumPy.
All 22 cases must remain in the comparison, including regressions. Compilation,
image decoding, hashing and correctness checks are outside timing; the numerical
dark-correction/clamp/reduction pipeline is inside its timed case. Functions do
not use parallel JIT even though the harness sets `NUMBA_NUM_THREADS=2`.

To validate the published historical trial files without rerunning benchmarks:

```sh
"$numba_python" "$numba_evidence/summarize_results.py" \
  --source "$numba_work/patched/numba/np/arraymath.py"
```

This checks recorded samples, identities and source compatibility; it is not
another-machine performance reproduction. Float/complex reductions remain on
the original execution path. These measurements concern Numba, not QQfenx.

## Publication status

The code and evidence are published in the author's fork, and
[PR #10820](https://github.com/numba/numba/pull/10820) tracks upstream submission.
The contributor reported completing an outside review and authorized submission.
Another-machine reproduction remains pending; submission is not endorsement.
